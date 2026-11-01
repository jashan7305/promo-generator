"""The local intelligence and rendering pipeline for Promo Generator.

The original implementation used shared temporary filenames and selected the
nearest three transcript lines.  This module keeps the local/offline model
approach, but makes each generation isolated, confidence-aware, diverse, and
safe to run concurrently.
"""

from __future__ import annotations

import base64
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations
from urllib.parse import parse_qs, urlparse
import uuid
import wave
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Iterable, cast

import ffmpeg
import numpy as np
import requests


BASE_DIR = Path(__file__).resolve().parents[1]
STORAGE_DIR = Path(os.getenv("PROMO_STORAGE_DIR", BASE_DIR / "storage"))
GENERATED_DIR = STORAGE_DIR / "generated"
MODEL_DIR = Path(os.getenv("PROMO_MODEL_DIR", BASE_DIR / "models"))
WHISPER_MODEL_NAME = os.getenv("PROMO_WHISPER_MODEL", "base")
EMBEDDING_MODEL_NAME = os.getenv("PROMO_EMBEDDING_MODEL", "Xenova/e5-small-v2")
VISION_API_KEY = os.getenv("PROMO_VISION_API_KEY") or os.getenv("OPENAI_API_KEY")
VISION_API_MODEL = os.getenv("PROMO_VISION_MODEL", "gpt-4.1-mini")
GEMINI_API_KEY = os.getenv("PROMO_GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("PROMO_GEMINI_MODEL", "gemini-3.8-flash")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_MODEL = os.getenv("PROMO_ELEVENLABS_MODEL", "scribe_v2")
API_TRANSCRIBE_MODEL = os.getenv("PROMO_API_TRANSCRIBE_MODEL", "whisper-1")
API_TIMEOUT_SECONDS = int(os.getenv("PROMO_API_TIMEOUT_SECONDS", "120"))
YOUTUBE_MAX_DURATION = int(os.getenv("PROMO_YOUTUBE_MAX_DURATION", "7200"))
MAX_REPAIR_ITERATIONS = max(1, min(5, int(os.getenv("PROMO_MAX_REPAIR_ITERATIONS", "3"))))
API_TRANSCRIBE_MAX_CHUNK_BYTES = 18 * 1024 * 1024

GENERATED_DIR.mkdir(parents=True, exist_ok=True)

_model_lock = Lock()
_transcription_model = None
_embedding_tokenizer = None
_embedding_session = None

ProgressCallback = Callable[[float, str], None]


def _progress(callback: ProgressCallback | None, value: float, stage: str) -> None:
    if callback:
        callback(round(max(0.0, min(1.0, value)), 3), stage)


def _get_transcription_model():
    global _transcription_model
    if _transcription_model is None:
        with _model_lock:
            if _transcription_model is None:
                from faster_whisper import WhisperModel

                _transcription_model = WhisperModel(
                    WHISPER_MODEL_NAME,
                    device=os.getenv("PROMO_WHISPER_DEVICE", "cpu"),
                    compute_type=os.getenv("PROMO_WHISPER_COMPUTE_TYPE", "int8"),
                )
    return _transcription_model


def _get_embedding_model():
    global _embedding_tokenizer, _embedding_session
    if _embedding_tokenizer is None or _embedding_session is None:
        with _model_lock:
            if _embedding_tokenizer is None or _embedding_session is None:
                from huggingface_hub import snapshot_download
                from onnxruntime import InferenceSession
                from transformers import AutoTokenizer

                local_dir = MODEL_DIR / "e5-small-v2"
                model_path = local_dir / "onnx" / "model.onnx"
                if not model_path.exists():
                    snapshot_download(EMBEDDING_MODEL_NAME, local_dir=str(local_dir))
                _embedding_tokenizer = AutoTokenizer.from_pretrained(str(local_dir))
                _embedding_session = InferenceSession(
                    str(model_path), providers=["CPUExecutionProvider"]
                )
    return _embedding_tokenizer, _embedding_session


def _normalise(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-12)


def embed_texts(texts: str | Iterable[str]) -> np.ndarray:
    """Embed text with mean pooling, preserving the attention mask.

    The old code averaged token embeddings without the mask, which made
    padding influence similarity.  Correct pooling noticeably improves short
    dialogue ranking.
    """

    if isinstance(texts, str):
        texts = [texts]
    texts = list(texts)
    if not texts:
        return np.empty((0, 384), dtype=np.float32)

    tokenizer, session = _get_embedding_model()
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="np",
    )
    outputs = session.run(None, dict(inputs))
    token_embeddings = outputs[0]
    mask = inputs["attention_mask"][..., None].astype(np.float32)
    pooled = (token_embeddings * mask).sum(axis=1) / np.maximum(mask.sum(axis=1), 1.0)
    return _normalise(pooled)


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9']+", text.lower())
        if len(token) > 2
    }


def _has_repeated_phrase(text: str) -> bool:
    tokens = re.findall(r"[a-z0-9']+", text.lower())
    return any(
        len(tokens) >= width * 2 and tokens[-width:] == tokens[-width * 2 : -width]
        for width in range(1, max(2, len(tokens) // 2 + 1))
    )


def _is_transcription_artifact(text: str) -> bool:
    normalized = re.sub(r"[^a-z0-9']+", " ", text.lower()).strip()
    return normalized in {
        "we're takes",
        "we re takes",
        "thanks for watching",
        "thank you for watching",
        "please subscribe",
        "subscribe to our channel",
    }


def _lexical_similarity(text: str, theme: str) -> float:
    theme_tokens = _tokens(theme)
    text_tokens = _tokens(text)
    if not theme_tokens or not text_tokens:
        return 0.0
    return len(theme_tokens & text_tokens) / len(theme_tokens)


def _safe_confidence(dialogue: dict) -> float:
    value = dialogue.get("confidence", 0.75)
    try:
        return max(0.05, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.75


def _quality_score(text: str) -> float:
    words = _tokens(text)
    if not words:
        return 0.05
    length_score = min(1.0, len(words) / 10)
    complete_score = 1.0 if re.search(r"[.!?…]$", text.strip()) else 0.58
    tokens = re.findall(r"[a-z0-9']+", text.lower())
    unique_ratio = len(set(tokens)) / max(1, len(tokens))
    repeated_phrase = _has_repeated_phrase(text)
    repetition_score = 0.08 if repeated_phrase else unique_ratio
    return round(0.45 * length_score + 0.35 * complete_score + 0.20 * repetition_score, 4)


def _music_match_score(
    candidate: dict,
    music_profile: dict | None,
    target_duration: int,
    video_duration: float | None,
) -> float:
    """Estimate whether a source moment fits the soundtrack's rhythm."""

    if not music_profile or not video_duration or video_duration <= 0:
        return 0.5
    beats = music_profile.get("beats", [])
    music_duration = max(float(music_profile.get("duration", 0.0)), 0.1)
    source_center = (float(candidate.get("start", 0.0)) + float(candidate.get("end", 0.0))) / 2
    target_position = ((source_center / video_duration) * float(target_duration)) % music_duration
    beat_score = 0.5
    if beats:
        distance = min(abs(float(beat) - target_position) for beat in beats)
        beat_score = math.exp(-distance / 0.22)
    profile = music_profile.get("energy", {"values": np.array([]), "window": 0.4})
    music_energy = sample_audio_energy(
        profile,
        target_position,
        min(
            music_duration,
            target_position + max(0.2, float(candidate.get("end", 0.0)) - float(candidate.get("start", 0.0))),
        ),
    )
    video_energy = max(
        float(candidate.get("energy_score", 0.5)),
        float(candidate.get("motion_score", 0.5)),
        float(candidate.get("visual_score", 0.5)),
    )
    energy_score = 1.0 - abs(video_energy - music_energy)
    return round(max(0.0, min(1.0, 0.58 * beat_score + 0.42 * energy_score)), 4)


def score_dialogues(
    dialogues: list[dict],
    theme: str,
    strategy: str = "theme",
    music_profile: dict | None = None,
    target_duration: int = 30,
    video_duration: float | None = None,
) -> list[dict]:
    """Add semantic, narrative, audio, and visual importance scores.

    A theme match alone is not enough for a trailer.  The best line can be a
    low-energy continuation, an incomplete sentence, or a moment with no
    visual movement.  The extra signals let the selector keep the important
    story beat even when its exact words do not match the theme.
    """

    if not dialogues:
        return []

    lexical = np.array([_lexical_similarity(d["text"], theme) for d in dialogues])
    try:
        # E5 models are trained with query/passage prefixes.  Supplying them
        # makes the theme-to-dialogue comparison more faithful than embedding
        # both strings as undifferentiated text.
        dialogue_embeddings = embed_texts([f"passage: {d['text']}" for d in dialogues])
        theme_embedding = embed_texts(f"query: {theme}")[0]
        semantic = np.clip(dialogue_embeddings @ theme_embedding, 0.0, 1.0)
    except Exception:
        # The embedding model is optional at runtime.  A lexical fallback keeps
        # the product usable on a first run with no network/model cache.
        semantic = lexical

    scored = []
    weights = {
        "theme": (0.46, 0.13, 0.09, 0.07, 0.07, 0.06),
        "best-overall": (0.16, 0.04, 0.14, 0.19, 0.18, 0.17),
        "story-arc": (0.31, 0.09, 0.11, 0.09, 0.11, 0.17),
        "visual-montage": (0.07, 0.02, 0.07, 0.27, 0.32, 0.13),
    }.get(strategy, (0.46, 0.13, 0.09, 0.07, 0.07, 0.06))
    for index, dialogue in enumerate(dialogues):
        item = dict(dialogue)
        item["semantic_score"] = float(semantic[index])
        item["lexical_score"] = float(lexical[index])
        item["quality_score"] = _quality_score(dialogue["text"])
        item["coherence_score"] = (
            0.05
            if _has_repeated_phrase(dialogue["text"])
            or _is_transcription_artifact(dialogue["text"])
            or item["quality_score"] < 0.25
            else 1.0
        )
        item["energy_score"] = max(0.0, min(1.0, float(dialogue.get("energy_score", 0.5))))
        item["visual_score"] = max(0.0, min(1.0, float(dialogue.get("visual_score", 0.5))))
        item["motion_score"] = max(0.0, min(1.0, float(dialogue.get("motion_score", 0.5))))
        item["api_score"] = max(0.0, min(1.0, float(dialogue.get("api_score", 0.5))))
        item["music_score"] = _music_match_score(
            dialogue,
            music_profile,
            target_duration,
            video_duration,
        )
        source = str(item.get("source", ""))
        if source in {"visual", "thrill", "opening"}:
            source_bonus = {
                "opening": 0.20,
                "thrill": 0.12,
                "visual": 0.0,
            }.get(source, 0.0)
            item["score"] = round(
                0.32 * item["energy_score"]
                + 0.30 * item["visual_score"]
                + 0.23 * item["motion_score"]
                + 0.15 * item["quality_score"]
                + 0.10 * item["api_score"],
                4,
            )
            item["score"] = round(
                min(1.0, item["score"] + 0.16 * item["music_score"] + source_bonus),
                4,
            )
        else:
            item["score"] = round(
                weights[0] * semantic[index]
                + weights[1] * lexical[index]
                + weights[2] * _safe_confidence(dialogue)
                + weights[3] * item["energy_score"]
                + weights[4] * item["visual_score"]
                + weights[5] * item["quality_score"]
                + 0.12 * item["motion_score"]
                + 0.12 * item["coherence_score"]
                + 0.10 * item["api_score"]
                + 0.12 * item["music_score"],
                4,
            )
        scored.append(item)
    return scored


def _overlap_ratio(first: tuple[float, float], second: tuple[float, float]) -> float:
    overlap = max(0.0, min(first[1], second[1]) - max(first[0], second[0]))
    shorter = max(0.001, min(first[1] - first[0], second[1] - second[0]))
    return overlap / shorter


def _group_dialogue_contexts(
    candidates: list[dict],
    max_gap: float = 2.5,
    max_duration: float = 20.0,
) -> list[dict]:
    """Make selectable dialogue beats out of adjacent complete sentences.

    A transcript sentence is not automatically a usable trailer quote.  The
    line immediately before it can establish who is speaking, and the line
    immediately after it can complete the thought.  Ranking individual
    sentences caused the fitter to keep a technically complete line while
    throwing away the context that made it understandable.  Group only
    non-overlapping transcript sentences separated by a short pause; visual
    and API-only moments remain independent candidates.
    """

    dialogue = [
        candidate
        for candidate in candidates
        if candidate.get("source") not in {"visual", "thrill", "opening", "api"}
        and str(candidate.get("text", "")).strip()
        not in {"Visual beat", "Visual story beat"}
    ]
    other = [candidate for candidate in candidates if candidate not in dialogue]
    normalized: list[dict] = []
    approximate_groups: dict[tuple[float, float, str], dict] = {}
    for candidate in dialogue:
        if candidate.get("timing_approximate"):
            context_start = float(candidate.get("context_start", candidate["start"]))
            context_end = float(candidate.get("context_end", candidate["end"]))
            speaker = candidate.get("speaker_id", candidate.get("speaker"))
            key = (context_start, context_end, str(speaker))
            if key not in approximate_groups:
                item = dict(candidate)
                item["start"] = context_start
                item["end"] = context_end
                item["text"] = str(candidate.get("context_text") or candidate["text"])
                item["context_start"] = context_start
                item["context_end"] = context_end
                item["context_member_count"] = 1
                approximate_groups[key] = item
            continue
        normalized.append(dict(candidate))
    normalized.extend(approximate_groups.values())
    ordered = sorted(normalized, key=lambda item: (float(item["start"]), float(item["end"])))
    groups: list[list[dict]] = []
    current: list[dict] = []

    def flush() -> None:
        if current:
            groups.append(current.copy())
            current.clear()

    for candidate in ordered:
        if not current:
            current.append(candidate)
            continue
        previous = current[-1]
        gap = float(candidate["start"]) - float(previous["end"])
        combined_duration = float(candidate["end"]) - float(current[0]["start"])
        previous_speaker = previous.get("speaker_id", previous.get("speaker"))
        candidate_speaker = candidate.get("speaker_id", candidate.get("speaker"))
        speaker_continues = (
            previous_speaker is None
            or candidate_speaker is None
            or previous_speaker == candidate_speaker
        )
        if (
            speaker_continues
            and len(current) < 2
            and 0.0 <= gap <= max_gap
            and combined_duration <= max_duration
        ):
            current.append(candidate)
        else:
            flush()
            current.append(candidate)
    flush()

    grouped: list[dict] = []
    for group in groups:
        first = dict(group[0])
        if len(group) == 1:
            first["context_member_count"] = 1
            first["context_start"] = float(first.get("context_start", first["start"]))
            first["context_end"] = float(first.get("context_end", first["end"]))
            grouped.append(first)
            continue

        merged = dict(first)
        merged["start"] = round(float(group[0]["start"]), 3)
        merged["end"] = round(float(group[-1]["end"]), 3)
        merged["text"] = " ".join(str(item.get("text", "")).strip() for item in group).strip()
        merged["words"] = [
            word
            for item in group
            for word in item.get("words", [])
        ]
        merged["confidence"] = round(min(_safe_confidence(item) for item in group), 3)
        for key in ("energy_score", "visual_score", "motion_score", "api_score"):
            values = [float(item.get(key, 0.5)) for item in group if key in item]
            if values:
                merged[key] = round(max(values), 4)
        merged["context_member_count"] = len(group)
        merged["timing_approximate"] = any(item.get("timing_approximate", False) for item in group)
        merged["word_timing_available"] = all(item.get("word_timing_available", bool(item.get("words"))) for item in group)
        merged["context_start"] = min(
            float(item.get("context_start", item["start"])) for item in group
        )
        merged["context_end"] = max(
            float(item.get("context_end", item["end"])) for item in group
        )
        grouped.append(merged)

    return sorted(grouped + [dict(item) for item in other], key=lambda item: float(item["start"]))


def build_trailer_structure_plan(
    candidates: list[dict],
    request: dict,
    target_duration: int,
    video_duration: float | None,
    music_profile: dict | None = None,
) -> dict:
    """Plan narrative functions against real candidates, without inventing times.

    Region and event matches are editing preferences, not claims about plot.
    Offline candidates without visual descriptions receive lower confidence.
    """
    prompt = str(request.get("prompt", "")).lower()
    explicit_chronology = bool(re.search(r"\b(chronological|in source order|tell it in order)\b", prompt))
    nonlinear = bool(request.get("nonlinear_order")) or any(
        phrase in prompt
        for phrase in ("rewind", "start near the ending", "start at the end", "back to the beginning", "cold open")
    )
    style = str(request.get("structure_style", ""))
    if explicit_chronology:
        style = "chronological"
    elif nonlinear or re.search(r"\b(thriller|suspense|mystery|shocking opening|mysterious)\b", prompt):
        style = "nonlinear_mystery"
    elif not style:
        if request.get("dialogue_preference") == "prioritize":
            style = "dialogue_progression"
        elif request.get("pacing") == "fast":
            style = "action_escalation"
        elif request.get("ending") == "emotional":
            style = "emotional_progression"
        else:
            style = "cinematic_arc"

    if style == "chronological":
        return {"style": style, "beats": [], "confidence": "preference"}
    if style == "dialogue_progression":
        roles = (
            ["hook", "context", "emotional_payoff"]
            if target_duration <= 30 and request.get("ending") == "emotional"
            else ["hook", "context", "final_sting"]
            if target_duration <= 30
            else ["hook", "context", "exchange", "emotional_payoff"]
        )
    elif style == "action_escalation":
        roles = ["hook", "setup", "escalation", "peak", "final_sting"]
    elif style == "emotional_progression":
        roles = ["hook", "context", "connection", "emotional_payoff"]
    elif style == "nonlinear_mystery":
        roles = (
            ["cold_open", "setup", "final_sting"]
            if target_duration <= 15
            else ["cold_open", "setup", "mystery", "final_sting"]
            if target_duration <= 30
            else ["cold_open", "setup", "mystery", "escalation", "final_sting"]
        )
    else:
        roles = ["hook", "setup", "turn", "payoff"]
    if request.get("ending") == "title":
        roles = [role for role in roles if role not in {"final_sting", "payoff", "emotional_payoff"}]
        roles.append("title")

    duration = max(1.0, float(video_duration or 1.0))
    music_values = np.asarray((music_profile or {}).get("energy", {}).get("values", []), dtype=np.float32)
    music_window = max(0.01, float((music_profile or {}).get("energy", {}).get("window", 0.2)))
    selected: set[int] = set()
    beats = []
    for index, role in enumerate(roles):
        chosen_index = None
        chosen_score = float("-inf")
        chosen_music_target = None
        for candidate_index, item in enumerate(candidates):
            if candidate_index in selected:
                continue
            start = max(0.0, float(item.get("start", 0.0)))
            end = min(duration, float(item.get("end", start)))
            if end <= start or item.get("coherence_score", 1.0) < 0.4:
                continue
            if any(
                _overlap_ratio((start, end), (float(candidates[chosen_index]["start"]), float(candidates[chosen_index]["end"]))) > 0.35
                for chosen_index in selected
            ):
                continue
            position = (start + end) / (2 * duration)
            text = " ".join((str(item.get("text", "")), str(item.get("story_role", "")), str(item.get("visual_description", "")), str(item.get("api_reason", "")))).lower()
            tokens = _tokens(text)
            source = item.get("source")
            event = item.get("event_type")
            dialogue = source not in {"visual", "thrill", "opening", "api"} and item.get("text") not in {"Visual beat", "Visual story beat", "Visual thrill beat"}
            explicit_title = event == "title_card"
            if role == "title" and not explicit_title and position < 0.55:
                continue
            dialogue_emotional_ending = (
                role == "emotional_payoff"
                and request.get("dialogue_preference") == "prioritize"
            )
            if dialogue_emotional_ending and not dialogue:
                continue
            region_score = 0.5
            role_evidence = 0.0
            if role in {"cold_open", "hook"}:
                region_score = position if style == "nonlinear_mystery" else 1.0 - abs(position - 0.55) * 0.5
                role_evidence += 0.22 * float(event in {"reveal", "action", "reaction"})
                role_evidence += 0.12 * float(any(word in tokens for word in {"why", "what", "danger", "secret", "dead", "help"}))
                role_evidence += 0.08 * float(item.get("visual_score", 0.5))
            elif role in {"setup", "context"}:
                region_score = 1.0 - min(1.0, position * 1.15)
                role_evidence += 0.16 * dialogue + 0.12 * float(event == "establishing")
            elif role in {"mystery", "turn", "exchange"}:
                region_score = 1.0 - min(1.0, abs(position - 0.5) * 2.0)
                role_evidence += 0.22 * float(event in {"reveal", "interaction", "reaction"})
                role_evidence += 0.12 * float("?" in str(item.get("text", "")) or any(word in tokens for word in {"secret", "truth", "follow", "who", "why"}))
                role_evidence += 0.08 * dialogue
            elif role in {"escalation", "peak"}:
                region_score = position
                role_evidence += 0.2 * float(event in {"action", "confrontation", "reveal"})
                role_evidence += 0.12 * max(float(item.get("motion_score", 0.0)), float(item.get("energy_score", 0.0)))
            elif role in {"final_sting", "payoff", "emotional_payoff", "title"}:
                region_score = position
                role_evidence += 0.24 * float(event in {"reveal", "reaction"})
                role_evidence += 0.16 * float(explicit_title)
                role_evidence += 0.12 * float(any(word in tokens for word in {"goodbye", "promise", "home", "together", "die", "sorry", "love"}))
            elif role == "connection":
                region_score = 1.0 - abs(position - 0.5)
                role_evidence += 0.2 * dialogue + 0.15 * float(event in {"interaction", "reaction"})
            if role in {"cold_open", "final_sting", "peak", "escalation", "payoff", "emotional_payoff"} and position < 0.55 and not dialogue_emotional_ending:
                continue
            if role in {"setup", "context"} and position > 0.70:
                continue
            if role in {"mystery", "turn", "exchange"} and not 0.22 <= position <= 0.82:
                continue
            music_target = None
            music_fit = 0.0
            if len(music_values):
                phase = index / max(1, len(roles) - 1)
                music_time = phase * min(float(target_duration), len(music_values) * music_window)
                music_target = float(music_values[min(len(music_values) - 1, int(music_time / music_window))])
                candidate_energy = 0.55 * float(item.get("motion_score", 0.0)) + 0.45 * float(item.get("energy_score", 0.0))
                music_fit = 0.12 * (1.0 - abs(candidate_energy - music_target))
            score = 0.62 * float(item.get("score", 0.0)) + 0.30 * region_score + role_evidence + music_fit
            if role == "final_sting" and explicit_title:
                score += 0.4
            if score > chosen_score:
                chosen_index, chosen_score = candidate_index, score
                chosen_music_target = music_target
        if chosen_index is not None:
            selected.add(chosen_index)
            chosen = candidates[chosen_index]
            actual_role = "final_sting" if role == "title" and chosen.get("event_type") != "title_card" else role
            region = "late" if role in {"cold_open", "final_sting", "peak", "escalation", "payoff", "emotional_payoff"} else ("early" if role in {"setup", "context"} else "middle")
            confidence = 0.9 if chosen.get("visual_description") or chosen.get("story_role") else (0.68 if chosen.get("source") not in {"visual", "thrill", "opening", "api"} else 0.48)
            beats.append({
                "role": actual_role,
                "requested_role": "title" if role == "title" else None,
                "purpose": "final sting fallback" if role == "title" and actual_role != "title" else role.replace("_", " "),
                "emotional_effect": "intrigue" if role in {"cold_open", "mystery", "hook"} else "rising stakes" if role in {"escalation", "peak"} else "resolution",
                "preferred_region": region,
                "approx_duration": round(min(float(target_duration) / max(1, len(roles)), max(0.5, float(chosen["end"]) - float(chosen["start"]))), 2),
                "candidate_index": chosen_index,
                "candidate_source_interval": [float(chosen["start"]), float(chosen["end"])],
                "confidence": confidence,
                "music_energy_target": round(chosen_music_target, 3) if chosen_music_target is not None else None,
                "status": "fallback" if role == "title" and actual_role != "title" else "selected",
                "rationale": "No verified title card was available; used a late candidate instead." if role == "title" and actual_role != "title" else "Matched available candidate evidence and preferred source region.",
            })
        else:
            if role == "title":
                fallback = max(
                    ((i, item) for i, item in enumerate(candidates) if i not in selected and item.get("event_type") != "title_card"),
                    key=lambda pair: float(pair[1].get("score", 0.0)) + 0.25 * float(pair[1].get("start", 0.0)) / duration,
                    default=None,
                )
                if fallback:
                    chosen_index, chosen = fallback
                    selected.add(chosen_index)
                    beats.append({"role": "final_sting", "requested_role": "title", "purpose": "final sting fallback", "preferred_region": "late", "approx_duration": round(min(float(target_duration) / max(1, len(roles)), float(chosen["end"]) - float(chosen["start"])), 2), "candidate_index": chosen_index, "candidate_source_interval": [float(chosen["start"]), float(chosen["end"])], "confidence": 0.45, "status": "fallback", "rationale": "No verified title card was available; used a late candidate instead."})
                    continue
            beats.append({"role": role, "purpose": role.replace("_", " "), "preferred_region": "unknown", "approx_duration": 0.0, "candidate_index": None, "confidence": 0.0, "status": "unavailable", "rationale": "No valid candidate supports this beat."})
    # A late-source cold open followed by early setup is an intentional rewind;
    # retain the plan's beat sequence rather than sorting these source times.
    return {"style": style, "beats": beats, "confidence": "estimated"}


def select_highlights(
    dialogues: list[dict],
    theme: str,
    target_duration: int = 30,
    energy: str = "balanced",
    strategy: str = "theme",
    video_duration: float | None = None,
    music_profile: dict | None = None,
    request: dict | None = None,
) -> list[dict]:
    """Select high-signal, non-repeating dialogue moments.

    Selection is relevance-first but deliberately avoids a promo made from
    three adjacent lines or three nearly identical sentences.  The returned
    moments are put back into chronological order for a coherent narrative.
    """

    scored = score_dialogues(
        _group_dialogue_contexts(dialogues),
        theme,
        strategy,
        music_profile=music_profile,
        target_duration=target_duration,
        video_duration=video_duration,
    )
    if not scored:
        return []

    clip_budget = {15: 3, 30: 4, 45: 5, 60: 6, 90: 7, 120: 8}.get(target_duration, 4)
    if energy == "cinematic":
        clip_budget = max(3, clip_budget - 1)
    elif energy == "high-impact":
        clip_budget += 1 + (1 if target_duration <= 30 else 0)
    if strategy == "visual-montage" and energy != "high-impact":
        # Dialogue-free stories need enough distinct visual beats to satisfy
        # the duration target without stretching a handful of shots into long
        # chronological blocks.
        clip_budget += 1

    ranked = sorted(scored, key=lambda item: item["score"], reverse=True)
    chosen: list[dict] = []
    structure_plan = build_trailer_structure_plan(
        scored, request or {}, target_duration, video_duration, music_profile
    ) if request else None

    def acceptable(candidate: dict) -> bool:
        candidate_window = (
            float(candidate.get("context_start", candidate["start"])),
            float(candidate.get("context_end", candidate["end"])),
        ) if candidate.get("timing_approximate") else (float(candidate["start"]), float(candidate["end"]))
        if any(
            _overlap_ratio(
                candidate_window,
                (
                    float(existing.get("context_start", existing["start"])),
                    float(existing.get("context_end", existing["end"])),
                ) if existing.get("timing_approximate") else (float(existing["start"]), float(existing["end"])),
            ) > 0.35
            for existing in chosen
        ):
            return False
        candidate_words = _tokens(candidate["text"])
        for existing in chosen:
            existing_words = _tokens(existing["text"])
            candidate_is_visual = candidate.get("source") in {"visual", "thrill", "opening"}
            existing_is_visual = existing.get("source") in {"visual", "thrill", "opening"}
            if (
                candidate_is_visual
                and existing_is_visual
                and abs(float(candidate["start"]) - float(existing["start"])) < 4.0
            ):
                return False
            if not candidate_is_visual and not existing_is_visual and candidate_words and existing_words:
                similarity = len(candidate_words & existing_words) / max(1, len(candidate_words | existing_words))
                if similarity > 0.62:
                    return False
        if candidate.get("coherence_score", 1.0) < 0.4:
            return False
        return True

    def choose(candidate: dict) -> None:
        if len(chosen) < clip_budget and acceptable(candidate):
            chosen.append(candidate)

    if structure_plan and structure_plan.get("style") != "chronological":
        # Assign narrative functions before the legacy coverage/relevance fill.
        # The plan contains indices into the scored, timestamp-validated pool.
        for beat in structure_plan["beats"]:
            candidate_index = beat.get("candidate_index")
            if candidate_index is None or len(chosen) >= clip_budget:
                continue
            candidate = dict(scored[candidate_index])
            if not acceptable(candidate):
                continue
            candidate["timeline_role"] = beat["role"]
            candidate["role_rationale"] = beat["rationale"]
            chosen.append(candidate)

    opening_candidates = [item for item in ranked if item.get("source") == "opening"]
    if opening_candidates:
        # The first production/company card or opening studio image is a
        # trailer convention worth preserving, not a disposable highlight.
        choose(min(opening_candidates, key=lambda item: item["start"]))

    thrill_candidates = [
        item for item in ranked
        if item.get("source") == "thrill"
        and float(item["start"]) > 0.5
    ]
    thrill_count = 1 if target_duration <= 15 else 2
    selected_thrills = 0
    for candidate in sorted(thrill_candidates, key=lambda item: item["score"], reverse=True):
        before = len(chosen)
        choose(candidate)
        if len(chosen) > before:
            selected_thrills += 1
        if selected_thrills >= thrill_count:
            break

    # Reserve one moment in each third of the trailer.  This prevents a dense
    # middle section from crowding out the ending or opening reveal.
    if video_duration and video_duration > 12 and clip_budget >= 3:
        opening_candidates = [
            candidate
            for candidate in ranked
            if float(candidate["start"]) <= video_duration * 0.18
            and (candidate.get("quality_score", 0) >= 0.82 or candidate.get("semantic_score", 0) >= 0.68)
        ]
        if opening_candidates:
            # The first intelligible beat is more valuable than a higher
            # scoring line that arrives halfway through the first act.
            choose(min(opening_candidates, key=lambda item: item["start"]))

        for lower, upper in (
            (0.0, video_duration / 3),
            (video_duration / 3, video_duration * 2 / 3),
            (video_duration * 2 / 3, video_duration + 0.01),
        ):
            for candidate in ranked:
                if (
                    candidate.get("source") not in {"visual", "thrill", "opening"}
                    and lower <= float(candidate["start"]) < upper
                ):
                    before = len(chosen)
                    choose(candidate)
                    if len(chosen) > before:
                        break

    for candidate in ranked:
        if len(chosen) >= clip_budget:
            break
        choose(candidate)

    if len(chosen) < min(2, len(ranked)):
        fallback = [item for item in ranked if item.get("coherence_score", 1.0) >= 0.4]
        chosen = fallback[: min(clip_budget, len(fallback))]
    if structure_plan and structure_plan.get("style") != "chronological":
        planned_roles = {beat["role"] for beat in structure_plan["beats"]}
        terminal_role = "final_sting" if "final_sting" in planned_roles else (
            "payoff" if "payoff" in planned_roles else "emotional_payoff" if "emotional_payoff" in planned_roles else None
        )
        if terminal_role and not any(item.get("timeline_role") == terminal_role for item in chosen):
            assigned = [item for item in chosen if item.get("timeline_role")]
            alternatives = [
                item for item in chosen
                if not item.get("timeline_role")
                and float(item["start"]) / max(1.0, float(video_duration or 1.0)) >= 0.55
                and not (terminal_role == "emotional_payoff" and request.get("dialogue_preference") == "prioritize" and item.get("source") in {"visual", "thrill", "opening", "api"})
                and not any(_overlap_ratio((float(item["start"]), float(item["end"])), (float(other["start"]), float(other["end"]))) > 0.35 for other in assigned)
            ]
            if alternatives:
                candidate = max(alternatives, key=lambda item: float(item.get("score", 0.0)) + 0.2 * float(item["start"]) / max(1.0, float(video_duration or 1.0)))
                candidate["timeline_role"] = terminal_role
                candidate["role_rationale"] = "Fallback ending selected from a distinct late-source candidate after the preferred ending candidate conflicted."
        role_order = {beat["role"]: index for index, beat in enumerate(structure_plan["beats"])}
        ordered = sorted(
            chosen,
            key=lambda item: (
                role_order.get(item.get("timeline_role"), len(role_order)),
                float(item["start"]),
            ),
        )
        for index, item in enumerate(ordered):
            item["timeline_order"] = index
        return ordered
    return sorted(chosen, key=lambda item: item["start"])


def merge_dialogue_segments(dialogues: list[dict]) -> list[dict]:
    """Turn Whisper fragments into complete dialogue beats.

    Whisper often emits a final two-word continuation as a separate segment.
    Treating that fragment as a standalone highlight creates cuts like
    ``"we can"`` followed by ``"do."``.  Merge only close/obviously
    unfinished fragments, while keeping independent sentences separate.
    """

    if not dialogues:
        return []
    merged: list[dict] = []
    current = dict(sorted(dialogues, key=lambda item: item["start"])[0])
    for following in sorted(dialogues, key=lambda item: item["start"])[1:]:
        gap = float(following["start"]) - float(current["end"])
        current_text = current["text"].strip()
        following_words = _tokens(following["text"])
        unfinished = not re.search(r"[.!?…]$", current_text)
        tiny_continuation = len(following_words) <= 3
        starts_lowercase = following["text"].strip()[:1].islower()
        current_speaker = current.get("speaker_id", current.get("speaker"))
        following_speaker = following.get("speaker_id", following.get("speaker"))
        speaker_continues = (
            current_speaker is None
            or following_speaker is None
            or current_speaker == following_speaker
        )
        should_merge = (
            speaker_continues
            and 0.0 <= gap <= 4.0
            and unfinished
            and (gap <= 0.85 or tiny_continuation or starts_lowercase)
        )
        if should_merge:
            current["end"] = max(float(current["end"]), float(following["end"]))
            current["text"] = f"{current_text} {following['text'].strip()}".strip()
            current["words"] = current.get("words", []) + following.get("words", [])
            current["confidence"] = round(
                min(_safe_confidence(current), _safe_confidence(following)), 3
            )
        else:
            merged.append(current)
            current = dict(following)
    merged.append(current)
    return merged


def split_dialogue_sentences(dialogues: list[dict]) -> list[dict]:
    """Split long Whisper blocks into selectable complete thoughts.

    Whisper can return one 20-second block containing several sentences. Using
    that block as one anchor makes a 15/30-second teaser discard all other
    story beats. Sentence-level timing gives the selector useful, safe anchors
    without chopping a selected sentence in half later.
    """

    output = []
    for dialogue in dialogues:
        text = dialogue["text"].strip()
        matches = re.findall(r"[^.!?…]+(?:[.!?…]+|$)", text)
        sentences = [sentence.strip() for sentence in matches if sentence.strip()]
        if len(sentences) <= 1:
            output.append(dict(dialogue))
            continue
        word_timestamps = dialogue.get("words", [])
        words = [max(1, len(_tokens(sentence))) for sentence in sentences]
        if word_timestamps:
            sentence_ranges = []
            word_cursor = 0
            for sentence in sentences:
                sentence_word_count = max(1, len(_tokens(sentence)))
                sentence_ranges.append(
                    word_timestamps[word_cursor : word_cursor + sentence_word_count]
                )
                word_cursor += sentence_word_count
            for sentence, sentence_words in zip(sentences, sentence_ranges):
                if sentence_words:
                    item = dict(dialogue)
                    item["start"] = round(float(sentence_words[0]["start"]), 3)
                    item["end"] = round(float(sentence_words[-1]["end"]), 3)
                    item["text"] = sentence
                    item["words"] = sentence_words
                    item["timing_approximate"] = False
                    item["word_timing_available"] = True
                    output.append(item)
            if sentence_ranges and any(sentence_ranges):
                continue
        total_words = sum(words)
        duration = max(0.2, float(dialogue["end"]) - float(dialogue["start"]))
        cursor = float(dialogue["start"])
        for index, sentence in enumerate(sentences):
            portion = duration * words[index] / total_words
            end = float(dialogue["end"]) if index == len(sentences) - 1 else cursor + portion
            item = dict(dialogue)
            item["start"] = round(cursor, 3)
            item["end"] = round(end, 3)
            item["text"] = sentence
            # Proportional timings can rank thoughts, but are not word
            # alignment. Keep the parent speech interval as the safe anchor.
            item["context_start"] = float(dialogue.get("context_start", dialogue["start"]))
            item["context_end"] = float(dialogue.get("context_end", dialogue["end"]))
            item["context_text"] = text
            item["timing_approximate"] = True
            item["word_timing_available"] = False
            output.append(item)
            cursor = end
    return output


def probe_video(video_path: str | Path) -> dict:
    """Probe a video and return stable metadata or a user-facing error."""

    try:
        probe = ffmpeg.probe(str(video_path))
    except FileNotFoundError as error:
        raise RuntimeError("FFmpeg is required to inspect videos. Install it or use the Docker setup.") from error
    except ffmpeg.Error as error:
        details = error.stderr.decode(errors="replace") if error.stderr else ""
        raise ValueError("The uploaded file is not a readable video.") from RuntimeError(details)

    video_stream = next((stream for stream in probe.get("streams", []) if stream.get("codec_type") == "video"), None)
    if not video_stream:
        raise ValueError("The uploaded file does not contain a video stream.")

    try:
        duration = float(probe["format"]["duration"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("The video duration could not be determined.") from error
    if duration < 1:
        raise ValueError("The video is too short to create a promo.")

    return {
        "duration": duration,
        "width": int(video_stream.get("width") or 0),
        "height": int(video_stream.get("height") or 0),
        "has_audio": any(stream.get("codec_type") == "audio" for stream in probe.get("streams", [])),
    }


def probe_audio(audio_path: str | Path) -> dict:
    """Probe an uploaded soundtrack and reject files without usable audio."""

    try:
        probe = ffmpeg.probe(str(audio_path))
    except FileNotFoundError as error:
        raise RuntimeError("FFmpeg is required to inspect soundtrack files.") from error
    except ffmpeg.Error as error:
        details = error.stderr.decode(errors="replace") if error.stderr else ""
        raise ValueError("The uploaded soundtrack is not readable.") from RuntimeError(details)

    audio_stream = next(
        (stream for stream in probe.get("streams", []) if stream.get("codec_type") == "audio"),
        None,
    )
    if not audio_stream:
        raise ValueError("The uploaded soundtrack does not contain an audio stream.")
    try:
        duration = float(audio_stream.get("duration") or probe["format"]["duration"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("The soundtrack duration could not be determined.") from error
    if duration < 0.5:
        raise ValueError("The soundtrack is too short to use in a promo.")
    return {
        "duration": duration,
        "codec": str(audio_stream.get("codec_name") or "unknown"),
        "sample_rate": int(audio_stream.get("sample_rate") or 0),
        "channels": int(audio_stream.get("channels") or 0),
    }


def normalise_youtube_url(value: str) -> str:
    """Accept a single YouTube video URL, never a playlist/search URL."""

    parsed = urlparse(value.strip())
    host = parsed.netloc.lower().split(":", 1)[0]
    if host in {"youtu.be", "www.youtu.be"}:
        video_id = parsed.path.strip("/").split("/", 1)[0]
    elif host in {"youtube.com", "www.youtube.com", "m.youtube.com"}:
        if parsed.path == "/watch":
            video_id = parse_qs(parsed.query).get("v", [""])[0]
        elif parsed.path.startswith("/shorts/"):
            video_id = parsed.path.split("/", 2)[2]
        elif parsed.path.startswith("/embed/"):
            video_id = parsed.path.split("/", 2)[2]
        else:
            video_id = ""
    else:
        video_id = ""
    video_id = video_id.split("?", 1)[0].split("&", 1)[0]
    if not re.fullmatch(r"[A-Za-z0-9_-]{6,}", video_id):
        raise ValueError("Paste a direct YouTube video URL, not a playlist or search URL.")
    return f"https://www.youtube.com/watch?v={video_id}"


def download_youtube_video(url: str, target_dir: str | Path) -> dict:
    """Download one user-supplied YouTube video into an isolated workspace."""

    canonical_url = normalise_youtube_url(url)
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    output_template = str(target_dir / "source.%(ext)s")
    try:
        import yt_dlp
    except ImportError as error:
        raise RuntimeError("YouTube input requires the yt-dlp dependency.") from error

    options = {
        "format": "bv*+ba/b",
        "merge_output_format": "mp4",
        "outtmpl": output_template,
        "noplaylist": True,
        "restrictfilenames": True,
        "quiet": True,
        "no_warnings": True,
        "socket_timeout": 30,
        "retries": 2,
        "js_runtimes": {"node": {}},
        "remote_components": {"ejs:npm"},
    }
    try:
        youtube_dl = cast(Any, yt_dlp.YoutubeDL)
        with youtube_dl(options) as downloader:
            info = downloader.extract_info(canonical_url, download=True)
    except Exception as error:
        raise RuntimeError(
            "YouTube could not be downloaded. Check the URL, network access, and that you have permission to use the video."
        ) from error

    files = sorted(target_dir.glob("source.*"))
    video_path = next((path for path in files if path.suffix.lower() in {".mp4", ".mkv", ".webm", ".mov"}), None)
    if not video_path:
        raise RuntimeError("YouTube returned no usable video file.")
    metadata = probe_video(video_path)
    if metadata["duration"] > YOUTUBE_MAX_DURATION:
        video_path.unlink(missing_ok=True)
        raise ValueError(f"YouTube videos are limited to {YOUTUBE_MAX_DURATION // 60} minutes.")
    return {
        "path": str(video_path),
        "title": str(info.get("title", "YouTube video")),
        "webpage_url": str(info.get("webpage_url", canonical_url)),
        "metadata": metadata,
    }


def extract_audio(video_path: str | Path, workspace: str | Path) -> str:
    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    audio_path = workspace / "analysis.wav"
    (
        ffmpeg.input(str(video_path))
        .output(str(audio_path), ac=1, ar=16000, format="wav")
        .overwrite_output()
        .run(quiet=True, capture_stderr=True)
    )
    return str(audio_path)


def transcribe_audio(audio_path: str, progress_callback: ProgressCallback | None = None) -> list[dict]:
    model = _get_transcription_model()
    segments, _info = model.transcribe(
        audio_path,
        beam_size=5,
        vad_filter=True,
        condition_on_previous_text=False,
        word_timestamps=True,
    )

    results = []
    for segment in segments:
        text = segment.text.strip()
        if not text:
            continue
        avg_logprob = getattr(segment, "avg_logprob", -0.7)
        no_speech_prob = getattr(segment, "no_speech_prob", 0.0)
        confidence = math.exp(min(0.0, float(avg_logprob))) * (1.0 - float(no_speech_prob))
        results.append(
            {
                "start": round(float(segment.start), 3),
                "end": round(float(segment.end), 3),
                "text": text,
                "confidence": round(max(0.05, min(1.0, confidence)), 3),
                "words": [
                    {
                        "start": round(float(word.start), 3),
                        "end": round(float(word.end), 3),
                        "word": word.word.strip(),
                    }
                    for word in (getattr(segment, "words", None) or [])
                    if getattr(word, "word", "").strip()
                ],
            }
        )
        if progress_callback:
            progress_callback(results[-1]["end"], "Transcribing dialogue")
    return results


def read_audio_energy(audio_path: str, window_seconds: float = 0.4) -> dict:
    """Build a lightweight RMS energy profile from the analysis WAV."""

    try:
        with wave.open(audio_path, "rb") as audio:
            sample_rate = audio.getframerate()
            sample_width = audio.getsampwidth()
            channels = audio.getnchannels()
            raw = audio.readframes(audio.getnframes())
    except (OSError, wave.Error):
        return {"values": np.array([], dtype=np.float32), "window": window_seconds}

    if sample_width == 2:
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 1:
        samples = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128) / 128.0
    else:
        return {"values": np.array([], dtype=np.float32), "window": window_seconds}
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)

    frame_size = max(1, int(sample_rate * window_seconds))
    frame_count = max(1, math.ceil(len(samples) / frame_size))
    values = np.array(
        [
            float(np.sqrt(np.mean(np.square(samples[index * frame_size : (index + 1) * frame_size]))))
            for index in range(frame_count)
        ],
        dtype=np.float32,
    )
    if len(values):
        low, high = np.percentile(values, [10, 95])
        values = np.clip((values - low) / max(high - low, 1e-6), 0.0, 1.0)
    return {"values": values, "window": window_seconds}


def sample_audio_energy(profile: dict, start: float, end: float) -> float:
    values = profile.get("values", np.array([]))
    if len(values) == 0:
        return 0.5
    window = float(profile.get("window", 0.4))
    first = min(len(values) - 1, max(0, int(start / window)))
    last = min(len(values), max(first + 1, math.ceil(end / window)))
    return round(float(np.mean(values[first:last])), 4)


def detect_music_beats(audio_path: str, duration: float | None = None) -> list[float]:
    """Find practical cut points from positive RMS-energy onsets.

    This deliberately uses the already extracted WAV and NumPy rather than a
    heavyweight music library. It is not intended to replace a DAW beat grid;
    it gives the editor stable musical accents for cut selection and soundtrack
    reporting, including for short uploads and songs with a changing tempo.
    """

    profile = read_audio_energy(audio_path, window_seconds=0.05)
    values = np.asarray(profile.get("values", np.array([])), dtype=np.float32)
    if len(values) < 3:
        return []
    onset = np.maximum(0.0, np.diff(values, prepend=values[0]))
    threshold = max(0.08, float(np.percentile(onset, 72)))
    candidates: list[tuple[float, float]] = []
    window = float(profile.get("window", 0.05))
    for index in range(1, len(onset) - 1):
        if onset[index] < threshold or onset[index] < onset[index - 1] or onset[index] < onset[index + 1]:
            continue
        candidates.append((index * window, float(onset[index])))

    beats: list[float] = []
    for timestamp, _strength in sorted(candidates, key=lambda item: item[1], reverse=True):
        if all(abs(timestamp - beat) >= 0.22 for beat in beats):
            beats.append(timestamp)
    beats.sort()
    if duration is not None:
        beats = [round(max(0.0, min(float(duration), beat)), 3) for beat in beats]
    return beats


def analyze_music(audio_path: str | Path, duration: float | None = None) -> dict:
    """Estimate track rhythm and broad energy sections without a new dependency."""

    if duration is None:
        duration = probe_audio(audio_path)["duration"]
    analysis_path = Path(audio_path)
    profile = read_audio_energy(str(analysis_path), window_seconds=0.2)
    beats = detect_music_beats(str(analysis_path), duration)
    intervals = np.diff(np.asarray(beats, dtype=np.float32)) if len(beats) > 2 else np.array([])
    plausible = intervals[(intervals >= 0.28) & (intervals <= 1.5)]
    tempo = None
    tempo_confidence = 0.0
    beat_period = None
    if len(plausible) >= 3:
        beat_period = float(np.median(plausible))
        tempo = round(60.0 / beat_period, 1)
        regularity = float(np.mean(np.abs(plausible - beat_period) <= beat_period * 0.18))
        tempo_confidence = round(min(0.85, regularity * min(1.0, len(plausible) / 10)), 3)

    values = np.asarray(profile.get("values", []), dtype=np.float32)
    window = float(profile.get("window", 0.2))
    section_width = max(1, int(round(4.0 / max(window, 0.01))))
    sections = []
    for first in range(0, len(values), section_width):
        last = min(len(values), first + section_width)
        mean_energy = float(np.mean(values[first:last]))
        next_mean = float(np.mean(values[last:last + section_width])) if last < len(values) else mean_energy
        if mean_energy >= 0.68:
            label = "high_energy"
        elif next_mean - mean_energy >= 0.16:
            label = "build"
        elif mean_energy <= 0.28:
            label = "quiet"
        else:
            label = "steady"
        sections.append({
            "start": round(first * window, 3),
            "end": round(min(float(duration), last * window), 3),
            "label": label,
            "energy": round(mean_energy, 3),
            "confidence": round(min(0.8, abs(next_mean - mean_energy) + 0.35), 3),
        })
    climax_section = max(sections, key=lambda item: item["energy"], default=None)
    return {
        "duration": round(float(duration), 3),
        "beats": beats,
        "energy": profile,
        "beat_count": len(beats),
        "tempo_bpm": tempo,
        "tempo_confidence": tempo_confidence,
        "downbeats": [round(beat, 3) for index, beat in enumerate(beats) if index % 4 == 0],
        "sections": sections,
        "climax": climax_section,
        "natural_end_candidates": beats[-4:] if beats else ([round(float(duration), 3)] if duration else []),
    }


def choose_music_excerpt(
    music_profile: dict,
    desired_duration: float,
    pacing: str = "gradual",
) -> dict:
    """Choose a track window with an energy shape that suits the edit."""
    track_duration = max(0.0, float(music_profile.get("duration", 0.0)))
    desired_duration = max(0.0, float(desired_duration))
    if not track_duration or not desired_duration:
        return {"start": 0.0, "end": 0.0, "loop": False, "confidence": 0.0}
    if track_duration <= desired_duration:
        return {
            "start": 0.0,
            "end": round(track_duration, 3),
            "loop": True,
            "confidence": 0.5,
            "reason": "The track is shorter than the promo and will be looped with a fade.",
        }

    profile = music_profile.get("energy", {})
    values = np.asarray(profile.get("values", []), dtype=np.float32)
    window = max(0.01, float(profile.get("window", 0.2)))
    if len(values) == 0:
        return {"start": 0.0, "end": round(desired_duration, 3), "loop": False, "confidence": 0.0, "reason": "No usable energy profile; using the track opening."}

    width = max(1, int(desired_duration / window))
    step = max(1, int(0.5 / window))
    max_first = max(0, len(values) - width)
    choices = []
    for first in range(0, max_first + 1, step):
        excerpt = values[first:first + width]
        if len(excerpt) < width:
            continue
        third = max(1, len(excerpt) // 3)
        opening = float(np.mean(excerpt[:third]))
        ending = float(np.mean(excerpt[-third:]))
        peak_pos = float(np.argmax(np.convolve(excerpt, np.ones(min(5, len(excerpt))) / min(5, len(excerpt)), mode="valid"))) / max(1, len(excerpt) - 1)
        build_score = max(-0.5, min(0.5, ending - opening))
        peak_score = 1.0 - abs(peak_pos - 0.78)
        mean_score = float(np.mean(excerpt))
        excerpt_end = min(track_duration, first * window + desired_duration)
        beats = music_profile.get("beats", [])
        end_alignment = 0.5
        if beats:
            nearest = min(abs(float(beat) - excerpt_end) for beat in beats)
            end_alignment = math.exp(-nearest / 0.35)
        if pacing == "slow":
            score = 0.50 * (1.0 - mean_score) + 0.22 * peak_score + 0.18 * max(0.0, build_score) + 0.10 * end_alignment
        elif pacing == "fast":
            score = 0.40 * mean_score + 0.28 * peak_score + 0.22 * max(0.0, build_score) + 0.10 * end_alignment
        else:
            score = 0.32 * mean_score + 0.32 * peak_score + 0.26 * max(0.0, build_score) + 0.10 * end_alignment
        choices.append((score, first * window, opening, ending))
    if not choices:
        start = 0.0
        score, opening, ending = 0.0, 0.0, 0.0
    else:
        score, start, opening, ending = max(choices)
    return {
        "start": round(start, 3),
        "end": round(min(track_duration, start + desired_duration), 3),
        "loop": False,
        "confidence": round(min(0.9, max(0.0, score)), 3),
        "opening_energy": round(opening, 3),
        "ending_energy": round(ending, 3),
        "reason": "Selected from the analyzed energy profile to fit the requested pacing.",
    }


def read_motion_profile(
    video_path: str | Path,
    duration: float,
    sample_rate: float = 2.0,
    width: int = 160,
    height: int = 90,
) -> dict:
    """Decode a tiny grayscale stream and measure frame-to-frame motion.

    This keeps silent films and action-only footage useful without requiring
    OpenCV or a heavyweight local vision model. The raw stream is deliberately
    small: a two-minute film at two frames per second is only a few MB.
    """

    ffmpeg_binary = shutil.which("ffmpeg")
    if not ffmpeg_binary:
        return {"values": np.array([], dtype=np.float32), "window": 1 / sample_rate}
    command = [
        ffmpeg_binary,
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-vf",
        f"fps={sample_rate},scale={width}:{height}:flags=fast_bilinear,format=gray",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gray",
        "pipe:1",
    ]
    try:
        completed = subprocess.run(command, capture_output=True, check=True, timeout=max(60, int(duration * 2)))
    except (OSError, subprocess.SubprocessError):
        return {"values": np.array([], dtype=np.float32), "window": 1 / sample_rate}

    frame_size = width * height
    frame_count = len(completed.stdout) // frame_size
    if frame_count < 2:
        return {"values": np.array([], dtype=np.float32), "window": 1 / sample_rate}
    frames = np.frombuffer(completed.stdout[: frame_count * frame_size], dtype=np.uint8)
    frames = frames.reshape(frame_count, height, width).astype(np.float32)
    deltas = np.mean(np.abs(np.diff(frames, axis=0)), axis=(1, 2)) / 255.0
    low, high = np.percentile(deltas, [10, 95])
    values = np.clip((deltas - low) / max(high - low, 1e-6), 0.0, 1.0).astype(np.float32)
    return {"values": values, "window": 1 / sample_rate}


def sample_motion(profile: dict, start: float, end: float) -> float:
    values = profile.get("values", np.array([]))
    if len(values) == 0:
        return 0.0
    window = float(profile.get("window", 0.5))
    first = min(len(values) - 1, max(0, int(start / window) - 1))
    last = min(len(values), max(first + 1, math.ceil(end / window)))
    return round(float(np.mean(values[first:last])), 4)


def detect_shot_boundaries(video_path: str | Path, duration: float) -> list[float]:
    """Detect meaningful visual cuts without adding an OpenCV dependency.

    Uses adaptive threshold based on scene change distribution for better
    detection across different video types (action, dialogue, documentary).
    """

    ffmpeg_binary = shutil.which("ffmpeg")
    if not ffmpeg_binary:
        return [0.0, duration]

    # First pass: collect all scene scores
    command = [
        ffmpeg_binary,
        "-hide_banner",
        "-i",
        str(video_path),
        "-vf",
        "select='gt(scene,0)',showinfo",
        "-an",
        "-f",
        "null",
        "-",
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=False, timeout=max(60, int(duration * 2)))
    except (OSError, subprocess.TimeoutExpired):
        return [0.0, duration]

    # Parse scene scores and timestamps
    scene_data = []
    for line in completed.stderr.split('\n'):
        if 'pts_time:' in line and 'scene_score' in line:
            try:
                pts_match = re.search(r"pts_time:([0-9.]+)", line)
                score_match = re.search(r"scene_score:([0-9.]+)", line)
                if pts_match and score_match:
                    timestamp = float(pts_match.group(1))
                    score = float(score_match.group(1))
                    scene_data.append((timestamp, score))
            except (ValueError, IndexError):
                continue

    if not scene_data:
        return [0.0, duration]

    # Adaptive threshold: use 75th percentile of scores, minimum 0.15, maximum 0.45
    scores = [s for _, s in scene_data]
    scores.sort()
    percentile_idx = int(len(scores) * 0.75)
    adaptive_threshold = max(0.15, min(0.45, scores[percentile_idx]))

    # Also consider fixed threshold as fallback
    fixed_threshold = 0.32
    threshold = min(adaptive_threshold, fixed_threshold)

    boundaries = [0.0]
    for timestamp, score in scene_data:
        if score >= threshold and 0.5 < timestamp < duration - 0.5:
            if timestamp - boundaries[-1] >= 0.7:
                boundaries.append(timestamp)
    boundaries.append(duration)

    # If too few boundaries, lower threshold and retry
    if len(boundaries) <= 2 and len(scene_data) > 2:
        lower_threshold = max(0.1, threshold * 0.5)
        boundaries = [0.0]
        for timestamp, score in scene_data:
            if score >= lower_threshold and 0.5 < timestamp < duration - 0.5:
                if timestamp - boundaries[-1] >= 0.5:
                    boundaries.append(timestamp)
        boundaries.append(duration)

    return boundaries


def sample_visual_change(boundaries: list[float], start: float, end: float) -> float:
    if len(boundaries) <= 2:
        return 0.5
    center = (start + end) / 2
    nearby = sum(1 for boundary in boundaries[1:-1] if start - 1.5 <= boundary <= end + 1.5)
    near_center = any(abs(boundary - center) <= 1.5 for boundary in boundaries[1:-1])
    return round(min(1.0, 0.35 + nearby * 0.12 + (0.28 if near_center else 0.0)), 4)


def build_visual_candidates(
    boundaries: list[float],
    duration: float,
    energy_profile: dict,
    motion_profile: dict,
    dialogues: list[dict],
) -> list[dict]:
    """Create optional action/montage beats for moments with no clear speech."""

    candidates = []
    for start, end in zip(boundaries, boundaries[1:]):
        if end - start < 0.65 or end - start > 8.0:
            continue
        overlap = max(
            (
                max(0.0, min(end, dialogue["end"]) - max(start, dialogue["start"]))
                / max(0.1, min(end - start, dialogue["end"] - dialogue["start"]))
                for dialogue in dialogues
            ),
            default=0.0,
        )
        if overlap > 0.35:
            continue
        energy = sample_audio_energy(energy_profile, start, end) if energy_profile.get("values") is not None and len(energy_profile.get("values", [])) else 0.0
        motion = sample_motion(motion_profile, start, end)
        visual = max(sample_visual_change(boundaries, start, end), 0.45 + 0.55 * motion)
        # Keep calm, dialogue-free shots in the candidate pool. Quiet reveals
        # and reactions cannot be selected later if this early gate drops them.
        source = "thrill" if motion >= 0.45 or energy >= 0.50 or visual >= 0.80 else "visual"
        candidates.append(
            {
                "start": round(start, 3),
                "end": round(end, 3),
                "text": "Visual beat",
                "confidence": 0.95,
                "energy_score": energy,
                "visual_score": visual,
                "motion_score": motion,
                "quality_score": 0.75,
                "source": source,
            }
        )
    # Scene detection can miss fast action inside a long continuous shot.
    # Add short motion/energy windows as safe cutaway candidates, but only in
    # dialogue-free space so the visual lane never cuts through a spoken line.
    existing_thrills = [item for item in candidates if item.get("source") == "thrill"]
    if len(existing_thrills) < 2:
        window = 2.2
        stride = 1.1
        cursor = 0.0
        while cursor < duration:
            end = min(duration, cursor + window)
            if end - cursor < 1.0:
                break
            overlap = max(
                (
                    max(0.0, min(end, dialogue["end"]) - max(cursor, dialogue["start"]))
                    / max(0.1, min(end - cursor, dialogue["end"] - dialogue["start"]))
                    for dialogue in dialogues
                ),
                default=0.0,
            )
            energy = sample_audio_energy(energy_profile, cursor, end) if len(energy_profile.get("values", [])) else 0.0
            motion = sample_motion(motion_profile, cursor, end)
            visual = sample_visual_change(boundaries, cursor, end)
            duplicate = any(
                _overlap_ratio((cursor, end), (item["start"], item["end"])) > 0.7
                for item in candidates
            )
            if overlap <= 0.20 and not duplicate and (motion >= 0.55 or energy >= 0.62):
                candidates.append(
                    {
                        "start": round(cursor, 3),
                        "end": round(end, 3),
                        "text": "Visual thrill beat",
                        "confidence": 0.9,
                        "energy_score": energy,
                        "visual_score": max(visual, 0.62),
                        "motion_score": motion,
                        "quality_score": 0.78,
                        "coherence_score": 1.0,
                        "source": "thrill",
                    }
                )
            cursor += stride
    if not candidates:
        # A silent/static film can have no FFmpeg scene cut at all. Keep a
        # small set of evenly distributed visual windows so the generator has
        # something meaningful to rank instead of failing on an empty speech
        # transcript.
        window = min(6.0, max(2.0, duration / 4))
        cursor = 0.0
        while cursor < duration:
            end = min(duration, cursor + window)
            if end - cursor >= 1.0:
                motion = sample_motion(motion_profile, cursor, end)
                candidates.append(
                    {
                        "start": round(cursor, 3),
                        "end": round(end, 3),
                        "text": "Visual story beat",
                        "confidence": 0.8,
                        "energy_score": sample_audio_energy(energy_profile, cursor, end)
                        if len(energy_profile.get("values", []))
                        else 0.0,
                        "visual_score": round(0.55 + 0.45 * motion, 4),
                        "motion_score": motion,
                        "quality_score": 0.7,
                        "coherence_score": 1.0,
                        "source": "visual",
                    }
                )
            cursor += window
    return candidates


def build_opening_candidate(
    boundaries: list[float],
    duration: float,
    energy_profile: dict,
    motion_profile: dict,
) -> dict | None:
    """Offer a short opening shot as an unverified visual candidate.

    The local pipeline has no OCR or scene classifier, so it must not label an
    arbitrary first shot as a studio card or force it into the final edit.
    Optional storyboard analysis can explicitly identify a title card.
    """

    usable_boundaries = [point for point in boundaries[1:] if point > 0.8]
    if not usable_boundaries or duration <= 0.8:
        return None
    first_cut = usable_boundaries[0]
    if first_cut < 1.5 and len(usable_boundaries) > 1:
        first_cut = usable_boundaries[1]
    end = min(duration, max(1.5, min(first_cut, 3.5)))
    if end <= 0.8:
        return None
    return {
        "start": 0.0,
        "end": round(end, 3),
        "text": "Opening visual beat",
        "confidence": 0.75,
        "energy_score": sample_audio_energy(energy_profile, 0.0, end),
        "visual_score": sample_visual_change(boundaries, 0.0, end),
        "motion_score": sample_motion(motion_profile, 0.0, end),
        "quality_score": 0.75,
        "coherence_score": 1.0,
        "api_score": 0.5,
        "source": "visual",
        "preserve_opening": False,
        "context_start": 0.0,
        "context_end": round(end, 3),
    }


def _storyboard_times(duration: float, boundaries: list[float], max_frames: int = 18) -> list[float]:
    points = [
        (start + end) / 2
        for start, end in zip(boundaries, boundaries[1:])
        if end - start >= 0.8
    ]
    if len(points) < max_frames:
        points.extend(np.linspace(0.5, max(0.5, duration - 0.5), max_frames).tolist())
    unique = sorted({round(max(0.2, min(duration - 0.2, point)), 2) for point in points})
    if len(unique) <= max_frames:
        return unique
    indexes = np.linspace(0, len(unique) - 1, max_frames).astype(int)
    return [unique[index] for index in sorted(set(indexes))]


def extract_storyboard(
    video_path: str | Path,
    workspace: str | Path,
    duration: float,
    boundaries: list[float],
    max_frames: int = 18,
) -> list[dict]:
    """Extract small JPEG storyboard frames for a vision-capable API.

    Uses shot boundaries, high-motion moments, and uniform sampling for
    comprehensive visual coverage.
    """

    workspace = Path(workspace)
    storyboard_dir = workspace / "storyboard"
    storyboard_dir.mkdir(parents=True, exist_ok=True)

    # Get diverse timestamps: shot boundaries midpoints + high motion + uniform
    timestamps = _storyboard_times(duration, boundaries, max_frames)

    # Ensure we have timestamps even if boundaries detection was minimal
    if len(timestamps) < 3:
        timestamps = np.linspace(0.5, max(0.5, duration - 0.5), min(max_frames, 8)).tolist()

    frames = []
    for index, timestamp in enumerate(timestamps):
        frame_path = storyboard_dir / f"frame-{index:02d}.jpg"
        try:
            (
                ffmpeg.input(str(video_path), ss=timestamp)
                .output(
                    str(frame_path),
                    vframes=1,
                    vf="scale=640:-2:flags=fast_bilinear",
                    q=5,
                    format="image2",
                )
                .overwrite_output()
                .run(quiet=True, capture_stderr=True)
            )
            encoded = base64.b64encode(frame_path.read_bytes()).decode("ascii")
        except (OSError, ffmpeg.Error):
            continue
        frames.append({"time": timestamp, "data_url": f"data:image/jpeg;base64,{encoded}"})

    # If no frames extracted, create at least one at the middle
    if not frames and duration > 1:
        mid_frame = storyboard_dir / "frame-mid.jpg"
        try:
            mid_time = duration / 2
            (
                ffmpeg.input(str(video_path), ss=mid_time)
                .output(
                    str(mid_frame),
                    vframes=1,
                    vf="scale=640:-2:flags=fast_bilinear",
                    q=5,
                    format="image2",
                )
                .overwrite_output()
                .run(quiet=True, capture_stderr=True)
            )
            encoded = base64.b64encode(mid_frame.read_bytes()).decode("ascii")
            frames.append({"time": mid_time, "data_url": f"data:image/jpeg;base64,{encoded}"})
        except (OSError, ffmpeg.Error):
            pass

    return frames


def _provider_error(provider: str, status_code: int) -> RuntimeError:
    if status_code in {401, 403}:
        detail = "API key was rejected or lacks permission."
    elif status_code == 404:
        detail = "The configured model or endpoint is unavailable. Check the model setting."
    elif status_code in {402, 429}:
        detail = "The provider rate limit or quota was reached. Check the provider account and retry later."
    elif status_code in {400, 413, 415, 422}:
        detail = "The provider rejected the request or media format."
    elif status_code >= 500:
        detail = "The provider is temporarily unavailable. Retry later."
    else:
        detail = "The provider request failed."
    return RuntimeError(f"{provider} request failed (HTTP {status_code}). {detail}")


def _api_json_response(response: requests.Response, provider: str = "Vision API") -> dict:
    if not response.ok:
        # Provider error bodies can echo credential fragments. Keep diagnostics
        # useful without copying any response content into job logs.
        raise _provider_error(provider, response.status_code)
    try:
        payload = response.json()
    except (ValueError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{provider} returned a malformed response.") from error
    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as error:
        raise RuntimeError(f"{provider} returned a response without message content.") from error
    if isinstance(content, list):
        content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
    try:
        return json.loads(content)
    except (TypeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{provider} returned malformed JSON.") from error


def _api_provider_config(provider: str, api_key: str | None) -> tuple[str, str, str]:
    """Return endpoint, credential, and model for request-scoped provider settings."""
    if provider not in {"gemini", "openai"}:
        raise ValueError("Unknown vision provider.")
    key = api_key or (GEMINI_API_KEY if provider == "gemini" else VISION_API_KEY)
    if not key:
        raise ValueError(f"No API key is configured for {provider}.")
    if provider == "gemini":
        return ("https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
                key, GEMINI_MODEL)
    return ("https://api.openai.com/v1/chat/completions", key, VISION_API_MODEL)


def _provider_key(provider: str, api_key: str | None = None) -> str | None:
    if api_key:
        return api_key
    if provider == "gemini":
        return GEMINI_API_KEY
    if provider == "elevenlabs":
        return ELEVENLABS_API_KEY
    return VISION_API_KEY


def _post_provider_request(provider: str, endpoint: str, **kwargs) -> requests.Response:
    try:
        return requests.post(endpoint, timeout=API_TIMEOUT_SECONDS, **kwargs)
    except requests.exceptions.Timeout as error:
        raise RuntimeError(f"{provider} request timed out. Retry with a shorter source or try again later.") from error
    except requests.exceptions.ConnectionError as error:
        raise RuntimeError(f"Could not connect to {provider}. Check the network and retry.") from error


def transcribe_audio_with_elevenlabs(
    audio_path: str | Path,
    api_key: str | None = None,
    duration: float | None = None,
    progress_callback: ProgressCallback | None = None,
) -> list[dict]:
    """Transcribe extracted mono 16 kHz PCM WAV and preserve returned timing evidence."""
    key = _provider_key("elevenlabs", api_key)
    if not key:
        raise ValueError("ElevenLabs transcription is selected, but no ElevenLabs API key was provided.")
    source = Path(audio_path)
    try:
        with wave.open(str(source), "rb") as audio:
            if (audio.getnchannels(), audio.getsampwidth(), audio.getframerate()) != (1, 2, 16000):
                raise ValueError("ElevenLabs requires the extracted mono 16 kHz, 16-bit PCM audio track.")
    except (OSError, wave.Error) as error:
        raise ValueError("ElevenLabs requires a readable extracted WAV audio track.") from error

    try:
        with source.open("rb") as audio_file:
            response = _post_provider_request(
                "ElevenLabs",
                "https://api.elevenlabs.io/v1/speech-to-text",
                headers={"xi-api-key": key},
                files={"file": (source.name, audio_file, "audio/wav")},
                data={"model_id": ELEVENLABS_MODEL, "file_format": "pcm_s16le_16",
                      "timestamps_granularity": "word", "diarize": "true"},
            )
    except requests.exceptions.RequestException as error:
        raise RuntimeError("ElevenLabs could not process the audio request.") from error
    if not response.ok:
        raise _provider_error("ElevenLabs", response.status_code)
    try:
        payload = response.json()
    except (ValueError, json.JSONDecodeError) as error:
        raise RuntimeError("ElevenLabs returned a malformed transcription response.") from error
    if not isinstance(payload, dict):
        raise RuntimeError("ElevenLabs returned a malformed transcription response.")

    limit = float(duration) if duration is not None else None
    raw_words = payload.get("words")
    results: list[dict] = []
    if isinstance(raw_words, list) and raw_words:
        phrase: list[dict] = []
        def flush_phrase() -> None:
            if phrase:
                results.append(_elevenlabs_phrase(phrase))
                phrase.clear()

        for word in raw_words:
            if not isinstance(word, dict) or word.get("type", "word") != "word":
                continue
            text = str(word.get("text", "")).strip()
            try:
                start, end = float(word["start"]), float(word["end"])
            except (KeyError, TypeError, ValueError):
                flush_phrase()
                continue
            if (not text or not math.isfinite(start) or not math.isfinite(end) or
                    start < 0 or end <= start or (limit is not None and end > limit + 0.1)):
                flush_phrase()
                continue
            speaker_id = word.get("speaker_id")
            if phrase and (start - phrase[0]["start"] > 2.5 or speaker_id != phrase[0].get("speaker_id") or len(phrase) >= 40):
                flush_phrase()
            try:
                logprob = float(word["logprob"]) if word.get("logprob") is not None else None
            except (TypeError, ValueError):
                logprob = None
            phrase.append({"start": start, "end": end, "text": text,
                           "speaker_id": speaker_id, "logprob": logprob})
        flush_phrase()
        for segment in results:
            segment["word_timing_available"] = True
            segment["timing_approximate"] = False
    else:
        # Some models/configurations return timed chunks or only plain text.
        # Preserve measured chunk bounds; for text-only results use the full
        # source interval and mark timing approximate without inventing words.
        chunks = payload.get("chunks")
        if isinstance(chunks, list):
            for chunk in chunks:
                if not isinstance(chunk, dict):
                    continue
                text = str(chunk.get("text", "")).strip()
                try:
                    start, end = float(chunk["start"]), float(chunk["end"])
                except (KeyError, TypeError, ValueError):
                    continue
                if not text or not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start or (limit is not None and end > limit + 0.1):
                    continue
                results.append({"start": start, "end": end, "context_start": start,
                                "context_end": end, "text": text, "words": [],
                                "confidence": None, "confidence_available": False,
                                "speaker_id": chunk.get("speaker_id"),
                                "timing_approximate": False, "word_timing_available": False})
        elif payload.get("text") and limit and limit > 0:
            results.append({"start": 0.0, "end": limit, "context_start": 0.0,
                            "context_end": limit, "text": str(payload["text"]).strip(),
                            "words": [], "confidence": None, "confidence_available": False,
                            "timing_approximate": True, "word_timing_available": False})
    _progress(progress_callback, 0.95, "ElevenLabs transcription complete")
    return results


def _elevenlabs_phrase(words: list[dict]) -> dict:
    start, end = words[0]["start"], words[-1]["end"]
    mapped_words = [{"start": round(item["start"], 3), "end": round(item["end"], 3),
                     "word": item["text"]} for item in words]
    logprobs = [item["logprob"] for item in words if item["logprob"] is not None]
    # The API exposes log probabilities, not calibrated segment confidence.
    # Keep the pipeline's ranking default and explicitly flag the distinction.
    return {"start": round(start, 3), "end": round(end, 3),
            "context_start": round(start, 3), "context_end": round(end, 3),
            "text": " ".join(item["text"] for item in words), "words": mapped_words,
            "confidence": None, "confidence_available": False,
            "speaker_id": words[0].get("speaker_id"),
            "provider_logprob": round(sum(logprobs) / len(logprobs), 4) if logprobs else None,
            "timing_approximate": False, "word_timing_available": True}


def transcribe_audio_with_provider(
    provider: str,
    audio_path: str | Path,
    *,
    api_key: str | None = None,
    duration: float | None = None,
    progress_callback: ProgressCallback | None = None,
) -> list[dict]:
    if provider == "local":
        return transcribe_audio(str(audio_path), progress_callback)
    if provider == "elevenlabs":
        return transcribe_audio_with_elevenlabs(
            audio_path, api_key=api_key, duration=duration, progress_callback=progress_callback
        )
    if provider == "openai":
        return transcribe_audio_with_api(
            str(audio_path), progress_callback, duration=duration, api_key=api_key,
        )
    raise ValueError("Unknown transcription provider.")


def transcribe_audio_with_api(
    audio_path: str,
    progress_callback: ProgressCallback | None = None,
    duration: float | None = None,
    api_key: str | None = None,
) -> list[dict]:
    api_key = _provider_key("openai", api_key)
    if not api_key:
        raise RuntimeError("No API key is configured for API audio analysis.")
    source_path = Path(audio_path)
    results = []
    with tempfile.TemporaryDirectory(prefix="api-audio-", dir=source_path.parent) as chunk_dir:
        chunks = []
        try:
            with wave.open(str(source_path), "rb") as source_audio:
                parameters = source_audio.getparams()
                bytes_per_frame = parameters.nchannels * parameters.sampwidth
                frames_per_chunk = max(
                    1,
                    API_TRANSCRIBE_MAX_CHUNK_BYTES // max(1, bytes_per_frame),
                )
                offset = 0.0
                index = 0
                while True:
                    frames = source_audio.readframes(frames_per_chunk)
                    if not frames:
                        break
                    chunk_path = Path(chunk_dir) / f"chunk-{index:04d}.wav"
                    with wave.open(str(chunk_path), "wb") as chunk_audio:
                        chunk_audio.setparams(parameters)
                        chunk_audio.writeframes(frames)
                    chunk_duration = len(frames) / max(1, bytes_per_frame * parameters.framerate)
                    chunks.append((chunk_path, offset, chunk_duration))
                    offset += chunk_duration
                    index += 1
        except (OSError, wave.Error) as error:
            raise ValueError("API transcription requires a readable extracted WAV audio track.") from error

        for index, (chunk_path, offset, chunk_duration) in enumerate(chunks):
            with chunk_path.open("rb") as audio_file:
                response = _post_provider_request(
                    "OpenAI",
                    "https://api.openai.com/v1/audio/transcriptions",
                    headers={"Authorization": f"Bearer {api_key}"},
                    files={"file": (chunk_path.name, audio_file, "audio/wav")},
                    data={
                        "model": API_TRANSCRIBE_MODEL,
                        "response_format": "verbose_json",
                        "timestamp_granularities[]": "segment",
                    },
                )
            if not response.ok:
                raise _provider_error("OpenAI transcription", response.status_code)
            try:
                response_payload = response.json()
            except (ValueError, json.JSONDecodeError) as error:
                raise RuntimeError("OpenAI transcription returned malformed JSON.") from error
            for segment in response_payload.get("segments", []):
                text = str(segment.get("text", "")).strip()[:1000]
                try:
                    local_start = float(segment.get("start", 0))
                    local_end = float(segment.get("end", 0))
                except (TypeError, ValueError):
                    continue
                if not math.isfinite(local_start) or not math.isfinite(local_end):
                    continue
                if local_start < -0.1 or local_end > chunk_duration + 0.1:
                    continue
                start = offset + max(0.0, local_start)
                end = offset + min(chunk_duration, local_end)
                if duration is not None:
                    start = max(0.0, min(float(duration), start))
                    end = max(0.0, min(float(duration), end))
                if not text or end <= start:
                    continue
                results.append({
                    "start": round(start, 3),
                    "end": round(end, 3),
                    "context_start": round(start, 3),
                    "context_end": round(end, 3),
                    "text": text,
                    "confidence": 0.92,
                    "timing_approximate": True,
                    "word_timing_available": False,
                })
            _progress(
                progress_callback,
                0.5 + 0.45 * ((index + 1) / max(1, len(chunks))),
                "API transcription complete",
            )
    return results


def analyze_storyboard_with_api(
    storyboard: list[dict],
    theme: str,
    duration: float,
    dialogues: list[dict],
    strategy: str,
    request: dict | None = None,
    provider: str = "openai",
    api_key: str | None = None,
) -> dict:
    """Ask a vision model to reason over the storyboard and transcript.

    The API receives sampled frames rather than the original video file. That
    keeps the request bounded while still giving it visual context. Audio is
    represented by the timestamped transcript and local energy features.
    """

    api_key = _provider_key(provider, api_key)
    if not api_key:
        raise RuntimeError("No API key is configured for vision analysis.")
    transcript = "\n".join(
        f"[{item['start']:.1f}-{item['end']:.1f}] {item['text']}" for item in dialogues
    )[:18000]
    frame_text = "\n".join(f"Frame {index + 1}: {frame['time']:.2f}s" for index, frame in enumerate(storyboard))
    content: list[dict] = [
        {
            "type": "text",
            "text": (
                "You are an experienced film editor. Analyze this storyboard and transcript for a short film or trailer. "
                "Choose moments that can survive as a coherent teaser, not merely keyword matches. Preserve complete dialogue, "
                "avoid duplicate or corrupted lines, cover the film's setup, turn, and payoff when possible, and include silent "
                "visual beats when they carry story. Describe the visible event briefly and label its type. "
                "Treat descriptions as interpretations of sampled frames, not verified ground truth. "
                "Only propose moments supported by the supplied sampled frame times or transcript intervals; do not invent source timestamps. "
                "Return only the requested JSON.\n\n"
                f"Theme: {theme}\nSelection strategy: {strategy}\nVideo duration: {duration:.1f}s\n"
                f"Edit request: {json.dumps(request or {}, ensure_ascii=False)[:3000]}\n"
                f"{frame_text}\n\nTranscript:\n{transcript or '[No dialogue detected — rely on visual storytelling and sound/motion cues].'}"
            ),
        }
    ]
    content.extend(
        {
            "type": "image_url",
            "image_url": {"url": frame["data_url"], "detail": "low"},
        }
        for frame in storyboard
    )
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "moments": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "start": {"type": "number"},
                        "end": {"type": "number"},
                        "importance": {"type": "number"},
                        "reason": {"type": "string"},
                        "story_role": {"type": "string"},
                        "transcript_quote": {"type": "string"},
                        "visual_description": {"type": "string"},
        "event_type": {"type": "string", "enum": ["title_card", "establishing", "dialogue", "interaction", "product", "action", "reaction", "atmosphere", "reveal", "other"]},
                    },
                    "required": ["start", "end", "importance", "reason", "story_role", "transcript_quote", "visual_description", "event_type"],
                },
            },
            "script": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "title": {"type": "string"},
                    "logline": {"type": "string"},
                    "voiceover": {"type": "array", "items": {"type": "string"}},
                    "text_cards": {"type": "array", "items": {"type": "string"}},
                    "beats": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["title", "logline", "voiceover", "text_cards", "beats"],
            },
        },
        "required": ["moments", "script"],
    }
    endpoint, api_key, model = _api_provider_config(provider, api_key)
    response = _post_provider_request(
        "Gemini" if provider == "gemini" else "OpenAI",
        endpoint,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": model,
            "temperature": 0.2,
            "messages": [
                {"role": "system", "content": "Return strict JSON matching the supplied schema."},
                {"role": "user", "content": content},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "trailer_analysis", "strict": True, "schema": schema},
            },
        },
    )
    result = _api_json_response(response, "Gemini" if provider == "gemini" else "OpenAI")
    return _validate_storyboard_response(result, duration, "Gemini" if provider == "gemini" else "OpenAI")


def _validate_storyboard_response(result: dict, duration: float, provider: str = "Vision API") -> dict:
    if not isinstance(result, dict) or not isinstance(result.get("moments"), list):
        raise RuntimeError(f"{provider} returned an invalid storyboard structure.")
    script = result.get("script")
    script_fields = {"title": str, "logline": str, "voiceover": list,
                     "text_cards": list, "beats": list}
    if not isinstance(script, dict) or any(not isinstance(script.get(name), kind) for name, kind in script_fields.items()):
        raise RuntimeError(f"{provider} returned an invalid trailer script structure.")
    for name in ("voiceover", "text_cards", "beats"):
        if any(not isinstance(item, str) for item in script[name]):
            raise RuntimeError(f"{provider} returned invalid text in the trailer script.")
    valid_moments = []
    event_types = {"title_card", "establishing", "dialogue", "interaction", "product",
                   "action", "reaction", "atmosphere", "reveal", "other"}
    text_fields = ("reason", "story_role", "transcript_quote", "visual_description", "event_type")
    for moment in result["moments"]:
        if not isinstance(moment, dict) or any(field not in moment for field in
                ("start", "end", "importance", *text_fields)):
            raise RuntimeError(f"{provider} returned a storyboard moment with missing fields.")
        if any(not isinstance(moment[field], str) for field in text_fields):
            raise RuntimeError(f"{provider} returned invalid text in a storyboard moment.")
        if moment["event_type"] not in event_types:
            raise RuntimeError(f"{provider} returned an unsupported storyboard event type.")
        try:
            if isinstance(moment["start"], bool) or isinstance(moment["end"], bool) or isinstance(moment["importance"], bool):
                raise ValueError("boolean is not a numeric moment field")
            start, end = float(moment["start"]), float(moment["end"])
            importance = float(moment["importance"])
        except (KeyError, TypeError, ValueError):
            continue
        if (not all(math.isfinite(value) for value in (start, end, importance)) or
                start < 0 or end <= start or end > duration or not 0 <= importance <= 1):
            continue
        valid_moments.append(moment)
    return {"moments": valid_moments, "script": script}


def apply_api_moments(candidates: list[dict], api_moments: list[dict], duration: float) -> list[dict]:
    """Use API suggestions as a second opinion, never as unchecked timestamps."""

    output = [dict(candidate) for candidate in candidates]
    if not math.isfinite(float(duration)) or duration <= 0.5:
        return output
    for moment in api_moments:
        if not isinstance(moment, dict):
            continue
        try:
            proposed_start = float(moment.get("start", 0))
            proposed_end = float(moment.get("end", proposed_start + 1))
            raw_importance = float(moment.get("importance", 0.5))
        except (TypeError, ValueError):
            continue
        if not all(math.isfinite(value) for value in (proposed_start, proposed_end, raw_importance)):
            continue
        if proposed_start < -0.1 or proposed_start > duration or proposed_end < 0 or proposed_end > duration + 0.1:
            continue
        start = max(0.0, min(duration, proposed_start))
        end = max(0.0, min(duration, proposed_end))
        if end <= start:
            continue
        if end - start < 0.5:
            center = (start + end) / 2
            start = max(0.0, min(duration - 0.5, center - 0.25))
            end = start + 0.5
        importance = max(0.0, min(1.0, raw_importance))
        overlaps = []
        for candidate in output:
            try:
                candidate_start, candidate_end = float(candidate["start"]), float(candidate["end"])
            except (KeyError, TypeError, ValueError):
                continue
            if (not math.isfinite(candidate_start) or not math.isfinite(candidate_end) or
                    candidate_start < 0 or candidate_end <= candidate_start or candidate_end > duration):
                continue
            ratio = _overlap_ratio((start, end), (candidate_start, candidate_end))
            if ratio >= 0.2:
                overlaps.append((ratio, candidate))
        # The model may rank local evidence but cannot introduce source time
        # intervals absent from shot, opening, or dialogue analysis.
        if not overlaps:
            continue
        _ratio, target = max(overlaps, key=lambda item: item[0])
        target["api_score"] = max(float(target.get("api_score", 0.5)), importance)
        target["api_reason"] = str(moment.get("reason", ""))[:400]
        target["story_role"] = str(moment.get("story_role", ""))[:120]
        target["visual_description"] = str(moment.get("visual_description", ""))[:400]
        target["event_type"] = str(moment.get("event_type", "other"))
        if target["event_type"] == "title_card" and start <= 6.0:
            target["source"] = "opening"
            target["preserve_opening"] = True
            target["text"] = "Identified title card"
    return output


def build_local_script(theme: str, selected: list[dict], has_dialogue: bool) -> dict:
    quotes = [
        item["text"]
        for item in selected
        if item.get("text")
        and item["text"] not in {"Visual beat", "Visual story beat", "Visual thrill beat"}
        and not item.get("opening_preserved")
        and not any(source in {"visual", "thrill", "opening"} for source in item.get("sources", []))
    ]
    beats = [
        "Open with the first clear visual or line that establishes the world.",
        "Escalate through the strongest conflict or emotional turn.",
        "End on the most unresolved, memorable image or promise.",
    ]
    return {
        "title": f"{theme.title()} — teaser outline",
        "logline": f"A teaser about {theme}, built from the strongest available story and visual beats.",
        "voiceover": quotes if has_dialogue else ["Use natural sound, score, and visual rhythm instead of forced narration."],
        "text_cards": [theme.title(), "Some choices change everything."] if not has_dialogue else [],
        "beats": beats,
    }


def build_edit_plan(
    prompt: str,
    target_duration: int,
    energy: str,
    strategy: str,
    analysis_mode: str,
    music_enabled: bool = False,
    request: dict | None = None,
) -> dict:
    """Create the explicit plan an agent follows before touching media."""

    structure = {
        "cinematic": ["hook", "world", "pressure", "unresolved payoff"],
        "high-impact": ["hook", "escalation", "impact montage", "hard button"],
    }.get(energy, ["hook", "setup", "turn", "payoff"])
    return {
        "prompt": prompt,
        "time_budget_seconds": target_duration,
        "structure": structure,
        "tools": [
            "probe_video",
            "transcribe_audio_with_word_timestamps",
            "detect_scene_boundaries",
            "measure_audio_energy",
            "analyse_music_beats" if music_enabled else "no_external_soundtrack",
            "measure_visual_motion",
            "storyboard_vision" if analysis_mode in {"auto", "api", "gemini"} else "local_visual_scoring",
            "draft_selection",
            "dialogue_continuity_critic",
            "ffmpeg_stitch",
            "render_decode_qa",
        ],
        "strategy": strategy,
        "request": request or {},
        "guardrails": [
            "never cut inside a selected word-timed dialogue beat",
            "keep the rendered duration within the requested target",
            "never use a repeated transcription artifact as a story beat",
            "cover multiple story regions when the source supports it",
            "prefer a complete thought over a higher keyword score",
        ],
    }


def interpret_edit_request(
    prompt: str,
    target_duration: int = 30,
    energy: str = "balanced",
    strategy: str = "theme",
    aspect_ratio: str = "source",
    music_enabled: bool = False,
) -> dict:
    """Convert a natural-language edit request into bounded local controls.

    This parser is deliberately offline and deterministic. It extracts only
    supported choices; the selected source moments still come from measured
    media candidates and never from prompt-generated timestamps.
    """
    prompt = re.sub(r"\s+", " ", prompt).strip()
    if not prompt or len(prompt) > 500:
        raise ValueError("Describe the promo in 1 to 500 characters.")

    duration_pattern = r"\b(15|30|45|60|90|120)(?:[- ]?)(?:s|sec|secs|seconds?)\b"
    duration_match = re.search(duration_pattern, prompt, re.I)
    if duration_match:
        target_duration = int(duration_match.group(1))
    ratio_match = re.search(r"\b(9\s*:\s*16|vertical|portrait|1\s*:\s*1|square)\b", prompt, re.I)
    if ratio_match:
        ratio_value = re.sub(r"\s+", "", ratio_match.group(1).lower())
        aspect_ratio = "vertical" if ratio_value in {"vertical", "portrait", "9:16"} else "square"

    lowered = prompt.lower()
    pacing = "gradual"
    if re.search(r"\b(fast[- ]paced|high[- ]impact|rapid|punchy|energetic)\b", lowered):
        pacing, energy = "fast", "high-impact"
    elif re.search(r"\b(calm|slow|quiet|emotional|intimate|reflective)\b", lowered):
        pacing, energy = "slow", "cinematic"
    elif re.search(r"\b(cinematic|dramatic|build(?:s|ing)? tension|gradual(?:ly)? builds?)\b", lowered):
        pacing, energy = "gradual", "cinematic"

    if re.search(r"\b(action|chase|fight|product demo|customer reactions|montage)\b", lowered):
        strategy = "best-overall" if strategy == "theme" else strategy
    if re.search(r"\b(less dialogue|no dialogue|without dialogue|music[- ]driven)\b", lowered):
        strategy = "visual-montage"
    elif re.search(r"\b(story arc|story first|dialogue|complete lines|conversation)\b", lowered):
        strategy = "story-arc" if strategy == "theme" else strategy

    dialogue_preference = "natural"
    if re.search(r"\b(less dialogue|no dialogue|without dialogue)\b", lowered):
        dialogue_preference = "minimize"
    elif re.search(r"\b(dialogue|complete lines|natural dialogue)\b", lowered):
        dialogue_preference = "prioritize"

    ending = "source_order"
    if re.search(r"\b(emotional|heartfelt|moving)\b", lowered):
        ending = "emotional"
    elif re.search(r"\b(reveal|final reveal)\b", lowered):
        ending = "reveal"
    elif re.search(r"\b(unresolved|mystery|suspense|intriguing)\b", lowered):
        ending = "unresolved"
    elif re.search(r"\b(musical|music(?:al)? ending|track ending)\b", lowered):
        ending = "musical"
    elif re.search(r"\b(memorable|strong|powerful) ending\b", lowered):
        ending = "memorable"
    elif re.search(r"\b(title card|end on the title|title ending)\b", lowered):
        ending = "title"

    if len(prompt.split()) > 5 and re.match(r"^(?:make|create|build|give me|please)\b", lowered):
        theme = re.sub(
            r"\b(make|create|build|give me|a|an|the|promo|promotional video|trailer|short film|social media|highlighting|highlight|prioritizing|prioritise|starts with|start with|ends on|end on|that|and|with|for|using|use|less|more)\b",
            " ",
            prompt,
            flags=re.I,
        )
        theme = re.sub(duration_pattern, " ", theme, flags=re.I)
        theme = re.sub(r"\s+", " ", theme).strip(" ,.-") or prompt
    else:
        theme = prompt

    desired = []
    for token in ("product", "customer", "reaction", "scenery", "action", "character", "reveal", "dialogue", "emotional", "landscape"):
        if token in lowered:
            desired.append(token)
    explicit_chronology = bool(re.search(r"\b(chronological|in source order|tell it in order)\b", lowered))
    nonlinear_order = bool(re.search(r"\b(rewind|start near the ending|start at the end|back to the beginning|cold open|shocking opening)\b", lowered))
    if explicit_chronology:
        structure_style = "chronological"
    elif nonlinear_order or re.search(r"\b(thriller|suspense|mystery|mysterious)\b", lowered):
        structure_style = "nonlinear_mystery"
    elif dialogue_preference == "prioritize":
        structure_style = "dialogue_progression"
    elif pacing == "fast":
        structure_style = "action_escalation"
    elif ending == "emotional":
        structure_style = "emotional_progression"
    else:
        structure_style = "cinematic_arc"
    return {
        "prompt": prompt,
        "theme": theme[:160],
        "target_duration": target_duration,
        "energy": energy,
        "strategy": strategy,
        "aspect_ratio": aspect_ratio,
        "pacing": pacing,
        "dialogue_preference": dialogue_preference,
        "ending": ending,
        "desired_content": desired,
        "hook_first": bool(re.search(r"\b(hook|starts with|start with|opening)\b", lowered)),
        "structure_style": structure_style,
        "nonlinear_order": nonlinear_order and not explicit_chronology,
        "music_role": "structural_guide" if music_enabled and re.search(r"\b(structural guide|music[- ]driven|follow the music|music as the guide)\b", lowered) else ("score" if music_enabled else "source_audio"),
    }


def interpret_edit_request_with_api(prompt: str, defaults: dict, provider: str = "openai", api_key: str | None = None) -> dict:
    """Use the configured language model for creative intent, with no media upload."""
    api_key = _provider_key(provider, api_key)
    if not api_key:
        raise RuntimeError("No API key is configured for request interpretation.")
    fields = {
        "theme": {"type": "string"},
        "target_duration": {"type": "integer", "enum": [15, 30, 45, 60, 90, 120]},
        "energy": {"type": "string", "enum": ["balanced", "cinematic", "high-impact"]},
        "strategy": {"type": "string", "enum": ["theme", "best-overall", "story-arc", "visual-montage"]},
        "aspect_ratio": {"type": "string", "enum": ["source", "vertical", "square"]},
        "pacing": {"type": "string", "enum": ["gradual", "slow", "fast"]},
        "dialogue_preference": {"type": "string", "enum": ["natural", "prioritize", "minimize"]},
        "ending": {"type": "string", "enum": ["source_order", "memorable", "emotional", "reveal", "unresolved", "musical", "title"]},
        "desired_content": {"type": "array", "items": {"type": "string"}},
        "hook_first": {"type": "boolean"},
        "music_role": {"type": "string", "enum": ["source_audio", "score", "structural_guide"]},
    }
    schema = {
        "type": "object",
        "properties": fields,
        "required": list(fields),
        "additionalProperties": False,
    }
    endpoint, api_key, model = _api_provider_config(provider, api_key)
    provider_label = "Gemini" if provider == "gemini" else "OpenAI"
    response = _post_provider_request(
        provider_label,
        endpoint,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": model,
            "temperature": 0,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "framecut_edit_request", "strict": True, "schema": schema},
            },
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Interpret the user's creative edit request as bounded editing controls. "
                        "Do not invent scenes, source timestamps, or claims about media that has not been analyzed. "
                        "Use supplied defaults when the request is ambiguous."
                    ),
                },
                {"role": "user", "content": f"Defaults: {json.dumps(defaults)}\nRequest: {prompt}"},
            ],
        },
    )
    parsed = _api_json_response(response, provider_label)
    if not isinstance(parsed, dict):
        raise RuntimeError(f"{provider_label} returned an invalid edit interpretation.")
    if set(parsed) != set(fields):
        raise RuntimeError(f"{provider_label} returned an edit interpretation with missing or unexpected fields.")
    result = dict(defaults)
    result.update(parsed)
    result["prompt"] = prompt
    if isinstance(result.get("target_duration"), bool) or result.get("target_duration") not in {15, 30, 45, 60, 90, 120}:
        raise ValueError("The editing model returned an unsupported duration.")
    if result.get("energy") not in {"balanced", "cinematic", "high-impact"}:
        raise ValueError("The editing model returned an unsupported creative direction.")
    if result.get("strategy") not in {"theme", "best-overall", "story-arc", "visual-montage"}:
        raise ValueError("The editing model returned an unsupported selection strategy.")
    if result.get("aspect_ratio") not in {"source", "vertical", "square"}:
        raise ValueError("The editing model returned an unsupported output format.")
    if result.get("ending") not in {"source_order", "memorable", "emotional", "reveal", "unresolved", "musical", "title"}:
        raise ValueError("The editing model returned an unsupported ending style.")
    if result.get("pacing") not in {"gradual", "slow", "fast"}:
        raise ValueError("The editing model returned an unsupported pacing style.")
    if result.get("dialogue_preference") not in {"natural", "prioritize", "minimize"}:
        raise ValueError("The editing model returned an unsupported dialogue preference.")
    if result.get("music_role") not in {"source_audio", "score", "structural_guide"}:
        raise ValueError("The editing model returned an unsupported music role.")
    if not isinstance(result.get("theme"), str):
        raise ValueError("The editing model returned an invalid theme.")
    if not isinstance(result.get("desired_content"), list) or any(
        not isinstance(value, str) for value in result["desired_content"]
    ):
        raise ValueError("The editing model returned invalid desired content.")
    if not isinstance(result.get("hook_first"), bool):
        raise ValueError("The editing model returned an invalid hook preference.")
    theme = str(result.get("theme", defaults["theme"]))[:160].strip()
    result["theme"] = theme or defaults["theme"]
    result["desired_content"] = [str(value)[:80] for value in result.get("desired_content", [])[:12]]
    result["hook_first"] = bool(result.get("hook_first", defaults["hook_first"]))
    return result


def critique_edit(
    highlights: list[dict],
    raw_clips: list[tuple[float, float]],
    clips: list[tuple[float, float]],
    target_duration: int,
    video_duration: float,
    request: dict | None = None,
) -> dict:
    """Critique a draft cut before it reaches the renderer."""

    issues = []
    findings = []
    if not clips:
        issues.append("no_renderable_clips")
    rendered_duration = sum(end - start for start, end in clips)
    duration_limit = float(target_duration) + 0.25
    if rendered_duration > duration_limit:
        issues.append("duration_over_budget")
        findings.append({
            "timestamp": 0.0,
            "severity": "high",
            "explanation": f"Draft is {rendered_duration - target_duration:.2f}s over the requested duration.",
            "suggested_repair": "Drop a complete lower-scoring beat or reduce only non-dialogue context.",
        })
    output_cursor = 0.0
    for source_start, source_end in clips:
        if source_start < 0 or source_end <= source_start or source_end > video_duration + 0.05:
            issues.append("invalid_source_interval")
            findings.append({
                "timestamp": round(output_cursor, 3),
                "severity": "critical",
                "explanation": f"Invalid source interval {source_start:.3f}–{source_end:.3f}s for a {video_duration:.3f}s source.",
                "suggested_repair": "Clamp or remove this timeline item before rendering.",
            })
        output_cursor += max(0.0, source_end - source_start)
    if len(highlights) >= 3:
        regions = {min(2, int((item["start"] / max(video_duration, 1)) * 3)) for item in highlights}
        if len(regions) < 2:
            issues.append("story_coverage_too_narrow")
    if request and request.get("structure_style") == "nonlinear_mystery" and len(highlights) >= 3:
        timed = sorted(highlights, key=lambda item: item.get("timeline_order", 0))
        source_positions = [float(item["start"]) for item in timed]
        if len(source_positions) > 1 and all(right > left for left, right in zip(source_positions, source_positions[1:])):
            issues.append("planned_nonlinear_order_lost")
            findings.append({
                "timestamp": 0.0,
                "severity": "medium",
                "explanation": "The planned nonlinear beats ended up in source chronology.",
                "suggested_repair": "Restore the validated beat order without changing source intervals.",
            })
    for highlight in highlights:
        if highlight.get("source") in {"visual", "thrill", "opening", "api"}:
            continue
        anchor = (
            float(highlight.get("context_start", highlight["start"])),
            float(highlight.get("context_end", highlight["end"])),
        )
        related_final_clips = [
            clip
            for clip in clips
            if _overlap_ratio(anchor, clip) > 0.2
        ]
        # A highlight can be deliberately dropped when the complete dialogue
        # beats cannot all fit in the requested duration.  Only report a trim
        # when that highlight still intersects the rendered cut and its words
        # are actually clipped.  The old zip-based check inspected the first
        # N highlights after fitting, not the highlights that survived.
        required_start = float(highlight.get("context_start", highlight["start"])) if highlight.get("timing_approximate") else float(highlight["start"])
        required_end = float(highlight.get("context_end", highlight["end"])) if highlight.get("timing_approximate") else float(highlight["end"])
        if related_final_clips and not any(
            clip[0] <= required_start + 0.05
            and clip[1] >= required_end - 0.05
            for clip in related_final_clips
        ):
            issues.append("dialogue_anchor_trimmed")
            output_time = 0.0
            for clip in clips:
                if _overlap_ratio(anchor, clip) > 0.2:
                    break
                output_time += max(0.0, clip[1] - clip[0])
            findings.append({
                "timestamp": round(output_time, 3),
                "severity": "critical",
                "explanation": f"Selected dialogue beat may be clipped: {str(highlight.get('text', ''))[:120]}",
                "suggested_repair": "Keep its full context interval or replace it with a shorter complete beat.",
            })
            break
        if _is_transcription_artifact(highlight.get("text", "")):
            issues.append("transcription_artifact")
            break
    return {
        "passed": not issues,
        "issues": issues,
        "duration": round(rendered_duration, 3),
        "clip_count": len(clips),
        "findings": findings,
    }


def validate_render(
    output_path: str | Path,
    has_audio: bool,
    expected_duration: float | None = None,
    expected_aspect_ratio: str | None = None,
) -> dict:
    """Probe and fully decode the final file; check streams, duration, and frame shape."""

    try:
        probe = ffmpeg.probe(str(output_path))
        streams = probe.get("streams", [])
        video = next((item for item in streams if item.get("codec_type") == "video"), None)
        audio = next((item for item in streams if item.get("codec_type") == "audio"), None)
        if not video:
            return {"passed": False, "issues": ["missing_video_stream"]}
        video_duration = float(video.get("duration") or probe["format"].get("duration") or 0)
        audio_duration = float(audio.get("duration") or 0) if audio else 0.0
        width = int(video.get("width") or 0)
        height = int(video.get("height") or 0)
        issues = []
        if video_duration <= 0:
            issues.append("empty_video")
        if has_audio and not audio:
            issues.append("missing_audio_stream")
        if audio and abs(video_duration - audio_duration) > 0.35:
            issues.append("audio_video_duration_mismatch")
        if expected_duration is not None and abs(video_duration - expected_duration) > 0.35:
            issues.append("target_duration_mismatch")
        expected_ratio = {"vertical": 9 / 16, "square": 1.0}.get(expected_aspect_ratio)
        actual_ratio = width / height if height else 0.0
        if expected_ratio and (not actual_ratio or abs(actual_ratio - expected_ratio) > 0.025):
            issues.append("aspect_ratio_mismatch")
        try:
            ffmpeg.input(str(output_path)).output(os.devnull, format="null").run(
                quiet=True, capture_stderr=True
            )
            decode_passed = True
        except ffmpeg.Error:
            decode_passed = False
            issues.append("full_decode_failed")
        return {
            "passed": not issues,
            "issues": issues,
            "video_duration": round(video_duration, 3),
            "audio_duration": round(audio_duration, 3),
            "expected_duration": round(expected_duration, 3) if expected_duration is not None else None,
            "width": width,
            "height": height,
            "aspect_ratio": round(actual_ratio, 4) if actual_ratio else None,
            "decode_passed": decode_passed,
        }
    except (OSError, KeyError, TypeError, ValueError, ffmpeg.Error) as error:
        return {"passed": False, "issues": ["render_probe_failed", str(error)]}


def snap_clips_to_shots(
    clips: list[tuple[float, float]],
    boundaries: list[float],
    max_clip_duration: float,
) -> list[tuple[float, float]]:
    """Use nearby scene boundaries without allowing a shot to take over the cut."""

    if len(boundaries) <= 2:
        return clips
    snapped = []
    for start, end in clips:
        nearby_start = max((point for point in boundaries if point <= start and start - point <= 1.2), default=start)
        nearby_end = min((point for point in boundaries if point >= end and point - end <= 1.2), default=end)
        if nearby_end - nearby_start <= max_clip_duration:
            start, end = nearby_start, nearby_end
        snapped.append((round(start, 3), round(end, 3)))
    return snapped


def expand_timestamps(
    dialogues: list[dict],
    secs: float = 2.5,
    video_duration: float | None = None,
    max_clip_duration: float = 10.0,
) -> list[tuple[float, float]]:
    clips = []
    for dialogue in dialogues:
        anchor_start = float(dialogue.get("context_start", dialogue["start"]))
        anchor_end = float(dialogue.get("context_end", dialogue["end"]))
        start = max(0.0, anchor_start - secs)
        end = anchor_end + secs
        if video_duration is not None:
            end = min(video_duration, end)
        is_dialogue = dialogue.get("source") not in {"visual", "thrill", "opening", "api"}
        if end - start > max_clip_duration and not is_dialogue:
            anchor_start = max(0.0, anchor_start)
            anchor_end = min(video_duration or anchor_end, anchor_end)
            center = (anchor_start + anchor_end) / 2
            start = max(0.0, center - max_clip_duration / 2)
            end = min(video_duration or end, start + max_clip_duration)
        if end > start:
            clips.append((round(start, 3), round(end, 3)))
    return clips


def resolve_dialogue_boundaries(
    clips: list[tuple[float, float]],
    anchors: list[dict],
    safety_margin: float = 0.06,
    video_duration: float | None = None,
    speech_intervals: list[tuple[float, float]] | None = None,
) -> list[tuple[float, float]]:
    """Move cuts out of aligned words while keeping source time explicit.

    A boundary inside a word expands to the word edge plus a small margin.
    Approximate sentence timings are deliberately ignored; their parent
    context interval is used by selection and duration fitting instead.
    """
    resolved = []
    for index, (start, end) in enumerate(clips):
        left, right = float(start), float(end)
        anchor = anchors[index] if index < len(anchors) else {}
        if not anchor.get("timing_approximate", False):
            for word in anchor.get("words", []):
                word_start = float(word.get("start", left))
                word_end = float(word.get("end", right))
                if word_start < left < word_end:
                    left = max(0.0, word_start - safety_margin)
                if word_start < right < word_end:
                    right = word_end + safety_margin
        for speech_start, speech_end in speech_intervals or []:
            speech_start, speech_end = float(speech_start), float(speech_end)
            if speech_start < left < speech_end:
                left = max(0.0, speech_start - safety_margin)
            if speech_start < right < speech_end:
                right = speech_end + safety_margin
        if video_duration is not None:
            right = min(float(video_duration), right)
        resolved.append((round(left, 3), round(right, 3)))
    return resolved


def source_dialogue_intervals(dialogues: list[dict]) -> list[tuple[float, float]]:
    """Return conservative original-media speech ranges for cut validation."""
    intervals = []
    seen = set()
    for dialogue in dialogues:
        if dialogue.get("timing_approximate"):
            start = float(dialogue.get("context_start", dialogue["start"]))
            end = float(dialogue.get("context_end", dialogue["end"]))
        elif dialogue.get("words"):
            words = dialogue["words"]
            start = float(words[0].get("start", dialogue["start"]))
            end = float(words[-1].get("end", dialogue["end"]))
        else:
            start, end = float(dialogue["start"]), float(dialogue["end"])
        key = (round(start, 3), round(end, 3))
        if end > start and key not in seen:
            intervals.append(key)
            seen.add(key)
    return sorted(intervals)


def source_audio_suppression_intervals(
    clips: list[tuple[float, float]],
    dialogues: list[dict],
    selected_dialogue: list[dict],
) -> list[tuple[float, float]]:
    """Mute only an unselected utterance fragment created by a picture cut."""
    preserved = []
    for item in selected_dialogue:
        if item.get("source") in {"visual", "thrill", "opening", "api"}:
            continue
        preserved.append((
            float(item.get("context_start", item["start"])),
            float(item.get("context_end", item["end"])),
        ))

    suppressed = []
    output_cursor = 0.0
    for clip_start, clip_end in clips:
        for speech_start, speech_end in source_dialogue_intervals(dialogues):
            overlap_start = max(clip_start, speech_start)
            overlap_end = min(clip_end, speech_end)
            if overlap_end <= overlap_start:
                continue
            selected = any(
                selected_start <= speech_start + 0.05 and selected_end >= speech_end - 0.05
                for selected_start, selected_end in preserved
            )
            complete = clip_start <= speech_start - 0.05 and clip_end >= speech_end + 0.05
            if selected or complete:
                continue
            suppressed.append((
                round(output_cursor + overlap_start - clip_start, 3),
                round(output_cursor + overlap_end - clip_start, 3),
            ))
        output_cursor += max(0.0, clip_end - clip_start)

    merged = []
    for start, end in sorted(suppressed):
        if merged and start <= merged[-1][1] + 0.05:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def validate_dialogue_timeline(
    clips: list[tuple[float, float]],
    dialogues: list[dict],
    selected_dialogue: list[dict],
    source_mute_intervals: list[tuple[float, float]] | None = None,
    audio_mode: str = "source",
) -> dict:
    """Validate source-to-output speech containment and partial-line muting.

    This is a deterministic timeline audit; it does not pretend to recognize
    speech in the encoded output. Final listening or ASR remains a separate QA.
    """
    issues = []
    selected_ranges = [
        source_dialogue_intervals([item])
        for item in selected_dialogue
        if item.get("source") not in {"visual", "thrill", "opening", "api"}
        and any(_overlap_ratio((float(item.get("context_start", item["start"])), float(item.get("context_end", item["end"]))), clip) > 0.05 for clip in clips)
    ]
    required = [pair for group in selected_ranges for pair in group]
    output_map = []
    cursor = 0.0
    for source_start, source_end in clips:
        output_map.append({"source_start": source_start, "source_end": source_end, "output_start": cursor, "output_end": cursor + source_end - source_start})
        cursor += max(0.0, source_end - source_start)
    for speech_start, speech_end in required:
        if not any(start <= speech_start + 0.05 and end >= speech_end - 0.05 for start, end in clips):
            issues.append({"code": "selected_dialogue_not_contained", "source_start": speech_start, "source_end": speech_end})
    if audio_mode in {"source", "mixed", "dialogue"}:
        selected = [interval for group in selected_ranges for interval in group]
        for speech_start, speech_end in source_dialogue_intervals(dialogues):
            complete = any(start <= speech_start - 0.05 and end >= speech_end + 0.05 for start, end in clips)
            is_selected = any(
                start <= speech_start + 0.05 and end >= speech_end - 0.05
                and any(clip_start <= speech_start + 0.05 and clip_end >= speech_end - 0.05 for clip_start, clip_end in clips)
                for start, end in selected
            )
            if complete or is_selected:
                continue
            for clip_index, (clip_start, clip_end) in enumerate(clips):
                overlap_start, overlap_end = max(clip_start, speech_start), min(clip_end, speech_end)
                if overlap_end <= overlap_start:
                    continue
                mapped_start = sum(end - start for start, end in clips[:clip_index]) + overlap_start - clip_start
                mapped_end = mapped_start + overlap_end - overlap_start
                if not any(mute_start <= mapped_start + 0.03 and mute_end >= mapped_end - 0.03 for mute_start, mute_end in (source_mute_intervals or [])):
                    issues.append({"code": "partial_unselected_dialogue_not_muted", "source_start": overlap_start, "source_end": overlap_end, "output_start": mapped_start, "output_end": mapped_end})
    return {"passed": not issues, "issues": issues, "output_duration": round(cursor, 3), "source_to_output": output_map, "method": "transcript interval containment and mute mapping; no output ASR performed"}


def validate_planned_structure(clips: list[tuple[float, float]], highlights: list[dict], style: str) -> dict:
    """Check objective beat-order invariants on the final source/output map."""
    roles = []
    for clip in clips:
        related = [
            item for item in highlights
            if _overlap_ratio(
                (float(item.get("context_start", item["start"])), float(item.get("context_end", item["end"]))),
                clip,
            ) > 0.2
        ]
        roles.append(next((item["timeline_role"] for item in related if item.get("timeline_role")), None))
    issues = []
    terminal_positions = [index for index, role in enumerate(roles) if role in {"final_sting", "payoff", "emotional_payoff", "title"}]
    if terminal_positions and terminal_positions[-1] != len(clips) - 1:
        issues.append("terminal_beat_not_last")
    starts = [start for start, _ in clips]
    if style == "nonlinear_mystery" and len(starts) >= 3 and all(right >= left for left, right in zip(starts, starts[1:])):
        issues.append("nonlinear_order_lost")
    for left_index, left in enumerate(clips):
        for right in clips[left_index + 1:]:
            if _overlap_ratio(left, right) > 0.95:
                issues.append("repeated_source_moment")
                break
        if "repeated_source_moment" in issues:
            break
    return {"passed": not issues, "issues": issues, "timeline_roles": roles, "source_intervals": [[round(a, 3), round(b, 3)] for a, b in clips], "method": "deterministic role and timestamp checks; no visual or narrative perception performed"}


def separate_overlapping_clip_contexts(
    clips: list[tuple[float, float]],
    anchors: list[dict],
) -> list[tuple[float, float]]:
    """Trim only overlapping context, keeping distinct selected anchors intact."""
    if len(clips) != len(anchors) or len(clips) < 2:
        return clips
    result = [[float(start), float(end)] for start, end in clips]
    order = sorted(range(len(result)), key=lambda index: result[index][0])
    for left_index, right_index in zip(order, order[1:]):
        if result[left_index][1] <= result[right_index][0]:
            continue
        left = anchors[left_index]
        right = anchors[right_index]
        left_end = float(left.get("context_end", left["end"]))
        right_start = float(right.get("context_start", right["start"]))
        if left_end < right_start:
            pivot = (left_end + right_start) / 2
            result[left_index][1] = min(result[left_index][1], pivot)
            result[right_index][0] = max(result[right_index][0], pivot)
    return [(round(start, 3), round(end, 3)) for start, end in result]


def merge_clips(clips: list[tuple[float, float]], max_gap: float = 0.8) -> list[tuple[float, float]]:
    if not clips:
        return []
    merged: list[list[float]] = []
    for start, end in sorted(clips):
        if not merged or start - merged[-1][1] > max_gap:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(round(start, 3), round(end, 3)) for start, end in merged]


def _fit_clips_to_budget(
    clips: list[tuple[float, float]],
    target_duration: int,
    anchors: list[dict] | None = None,
) -> list[tuple[float, float]]:
    """Fit context into the target while never trimming selected dialogue first."""

    if not clips:
        return []
    total = sum(end - start for start, end in clips)
    if total <= target_duration:
        return clips

    if anchors and len(anchors) == len(clips):
        anchors = [
            (
                {
                    **item,
                    "start": clip[0],
                    "end": clip[1],
                    "context_start": clip[0],
                    "context_end": clip[1],
                }
                if item.get("source") in {"visual", "thrill", "opening", "api"}
                else {
                    **item,
                    "start": float(item.get("context_start", item["start"])),
                    "end": float(item.get("context_end", item["end"])),
                    "context_start": float(item.get("context_start", item["start"])),
                    "context_end": float(item.get("context_end", item["end"])),
                }
                if item.get("timing_approximate")
                else item
            )
            for item, clip in zip(anchors, clips)
        ]
        anchor_durations = [
            max(
                0.5,
                float(item.get("context_end", item["end"]))
                - float(item.get("context_start", item["start"])),
            )
            for item in anchors
        ]
        # A complete dialogue group is indivisible. The hard target is applied
        # to the surrounding context; a line that cannot fit is dropped rather
        # than center-cropped into an incoherent fragment.
        budget = float(target_duration)
        anchor_duration = sum(anchor_durations)
        if anchor_duration <= budget:
            context_budget = budget - anchor_duration
            contexts = [
                max(0.0, (float(anchor["start"]) - clip[0]) + (clip[1] - float(anchor["end"])))
                for clip, anchor in zip(clips, anchors)
            ]
            context_total = sum(contexts)
            fitted = []
            for clip, anchor, context in zip(clips, anchors, contexts):
                allowed = context_budget * (context / context_total) if context_total else 0.0
                left_context = min(max(0.0, float(anchor["start"]) - clip[0]), allowed / 2)
                right_context = min(
                    max(0.0, clip[1] - float(anchor["end"])),
                    max(0.0, allowed - left_context),
                )
                fitted.append(
                    (
                        round(max(0.0, float(anchor["start"]) - left_context), 3),
                        round(float(anchor["end"]) + right_context, 3),
                    )
                )
            return fitted

        # The dialogue itself is longer than the requested promo. Keep the
        # strongest complete beats rather than cutting sentences in half.
        ranked_indices = sorted(
            range(len(anchors)),
            key=lambda index: (
                float(anchors[index].get("score", 0))
                + (0.06 if anchors[index].get("context_member_count", 1) > 1 else 0.0)
                + 0.38 * max(0, int(anchors[index].get("context_member_count", 1)) - 1)
                + (0.04 if anchors[index].get("source") != "visual" else 0.0)
                - (0.16 if anchors[index].get("context_member_count", 1) == 1 else 0.0)
            ),
            reverse=True,
        )
        best_subset: tuple[int, ...] = ()
        best_value = float("-inf")
        mandatory = {
            index for index, anchor in enumerate(anchors)
            if anchor.get("source") == "opening" or anchor.get("preserve_opening")
        }
        for size in range(1, len(anchors) + 1):
            for subset in combinations(ranked_indices, size):
                if mandatory and not mandatory.issubset(subset):
                    continue
                duration = sum(anchor_durations[index] for index in subset)
                if duration > budget:
                    continue
                regions = {
                    min(2, int((float(anchors[index]["start"]) / max(1.0, max(item["end"] for item in anchors))) * 3))
                    for index in subset
                }
                value = sum(
                    float(anchors[index].get("score", 0))
                    + (0.06 if anchors[index].get("context_member_count", 1) > 1 else 0.0)
                    + 0.38 * max(0, int(anchors[index].get("context_member_count", 1)) - 1)
                    + (0.04 if anchors[index].get("source") != "visual" else 0.0)
                    - (0.16 if anchors[index].get("context_member_count", 1) == 1 else 0.0)
                    for index in subset
                )
                value += 0.05 * len(regions)
                value += min(0.08, duration / max(budget, 1.0) * 0.08)
                if value > best_value:
                    best_value = value
                    best_subset = subset
        if not best_subset and anchors:
            # A complete dialogue beat longer than the requested promo cannot
            # be made safe by center-cropping it. Drop it and let the selector
            # choose a shorter beat or report that the target is impossible.
            return []
        kept_clips = [clips[index] for index in sorted(best_subset)]
        kept_anchors = [anchors[index] for index in sorted(best_subset)]
        return _fit_clips_to_budget(kept_clips, target_duration, kept_anchors)

    per_clip_budget = target_duration / len(clips)
    fitted = []
    for start, end in clips:
        duration = min(end - start, per_clip_budget)
        center = (start + end) / 2
        fitted_start = max(0.0, center - duration / 2)
        fitted_end = fitted_start + duration
        fitted.append((round(fitted_start, 3), round(fitted_end, 3)))
    return fitted


def extend_clips_to_duration(
    clips: list[tuple[float, float]],
    target_duration: int,
    video_duration: float,
    max_context_per_clip: float | None = None,
) -> list[tuple[float, float]]:
    """Use unselected source context to reach the target without cutting beats."""

    if not clips:
        return []
    target = min(float(target_duration), max(0.0, float(video_duration)))
    ordered = [[float(start), float(end)] for start, end in sorted(clips)]
    current = sum(end - start for start, end in ordered)
    if current >= target - 0.001:
        return [(round(start, 3), round(end, 3)) for start, end in ordered]
    remaining = target - current

    context_added = [0.0 for _ in ordered]
    # Use nearby source context conservatively. A long gap is not permission to
    # turn one selected moment into a lengthy unrelated shot.
    for index, clip in enumerate(ordered):
        if remaining <= 0.001:
            break
        right_limit = ordered[index + 1][0] if index + 1 < len(ordered) else float(video_duration)
        available = max(0.0, right_limit - clip[1])
        allowed = float("inf") if max_context_per_clip is None else max(0.0, max_context_per_clip - context_added[index])
        amount = min(remaining, available, allowed)
        clip[1] += amount
        context_added[index] += amount
        remaining -= amount

    # If the gaps were not enough, use the source edges. This is safe because
    # no selected word-timed anchor is shortened.
    for index, clip in enumerate(ordered):
        if remaining <= 0.001:
            break
        left_limit = ordered[index - 1][1] if index else 0.0
        available = max(0.0, clip[0] - left_limit)
        allowed = float("inf") if max_context_per_clip is None else max(0.0, max_context_per_clip - context_added[index])
        amount = min(remaining, available, allowed)
        clip[0] -= amount
        context_added[index] += amount
        remaining -= amount

    return [(round(max(0.0, start), 3), round(min(video_duration, end), 3)) for start, end in ordered]


def _selected_clip_metadata(
    highlights: list[dict],
    raw_clips: list[tuple[float, float]],
    clips: list[tuple[float, float]],
    dialogues: list[dict] | None = None,
    source_mute_intervals: list[tuple[float, float]] | None = None,
    include_source_dialogue: bool = True,
) -> list[dict]:
    selected = []
    output_cursor = 0.0
    for clip_start, clip_end in clips:
        related = []
        for highlight in highlights:
            anchor_start = float(highlight.get("context_start", highlight["start"]))
            anchor_end = float(highlight.get("context_end", highlight["end"]))
            overlap = max(0.0, min(clip_end, anchor_end) - max(clip_start, anchor_start))
            if overlap > 0.2:
                related.append(highlight)
        if not related and highlights:
            midpoint = (clip_start + clip_end) / 2
            related = [min(highlights, key=lambda item: abs(((item["start"] + item["end"]) / 2) - midpoint))]
        duration = max(0.0, clip_end - clip_start)
        dialogue_context = []
        seen_dialogue = set()
        for dialogue in dialogues or []:
            speech_ranges = source_dialogue_intervals([dialogue])
            if not speech_ranges:
                continue
            source_start, source_end = speech_ranges[0]
            overlap_start = max(clip_start, source_start)
            overlap_end = min(clip_end, source_end)
            if overlap_end - overlap_start <= 0.05:
                continue
            speaker = dialogue.get("speaker_id", dialogue.get("speaker"))
            text = str(dialogue.get("context_text") or dialogue.get("text", "")).strip()
            output_start = round(output_cursor + overlap_start - clip_start, 3)
            output_end = round(output_cursor + overlap_end - clip_start, 3)
            suppressed = not include_source_dialogue or any(
                mute_start <= output_start + 0.05 and mute_end >= output_end - 0.05
                for mute_start, mute_end in (source_mute_intervals or [])
            )
            key = (round(source_start, 3), round(source_end, 3), str(speaker), text)
            if key in seen_dialogue:
                continue
            seen_dialogue.add(key)
            dialogue_context.append({
                "text": text,
                "speaker_id": speaker,
                "source_start": round(source_start, 3),
                "source_end": round(source_end, 3),
                "output_start": output_start,
                "output_end": output_end,
                "timing_approximate": bool(dialogue.get("timing_approximate", False)),
                "audible": not suppressed,
            })
        related_text = [str(item.get("text", "")).strip() for item in related]
        related_text.extend(item["text"] for item in dialogue_context if item["audible"])
        cut_text = []
        seen_text = set()
        for text in related_text:
            normalized_text = re.sub(r"[^a-z0-9']+", " ", text.casefold()).strip()
            if not text or not normalized_text:
                continue
            redundant = any(
                normalized_text == previous
                or (
                    min(len(normalized_text.split()), len(previous.split())) >= 4
                    and (normalized_text in previous or previous in normalized_text)
                )
                for previous in seen_text
            )
            if not redundant:
                cut_text.append(text)
                seen_text.add(normalized_text)
        selected.append(
            {
                # Keep the old source-time fields for API compatibility, and
                # make output-time coordinates explicit for timeline review.
                "start": round(clip_start, 2),
                "end": round(clip_end, 2),
                "source_start": round(clip_start, 3),
                "source_end": round(clip_end, 3),
                "output_start": round(output_cursor, 3),
                "output_end": round(output_cursor + duration, 3),
                "text": " ".join(cut_text),
                "dialogue": dialogue_context,
                "speakers": sorted({str(item["speaker_id"]) for item in dialogue_context if item["audible"] and item.get("speaker_id") is not None}),
                "score": round(max((item.get("score", 0) for item in related), default=0), 4),
                "sources": sorted({str(item.get("source", "dialogue")) for item in related} | ({"dialogue"} if any(item["audible"] for item in dialogue_context) else set())),
                "story_roles": sorted({str(item["story_role"]) for item in related if item.get("story_role")}),
                "timeline_roles": sorted({str(item["timeline_role"]) for item in related if item.get("timeline_role")}),
                "visual_description": " ".join(item["visual_description"] for item in related if item.get("visual_description")),
                "event_types": sorted({str(item["event_type"]) for item in related if item.get("event_type")}),
                "opening_preserved": any(item.get("preserve_opening") for item in related),
                "rationale": next(
                    (str(item["api_reason"]) for item in related if item.get("api_reason")),
                    next((str(item["story_role"]) for item in related if item.get("story_role")),
                    next((str(item["role_rationale"]) for item in related if item.get("role_rationale")),
                    "Selected for its dialogue and audiovisual relevance." if any(
                        item.get("source") not in {"visual", "thrill", "opening"}
                        for item in related
                    ) else "Selected as a visual beat with detected dialogue context." if dialogue_context else "Selected as a visual beat.")),
                ),
            }
        )
        output_cursor += duration
    return selected


def dialogue_timeline_intervals(
    clips: list[tuple[float, float]],
    highlights: list[dict],
    safety_margin: float = 0.12,
    suppressed_intervals: list[tuple[float, float]] | None = None,
) -> list[tuple[float, float]]:
    """Map selected source-dialogue ranges onto output time for soundtrack ducking."""
    dialogue = [
        item for item in highlights
        if item.get("source") not in {"visual", "thrill", "opening", "api"}
    ]
    mapped = []
    output_cursor = 0.0
    for clip_start, clip_end in clips:
        for item in dialogue:
            source_start = float(item.get("context_start", item["start"]))
            source_end = float(item.get("context_end", item["end"]))
            overlap_start = max(clip_start, source_start)
            overlap_end = min(clip_end, source_end)
            if overlap_end <= overlap_start:
                continue
            start = max(0.0, output_cursor + overlap_start - clip_start - safety_margin)
            end = output_cursor + overlap_end - clip_start + safety_margin
            mapped.append((round(start, 3), round(end, 3)))
        output_cursor += max(0.0, clip_end - clip_start)
    merged = []
    for start, end in sorted(mapped):
        if merged and start <= merged[-1][1] + 0.08:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    for mute_start, mute_end in sorted(suppressed_intervals or []):
        mute_start = max(0.0, float(mute_start) - safety_margin)
        mute_end = float(mute_end) + safety_margin
        remaining = []
        for start, end in merged:
            if mute_end <= start or mute_start >= end:
                remaining.append((start, end))
                continue
            if start < mute_start:
                remaining.append((start, mute_start))
            if mute_end < end:
                remaining.append((mute_end, end))
        merged = remaining
    return merged


def order_timeline_for_request(
    clips: list[tuple[float, float]],
    highlights: list[dict],
    request: dict,
) -> list[tuple[float, float]]:
    """Apply requested hook and ending roles, retaining source order otherwise."""
    if not clips:
        return clips
    role_order = {
        "cold_open": 0, "hook": 0, "setup": 1, "context": 1,
        "mystery": 2, "exchange": 2, "connection": 2, "turn": 2,
        "escalation": 3, "peak": 3, "emotional_payoff": 4,
        "payoff": 4, "final_sting": 5, "title": 6,
    }

    def clip_index(candidate: dict) -> int | None:
        anchor = (
            float(candidate.get("context_start", candidate["start"])),
            float(candidate.get("context_end", candidate["end"])),
        )
        index = max(range(len(clips)), key=lambda i: _overlap_ratio(anchor, clips[i]))
        return index if _overlap_ratio(anchor, clips[index]) > 0.2 else None

    planned = [item for item in highlights if item.get("timeline_role") in role_order]
    if planned:
        ranked = []
        used = set()
        terminal_roles = {"final_sting", "payoff", "emotional_payoff", "title"}
        terminal_rank = max((role_order[item["timeline_role"]] for item in planned if item["timeline_role"] in terminal_roles), default=len(role_order))
        for candidate in planned:
            index = clip_index(candidate)
            if index is not None and index not in used:
                used.add(index)
                ranked.append((role_order[candidate["timeline_role"]], float(candidate["start"]), index))
        support_rank = max(0.0, terminal_rank - 0.5)
        ranked.extend((support_rank, float(clips[index][0]), index) for index in range(len(clips)) if index not in used)
        return [clips[index] for _, _, index in sorted(ranked)]

    eligible = [
        item for item in highlights
        if item.get("source") != "opening"
        and item.get("text") != "Opening visual beat"
    ]
    if not eligible:
        return clips

    hook_index = None
    if request.get("hook_first"):
        hook = max(eligible, key=lambda item: float(item.get("score", 0.0)))
        hook_index = clip_index(hook)
        if hook_index is not None:
            hook["timeline_role"] = "hook"

    ending = request.get("ending", "source_order")
    ending_candidates = eligible
    if ending == "reveal":
        ending_candidates = [item for item in eligible if item.get("event_type") == "reveal"]
    elif ending == "unresolved":
        ending_candidates = [item for item in eligible if str(item.get("text", "")).rstrip().endswith("?") or item.get("event_type") == "reveal"]
    elif ending == "emotional":
        emotional_terms = {"love", "loss", "die", "goodbye", "sorry", "family", "home", "miss", "forgive", "together", "promise", "hope", "sacrifice"}
        def emotion_score(item: dict) -> float:
            evidence = " ".join((str(item.get("text", "")), str(item.get("visual_description", "")))).lower()
            terms = len(emotional_terms.intersection(_tokens(evidence)))
            reaction = item.get("event_type") == "reaction"
            return min(1.0, terms / 3.0 + (0.5 if reaction else 0.0))
        ending_candidates = [item for item in eligible if emotion_score(item) >= 0.3]

    if hook_index is not None and len(clips) > 1:
        ending_candidates = [item for item in ending_candidates if clip_index(item) != hook_index]
    ending_candidate = None
    if ending_candidates and ending == "memorable":
        ending_candidate = max(ending_candidates, key=lambda item: (float(item.get("score", 0.0)), float(item.get("start", 0.0))))
    elif ending_candidates and ending == "musical":
        ending_candidate = max(ending_candidates, key=lambda item: (float(item.get("music_score", 0.0)), float(item.get("start", 0.0))))
    elif ending_candidates and ending in {"reveal", "unresolved"}:
        ending_candidate = max(ending_candidates, key=lambda item: float(item.get("score", 0.0)))
    elif ending_candidates and ending == "emotional":
        ending_candidate = max(ending_candidates, key=lambda item: emotion_score(item) + 0.2 * float(item.get("score", 0.0)))

    ending_index = clip_index(ending_candidate) if ending_candidate else None
    if ending_index is not None and ending_candidate is not None:
        ending_candidate["timeline_role"] = "ending"

    ordered_indices = list(range(len(clips)))
    if hook_index is not None:
        ordered_indices.remove(hook_index)
        ordered_indices.insert(0, hook_index)
    if ending_index is not None:
        ordered_indices.remove(ending_index)
        ordered_indices.append(ending_index)
    return [clips[index] for index in ordered_indices]


def stitch_clips(
    video_path: str | Path,
    clips: list[tuple[float, float]],
    output_path: str | Path,
    has_audio: bool = True,
    aspect_ratio: str = "source",
    music_path: str | Path | None = None,
    music_start: float = 0.0,
    dialogue_intervals: list[tuple[float, float]] | None = None,
    source_mute_intervals: list[tuple[float, float]] | None = None,
    audio_mode: str = "auto",
) -> str:
    if not clips:
        raise ValueError("No usable moments were found in this video.")
    if audio_mode not in {"auto", "source", "music", "mixed", "dialogue"}:
        raise ValueError("Audio mode must be auto, source, music, mixed, or dialogue.")
    if audio_mode in {"music", "mixed"} and not music_path:
        raise ValueError("Upload a soundtrack to use music-only or mixed audio mode.")

    source = ffmpeg.input(str(video_path))
    video_streams = []
    audio_streams = []
    for start, end in clips:
        # Each `ss`/`t` input can retain timestamps from the original trailer.
        # Reset every clip before feeding the concat filter; otherwise browsers
        # may play the first section and then seek to a later timestamp at a
        # clip boundary. Trimming one decoded source also avoids independent
        # input seeks landing on incompatible keyframes.
        video_stream = (
            source.video
            .filter("trim", start=start, end=end)
            .filter("fps", fps=30)
            .filter("setpts", "PTS-STARTPTS")
        )
        if aspect_ratio == "vertical":
            video_stream = (
                video_stream
                .filter("scale", 1080, 1920, force_original_aspect_ratio="increase")
                .filter("crop", 1080, 1920, "(in_w-out_w)/2", "(in_h-out_h)/2")
            )
        elif aspect_ratio == "square":
            video_stream = (
                video_stream
                .filter("scale", 1080, 1080, force_original_aspect_ratio="increase")
                .filter("crop", 1080, 1080, "(in_w-out_w)/2", "(in_h-out_h)/2")
            )
        video_streams.append(video_stream)
        if has_audio:
            audio_streams.append(
                source.audio
                .filter("atrim", start=start, end=end)
                .filter("asetpts", "PTS-STARTPTS")
            )

    joined_video = ffmpeg.concat(*video_streams, v=1, a=0)
    joined_audio = ffmpeg.concat(*audio_streams, v=0, a=1) if has_audio else None
    total_duration = sum(max(0.0, end - start) for start, end in clips)
    output_options = {
        "vcodec": "libx264",
        "pix_fmt": "yuv420p",
        "movflags": "+faststart",
    }
    if audio_mode == "auto":
        audio_mode = "mixed" if music_path else "source"
    if music_path and audio_mode != "source":
        music_input = ffmpeg.input(str(music_path), stream_loop=-1, ss=max(0.0, music_start))
        music_track = (
            music_input.audio
            .filter("atrim", start=0, end=total_duration)
            .filter("asetpts", "PTS-STARTPTS")
        )
        base_music_gain = 0.20 if audio_mode == "dialogue" else 0.34
        dialogue_music_gain = 0.05 if audio_mode == "dialogue" else 0.10
        music_gain = f"{base_music_gain:.3f}"
        for dialogue_start, dialogue_end in sorted(dialogue_intervals or []):
            start = max(0.0, float(dialogue_start))
            end = min(total_duration, float(dialogue_end))
            if end <= start:
                continue
            ramp_in_start = max(0.0, start - 0.18)
            ramp_out_end = min(total_duration, end + 0.35)
            ramp_in_slope = (base_music_gain - dialogue_music_gain) / max(0.001, start - ramp_in_start)
            ramp_out_slope = (base_music_gain - dialogue_music_gain) / max(0.001, ramp_out_end - end)
            expression = (
                f"if(between(t,{start:.3f},{end:.3f}),{dialogue_music_gain:.3f},"
                f"if(between(t,{ramp_in_start:.3f},{start:.3f}),{base_music_gain:.3f}-(t-{ramp_in_start:.3f})*{ramp_in_slope:.3f},"
                f"if(between(t,{end:.3f},{ramp_out_end:.3f}),{dialogue_music_gain:.3f}+(t-{end:.3f})*{ramp_out_slope:.3f},"
                f"{music_gain})))"
            )
            music_gain = expression
        music_track = music_track.filter("volume", volume=music_gain, eval="frame")
        if audio_mode == "music" or joined_audio is None:
            final_audio = music_track
        elif audio_mode in {"mixed", "dialogue"}:
            source_gain = 0.82 if audio_mode == "mixed" else 0.95
            source_audio = joined_audio
            source_gain_expression = f"{source_gain:.3f}"
            for mute_start, mute_end in sorted(source_mute_intervals or []):
                start = max(0.0, float(mute_start))
                end = min(total_duration, float(mute_end))
                if end <= start:
                    continue
                ramp_in = max(0.0, start - 0.025)
                ramp_out = min(total_duration, end + 0.04)
                ramp_in_duration = max(0.001, start - ramp_in)
                ramp_out_duration = max(0.001, ramp_out - end)
                source_gain_expression = (
                    f"if(between(t,{start:.3f},{end:.3f}),0,"
                    f"if(between(t,{ramp_in:.3f},{start:.3f}),"
                    f"({source_gain_expression})*(1-(t-{ramp_in:.3f})/{ramp_in_duration:.3f}),"
                    f"if(between(t,{end:.3f},{ramp_out:.3f}),"
                    f"({source_gain_expression})*((t-{end:.3f})/{ramp_out_duration:.3f}),"
                    f"{source_gain_expression})))"
                )
            source_audio = source_audio.filter(
                "volume",
                source_gain_expression if source_mute_intervals else source_gain,
                **({"eval": "frame"} if source_mute_intervals else {}),
            )
            final_audio = ffmpeg.filter(
                [source_audio, music_track],
                "amix",
                inputs=2,
                duration="first",
                dropout_transition=2,
                normalize=0,
            )
        elif joined_audio is not None:
            source_audio = joined_audio
            source_gain_expression = "1.000"
            for mute_start, mute_end in sorted(source_mute_intervals or []):
                start = max(0.0, float(mute_start))
                end = min(total_duration, float(mute_end))
                if end <= start:
                    continue
                ramp_in = max(0.0, start - 0.025)
                ramp_out = min(total_duration, end + 0.04)
                ramp_in_duration = max(0.001, start - ramp_in)
                ramp_out_duration = max(0.001, ramp_out - end)
                source_gain_expression = (
                    f"if(between(t,{start:.3f},{end:.3f}),0,"
                    f"if(between(t,{ramp_in:.3f},{start:.3f}),"
                    f"({source_gain_expression})*(1-(t-{ramp_in:.3f})/{ramp_in_duration:.3f}),"
                    f"if(between(t,{end:.3f},{ramp_out:.3f}),"
                    f"({source_gain_expression})*((t-{end:.3f})/{ramp_out_duration:.3f}),"
                    f"{source_gain_expression})))"
                )
            if source_mute_intervals:
                source_audio = source_audio.filter("volume", source_gain_expression, eval="frame")
            final_audio = source_audio
        final_audio = final_audio.filter("afade", t="in", st=0, d=min(0.18, total_duration))
        final_audio = final_audio.filter(
            "afade", t="out", st=max(0.0, total_duration - 1.25), d=min(1.25, total_duration)
        ).filter("alimiter", limit=0.95)
        output_options.update({"acodec": "aac", "audio_bitrate": "192k"})
        output = ffmpeg.output(joined_video, final_audio, str(output_path), **output_options).overwrite_output()
    elif joined_audio is not None:
        joined_audio = joined_audio.filter("afade", t="in", st=0, d=min(0.12, total_duration))
        joined_audio = joined_audio.filter(
            "afade", t="out", st=max(0.0, total_duration - 0.65), d=min(0.65, total_duration)
        ).filter("alimiter", limit=0.95)
        output_options.update({"acodec": "aac", "audio_bitrate": "160k"})
        output = ffmpeg.output(joined_video, joined_audio, str(output_path), **output_options).overwrite_output()
    else:
        output = ffmpeg.output(joined_video, str(output_path), **output_options).overwrite_output()
    try:
        output.run(quiet=True, capture_stderr=True)
    except ffmpeg.Error as error:
        details = error.stderr.decode(errors="replace") if error.stderr else ""
        raise RuntimeError("FFmpeg could not assemble the selected moments.") from RuntimeError(details)
    return str(output_path)


def resolve_analysis_configuration(
    analysis_mode: str = "auto",
    transcription_provider: str | None = None,
    visual_provider: str | None = None,
    provider_key: str | None = None,
    gemini_api_key: str | None = None,
    elevenlabs_api_key: str | None = None,
) -> dict:
    """Resolve explicit independent providers while retaining legacy mode behavior."""
    if transcription_provider is None and visual_provider is None:
        if analysis_mode == "local":
            transcription_provider, visual_provider = "local", "local"
        elif analysis_mode == "gemini":
            transcription_provider, visual_provider = "local", "gemini"
        elif analysis_mode == "elevenlabs":
            transcription_provider, visual_provider = "elevenlabs", "local"
        elif analysis_mode in {"api", "auto"}:
            legacy_key = provider_key or VISION_API_KEY
            if analysis_mode == "api" or legacy_key:
                transcription_provider, visual_provider = "openai", "openai"
                gemini_api_key = elevenlabs_api_key = legacy_key
            else:
                transcription_provider, visual_provider = "local", "local"
        else:
            raise ValueError("Unknown analysis mode.")
    else:
        transcription_provider = transcription_provider or "local"
        visual_provider = visual_provider or "local"
        if analysis_mode not in {"auto", "local", "providers"}:
            raise ValueError("Choose providers using the independent provider controls.")

    if transcription_provider not in {"local", "elevenlabs", "openai"}:
        raise ValueError("Unknown transcription provider.")
    if visual_provider not in {"local", "gemini", "openai"}:
        raise ValueError("Unknown visual intelligence provider.")
    transcription_key = elevenlabs_api_key or (provider_key if transcription_provider == "elevenlabs" else None)
    if transcription_provider == "elevenlabs":
        transcription_key = transcription_key or ELEVENLABS_API_KEY
        if not transcription_key:
            raise ValueError("ElevenLabs transcription is selected, but no ElevenLabs API key was provided.")
    elif transcription_provider == "openai":
        transcription_key = provider_key or VISION_API_KEY
        if not transcription_key:
            raise ValueError("OpenAI transcription is selected, but PROMO_VISION_API_KEY is not configured.")
    visual_key = gemini_api_key or (provider_key if visual_provider == "gemini" else None)
    if visual_provider == "gemini":
        visual_key = visual_key or GEMINI_API_KEY
        # Allow missing key - logic layer will fallback to local analysis
    elif visual_provider == "openai":
        visual_key = provider_key or VISION_API_KEY
        if not visual_key:
            raise ValueError("OpenAI visual analysis is selected, but PROMO_VISION_API_KEY is not configured.")
    return {
        "transcription_provider": transcription_provider,
        "visual_provider": visual_provider,
        "transcription_key": transcription_key,
        "visual_key": visual_key,
        "legacy_auto": analysis_mode == "auto" and transcription_provider == "openai" and visual_provider == "openai",
    }


def generate_promo(
    video_path: str | Path,
    theme: str,
    target_duration: int = 30,
    energy: str = "balanced",
    aspect_ratio: str = "source",
    strategy: str = "theme",
    analysis_mode: str = "auto",
    music_path: str | Path | None = None,
    audio_mode: str = "auto",
    job_id: str | None = None,
    progress_callback: ProgressCallback | None = None,
    provider_key: str | None = None,
    transcription_provider: str | None = None,
    visual_provider: str | None = None,
    gemini_api_key: str | None = None,
    elevenlabs_api_key: str | None = None,
) -> dict:
    """Generate one isolated promo and return its metadata."""

    _progress(progress_callback, 0.01, "Interpreting edit request")
    request = interpret_edit_request(
        theme,
        target_duration=target_duration,
        energy=energy,
        strategy=strategy,
        aspect_ratio=aspect_ratio,
        music_enabled=bool(music_path),
    )
    theme = request["theme"]
    target_duration = request["target_duration"]
    energy = request["energy"]
    strategy = request["strategy"]
    aspect_ratio = request["aspect_ratio"]
    if target_duration not in {15, 30, 45, 60, 90, 120}:
        raise ValueError("Target duration must be 15, 30, 45, 60, 90, or 120 seconds.")
    if energy not in {"balanced", "cinematic", "high-impact"}:
        raise ValueError("Unknown creative direction.")
    if aspect_ratio not in {"source", "vertical", "square"}:
        raise ValueError("Unknown output format.")
    if strategy not in {"theme", "best-overall", "story-arc", "visual-montage"}:
        raise ValueError("Unknown selection strategy.")
    configuration = resolve_analysis_configuration(
        analysis_mode=analysis_mode,
        transcription_provider=transcription_provider,
        visual_provider=visual_provider,
        provider_key=provider_key,
        gemini_api_key=gemini_api_key,
        elevenlabs_api_key=elevenlabs_api_key,
    )
    transcription_provider = configuration["transcription_provider"]
    visual_provider = configuration["visual_provider"]
    transcription_key = configuration["transcription_key"]
    visual_key = configuration["visual_key"]
    if audio_mode not in {"auto", "source", "music", "mixed", "dialogue"}:
        raise ValueError("Unknown soundtrack mode.")
    if audio_mode in {"music", "mixed"} and not music_path:
        raise ValueError("Upload a soundtrack to use music-only or mixed audio mode.")
    if audio_mode == "source":
        music_path = None
    effective_audio_mode = audio_mode
    if effective_audio_mode == "auto":
        effective_audio_mode = "mixed" if music_path else "source"
    if visual_provider in {"gemini", "openai"}:
        try:
            request = interpret_edit_request_with_api(
                request["prompt"], request, visual_provider, visual_key
            )
        except Exception:
            if not configuration["legacy_auto"]:
                raise
    theme = request["theme"]
    target_duration = request["target_duration"]
    energy = request["energy"]
    strategy = request["strategy"]
    aspect_ratio = request["aspect_ratio"]

    metadata = probe_video(video_path)
    music_metadata = probe_audio(music_path) if music_path else None
    agent_plan = build_edit_plan(
        theme,
        target_duration,
        energy,
        strategy,
        "gemini" if visual_provider == "gemini" else "api" if visual_provider == "openai" else "local",
        music_enabled=bool(music_path),
        request=request,
    )
    agent_trace = [
        {"stage": "request_interpretation", "status": "complete", "details": request},
        {"stage": "edit_plan", "status": "complete", "details": agent_plan},
    ]
    generation_id = job_id or uuid.uuid4().hex
    workspace = GENERATED_DIR / generation_id
    workspace.mkdir(parents=True, exist_ok=True)
    output_path = workspace / "promo.mp4"

    _progress(progress_callback, 0.06, "Inspecting video")
    audio_path = None
    music_profile: dict | None = None
    raw_dialogues: list[dict] = []
    energy_profile = {"values": np.array([], dtype=np.float32), "window": 0.4}
    if metadata["has_audio"]:
        audio_path = extract_audio(video_path, workspace)
        energy_profile = read_audio_energy(audio_path)
    if music_path and music_metadata:
        _progress(progress_callback, 0.12, "Analysing soundtrack rhythm")
        music_workspace = workspace / "music-analysis"
        music_analysis_path = extract_audio(music_path, music_workspace)
        music_profile = analyze_music(music_analysis_path, music_metadata["duration"])
    _progress(progress_callback, 0.16, "Preparing audio and visual analysis")

    def transcription_progress(value: float, stage: str) -> None:
        # Transcription duration is not known up front; cap the visible phase
        # so a long trailer still feels responsive.
        _progress(progress_callback, 0.18 + min(0.30, value / max(metadata["duration"], 1.0) * 0.30), stage)

    transcription_backend = "local"
    raw_dialogues: list[dict] = []
    shot_boundaries: list[float] = []
    motion_profile: dict = {"values": np.array([], dtype=np.float32), "window": 0.5}

    # Run transcription and visual analysis in parallel
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {}

        # Submit transcription task
        if audio_path:
            if transcription_provider in {"elevenlabs", "openai"}:
                futures["transcribe_api"] = executor.submit(
                    transcribe_audio_with_provider,
                    transcription_provider, audio_path,
                    api_key=transcription_key,
                    duration=metadata["duration"],
                    progress_callback=transcription_progress,
                )
            if transcription_provider in {"local", "openai"} or not futures:
                futures["transcribe_local"] = executor.submit(
                    transcribe_audio_with_provider,
                    "local", audio_path,
                    duration=metadata["duration"],
                    progress_callback=transcription_progress,
                )

        # Submit visual analysis tasks
        futures["shot_boundaries"] = executor.submit(
            detect_shot_boundaries, video_path, metadata["duration"]
        )
        futures["motion_profile"] = executor.submit(
            read_motion_profile, video_path, metadata["duration"]
        )

        # Collect results
        for name, future in futures.items():
            try:
                result = future.result()
                if name == "transcribe_api":
                    raw_dialogues = result
                    transcription_backend = transcription_provider
                elif name == "transcribe_local" and not raw_dialogues:
                    raw_dialogues = result
                    transcription_backend = "local"
                elif name == "shot_boundaries":
                    shot_boundaries = result
                elif name == "motion_profile":
                    motion_profile = result
            except Exception as e:
                if name in {"transcribe_api", "transcribe_local"}:
                    if not configuration.get("legacy_auto", False) and name == "transcribe_api":
                        raise
                    # For local transcription failure, continue with empty dialogues
                elif name in {"shot_boundaries", "motion_profile"}:
                    # Visual analysis failures - use defaults
                    if name == "shot_boundaries":
                        shot_boundaries = [0.0, metadata["duration"]]
                    else:
                        motion_profile = {"values": np.array([], dtype=np.float32), "window": 0.5}

    visual_backend = "local"
    dialogues = split_dialogue_sentences(merge_dialogue_segments(raw_dialogues))
    _progress(progress_callback, 0.50, "Analysing visual rhythm")
    for dialogue in dialogues:
        dialogue["energy_score"] = sample_audio_energy(energy_profile, dialogue["start"], dialogue["end"])
        dialogue["visual_score"] = sample_visual_change(
            shot_boundaries, dialogue["start"], dialogue["end"]
        )
        dialogue["motion_score"] = sample_motion(motion_profile, dialogue["start"], dialogue["end"])

    visual_candidates = build_visual_candidates(
        shot_boundaries,
        metadata["duration"],
        energy_profile,
        motion_profile,
        dialogues,
    )
    opening_candidate = build_opening_candidate(
        shot_boundaries,
        metadata["duration"],
        energy_profile,
        motion_profile,
    )
    api_result: dict = {}
    if visual_provider in {"gemini", "openai"}:
        _progress(progress_callback, 0.58, "Analysing storyboard with vision API")
        # Check API key availability first
        if not visual_key:
            _progress(progress_callback, 0.60, "API key missing for vision provider; falling back to local analysis")
        else:
            try:
                storyboard = extract_storyboard(
                    video_path,
                    workspace,
                    metadata["duration"],
                    shot_boundaries,
                )
                if not storyboard:
                    raise RuntimeError("No usable storyboard frames could be extracted for the selected visual provider.")
                api_result = analyze_storyboard_with_api(
                    storyboard,
                    theme,
                    metadata["duration"],
                    dialogues,
                    strategy,
                    request=request,
                    provider=visual_provider,
                    api_key=visual_key,
                )
                visual_backend = visual_provider
            except Exception as e:
                # Graceful fallback to local analysis instead of failing
                _progress(progress_callback, 0.60, f"Vision API failed ({e}); falling back to local analysis")
                api_result = {}
    # If no API result (fallback or not requested), ensure visual_backend is local
    if visual_backend == "local" and visual_provider in {"gemini", "openai"}:
        pass  # Already local, local analysis already ran
    provider = "local" if transcription_backend == "local" and visual_backend == "local" else (
        f"{transcription_backend}-audio+{visual_backend}-visual" if audio_path else f"{visual_backend}-visual"
    )

    candidates = dialogues + visual_candidates
    if opening_candidate:
        candidates.insert(0, opening_candidate)
    if api_result:
        candidates = apply_api_moments(
            candidates,
            api_result.get("moments", []),
            metadata["duration"],
        )
    if not candidates:
        raise ValueError("No usable story or visual moments were found in this video.")
    effective_strategy = strategy
    if not dialogues and strategy == "theme":
        effective_strategy = "visual-montage"
    _progress(progress_callback, 0.68, "Ranking story moments")

    agent_trace.append({"stage": "evidence", "status": "complete", "details": {
        "dialogue_segments": len(dialogues),
        "visual_candidates": len(visual_candidates),
        "opening_preservation": bool(opening_candidate),
        "scene_boundaries": len(shot_boundaries),
        "api_used": bool(api_result),
        "content_map": {
            "dialogue_beats": len(dialogues),
            "quiet_visual_scenes": sum(item.get("source") == "visual" for item in visual_candidates),
            "desired_content_terms": request["desired_content"],
            "available_speakers": sorted({str(item.get("speaker_id", item.get("speaker"))) for item in dialogues if item.get("speaker_id", item.get("speaker")) is not None}),
        },
    }})
    context_seconds = {"cinematic": 2.4, "high-impact": 1.35}.get(energy, 1.9)
    max_clip_duration = {"cinematic": 11.0, "high-impact": 4.5}.get(energy, 6.0)
    highlights: list[dict] = []
    raw_clips: list[tuple[float, float]] = []
    clips: list[tuple[float, float]] = []
    critique: dict = {"passed": False, "issues": ["not_run"]}
    for iteration in range(1, MAX_REPAIR_ITERATIONS + 1):
        highlights = select_highlights(
            candidates,
            theme,
            target_duration,
            energy,
            strategy=effective_strategy,
            video_duration=metadata["duration"],
            music_profile=music_profile,
            request=request,
        )
        if not highlights:
            raise ValueError("No strong story moments were found in this video.")
        raw_clips = expand_timestamps(
            highlights,
            secs=context_seconds,
            video_duration=metadata["duration"],
            max_clip_duration=max_clip_duration,
        )
        raw_clips = snap_clips_to_shots(raw_clips, shot_boundaries, max_clip_duration)
        raw_clips = resolve_dialogue_boundaries(
            raw_clips,
            highlights,
            video_duration=metadata["duration"],
        )
        raw_clips = separate_overlapping_clip_contexts(raw_clips, highlights)
        clips = _fit_clips_to_budget(
            raw_clips,
            target_duration,
            highlights,
        )
        extension_cap = {"cinematic": 2.0, "high-impact": 0.9}.get(energy, 1.4)
        clips = extend_clips_to_duration(
            clips,
            target_duration,
            metadata["duration"],
            max_context_per_clip=extension_cap,
        )
        critique = critique_edit(
            highlights,
            raw_clips,
            clips,
            target_duration,
            metadata["duration"],
            request=request,
        )
        agent_trace.append({"stage": "draft_critic", "iteration": iteration, "status": "passed" if critique["passed"] else "repairing", "details": critique})
        if critique["passed"]:
            break
        if "dialogue_anchor_trimmed" in critique["issues"]:
            context_seconds = min(3.5, context_seconds + 0.55)
            max_clip_duration = min(14.0, max_clip_duration + 1.5)
        if "story_coverage_too_narrow" in critique["issues"]:
            effective_strategy = "story-arc"
        if "transcription_artifact" in critique["issues"]:
            candidates = [item for item in candidates if not _is_transcription_artifact(item.get("text", ""))]
    if not clips:
        raise ValueError("The selected moments were too short to assemble into a promo.")
    planned_beats = [
        {
            "role": item.get("timeline_role", "supporting_moment"),
            "purpose": item.get("role_rationale", item.get("timeline_role", "supporting moment")),
            "source_interval": [round(float(item["start"]), 3), round(float(item["end"]), 3)],
            "confidence": 0.68 if item.get("visual_description") or item.get("story_role") else 0.48,
            "candidate_text": str(item.get("text", ""))[:180],
        }
        for item in highlights
    ]
    agent_plan["trailer_structure"] = {
        "style": request.get("structure_style", "cinematic_arc"),
        "beats": planned_beats,
        "ordering": "planned narrative order; source time remains attached to each beat",
    }
    clips = order_timeline_for_request(clips, highlights, request)
    structure_report = validate_planned_structure(
        clips,
        highlights,
        request.get("structure_style", "cinematic_arc"),
    )
    agent_trace.append({"stage": "structure_qa", "status": "passed" if structure_report["passed"] else "failed", "details": structure_report})
    if not structure_report["passed"]:
        raise RuntimeError("Trailer structure validation failed: " + ", ".join(structure_report["issues"]))
    source_mutes = source_audio_suppression_intervals(
        clips,
        dialogues,
        highlights,
    )
    music_excerpt = choose_music_excerpt(
        music_profile or {},
        sum(end - start for start, end in clips),
        pacing=request["pacing"],
    ) if music_profile else None
    if music_excerpt:
        agent_trace.append({"stage": "music_timeline", "status": "complete", "details": music_excerpt})
    _progress(progress_callback, 0.78, "Cutting the promo")
    dialogue_intervals = dialogue_timeline_intervals(
        clips,
        dialogues,
        suppressed_intervals=source_mutes,
    )
    dialogue_report = validate_dialogue_timeline(
        clips,
        dialogues,
        highlights,
        source_mutes,
        effective_audio_mode,
    )
    if not dialogue_report["passed"]:
        raise RuntimeError("Dialogue timeline validation failed: " + ", ".join(sorted({item["code"] for item in dialogue_report["issues"]})))

    stitch_clips(
        video_path,
        clips,
        output_path,
        has_audio=metadata["has_audio"],
        aspect_ratio=aspect_ratio,
        music_path=music_path,
        music_start=music_excerpt["start"] if music_excerpt else 0.0,
        dialogue_intervals=dialogue_intervals if effective_audio_mode in {"mixed", "dialogue"} else [],
        source_mute_intervals=source_mutes if effective_audio_mode in {"source", "mixed", "dialogue"} else [],
        audio_mode=effective_audio_mode,
    )
    _progress(progress_callback, 0.96, "Finalising export")
    render_report = validate_render(
        output_path,
        metadata["has_audio"] or bool(music_path),
        expected_duration=sum(end - start for start, end in clips),
        expected_aspect_ratio=aspect_ratio,
    )
    render_report["dialogue_timeline"] = dialogue_report
    render_report["structure_timeline"] = structure_report
    agent_trace.append({"stage": "render_qa", "status": "passed" if render_report["passed"] else "failed", "details": render_report})
    if not render_report["passed"]:
        raise RuntimeError(f"Render QA failed: {', '.join(render_report['issues'])}")

    selected = _selected_clip_metadata(
        highlights,
        raw_clips,
        clips,
        dialogues=dialogues,
        source_mute_intervals=source_mutes,
        include_source_dialogue=effective_audio_mode in {"source", "mixed", "dialogue"},
    )
    script = api_result.get("script") or build_local_script(theme, selected, bool(dialogues))

    result = {
        "promo_path": str(output_path),
        "duration": round(sum(end - start for start, end in clips), 2),
        "clip_count": len(selected),
        "rendered_clip_count": len(clips),
        "transcript_segments": len(dialogues),
        "clips": selected,
        "opening_preserved": any(item.get("opening_preserved") for item in selected),
        "script": script,
        "theme": theme,
        "energy": energy,
        "aspect_ratio": aspect_ratio,
        "strategy": strategy,
        "analysis_mode": analysis_mode,
        "analysis_provider": provider,
        "has_dialogue": bool(dialogues),
        "music_used": bool(music_path),
        "audio_mode": effective_audio_mode,
        "music_duration": music_metadata["duration"] if music_metadata else None,
        "music_beat_count": music_profile.get("beat_count", 0) if music_profile else 0,
        "music_tempo_bpm": music_profile.get("tempo_bpm") if music_profile else None,
        "music_tempo_confidence": music_profile.get("tempo_confidence", 0.0) if music_profile else None,
        "music_sections": music_profile.get("sections", [])[:24] if music_profile else [],
        "music_excerpt": music_excerpt,
        "music_duck_intervals": len(dialogue_intervals),
        "source_audio_suppressed_intervals": len(source_mutes),
        "target_duration": target_duration,
        "agent_plan": agent_plan,
        "agent_trace": agent_trace,
        "iteration_count": len([item for item in agent_trace if item.get("stage") == "draft_critic"]),
        "render_report": render_report,
    }
    _progress(progress_callback, 1.0, "Promo ready")
    return result


def hello_world():
    return {"message": "hello world"}
