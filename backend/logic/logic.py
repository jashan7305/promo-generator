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
import uuid
import wave
from pathlib import Path
from threading import Lock
from typing import Callable, Iterable

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
API_TRANSCRIBE_MODEL = os.getenv("PROMO_API_TRANSCRIBE_MODEL", "whisper-1")
API_TIMEOUT_SECONDS = int(os.getenv("PROMO_API_TIMEOUT_SECONDS", "120"))

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


def score_dialogues(dialogues: list[dict], theme: str, strategy: str = "theme") -> list[dict]:
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
        item["api_score"] = max(0.0, min(1.0, float(dialogue.get("api_score", 0.5))))
        if item.get("source") == "visual":
            item["score"] = round(
                0.42 * item["energy_score"]
                + 0.38 * item["visual_score"]
                + 0.20 * item["quality_score"]
                + 0.10 * item["api_score"],
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
                + 0.12 * item["coherence_score"]
                + 0.10 * item["api_score"],
                4,
            )
        scored.append(item)
    return scored


def _overlap_ratio(first: tuple[float, float], second: tuple[float, float]) -> float:
    overlap = max(0.0, min(first[1], second[1]) - max(first[0], second[0]))
    shorter = max(0.001, min(first[1] - first[0], second[1] - second[0]))
    return overlap / shorter


def select_highlights(
    dialogues: list[dict],
    theme: str,
    target_duration: int = 30,
    energy: str = "balanced",
    strategy: str = "theme",
    video_duration: float | None = None,
) -> list[dict]:
    """Select high-signal, non-repeating dialogue moments.

    Selection is relevance-first but deliberately avoids a promo made from
    three adjacent lines or three nearly identical sentences.  The returned
    moments are put back into chronological order for a coherent narrative.
    """

    scored = score_dialogues(dialogues, theme, strategy)
    if not scored:
        return []

    clip_budget = {15: 3, 30: 4, 45: 5, 60: 6, 90: 7, 120: 8}.get(target_duration, 4)
    if energy == "cinematic":
        clip_budget = max(3, clip_budget - 1)
    elif energy == "high-impact":
        clip_budget += 1

    ranked = sorted(scored, key=lambda item: item["score"], reverse=True)
    chosen: list[dict] = []

    def acceptable(candidate: dict) -> bool:
        candidate_window = (float(candidate["start"]), float(candidate["end"]))
        if any(_overlap_ratio(candidate_window, (x["start"], x["end"])) > 0.35 for x in chosen):
            return False
        candidate_words = _tokens(candidate["text"])
        for existing in chosen:
            existing_words = _tokens(existing["text"])
            if candidate_words and existing_words:
                similarity = len(candidate_words & existing_words) / max(1, len(candidate_words | existing_words))
                if similarity > 0.62:
                    return False
        if candidate.get("coherence_score", 1.0) < 0.4:
            return False
        return True

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
            chosen.append(min(opening_candidates, key=lambda item: item["start"]))

        for lower, upper in (
            (0.0, video_duration / 3),
            (video_duration / 3, video_duration * 2 / 3),
            (video_duration * 2 / 3, video_duration + 0.01),
        ):
            for candidate in ranked:
                if lower <= float(candidate["start"]) < upper and acceptable(candidate):
                    chosen.append(candidate)
                    break

    for candidate in ranked:
        if len(chosen) >= clip_budget:
            break
        if acceptable(candidate):
            chosen.append(candidate)

    if len(chosen) < min(2, len(ranked)):
        fallback = [item for item in ranked if item.get("coherence_score", 1.0) >= 0.4]
        chosen = fallback[: min(clip_budget, len(fallback))]
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
        should_merge = unfinished and gap <= 4.0 and (gap <= 0.85 or tiny_continuation or starts_lowercase)
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
    """Detect meaningful visual cuts without adding an OpenCV dependency."""

    ffmpeg_binary = shutil.which("ffmpeg")
    if not ffmpeg_binary:
        return [0.0, duration]

    command = [
        ffmpeg_binary,
        "-hide_banner",
        "-i",
        str(video_path),
        "-vf",
        "select='gt(scene,0.32)',showinfo",
        "-an",
        "-f",
        "null",
        "-",
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
    except OSError:
        return [0.0, duration]

    timestamps = [float(value) for value in re.findall(r"pts_time:([0-9.]+)", completed.stderr)]
    boundaries = [0.0]
    for timestamp in timestamps:
        if 0.5 < timestamp < duration - 0.5 and timestamp - boundaries[-1] >= 0.7:
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
        if end - start < 1.2 or end - start > 8.0:
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
        if energy < 0.38 and motion < 0.22 and visual < 0.65:
            continue
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
                "source": "visual",
            }
        )
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
    """Extract small JPEG storyboard frames for a vision-capable API."""

    workspace = Path(workspace)
    storyboard_dir = workspace / "storyboard"
    storyboard_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for index, timestamp in enumerate(_storyboard_times(duration, boundaries, max_frames)):
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
    return frames


def _api_json_response(response: requests.Response) -> dict:
    if not response.ok:
        detail = response.text[:500]
        raise RuntimeError(f"Vision API request failed ({response.status_code}): {detail}")
    payload = response.json()
    content = payload.get("choices", [{}])[0].get("message", {}).get("content", "")
    if isinstance(content, list):
        content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
    try:
        return json.loads(content)
    except (TypeError, json.JSONDecodeError) as error:
        raise RuntimeError("Vision API returned an invalid structured response.") from error


def transcribe_audio_with_api(audio_path: str, progress_callback: ProgressCallback | None = None) -> list[dict]:
    if not VISION_API_KEY:
        raise RuntimeError("No API key is configured for API audio analysis.")
    with open(audio_path, "rb") as audio_file:
        response = requests.post(
            "https://api.openai.com/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {VISION_API_KEY}"},
            files={"file": ("analysis.wav", audio_file, "audio/wav")},
            data={
                "model": API_TRANSCRIBE_MODEL,
                "response_format": "verbose_json",
                "timestamp_granularities[]": "segment",
            },
            timeout=API_TIMEOUT_SECONDS,
        )
    if not response.ok:
        raise RuntimeError(f"API transcription failed ({response.status_code}): {response.text[:500]}")
    segments = response.json().get("segments", [])
    results = []
    for segment in segments:
        text = str(segment.get("text", "")).strip()
        if not text:
            continue
        results.append(
            {
                "start": round(float(segment.get("start", 0)), 3),
                "end": round(float(segment.get("end", 0)), 3),
                "text": text,
                "confidence": 0.92,
            }
        )
    _progress(progress_callback, 0.5, "API transcription complete")
    return results


def analyze_storyboard_with_api(
    storyboard: list[dict],
    theme: str,
    duration: float,
    dialogues: list[dict],
    strategy: str,
) -> dict:
    """Ask a vision model to reason over the storyboard and transcript.

    The API receives sampled frames rather than the original video file. That
    keeps the request bounded while still giving it visual context. Audio is
    represented by the timestamped transcript and local energy features.
    """

    if not VISION_API_KEY:
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
                "visual beats when they carry story. Return only the requested JSON.\n\n"
                f"Theme: {theme}\nSelection strategy: {strategy}\nVideo duration: {duration:.1f}s\n"
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
                    },
                    "required": ["start", "end", "importance", "reason", "story_role", "transcript_quote"],
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
    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {VISION_API_KEY}", "Content-Type": "application/json"},
        json={
            "model": VISION_API_MODEL,
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
        timeout=API_TIMEOUT_SECONDS,
    )
    return _api_json_response(response)


def apply_api_moments(candidates: list[dict], api_moments: list[dict], duration: float) -> list[dict]:
    """Use API suggestions as a second opinion, never as unchecked timestamps."""

    output = [dict(candidate) for candidate in candidates]
    for moment in api_moments:
        start = max(0.0, min(duration, float(moment.get("start", 0))))
        end = max(start + 0.5, min(duration, float(moment.get("end", start + 1))))
        importance = max(0.0, min(1.0, float(moment.get("importance", 0.5))))
        overlaps = [
            candidate
            for candidate in output
            if _overlap_ratio((start, end), (candidate["start"], candidate["end"])) >= 0.2
        ]
        if overlaps:
            target = max(overlaps, key=lambda item: _overlap_ratio((start, end), (item["start"], item["end"])))
            target["api_score"] = max(float(target.get("api_score", 0.5)), importance)
            target["api_reason"] = str(moment.get("reason", ""))
            target["story_role"] = str(moment.get("story_role", ""))
        else:
            output.append(
                {
                    "start": start,
                    "end": end,
                    "text": str(moment.get("transcript_quote") or "Visual story beat"),
                    "confidence": 0.9,
                    "energy_score": 0.5,
                    "visual_score": importance,
                    "quality_score": 0.75,
                    "coherence_score": 1.0,
                    "api_score": importance,
                    "api_reason": str(moment.get("reason", "")),
                    "story_role": str(moment.get("story_role", "")),
                    "source": "api",
                }
            )
    return output


def build_local_script(theme: str, selected: list[dict], has_dialogue: bool) -> dict:
    quotes = [item["text"] for item in selected if item.get("text") and item["text"] not in {"Visual beat", "Visual story beat"}]
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
        start = max(0.0, float(dialogue["start"]) - secs)
        end = float(dialogue["end"]) + secs
        if video_duration is not None:
            end = min(video_duration, end)
        if end - start > max_clip_duration:
            center = (start + end) / 2
            start = max(0.0, center - max_clip_duration / 2)
            end = min(video_duration or end, start + max_clip_duration)
        if end > start:
            clips.append((round(start, 3), round(end, 3)))
    return clips


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
        anchor_duration = sum(max(0.5, float(item["end"]) - float(item["start"])) for item in anchors)
        if anchor_duration <= target_duration:
            context_budget = target_duration - anchor_duration
            contexts = [
                max(0.0, (clip[0] - float(anchor["start"])) + (clip[1] - float(anchor["end"])))
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
        ranked = sorted(
            zip(clips, anchors),
            key=lambda pair: float(pair[1].get("score", 0)),
            reverse=True,
        )
        kept = []
        remaining = float(target_duration)
        for clip, anchor in ranked:
            duration = min(float(anchor["end"]) - float(anchor["start"]), remaining)
            if duration < 0.8:
                continue
            kept.append((float(anchor["start"]), float(anchor["start"]) + duration))
            remaining -= duration
            if remaining <= 0:
                break
        return sorted((round(start, 3), round(end, 3)) for start, end in kept)

    per_clip_budget = target_duration / len(clips)
    fitted = []
    for start, end in clips:
        duration = min(end - start, per_clip_budget)
        center = (start + end) / 2
        fitted_start = max(0.0, center - duration / 2)
        fitted_end = fitted_start + duration
        fitted.append((round(fitted_start, 3), round(fitted_end, 3)))
    return fitted


def _selected_clip_metadata(
    highlights: list[dict],
    raw_clips: list[tuple[float, float]],
    clips: list[tuple[float, float]],
) -> list[dict]:
    selected = []
    for clip_start, clip_end in clips:
        related = []
        for highlight, raw_clip in zip(highlights, raw_clips):
            overlap = max(0.0, min(clip_end, raw_clip[1]) - max(clip_start, raw_clip[0]))
            if overlap > 0.2:
                related.append(highlight)
        if not related and highlights:
            midpoint = (clip_start + clip_end) / 2
            related = [min(highlights, key=lambda item: abs(((item["start"] + item["end"]) / 2) - midpoint))]
        selected.append(
            {
                "start": round(clip_start, 2),
                "end": round(clip_end, 2),
                "text": " ".join(item["text"] for item in related),
                "score": round(max((item.get("score", 0) for item in related), default=0), 4),
            }
        )
    return selected


def stitch_clips(
    video_path: str | Path,
    clips: list[tuple[float, float]],
    output_path: str | Path,
    has_audio: bool = True,
    aspect_ratio: str = "source",
) -> str:
    if not clips:
        raise ValueError("No usable moments were found in this video.")

    source = ffmpeg.input(str(video_path))
    streams = []
    for start, end in clips:
        # Each `ss`/`t` input can retain timestamps from the original trailer.
        # Reset every clip before feeding the concat filter; otherwise browsers
        # may play the first section and then seek to a later timestamp at a
        # clip boundary.  Trimning one decoded source also avoids independent
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
        streams.append(video_stream)
        if has_audio:
            streams.append(
                source.audio
                .filter("atrim", start=start, end=end)
                .filter("asetpts", "PTS-STARTPTS")
            )

    joined = ffmpeg.concat(*streams, v=1, a=1 if has_audio else 0)
    output_options = {
        "vcodec": "libx264",
        "pix_fmt": "yuv420p",
        "movflags": "+faststart",
    }
    if has_audio:
        output_options.update({"acodec": "aac", "audio_bitrate": "160k"})
    output = joined.output(str(output_path), **output_options).overwrite_output()
    try:
        output.run(quiet=True, capture_stderr=True)
    except ffmpeg.Error as error:
        details = error.stderr.decode(errors="replace") if error.stderr else ""
        raise RuntimeError("FFmpeg could not assemble the selected moments.") from RuntimeError(details)
    return str(output_path)


def generate_promo(
    video_path: str | Path,
    theme: str,
    target_duration: int = 30,
    energy: str = "balanced",
    aspect_ratio: str = "source",
    strategy: str = "theme",
    analysis_mode: str = "auto",
    job_id: str | None = None,
    progress_callback: ProgressCallback | None = None,
) -> dict:
    """Generate one isolated promo and return its metadata."""

    theme = theme.strip()
    if not theme:
        raise ValueError("Add a theme so the generator knows what to look for.")
    if target_duration not in {15, 30, 45, 60, 90, 120}:
        raise ValueError("Target duration must be 15, 30, 45, 60, 90, or 120 seconds.")
    if energy not in {"balanced", "cinematic", "high-impact"}:
        raise ValueError("Unknown creative direction.")
    if aspect_ratio not in {"source", "vertical", "square"}:
        raise ValueError("Unknown output format.")
    if strategy not in {"theme", "best-overall", "story-arc", "visual-montage"}:
        raise ValueError("Unknown selection strategy.")
    if analysis_mode not in {"auto", "local", "api"}:
        raise ValueError("Unknown analysis mode.")
    if analysis_mode == "api" and not VISION_API_KEY:
        raise ValueError("API analysis was selected, but PROMO_VISION_API_KEY is not configured.")

    metadata = probe_video(video_path)
    generation_id = job_id or uuid.uuid4().hex
    workspace = GENERATED_DIR / generation_id
    workspace.mkdir(parents=True, exist_ok=True)
    output_path = workspace / "promo.mp4"

    _progress(progress_callback, 0.06, "Inspecting video")
    audio_path = None
    raw_dialogues: list[dict] = []
    energy_profile = {"values": np.array([], dtype=np.float32), "window": 0.4}
    if metadata["has_audio"]:
        audio_path = extract_audio(video_path, workspace)
        energy_profile = read_audio_energy(audio_path)
    _progress(progress_callback, 0.16, "Preparing audio and visual analysis")

    def transcription_progress(value: float, stage: str) -> None:
        # Transcription duration is not known up front; cap the visible phase
        # so a long trailer still feels responsive.
        _progress(progress_callback, 0.18 + min(0.30, value / max(metadata["duration"], 1.0) * 0.30), stage)

    provider = "local"
    api_requested = analysis_mode in {"api", "auto"} and bool(VISION_API_KEY)
    if audio_path and api_requested:
        try:
            raw_dialogues = transcribe_audio_with_api(audio_path, transcription_progress)
            provider = "openai-audio+local-visual"
        except Exception:
            if analysis_mode == "api":
                raise
    if audio_path and not raw_dialogues:
        raw_dialogues = transcribe_audio(audio_path, transcription_progress)
    dialogues = split_dialogue_sentences(merge_dialogue_segments(raw_dialogues))
    _progress(progress_callback, 0.50, "Analysing visual rhythm")

    shot_boundaries = detect_shot_boundaries(video_path, metadata["duration"])
    motion_profile = read_motion_profile(video_path, metadata["duration"])
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
    api_result: dict = {}
    if analysis_mode in {"api", "auto"} and VISION_API_KEY:
        _progress(progress_callback, 0.58, "Analysing storyboard with vision API")
        try:
            storyboard = extract_storyboard(
                video_path,
                workspace,
                metadata["duration"],
                shot_boundaries,
            )
            if storyboard:
                api_result = analyze_storyboard_with_api(
                    storyboard,
                    theme,
                    metadata["duration"],
                    dialogues,
                    strategy,
                )
                provider = "openai-audio+vision" if audio_path else "openai-vision"
        except Exception:
            if analysis_mode == "api":
                raise

    candidates = dialogues + visual_candidates
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

    highlights = select_highlights(
        candidates,
        theme,
        target_duration,
        energy,
        strategy=effective_strategy,
        video_duration=metadata["duration"],
    )
    if not highlights:
        raise ValueError("No strong story moments were found in this video.")
    context_seconds = {"cinematic": 2.4, "high-impact": 1.35}.get(energy, 1.9)
    max_clip_duration = {"cinematic": 11.0, "high-impact": 8.0}.get(energy, 9.5)
    raw_clips = expand_timestamps(
        highlights,
        secs=context_seconds,
        video_duration=metadata["duration"],
        max_clip_duration=max_clip_duration,
    )
    raw_clips = snap_clips_to_shots(raw_clips, shot_boundaries, max_clip_duration)
    clips = _fit_clips_to_budget(raw_clips, target_duration, highlights)
    clips = merge_clips(clips, max_gap=0.45)
    if not clips:
        raise ValueError("The selected moments were too short to assemble into a promo.")
    _progress(progress_callback, 0.78, "Cutting the promo")

    stitch_clips(
        video_path,
        clips,
        output_path,
        has_audio=metadata["has_audio"],
        aspect_ratio=aspect_ratio,
    )
    _progress(progress_callback, 0.96, "Finalising export")

    selected = _selected_clip_metadata(highlights, raw_clips, clips)
    script = api_result.get("script") or build_local_script(theme, selected, bool(dialogues))

    result = {
        "promo_path": str(output_path),
        "duration": round(sum(end - start for start, end in clips), 2),
        "clip_count": len(selected),
        "rendered_clip_count": len(clips),
        "transcript_segments": len(dialogues),
        "clips": selected,
        "script": script,
        "theme": theme,
        "energy": energy,
        "aspect_ratio": aspect_ratio,
        "strategy": strategy,
        "analysis_mode": analysis_mode,
        "analysis_provider": provider,
        "has_dialogue": bool(dialogues),
        "target_duration": target_duration,
    }
    _progress(progress_callback, 1.0, "Promo ready")
    return result


def hello_world():
    return {"message": "hello world"}
