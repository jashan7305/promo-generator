from __future__ import annotations

import asyncio
import os
import re
import shutil
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from logic import logic


router = APIRouter()

BASE_DIR = Path(__file__).resolve().parents[1]
STORAGE_DIR = Path(os.getenv("PROMO_STORAGE_DIR", BASE_DIR / "storage"))
UPLOAD_DIR = STORAGE_DIR / "uploads"
MUSIC_DIR = STORAGE_DIR / "music"
GENERATED_DIR = STORAGE_DIR / "generated"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
MUSIC_DIR.mkdir(parents=True, exist_ok=True)
GENERATED_DIR.mkdir(parents=True, exist_ok=True)

MAX_UPLOAD_BYTES = int(os.getenv("PROMO_MAX_UPLOAD_BYTES", 1024 * 1024 * 1024))
MAX_MUSIC_UPLOAD_BYTES = int(os.getenv("PROMO_MAX_MUSIC_UPLOAD_BYTES", 512 * 1024 * 1024))
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".m4v", ".webm", ".mkv"}
ALLOWED_AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".webm"}
CHUNK_SIZE = 1024 * 1024

executor = ThreadPoolExecutor(max_workers=max(1, int(os.getenv("PROMO_WORKERS", "1"))))
jobs: dict[str, dict] = {}
jobs_lock = threading.Lock()


def _set_job(job_id: str, **updates) -> None:
    with jobs_lock:
        if job_id in jobs:
            jobs[job_id].update(updates)


def _public_job(job: dict) -> dict:
    response = {
        "job_id": job["job_id"],
        "status": job["status"],
        "progress": job.get("progress", 0),
        "stage": job.get("stage", "Waiting to start"),
    }
    if job.get("status") == "completed":
        response["result"] = job["result"]
    if job.get("status") == "failed":
        response["error"] = job.get("error", "Generation failed.")
    return response


def _safe_upload_path(video_id: str) -> Path | None:
    if not re.fullmatch(r"[a-f0-9]{32}", video_id or ""):
        return None
    matches = list(UPLOAD_DIR.glob(f"{video_id}.*"))
    matches.extend((UPLOAD_DIR / video_id).glob("source.*"))
    return matches[0] if len(matches) == 1 else None


def _safe_legacy_path(video_path: str | None) -> Path | None:
    """Allow old clients to finish a request without allowing path traversal."""

    if not video_path:
        return None
    candidate = Path(video_path)
    if not candidate.is_absolute():
        candidate = BASE_DIR / candidate
    try:
        resolved = candidate.resolve()
        resolved.relative_to(UPLOAD_DIR.resolve())
        return resolved if resolved.is_file() else None
    except (OSError, ValueError):
        return None


def _safe_music_path(music_id: str) -> Path | None:
    if not re.fullmatch(r"[a-f0-9]{32}", music_id or ""):
        return None
    matches = list(MUSIC_DIR.glob(f"{music_id}.*"))
    return matches[0] if len(matches) == 1 else None


async def _save_upload(file: UploadFile, target: Path) -> int:
    return await _save_upload_with_limit(file, target, MAX_UPLOAD_BYTES, "Video")


async def _save_upload_with_limit(
    file: UploadFile,
    target: Path,
    limit: int,
    label: str,
) -> int:
    size = 0
    with target.open("wb") as destination:
        while chunk := await file.read(CHUNK_SIZE):
            size += len(chunk)
            if size > limit:
                target.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=413,
                    detail=f"{label} is too large. The limit is {limit // (1024 * 1024)} MB.",
                )
            destination.write(chunk)
    return size


@router.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    extension = Path(file.filename or "").suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail="Upload an MP4, MOV, M4V, WebM, or MKV video.",
        )

    video_id = uuid.uuid4().hex
    target = UPLOAD_DIR / f"{video_id}{extension}"
    try:
        size = await _save_upload(file, target)
        metadata = await asyncio.to_thread(logic.probe_video, target)
    except HTTPException:
        raise
    except ValueError as error:
        target.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception as error:
        target.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail="The upload could not be inspected.") from error

    return {
        "video_id": video_id,
        "video_path": str(target.relative_to(BASE_DIR)),
        "filename": file.filename,
        "size": size,
        "metadata": metadata,
    }


@router.post("/upload-music")
async def upload_music(file: UploadFile = File(...)):
    extension = Path(file.filename or "").suffix.lower()
    if extension not in ALLOWED_AUDIO_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail="Upload an MP3, WAV, M4A, AAC, FLAC, OGG, OPUS, or WebM soundtrack.",
        )

    music_id = uuid.uuid4().hex
    target = MUSIC_DIR / f"{music_id}{extension}"
    try:
        size = await _save_upload_with_limit(file, target, MAX_MUSIC_UPLOAD_BYTES, "Soundtrack")
        metadata = await asyncio.to_thread(logic.probe_audio, target)
    except HTTPException:
        raise
    except ValueError as error:
        target.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception as error:
        target.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail="The soundtrack could not be inspected.") from error

    return {
        "music_id": music_id,
        "music_path": str(target.relative_to(BASE_DIR)),
        "filename": file.filename,
        "size": size,
        "metadata": metadata,
    }


@router.post("/upload-url")
async def upload_video_url(url: str = Form(...)):
    video_id = uuid.uuid4().hex
    target_dir = UPLOAD_DIR / video_id
    try:
        downloaded = await asyncio.to_thread(logic.download_youtube_video, url, target_dir)
        video_path = Path(downloaded["path"])
        size = video_path.stat().st_size
        if size > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"Video is too large. The limit is {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.",
            )
    except HTTPException:
        shutil.rmtree(target_dir, ignore_errors=True)
        raise
    except ValueError as error:
        shutil.rmtree(target_dir, ignore_errors=True)
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception as error:
        shutil.rmtree(target_dir, ignore_errors=True)
        raise HTTPException(status_code=502, detail=str(error)) from error

    return {
        "video_id": video_id,
        "video_path": str(video_path.relative_to(BASE_DIR)),
        "filename": downloaded.get("title", "YouTube video"),
        "size": size,
        "source_url": downloaded.get("webpage_url", url),
        "metadata": downloaded["metadata"],
    }


def _run_generation(
    job_id: str,
    video_path: Path,
    music_path: Path | None,
    audio_mode: str,
    theme: str,
    target_duration: int,
    energy: str,
    aspect_ratio: str,
    strategy: str,
    analysis_mode: str,
) -> None:
    def progress(value: float, stage: str) -> None:
        _set_job(job_id, progress=value, stage=stage)

    try:
        _set_job(job_id, status="running", stage="Starting local analysis")
        result = logic.generate_promo(
            video_path,
            theme,
            target_duration=target_duration,
            energy=energy,
            aspect_ratio=aspect_ratio,
            strategy=strategy,
            analysis_mode=analysis_mode,
            music_path=music_path,
            audio_mode=audio_mode,
            job_id=job_id,
            progress_callback=progress,
        )
        relative_output = Path(result["promo_path"]).resolve().relative_to(GENERATED_DIR.resolve())
        result = {
            key: value for key, value in result.items() if key != "promo_path"
        }
        result["download_url"] = f"/api/download/promos/{relative_output.as_posix()}"
        _set_job(job_id, status="completed", progress=1.0, stage="Promo ready", result=result)
    except Exception as error:
        _set_job(job_id, status="failed", progress=1.0, stage="Generation stopped", error=str(error))


@router.post("/generate", status_code=202)
async def generate_promo_route(
    theme: str | None = Form(None),
    edit_prompt: str | None = Form(None),
    video_id: str | None = Form(None),
    music_id: str | None = Form(None),
    audio_mode: str = Form("auto"),
    video_path: str | None = Form(None),
    target_duration: int = Form(30),
    energy: str = Form("balanced"),
    aspect_ratio: str = Form("source"),
    strategy: str = Form("theme"),
    analysis_mode: str = Form("auto"),
):
    request_text = (edit_prompt if edit_prompt is not None else theme or "").strip()
    max_length = 500 if edit_prompt is not None else 160
    if not 2 <= len(request_text) <= max_length:
        label = "Edit prompt" if edit_prompt is not None else "Theme"
        raise HTTPException(status_code=422, detail=f"{label} must be between 2 and {max_length} characters.")
    if target_duration not in {15, 30, 45, 60, 90, 120}:
        raise HTTPException(status_code=422, detail="Target duration must be 15, 30, 45, 60, 90, or 120 seconds.")
    if energy not in {"balanced", "cinematic", "high-impact"}:
        raise HTTPException(status_code=422, detail="Unknown creative direction.")
    if aspect_ratio not in {"source", "vertical", "square"}:
        raise HTTPException(status_code=422, detail="Unknown output format.")
    if strategy not in {"theme", "best-overall", "story-arc", "visual-montage"}:
        raise HTTPException(status_code=422, detail="Unknown selection strategy.")
    if analysis_mode not in {"auto", "local", "api"}:
        raise HTTPException(status_code=422, detail="Unknown analysis mode.")
    if audio_mode not in {"auto", "source", "music", "mixed", "dialogue"}:
        raise HTTPException(status_code=422, detail="Unknown soundtrack mode.")

    resolved_path = _safe_upload_path(video_id or "") or _safe_legacy_path(video_path)
    if not resolved_path:
        raise HTTPException(status_code=404, detail="Uploaded video not found. Upload it again and retry.")
    resolved_music_path = _safe_music_path(music_id or "") if music_id else None
    if music_id and not resolved_music_path:
        raise HTTPException(status_code=404, detail="Uploaded soundtrack not found. Upload it again and retry.")
    if audio_mode in {"music", "mixed"} and not resolved_music_path:
        raise HTTPException(status_code=422, detail="Upload a soundtrack for the selected audio mode.")

    job_id = uuid.uuid4().hex
    with jobs_lock:
        jobs[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "progress": 0,
            "stage": "Queued for local analysis",
        }
    executor.submit(
        _run_generation,
        job_id,
        resolved_path,
        resolved_music_path,
        audio_mode,
        request_text,
        target_duration,
        energy,
        aspect_ratio,
        strategy,
        analysis_mode,
    )
    return {"job_id": job_id, "status_url": f"/api/jobs/{job_id}"}


@router.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Generation job not found.")
        return _public_job(job)


@router.get("/download/promos/{filename:path}")
async def download_promo(filename: str):
    candidate = (GENERATED_DIR / filename).resolve()
    try:
        candidate.relative_to(GENERATED_DIR.resolve())
    except ValueError as error:
        raise HTTPException(status_code=404, detail="File not found") from error
    if not candidate.is_file() or candidate.suffix.lower() != ".mp4":
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(candidate, media_type="video/mp4", filename="promo.mp4")


@router.get("/health")
def health():
    return {
        "status": "ok",
        "active_jobs": sum(job["status"] in {"queued", "running"} for job in jobs.values()),
        "vision_api_configured": bool(logic.VISION_API_KEY),
    }


@router.get("/hello")
def hello():
    return logic.hello_world()
