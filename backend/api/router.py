from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
import os, uuid
from logic import logic

router = APIRouter()

UPLOAD_DIR = "temp"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/upload")
async def upload_video(file: UploadFile = File(...)) -> dict[str, str]:
    """
    Uploads a video file for processing.

    Args:
        file: The uploaded video file.

    Returns:
        The path to the uploaded file.
    """
    try:
        file_ext = os.path.splitext(file.filename)[1]
        video_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}{file_ext}")
        with open(video_path, "wb") as f:
            f.write(await file.read())
        return {"video_path": video_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/generate")
async def generate_promo_route(video_path: str = Form(...), theme: str = Form(...)) -> dict[str, str]:
    """
    Generates a promo from the uploaded video.

    Args:
        video_path: Path to the uploaded video.
        theme: Theme for the promo.

    Returns:
        The path and download URL for the generated promo.
    """
    try:
        promo_path = logic.generate_promo(video_path, theme)
        filename = os.path.basename(promo_path)
        return {"promo_path": promo_path, "download_url": f"/api/download/promos/{filename}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/download/promos/{filename}")
async def download_promo(filename: str) -> FileResponse:
    """
    Downloads a processed promo file.

    Args:
        filename: Name of the promo file.

    Returns:
        The video file.
    """
    file_path = os.path.join("promos", "promo.mp4") if filename == "promo.mp4" else os.path.join("temp", filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path, media_type="video/mp4", filename=filename)

@router.get("/hello")
def hello() -> dict[str, str]:
    """
    Returns a simple greeting.

    Returns:
        A dictionary with a greeting message.
    """
    return logic.hello_world()
