from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
import os, uuid
from logic import logic

router = APIRouter()

UPLOAD_DIR = "temp"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    """
    Uploads a video file to the server.

    Args:
        file (UploadFile): The video file to be uploaded.

    Returns:
        dict: The path where the video is saved.
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
async def generate_promo_route(video_path: str = Form(...), theme: str = Form(...)):
    """
    Generates a promo video based on a uploaded video and a theme.

    Args:
        video_path (str): Path to the uploaded video.
        theme (str): The thematic focus for the promo.

    Returns:
        dict: Information about the generated promo and download URL.
    """
    try:
        promo_path = logic.generate_promo(video_path, theme)
        filename = os.path.basename(promo_path)
        return {"promo_path": promo_path, "download_url": f"/api/download/promos/{filename}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/download/promos/{filename}")
async def download_promo(filename: str):
    """
    Downloads a generated promo video.

    Args:
        filename (str): The name of the file to download.

    Returns:
        FileResponse: The video file response.
    """
    file_path = os.path.join("promos", "promo.mp4") if filename == "promo.mp4" else os.path.join("temp", filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path, media_type="video/mp4", filename=filename)

@router.get("/hello")
def hello():
    """
    Checks if the API is operational.

    Returns:
        dict: A message confirming the API is working.
    """
    return logic.hello_world()
