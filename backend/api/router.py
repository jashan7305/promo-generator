from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
import os, uuid
from logic import logic
import os

router = APIRouter()

UPLOAD_DIR = "temp"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/upload")
async def upload_video(file: UploadFile = File(...)):
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
    try:
        promo_path = logic.generate_promo(video_path, theme)
        filename = os.path.basename(promo_path)
        return {"promo_path": promo_path, "download_url": f"/api/download/{filename}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/download/{filename}")
async def download_promo(filename: str):
    file_path = os.path.join("promo.mp4") if filename == "promo.mp4" else os.path.join("temp", filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path, media_type="video/mp4", filename=filename)

@router.get("/hello")
def hello():
    return logic.hello_world()