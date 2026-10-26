from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api import router


BASE_DIR = Path(__file__).resolve().parent
frontend_dir = BASE_DIR.parent / "frontend"

app = FastAPI(title="Promo Generator")

app.include_router(router.router, prefix="/api")

app.mount("/frontend", StaticFiles(directory=frontend_dir, html=True), name="frontend")

@app.get("/", response_class=FileResponse)
def serve_index():
    return FileResponse(frontend_dir / "index.html")
