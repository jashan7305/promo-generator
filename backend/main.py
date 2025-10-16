from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from api import router
import os

frontend_dir = os.path.join(os.path.dirname(__file__), "../frontend")

app = FastAPI(title="Promo Generator")

app.include_router(router.router, prefix="/api")

app.mount("/frontend", StaticFiles(directory="../frontend", html=True), name="frontend")

@app.get("/", response_class=FileResponse)
def serve_index():
    return FileResponse(os.path.join(frontend_dir, "index.html"))