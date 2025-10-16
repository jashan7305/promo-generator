from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from api import router

app = FastAPI(title="Promo Generator")

app.include_router(router.router, prefix="/api")

app.mount("/frontend", StaticFiles(directory="../frontend", html=True), name="frontend")