from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from logic import logic

router = APIRouter()

@router.get("/hello")
def hello():
    return logic.hello_world()