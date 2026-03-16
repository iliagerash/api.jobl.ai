from fastapi import APIRouter

from app.api.v1.health import router as health_router
from app.api.v1.process import router as process_router

api_router = APIRouter()
api_router.include_router(health_router, tags=["health"])
api_router.include_router(process_router, tags=["process"])
