from fastapi import APIRouter

from app.api.v1.classify import router as classify_router
from app.api.v1.embed import router as embed_router
from app.api.v1.health import router as health_router
from app.api.v1.process import router as process_router

api_router = APIRouter()
api_router.include_router(health_router, tags=["health"])
api_router.include_router(process_router, tags=["process"])
api_router.include_router(embed_router, tags=["embed"])
api_router.include_router(classify_router, tags=["classify"])
