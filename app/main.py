import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.router import api_router
from app.core.config import settings

logger = logging.getLogger("jobl.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.normalizer = None

# Categorizer (LightGBM — optional)
    if settings.categorizer_model_path:
        try:
            from app.services.categorizer import JobCategorizer
            app.state.categorizer = JobCategorizer(settings.categorizer_model_path)
        except Exception:
            logger.exception("categorizer failed to load; category will be null")
            app.state.categorizer = None
    else:
        logger.warning("CATEGORIZER_MODEL_PATH not set; categorizer disabled")
        app.state.categorizer = None

    # Bi-encoder (ONNX — optional)
    if settings.biencoder_model_path:
        try:
            from app.services.biencoder import BiEncoder
            app.state.biencoder = BiEncoder(settings.biencoder_model_path)
        except Exception:
            logger.exception("biencoder failed to load; embedding will be null")
            app.state.biencoder = None
    else:
        logger.warning("BIENCODER_MODEL_PATH not set; biencoder disabled")
        app.state.biencoder = None

    # Extractor (GGUF — optional)
    if settings.extractor_model_path:
        try:
            from app.services.extractor import JobExtractor
            app.state.extractor = JobExtractor(settings.extractor_model_path)
        except Exception:
            logger.exception("extractor failed to load; /v1/extract will return 503")
            app.state.extractor = None
    else:
        logger.warning("EXTRACTOR_MODEL_PATH not set; extractor disabled")
        app.state.extractor = None

    # Skill extractor (ESCO taxonomy from Postgres)
    try:
        from app.services.skill_extractor import SkillExtractor
        from app.db.session import SessionLocal
        app.state.skill_extractor = SkillExtractor(SessionLocal)
    except Exception:
        logger.exception("skill extractor failed to load; skills will be empty")
        app.state.skill_extractor = None

    yield


app = FastAPI(title=settings.app_name, version=settings.app_version, lifespan=lifespan)
app.include_router(api_router, prefix=settings.api_prefix)
