import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.router import api_router
from app.core.config import settings

logger = logging.getLogger("jobl.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Normalizer (seq2seq model — optional)
    if settings.model_dir:
        try:
            from app.services.normalizer import JobTitleNormalizer
            app.state.normalizer = JobTitleNormalizer(settings)
        except Exception:
            logger.exception("normalizer failed to load; falling back to rules-only")
            app.state.normalizer = None
    else:
        logger.warning("MODEL_DIR not set; normalizer disabled (rules-only fallback)")
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

    yield


app = FastAPI(title=settings.app_name, version=settings.app_version, lifespan=lifespan)
app.include_router(api_router, prefix=settings.api_prefix)
