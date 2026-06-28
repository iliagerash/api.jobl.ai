import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.router import api_router
from app.core.config import settings

logger = logging.getLogger("jobl.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.normalizer = None
    app.state.categorizer = None
    app.state.biencoder = None
    app.state.skill_extractor = None
    app.state.salary_predictor = None
    app.state.profitability_predictor = None

    # Categorizer (LightGBM)
    if settings.categorizer_model_path:
        try:
            from app.services.categorizer import JobCategorizer
            app.state.categorizer = JobCategorizer(settings.categorizer_model_path)
        except Exception:
            logger.exception("categorizer failed to load")

    # Bi-encoder (ONNX)
    if settings.biencoder_model_path:
        try:
            from app.services.biencoder import BiEncoder
            app.state.biencoder = BiEncoder(settings.biencoder_model_path)
        except Exception:
            logger.exception("biencoder failed to load")

    # Intelligence models (salary + profitability)
    if settings.intelligence_models_path:
        try:
            from app.services.salary_predictor import SalaryPredictor
            app.state.salary_predictor = SalaryPredictor(settings.intelligence_models_path)
        except Exception:
            logger.exception("salary predictor failed to load")

        try:
            from app.services.profitability_predictor import ProfitabilityPredictor
            app.state.profitability_predictor = ProfitabilityPredictor(settings.intelligence_models_path)
        except Exception:
            logger.exception("profitability predictor failed to load")

    # Skill extractor (taxonomy from Postgres)
    try:
        from app.services.skill_extractor import SkillExtractor
        from app.db.session import SessionLocal
        app.state.skill_extractor = SkillExtractor(SessionLocal)
    except Exception:
        logger.exception("skill extractor failed to load")

    yield


app = FastAPI(title=settings.app_name, version=settings.app_version, lifespan=lifespan)
app.include_router(api_router, prefix=settings.api_prefix)
