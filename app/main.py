import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.api.router import api_router
from app.core.config import settings

logger = logging.getLogger("jobl.api")

_ALLOWED_IPS: frozenset[str] | None = (
    frozenset(ip.strip() for ip in settings.allowed_ips.split(",") if ip.strip())
    if settings.allowed_ips
    else None
)


class IPAllowlistMiddleware(BaseHTTPMiddleware):
    """Return 403 for any request whose TCP source IP is not in ALLOWED_IPS.

    Only the direct connection IP (request.client.host) is checked — not
    X-Forwarded-For, which can be trivially spoofed by the caller.
    When ALLOWED_IPS is unset the middleware is a no-op.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        if _ALLOWED_IPS is not None:
            client_ip = request.client.host if request.client else ""
            if client_ip not in _ALLOWED_IPS:
                logger.warning("Blocked request from %s", client_ip)
                return Response("Forbidden", status_code=403)
        return await call_next(request)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if _ALLOWED_IPS is not None:
        logger.info("IP allowlist active: %s", ", ".join(sorted(_ALLOWED_IPS)))
    else:
        logger.warning("IP allowlist disabled — all IPs are allowed (set ALLOWED_IPS to restrict)")

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
if _ALLOWED_IPS is not None:
    app.add_middleware(IPAllowlistMiddleware)
app.include_router(api_router, prefix=settings.api_prefix)
