import argparse
import logging
import time
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings
from app.normalizer import JobTitleNormalizer, pre_strip
from app.schemas import (
    NormalizeBatchRequest,
    NormalizeBatchResponse,
    NormalizeRequest,
    NormalizeResponse,
)


logger = logging.getLogger("jobl.inference.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        app.state.normalizer = JobTitleNormalizer(settings)
    except Exception:
        logger.exception("startup failed while initializing normalizer")
        raise
    logger.info("inference service started")
    try:
        yield
    finally:
        logger.info("inference service shutting down")


app = FastAPI(title="jobl-inference", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
def unhandled_exception_handler(_request: Request, _exc: Exception) -> JSONResponse:
    logger.exception("unhandled exception")
    return JSONResponse(status_code=500, content={"detail": "internal server error"})


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready")
def ready(request: Request) -> JSONResponse:
    normalizer: JobTitleNormalizer = request.app.state.normalizer
    if normalizer.is_ready():
        return JSONResponse(status_code=200, content={"status": "ready"})
    return JSONResponse(status_code=503, content={"status": "loading"})


@app.post("/normalize", response_model=NormalizeResponse)
def normalize(request_body: NormalizeRequest, request: Request) -> NormalizeResponse:
    started = time.perf_counter()
    normalizer: JobTitleNormalizer = request.app.state.normalizer
    title_normalized = normalizer.normalize(request_body.title_raw, request_body.language_code)
    if not title_normalized.strip():
        title_normalized = pre_strip(request_body.title_raw) or request_body.title_raw.strip()
    latency_ms = (time.perf_counter() - started) * 1000
    logger.debug(
        "normalize request title_raw=%r language_code=%r title_normalized=%r latency_ms=%.2f",
        request_body.title_raw,
        request_body.language_code,
        title_normalized,
        latency_ms,
    )
    return NormalizeResponse(title_normalized=title_normalized)


@app.post("/normalize/batch", response_model=NormalizeBatchResponse)
def normalize_batch(request_body: NormalizeBatchRequest, request: Request) -> NormalizeBatchResponse:
    started = time.perf_counter()
    normalizer: JobTitleNormalizer = request.app.state.normalizer
    titles = [item.title_raw for item in request_body.items]
    language_codes = [item.language_code for item in request_body.items]
    normalized_titles = normalizer.normalize_batch(titles, language_codes)
    results = []
    for raw_title, normalized_title in zip(titles, normalized_titles):
        resolved_title = normalized_title
        if not resolved_title.strip():
            resolved_title = pre_strip(raw_title) or raw_title.strip()
        results.append(NormalizeResponse(title_normalized=resolved_title))
    latency_ms = (time.perf_counter() - started) * 1000
    logger.debug("normalize batch request batch_size=%s latency_ms=%.2f", len(titles), latency_ms)
    return NormalizeBatchResponse(results=results)


def run() -> None:
    parser = argparse.ArgumentParser(description="Run Jobl inference API")
    parser.add_argument(
        "--workers",
        type=int,
        default=settings.workers,
        help="Number of uvicorn worker processes (default: WORKERS or 1)",
    )
    args = parser.parse_args()
    workers = max(1, int(args.workers))

    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        log_level=settings.log_level.lower(),
        workers=workers,
    )
