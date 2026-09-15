"""FastAPI app: JSON API under /api, the comparison UI at /.

    uvicorn seqnet.api:app --port 8000
"""

import logging
import time
from contextlib import asynccontextmanager
from typing import Annotated, Literal, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from seqnet import __version__, config
from seqnet.data import FEATURES, TIMESTEPS
from seqnet.service import ModelService

log = logging.getLogger("seqnet.api")

ModelName = Literal["rnn", "lstm"]
Pixel = Annotated[float, Field(ge=0.0, le=1.0)]
Row = Annotated[list[Pixel], Field(min_length=FEATURES, max_length=FEATURES)]
Image = Annotated[list[Row], Field(min_length=TIMESTEPS, max_length=TIMESTEPS)]


class PredictRequest(BaseModel):
    pixels: Image = Field(description="8x8 grid, row-major, intensities in [0, 1]")
    models: Optional[list[ModelName]] = None


class BatchPredictRequest(BaseModel):
    inputs: Annotated[list[Image], Field(min_length=1, max_length=config.MAX_BATCH)]
    models: Optional[list[ModelName]] = None


def create_app(artifact_dir=config.ARTIFACT_DIR, web_dir=config.WEB_DIR):
    service = ModelService(artifact_dir)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        service.load()
        if not service.ready:
            log.error("no models loaded - run `make train` first")
        yield

    app = FastAPI(title="RNN vs LSTM", version=__version__, lifespan=lifespan)
    app.state.service = service
    app.add_middleware(GZipMiddleware, minimum_size=1024)
    if config.CORS_ORIGINS:
        app.add_middleware(CORSMiddleware, allow_origins=config.CORS_ORIGINS,
                           allow_methods=["GET", "POST"], allow_headers=["*"])

    @app.middleware("http")
    async def access_log(request: Request, call_next):
        started = time.perf_counter()
        response = await call_next(request)
        if request.url.path.startswith("/api"):
            log.info("%s %s -> %d (%.1f ms)", request.method, request.url.path,
                     response.status_code, (time.perf_counter() - started) * 1000)
        return response

    def require(names):
        if not service.ready:
            raise HTTPException(503, "no models loaded - train them with `make train`")
        missing = [n for n in (names or []) if n not in service.models]
        if missing:
            raise HTTPException(503, f"model(s) not loaded: {', '.join(missing)}")

    @app.get("/api/health")
    def health():
        return {"status": "ok" if len(service.models) == 2 else
                          "degraded" if service.ready else "unavailable",
                "version": __version__, "models": sorted(service.models),
                "errors": service.errors}

    @app.get("/api/models")
    def models():
        require(None)
        return {name: {k: v for k, v in meta.items()} for name, meta in service.metadata.items()}

    @app.post("/api/predict")
    def predict(req: PredictRequest):
        require(req.models)
        X = np.asarray(req.pixels, dtype=np.float64)[None]
        results = service.predict(X, req.models)
        preds = {r["prediction"] for r in results.values()}
        return {"results": results, "agree": len(preds) == 1}

    @app.post("/api/predict/batch")
    def predict_batch(req: BatchPredictRequest):
        require(req.models)
        return service.predict_batch(np.asarray(req.inputs, dtype=np.float64), req.models)

    @app.get("/api/compare")
    def compare():
        require(None)
        return service.comparison

    @app.get("/api/samples/random")
    def random_sample(digit: Annotated[Optional[int], Query(ge=0, le=9)] = None):
        pool = np.arange(len(service.y_te))
        if digit is not None:
            pool = pool[service.y_te == digit]
        return service.sample(np.random.default_rng().choice(pool))

    @app.get("/api/samples/{index}")
    def sample(index: int):
        if not 0 <= index < len(service.y_te):
            raise HTTPException(404, f"sample index must be in [0, {len(service.y_te) - 1}]")
        return service.sample(index)

    if web_dir and web_dir.exists():
        app.mount("/", StaticFiles(directory=web_dir, html=True), name="web")
    return app


logging.basicConfig(level=config.LOG_LEVEL,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
app = create_app()
