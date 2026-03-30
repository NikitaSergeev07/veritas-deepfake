from __future__ import annotations

import asyncio
import hashlib
import io
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps
from pydantic import BaseModel

from backend.deepfake_runtime import DeepfakeDetectorService
from backend.audio_runtime import AudioDetectorService, SUPPORTED_EXTENSIONS as AUDIO_EXTENSIONS, SUPPORTED_CONTENT_TYPES as AUDIO_CONTENT_TYPES
from backend.video_runtime import VideoDetectorService, SUPPORTED_EXTENSIONS as VIDEO_EXTENSIONS, SUPPORTED_CONTENT_TYPES as VIDEO_CONTENT_TYPES

LOGGER = logging.getLogger("veritas")

SERVICE_NAME = "Veritas Deepfake API"
MAX_UPLOAD_MB = int(os.environ.get("IMAGE_WORKSPACE_MAX_UPLOAD_MB", "10"))
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/bmp",
    "image/tiff",
}
ALLOWED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
}

DETECTOR        = DeepfakeDetectorService()
AUDIO_DETECTOR  = AudioDetectorService()
VIDEO_DETECTOR  = VideoDetectorService()
_AUDIO_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="audio_infer")
_IMAGE_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="image_infer")
_VIDEO_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="video_infer")


class ViewScore(BaseModel):
    name: str
    fake_probability: float
    real_probability: float


class ModelScore(BaseModel):
    model_id: str
    display_name: str
    backend: str
    device: str
    weight: float
    fake_probability: float
    real_probability: float
    views: list[ViewScore]


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    max_upload_mb: int
    supported_extensions: list[str]
    supported_content_types: list[str]
    device: str
    model_status: str
    configured_models: list[str]
    loaded_models: list[str]
    view_strategy: list[str]
    load_errors: dict[str, str]


class ImageInspectionResponse(BaseModel):
    status: str
    filename: str
    content_type: str
    extension: str
    size_bytes: int
    size_mb: float
    width: int
    height: int
    format: str
    mode: str
    orientation: str
    aspect_ratio: float
    sha256: str
    uploaded_at: str
    summary: str


class PredictResponse(BaseModel):
    status: str
    filename: str
    label: str
    confidence: float
    fake_probability: float
    real_probability: float
    summary: str
    device: str
    primary_model: str
    fallback_model: str | None = None
    view_names: list[str]
    models: list[ModelScore]
    content_type: str
    extension: str
    size_bytes: int
    size_mb: float
    width: int
    height: int
    format: str
    mode: str
    orientation: str
    aspect_ratio: float
    sha256: str
    uploaded_at: str


class AudioHealthResponse(BaseModel):
    status: str
    service: str
    device: str
    model_status: str
    configured_models: list[str]
    loaded_models: list[str]
    base_models: list[str]
    meta_accuracy: float | None
    supported_extensions: list[str]
    supported_content_types: list[str]
    load_errors: dict[str, str]


class AudioPredictResponse(BaseModel):
    status: str
    filename: str
    label: str
    confidence: float
    fake_probability: float
    real_probability: float
    summary: str
    device: str
    primary_model: str
    base_scores: dict[str, float]
    size_bytes: int
    size_mb: float
    uploaded_at: str


class VideoSegment(BaseModel):
    start: float
    end: float
    fake_probability: float
    label: str


class VideoHealthResponse(BaseModel):
    status: str
    service: str
    device: str
    model_status: str
    configured_models: list[str]
    loaded_models: list[str]
    supported_extensions: list[str]
    supported_content_types: list[str]
    load_errors: dict[str, str]


class VideoPredictResponse(BaseModel):
    status: str
    filename: str
    label: str
    confidence: float
    fake_probability: float
    real_probability: float
    summary: str
    device: str
    primary_model: str
    duration: float
    frames_analyzed: int
    segments: list[VideoSegment]
    size_bytes: int
    size_mb: float
    uploaded_at: str


app = FastAPI(title=SERVICE_NAME, version="0.3.0")

_DEV_ORIGINS = os.environ.get("CORS_ORIGINS", "").split(",") if os.environ.get("CORS_ORIGINS") else []
_DEFAULT_ORIGINS = [
    "http://localhost:5173", "http://127.0.0.1:5173",
    "http://localhost:5174", "http://127.0.0.1:5174",
    "http://localhost:5175", "http://127.0.0.1:5175",
    "http://localhost:3000", "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_DEV_ORIGINS or _DEFAULT_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def preload_detector_if_requested() -> None:
    if os.environ.get("DEEPFAKE_PRELOAD", "0").strip().lower() in {"1", "true", "yes", "on"}:
        try:
            DETECTOR.ensure_loaded()
        except Exception:
            LOGGER.exception("Detector preload failed")
    if os.environ.get("AUDIO_PRELOAD", "0").strip().lower() in {"1", "true", "yes", "on"}:
        try:
            AUDIO_DETECTOR.ensure_loaded()
        except Exception:
            LOGGER.exception("Audio detector preload failed")
    if os.environ.get("VIDEO_PRELOAD", "0").strip().lower() in {"1", "true", "yes", "on"}:
        try:
            VIDEO_DETECTOR.ensure_loaded()
        except Exception:
            LOGGER.exception("Video detector preload failed")


def _guess_extension(filename: str | None) -> str:
    return Path(filename or "").suffix.lower()


def _validate_upload(file: UploadFile, data: bytes) -> tuple[str, str]:
    extension = _guess_extension(file.filename)
    content_type = (file.content_type or "").lower()

    if not data:
        raise HTTPException(status_code=400, detail="Empty file")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=400, detail=f"File is too large. Max size is {MAX_UPLOAD_MB} MB")
    if content_type and content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=400, detail="Only image files are supported")
    if extension and extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Only image files are supported")
    if not content_type and not extension:
        raise HTTPException(status_code=400, detail="Only image files are supported")

    return extension or "unknown", content_type or "application/octet-stream"


def _detect_orientation(width: int, height: int) -> str:
    if width == height:
        return "square"
    return "landscape" if width > height else "portrait"


def _extract_metadata(data: bytes, filename: str, content_type: str, extension: str) -> dict[str, Any]:
    try:
        with Image.open(io.BytesIO(data)) as loaded:
            image_format = loaded.format or extension.replace(".", "").upper() or "UNKNOWN"
            image = ImageOps.exif_transpose(loaded)
            image.load()
            image = image.copy()
            width, height = image.size
            mode = image.mode
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid image file") from exc

    orientation = _detect_orientation(width, height)
    size_mb = round(len(data) / (1024 * 1024), 4)
    aspect_ratio = round(width / height, 4) if height else 0.0
    sha256 = hashlib.sha256(data).hexdigest()
    uploaded_at = datetime.now(timezone.utc).isoformat()
    summary = f"{image_format} image, {width}x{height}px, {orientation}, {size_mb:.2f} MB"

    LOGGER.info("Accepted upload filename=%s format=%s size=%s bytes", filename, image_format, len(data))

    return {
        "filename": filename,
        "content_type": content_type,
        "extension": extension,
        "size_bytes": len(data),
        "size_mb": size_mb,
        "width": width,
        "height": height,
        "format": image_format,
        "mode": mode,
        "orientation": orientation,
        "aspect_ratio": aspect_ratio,
        "sha256": sha256,
        "uploaded_at": uploaded_at,
        "summary": summary,
        "image": image,
    }


@app.get("/")
def root() -> dict[str, str]:
    return {"service": SERVICE_NAME, "status": "ok", "docs": "/docs"}


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    detector_state = DETECTOR.health_snapshot()
    return HealthResponse(
        status="ok",
        service=SERVICE_NAME,
        version=str(app.version),
        max_upload_mb=MAX_UPLOAD_MB,
        supported_extensions=sorted(ALLOWED_EXTENSIONS),
        supported_content_types=sorted(ALLOWED_CONTENT_TYPES),
        device=str(detector_state["device"]),
        model_status=str(detector_state["model_status"]),
        configured_models=[str(item) for item in detector_state["configured_models"]],
        loaded_models=[str(item) for item in detector_state["loaded_models"]],
        view_strategy=[str(item) for item in detector_state["view_strategy"]],
        load_errors={str(key): str(value) for key, value in detector_state["load_errors"].items()},
    )


@app.post("/inspect", response_model=ImageInspectionResponse)
async def inspect_image(file: UploadFile = File(...)) -> ImageInspectionResponse:
    data = await file.read()
    extension, content_type = _validate_upload(file, data)
    filename = file.filename or "upload"
    metadata = _extract_metadata(data, filename=filename, content_type=content_type, extension=extension)
    metadata.pop("image", None)
    return ImageInspectionResponse(status="accepted", **metadata)


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)) -> PredictResponse:
    data = await file.read()
    extension, content_type = _validate_upload(file, data)
    filename = file.filename or "upload"
    metadata = _extract_metadata(data, filename=filename, content_type=content_type, extension=extension)
    image = metadata.pop("image")
    metadata.pop("filename", None)
    metadata.pop("summary", None)

    try:
        loop = asyncio.get_running_loop()
        prediction = await loop.run_in_executor(
            _IMAGE_EXECUTOR,
            lambda: DETECTOR.predict(image),
        )
    except Exception as exc:
        LOGGER.exception("Prediction failed for %s", filename)
        raise HTTPException(status_code=503, detail=f"Detector unavailable: {exc}") from exc

    return PredictResponse(
        status="ok",
        filename=filename,
        label=str(prediction["label"]),
        confidence=float(prediction["confidence"]),
        fake_probability=float(prediction["fake_probability"]),
        real_probability=float(prediction["real_probability"]),
        summary=str(prediction["summary"]),
        device=str(prediction["device"]),
        primary_model=str(prediction["primary_model"]),
        fallback_model=(str(prediction["fallback_model"]) if prediction.get("fallback_model") else None),
        view_names=[str(item) for item in prediction["view_names"]],
        models=[ModelScore(**item) for item in prediction["models"]],
        **metadata,
    )


MAX_AUDIO_MB    = int(os.environ.get("AUDIO_MAX_UPLOAD_MB", "50"))
MAX_AUDIO_BYTES = MAX_AUDIO_MB * 1024 * 1024


def _validate_audio_upload(file: UploadFile, data: bytes) -> tuple[str, str]:
    extension    = _guess_extension(file.filename)
    content_type = (file.content_type or "").lower()

    if not data:
        raise HTTPException(status_code=400, detail="Empty file")
    if len(data) > MAX_AUDIO_BYTES:
        raise HTTPException(status_code=400, detail=f"File is too large. Max size is {MAX_AUDIO_MB} MB")
    if extension and extension not in AUDIO_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Only audio files are supported (wav, mp3, flac, ogg, m4a)")

    return extension or ".wav", content_type or "application/octet-stream"


@app.get("/health/audio", response_model=AudioHealthResponse)
def health_audio() -> AudioHealthResponse:
    state = AUDIO_DETECTOR.health_snapshot()
    return AudioHealthResponse(
        status="ok",
        service=SERVICE_NAME,
        device=str(state["device"]),
        model_status=str(state["model_status"]),
        configured_models=[str(m) for m in state["configured_models"]],
        loaded_models=[str(m) for m in state["loaded_models"]],
        base_models=[str(m) for m in state["base_models"]],
        meta_accuracy=state.get("meta_accuracy"),
        supported_extensions=sorted(AUDIO_EXTENSIONS),
        supported_content_types=sorted(AUDIO_CONTENT_TYPES),
        load_errors={str(k): str(v) for k, v in state["load_errors"].items()},
    )


@app.post("/predict/audio", response_model=AudioPredictResponse)
async def predict_audio(file: UploadFile = File(...)) -> AudioPredictResponse:
    data = await file.read()
    extension, content_type = _validate_audio_upload(file, data)
    filename = file.filename or f"upload{extension}"

    size_bytes = len(data)
    size_mb    = round(size_bytes / (1024 * 1024), 4)
    uploaded_at = datetime.now(timezone.utc).isoformat()

    try:
        loop = asyncio.get_running_loop()
        prediction = await loop.run_in_executor(
            _AUDIO_EXECUTOR,
            lambda: AUDIO_DETECTOR.predict(data, filename),
        )
    except Exception as exc:
        LOGGER.exception("Audio prediction failed for %s", filename)
        raise HTTPException(status_code=503, detail=f"Audio detector unavailable: {exc}") from exc

    return AudioPredictResponse(
        status="ok",
        filename=filename,
        label=str(prediction["label"]),
        confidence=float(prediction["confidence"]),
        fake_probability=float(prediction["fake_probability"]),
        real_probability=float(prediction["real_probability"]),
        summary=str(prediction["summary"]),
        device=str(prediction["device"]),
        primary_model=str(prediction["primary_model"]),
        base_scores={str(k): float(v) for k, v in prediction["base_scores"].items()},
        size_bytes=size_bytes,
        size_mb=size_mb,
        uploaded_at=uploaded_at,
    )


MAX_VIDEO_MB    = int(os.environ.get("VIDEO_MAX_UPLOAD_MB", "100"))
MAX_VIDEO_BYTES = MAX_VIDEO_MB * 1024 * 1024


def _validate_video_upload(file: UploadFile, data: bytes) -> tuple[str, str]:
    extension    = _guess_extension(file.filename)
    content_type = (file.content_type or "").lower()

    if not data:
        raise HTTPException(status_code=400, detail="Empty file")
    if len(data) > MAX_VIDEO_BYTES:
        raise HTTPException(status_code=400, detail=f"File is too large. Max size is {MAX_VIDEO_MB} MB")
    if extension and extension not in VIDEO_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Only video files are supported (mp4, mov, avi, mkv, webm)")

    return extension or ".mp4", content_type or "application/octet-stream"


@app.get("/health/video", response_model=VideoHealthResponse)
def health_video() -> VideoHealthResponse:
    state = VIDEO_DETECTOR.health_snapshot()
    return VideoHealthResponse(
        status="ok",
        service=SERVICE_NAME,
        device=str(state["device"]),
        model_status=str(state["model_status"]),
        configured_models=[str(m) for m in state["configured_models"]],
        loaded_models=[str(m) for m in state["loaded_models"]],
        supported_extensions=sorted(VIDEO_EXTENSIONS),
        supported_content_types=sorted(VIDEO_CONTENT_TYPES),
        load_errors={str(k): str(v) for k, v in state["load_errors"].items()},
    )


@app.post("/predict/video", response_model=VideoPredictResponse)
async def predict_video(file: UploadFile = File(...)) -> VideoPredictResponse:
    data = await file.read()
    extension, content_type = _validate_video_upload(file, data)
    filename = file.filename or f"upload{extension}"

    size_bytes = len(data)
    size_mb    = round(size_bytes / (1024 * 1024), 4)
    uploaded_at = datetime.now(timezone.utc).isoformat()

    try:
        loop = asyncio.get_running_loop()
        prediction = await loop.run_in_executor(
            _VIDEO_EXECUTOR,
            lambda: VIDEO_DETECTOR.predict(data, filename),
        )
    except Exception as exc:
        LOGGER.exception("Video prediction failed for %s", filename)
        raise HTTPException(status_code=503, detail=f"Video detector unavailable: {exc}") from exc

    return VideoPredictResponse(
        status="ok",
        filename=filename,
        label=str(prediction["label"]),
        confidence=float(prediction["confidence"]),
        fake_probability=float(prediction["fake_probability"]),
        real_probability=float(prediction["real_probability"]),
        summary=str(prediction["summary"]),
        device=str(prediction["device"]),
        primary_model=str(prediction["primary_model"]),
        duration=float(prediction["duration"]),
        frames_analyzed=int(prediction["frames_analyzed"]),
        segments=[VideoSegment(**s) for s in prediction["segments"]],
        size_bytes=size_bytes,
        size_mb=size_mb,
        uploaded_at=uploaded_at,
    )
