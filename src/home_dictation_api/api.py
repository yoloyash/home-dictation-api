from __future__ import annotations

import os
import secrets
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Protocol

from fastapi import Depends, FastAPI, File, Form, Header, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse

from home_dictation_api.engine import AudioDecodeError, AudioTooLongError, ParakeetTranscriber, TranscriptionError


class TranscriberProtocol(Protocol):
    def transcribe_bytes(self, audio_bytes: bytes, max_tokens: int = 256): ...


class OpenAIAPIError(Exception):
    def __init__(
        self,
        *,
        status_code: int,
        message: str,
        error_type: str = "invalid_request_error",
        param: str | None = None,
        code: str | None = None,
    ) -> None:
        self.status_code = status_code
        self.payload = {
            "error": {
                "message": message,
                "type": error_type,
                "param": param,
                "code": code,
            }
        }
        super().__init__(message)


def build_default_transcriber() -> ParakeetTranscriber:
    return ParakeetTranscriber()


def get_public_model_name() -> str:
    return os.environ.get("PUBLIC_MODEL_NAME", "whisper-1")


def should_preload_model() -> bool:
    value = os.environ.get("PRELOAD_MODEL", "1").strip().lower()
    return value not in {"0", "false", "no", "off"}


def create_app(
    *,
    transcriber: TranscriberProtocol | None = None,
    api_key: str | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if should_preload_model():
            current = app.state.transcriber
            if current is None:
                current = build_default_transcriber()
                app.state.transcriber = current

            load = getattr(current, "load", None)
            if callable(load):
                load()

        yield

    app = FastAPI(title="home-dictation-api", version="0.1.0", lifespan=lifespan)
    app.state.transcriber = transcriber
    app.state.api_key = api_key

    @app.exception_handler(OpenAIAPIError)
    async def openai_error_handler(_: Request, exc: OpenAIAPIError) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content=exc.payload)

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
        param = None
        if exc.errors():
            location = exc.errors()[0].get("loc") or []
            if location:
                param = str(location[-1])

        error = OpenAIAPIError(
            status_code=400,
            message="Invalid request body.",
            param=param,
        )
        return JSONResponse(status_code=error.status_code, content=error.payload)

    def get_expected_api_key(request: Request) -> str | None:
        configured = request.app.state.api_key
        if configured is not None:
            return configured

        value = os.environ.get("API_KEY", "").strip()
        return value or None

    def require_bearer_auth(
        request: Request,
        authorization: str | None = Header(default=None),
    ) -> None:
        expected_api_key = get_expected_api_key(request)
        if not expected_api_key:
            return

        if not authorization or not authorization.startswith("Bearer "):
            raise OpenAIAPIError(
                status_code=401,
                message="Missing or invalid bearer token.",
                error_type="invalid_request_error",
                code="invalid_api_key",
            )

        actual_api_key = authorization[7:].strip()
        if not secrets.compare_digest(actual_api_key, expected_api_key):
            raise OpenAIAPIError(
                status_code=401,
                message="Incorrect API key provided.",
                error_type="invalid_request_error",
                code="invalid_api_key",
            )

    def get_transcriber(request: Request) -> TranscriberProtocol:
        current = request.app.state.transcriber
        if current is None:
            current = build_default_transcriber()
            request.app.state.transcriber = current
        return current

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz(request: Request):
        current = request.app.state.transcriber
        if current is None:
            return JSONResponse(status_code=503, content={"status": "starting"})

        is_loaded = getattr(current, "is_loaded", None)
        if is_loaded is False:
            return JSONResponse(status_code=503, content={"status": "starting"})

        return {"status": "ok"}

    @app.get("/v1/models")
    async def list_models(_: None = Depends(require_bearer_auth)) -> dict[str, object]:
        model_name = get_public_model_name()
        return {
            "object": "list",
            "data": [
                {
                    "id": model_name,
                    "object": "model",
                    "created": 0,
                    "owned_by": "home-dictation-api",
                }
            ],
        }

    @app.get("/v1/models/{model_name}")
    async def get_model(model_name: str, _: None = Depends(require_bearer_auth)) -> dict[str, object]:
        if model_name != get_public_model_name():
            raise OpenAIAPIError(
                status_code=404,
                message=f"The model `{model_name}` does not exist.",
                error_type="invalid_request_error",
                code="model_not_found",
            )

        return {
            "id": model_name,
            "object": "model",
            "created": 0,
            "owned_by": "home-dictation-api",
        }

    @app.post("/v1/audio/transcriptions")
    async def create_transcription(
        _: None = Depends(require_bearer_auth),
        file: UploadFile = File(...),
        model: str = Form(...),
        language: str | None = Form(default=None),
        prompt: str | None = Form(default=None),
        response_format: str = Form(default="json"),
        stream: str | None = Form(default=None),
        transcriber: TranscriberProtocol = Depends(get_transcriber),
    ):
        del model, language, prompt, stream

        if not file.filename:
            raise OpenAIAPIError(
                status_code=400,
                message="A file upload is required.",
                param="file",
            )

        audio_bytes = await file.read()
        if not audio_bytes:
            raise OpenAIAPIError(
                status_code=400,
                message="Uploaded audio file is empty.",
                param="file",
            )

        try:
            result = transcriber.transcribe_bytes(audio_bytes)
        except (AudioDecodeError, AudioTooLongError, TranscriptionError) as exc:
            raise OpenAIAPIError(
                status_code=400,
                message=str(exc),
                param="file",
            ) from exc
        except Exception as exc:
            raise OpenAIAPIError(
                status_code=500,
                message=f"Transcription failed: {exc}",
                error_type="server_error",
                code="internal_error",
            ) from exc

        if response_format == "json":
            return {"text": result.text}
        if response_format == "text":
            return PlainTextResponse(result.text)
        if response_format == "verbose_json":
            payload = asdict(result)
            payload["text"] = result.text
            return payload

        raise OpenAIAPIError(
            status_code=400,
            message=f"Unsupported response_format `{response_format}`.",
            param="response_format",
        )

    return app


app = create_app()
