from __future__ import annotations

from fastapi.testclient import TestClient

from home_dictation_api.api import create_app
from home_dictation_api.engine import AudioDecodeError, TranscriptionResult


class StubTranscriber:
    def __init__(self, *, text: str = "hello world", error: Exception | None = None) -> None:
        self.text = text
        self.error = error
        self.calls: list[bytes] = []

    def transcribe_bytes(self, audio_bytes: bytes, max_tokens: int = 256) -> TranscriptionResult:
        self.calls.append(audio_bytes)
        if self.error is not None:
            raise self.error

        return TranscriptionResult(
            text=self.text,
            audio_duration_seconds=1.0,
            inference_seconds=0.1,
            rtfx=10.0,
        )


class PreloadStubTranscriber(StubTranscriber):
    def __init__(self) -> None:
        super().__init__()
        self.load_calls = 0
        self.is_loaded = False

    def load(self) -> None:
        self.load_calls += 1
        self.is_loaded = True


def test_transcriptions_json_response() -> None:
    transcriber = StubTranscriber(text="test transcript")
    client = TestClient(create_app(transcriber=transcriber, api_key="test-key"))

    response = client.post(
        "/v1/audio/transcriptions",
        headers={"Authorization": "Bearer test-key"},
        files={"file": ("sample.wav", b"fake wav bytes", "audio/wav")},
        data={"model": "whisper-1", "response_format": "json"},
    )

    assert response.status_code == 200
    assert response.json() == {"text": "test transcript"}
    assert transcriber.calls == [b"fake wav bytes"]


def test_transcriptions_text_response() -> None:
    client = TestClient(create_app(transcriber=StubTranscriber(text="plain text"), api_key="test-key"))

    response = client.post(
        "/v1/audio/transcriptions",
        headers={"Authorization": "Bearer test-key"},
        files={"file": ("sample.wav", b"fake wav bytes", "audio/wav")},
        data={"model": "whisper-1", "response_format": "text"},
    )

    assert response.status_code == 200
    assert response.text == "plain text"
    assert response.headers["content-type"].startswith("text/plain")


def test_transcriptions_require_bearer_auth() -> None:
    client = TestClient(create_app(transcriber=StubTranscriber(), api_key="test-key"))

    response = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("sample.wav", b"fake wav bytes", "audio/wav")},
        data={"model": "whisper-1"},
    )

    assert response.status_code == 401
    assert response.json()["error"]["code"] == "invalid_api_key"


def test_transcriptions_return_openai_error_shape_for_bad_audio() -> None:
    client = TestClient(
        create_app(transcriber=StubTranscriber(error=AudioDecodeError("bad audio")), api_key="test-key")
    )

    response = client.post(
        "/v1/audio/transcriptions",
        headers={"Authorization": "Bearer test-key"},
        files={"file": ("sample.wav", b"bad bytes", "audio/wav")},
        data={"model": "whisper-1"},
    )

    assert response.status_code == 400
    assert response.json() == {
        "error": {
            "message": "bad audio",
            "type": "invalid_request_error",
            "param": "file",
            "code": None,
        }
    }


def test_app_preloads_transcriber_on_startup(monkeypatch) -> None:
    monkeypatch.setenv("PRELOAD_MODEL", "1")
    transcriber = PreloadStubTranscriber()

    with TestClient(create_app(transcriber=transcriber, api_key="test-key")) as client:
        response = client.get("/readyz")

    assert transcriber.load_calls == 1
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
