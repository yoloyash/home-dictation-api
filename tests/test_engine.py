from __future__ import annotations

import io
import wave

import numpy as np

from home_dictation_api import engine


def make_wav_bytes(samples: np.ndarray, *, sample_rate: int = engine.SAMPLE_RATE) -> bytes:
    pcm = np.clip(samples, -1.0, 0.9999695)
    pcm = np.round(pcm * 32767.0).astype("<i2")

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())

    return buffer.getvalue()


def test_load_audio_from_bytes_uses_wav_fast_path(monkeypatch) -> None:
    def fail_ffmpeg(*args, **kwargs):
        raise AssertionError("ffmpeg should not run for PCM WAV input")

    monkeypatch.setattr(engine, "_run_ffmpeg", fail_ffmpeg)

    samples = np.array([0.0, 0.25, -0.25, 0.5, -0.5], dtype=np.float32)
    decoded = engine.load_audio_from_bytes(make_wav_bytes(samples))

    np.testing.assert_allclose(decoded[: samples.size], samples, atol=2.0 / 32768.0)


def test_load_audio_from_bytes_falls_back_to_ffmpeg_for_non_wav(monkeypatch) -> None:
    calls: list[tuple[list[str], bytes | None]] = []

    def fake_ffmpeg(cmd: list[str], stdin: bytes | None = None) -> np.ndarray:
        calls.append((cmd, stdin))
        return np.array([0.125], dtype=np.float32)

    monkeypatch.setattr(engine, "_run_ffmpeg", fake_ffmpeg)

    decoded = engine.load_audio_from_bytes(b"not-a-wav")

    np.testing.assert_allclose(decoded, np.array([0.125], dtype=np.float32))
    assert calls == [
        (
            [
                "ffmpeg",
                "-loglevel",
                "error",
                "-i",
                "pipe:0",
                "-f",
                "f32le",
                "-acodec",
                "pcm_f32le",
                "-ac",
                "1",
                "-ar",
                str(engine.SAMPLE_RATE),
                "-",
            ],
            b"not-a-wav",
        )
    ]


def test_trim_silence_removes_quiet_edges() -> None:
    leading_silence = np.zeros(engine.SILENCE_TRIM_PAD_SAMPLES * 2, dtype=np.float32)
    trailing_silence = np.zeros(engine.SILENCE_TRIM_PAD_SAMPLES * 2, dtype=np.float32)
    speech = 0.2 * np.sin(np.linspace(0.0, 20.0 * np.pi, engine.SAMPLE_RATE // 4, dtype=np.float32))

    trimmed = engine.trim_silence(np.concatenate([leading_silence, speech, trailing_silence]))

    assert speech.size <= trimmed.size <= speech.size + (engine.SILENCE_TRIM_PAD_SAMPLES * 2) + (
        engine.SILENCE_TRIM_FRAME_SAMPLES * 2
    )
    assert trimmed.size < leading_silence.size + speech.size + trailing_silence.size


def test_transcribe_audio_skips_model_work_for_silence(monkeypatch) -> None:
    def fail_compile(*args, **kwargs):
        raise AssertionError("silent clips should not force model load")

    monkeypatch.setattr(engine, "compile_models", fail_compile)

    transcriber = engine.ParakeetTranscriber(model_dir="/tmp/model", device="CPU")
    result = transcriber.transcribe_audio(np.zeros(engine.SAMPLE_RATE // 2, dtype=np.float32))

    assert result.text == ""
    assert result.audio_duration_seconds == 0.5
    assert result.inference_seconds == 0.0
    assert result.rtfx == 0.0


def test_transcriber_reuses_infer_bundle_per_thread(monkeypatch) -> None:
    created_bundles: list[object] = []

    monkeypatch.setattr(engine, "compile_models", lambda model_dir, device: object())

    def fake_create_infer_bundle(loaded: object) -> object:
        bundle = object()
        created_bundles.append(bundle)
        return bundle

    monkeypatch.setattr(engine, "create_infer_bundle", fake_create_infer_bundle)
    monkeypatch.setattr(engine, "transcribe_short_audio", lambda bundle, audio, max_tokens: "ok")

    transcriber = engine.ParakeetTranscriber(model_dir="/tmp/model", device="CPU")
    audio = np.ones(engine.SAMPLE_RATE // 4, dtype=np.float32)

    first = transcriber.transcribe_audio(audio)
    second = transcriber.transcribe_audio(audio)

    assert first.text == "ok"
    assert second.text == "ok"
    assert len(created_bundles) == 1
