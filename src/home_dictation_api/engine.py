from __future__ import annotations

import io
import json
import os
import threading
import time
import wave
from dataclasses import dataclass
from pathlib import Path

import av
import httpx
import numpy as np
from openvino import Core

DEFAULT_MODEL_DIR = "/opt/models/parakeet-tdt-0.6b-v3-ov"
DEFAULT_DEVICE = "CPU"
SAMPLE_RATE = 16000
BLANK_TOKEN_ID = 8192
DURATION_BINS = (0, 1, 2, 3, 4)
MAX_PREPROCESSOR_SAMPLES = 240000
MEL_BINS = 128
ENCODER_HIDDEN_SIZE = 1024
DECODER_HIDDEN_SIZE = 640
DEFAULT_MAX_TOKENS = 256
SILENCE_TRIM_FRAME_SAMPLES = 160
SILENCE_TRIM_PAD_SAMPLES = 1600
SILENCE_TRIM_MIN_RMS = 0.001
SILENCE_TRIM_PEAK_RATIO = 0.1


class TranscriptionError(RuntimeError):
    """Raised when audio cannot be transcribed."""


class AudioDecodeError(TranscriptionError):
    """Raised when audio cannot be decoded into the model input format."""


class AudioTooLongError(TranscriptionError):
    """Raised when the current short-audio path cannot handle the input."""


@dataclass(frozen=True)
class TranscriptionResult:
    text: str
    audio_duration_seconds: float
    inference_seconds: float
    rtfx: float


@dataclass
class LoadedParakeet:
    preprocessor: object
    encoder: object
    decoder: object
    joint: object
    vocab: list[str]


@dataclass
class InferBundle:
    preprocessor_request: object
    encoder_request: object
    decoder_request: object
    joint_request: object
    vocab: list[str]


def resolve_model_dir(model_dir: str | None = None) -> Path:
    return Path(model_dir or os.environ.get("MODEL_DIR", DEFAULT_MODEL_DIR))


def resolve_device(device: str | None = None) -> str:
    return device or os.environ.get("OPENVINO_DEVICE", DEFAULT_DEVICE)


def should_trim_silence() -> bool:
    value = os.environ.get("TRIM_SILENCE", "1").strip().lower()
    return value not in {"0", "false", "no", "off"}


def _is_wav_bytes(audio_bytes: bytes) -> bool:
    return len(audio_bytes) >= 12 and audio_bytes[:4] == b"RIFF" and audio_bytes[8:12] == b"WAVE"


def _decode_pcm_samples(frames: bytes, sample_width: int) -> np.ndarray:
    if sample_width == 1:
        audio = np.frombuffer(frames, dtype=np.uint8).astype(np.float32)
        return (audio - 128.0) / 128.0

    if sample_width == 2:
        return np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0

    if sample_width == 3:
        raw = np.frombuffer(frames, dtype=np.uint8)
        if raw.size % 3 != 0:
            raise AudioDecodeError("unsupported 24-bit WAV payload length")

        raw = raw.reshape(-1, 3)
        audio = (
            raw[:, 0].astype(np.int32)
            | (raw[:, 1].astype(np.int32) << 8)
            | (raw[:, 2].astype(np.int32) << 16)
        )
        sign_bit = 1 << 23
        audio = (audio ^ sign_bit) - sign_bit
        return audio.astype(np.float32) / float(sign_bit)

    if sample_width == 4:
        return np.frombuffer(frames, dtype="<i4").astype(np.float32) / 2147483648.0

    raise AudioDecodeError(f"unsupported WAV sample width: {sample_width} bytes")


def _resample_audio(audio: np.ndarray, input_sample_rate: int, output_sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    if audio.size == 0 or input_sample_rate == output_sample_rate:
        return np.ascontiguousarray(audio, dtype=np.float32)

    target_size = max(1, int(round(audio.size * output_sample_rate / input_sample_rate)))
    source_positions = np.arange(audio.size, dtype=np.float64)
    target_positions = np.arange(target_size, dtype=np.float64) * input_sample_rate / output_sample_rate
    target_positions = np.clip(target_positions, 0.0, max(0.0, audio.size - 1))
    return np.interp(target_positions, source_positions, audio).astype(np.float32)


def load_audio_from_wav_bytes(audio_bytes: bytes) -> np.ndarray:
    try:
        with wave.open(io.BytesIO(audio_bytes), "rb") as wav_file:
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            sample_rate = wav_file.getframerate()
            frame_count = wav_file.getnframes()
            frames = wav_file.readframes(frame_count)
    except (wave.Error, EOFError) as exc:
        raise AudioDecodeError(f"failed to parse WAV audio: {exc}") from exc

    if channels <= 0:
        raise AudioDecodeError("WAV file has no channels")
    if sample_rate <= 0:
        raise AudioDecodeError("WAV file has an invalid sample rate")

    audio = _decode_pcm_samples(frames, sample_width)
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1, dtype=np.float32)

    return _resample_audio(audio, sample_rate)


def _normalize_pyav_audio(frame: av.AudioFrame) -> np.ndarray:
    audio = np.asarray(frame.to_ndarray())
    if np.issubdtype(audio.dtype, np.integer):
        info = np.iinfo(audio.dtype)
        max_value = float(max(abs(info.min), info.max))
        audio = audio.astype(np.float32) / max_value
    else:
        audio = audio.astype(np.float32, copy=False)

    if audio.ndim == 2:
        audio = audio.mean(axis=0, dtype=np.float32)

    return np.ascontiguousarray(audio, dtype=np.float32)


def _iter_resampled_frames(result: object) -> list[av.AudioFrame]:
    if result is None:
        return []
    if isinstance(result, list):
        return [frame for frame in result if frame is not None]
    return [result] if result is not None else []


def _load_audio_from_pyav_input(source: str | io.BytesIO) -> np.ndarray:
    try:
        with av.open(source) as container:
            audio_stream = next((stream for stream in container.streams if stream.type == "audio"), None)
            if audio_stream is None:
                raise AudioDecodeError("input does not contain an audio stream")

            resampler = av.audio.resampler.AudioResampler(format="fltp", layout="mono", rate=SAMPLE_RATE)
            chunks: list[np.ndarray] = []

            for frame in container.decode(audio=audio_stream.index):
                for resampled_frame in _iter_resampled_frames(resampler.resample(frame)):
                    chunk = _normalize_pyav_audio(resampled_frame)
                    if chunk.size > 0:
                        chunks.append(chunk)

            for resampled_frame in _iter_resampled_frames(resampler.resample(None)):
                chunk = _normalize_pyav_audio(resampled_frame)
                if chunk.size > 0:
                    chunks.append(chunk)
    except AudioDecodeError:
        raise
    except Exception as exc:
        raise AudioDecodeError(f"PyAV failed to decode audio: {exc}") from exc

    if not chunks:
        raise AudioDecodeError("decoded audio buffer is empty")

    return np.ascontiguousarray(np.concatenate(chunks), dtype=np.float32)


def _download_audio_bytes(url: str) -> bytes:
    try:
        response = httpx.get(url, follow_redirects=True, timeout=30.0)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise AudioDecodeError(f"failed to download audio: {exc}") from exc

    if not response.content:
        raise AudioDecodeError("downloaded audio file is empty")

    return response.content


def load_audio_from_source(source: str) -> np.ndarray:
    if source.startswith(("http://", "https://")):
        return load_audio_from_bytes(_download_audio_bytes(source))

    source_path = Path(source)
    if not source_path.is_file():
        raise AudioDecodeError(f"audio source does not exist: {source}")

    return load_audio_from_bytes(source_path.read_bytes())


def load_audio_from_bytes(audio_bytes: bytes) -> np.ndarray:
    if _is_wav_bytes(audio_bytes):
        try:
            return load_audio_from_wav_bytes(audio_bytes)
        except AudioDecodeError:
            pass

    return _load_audio_from_pyav_input(io.BytesIO(audio_bytes))


def trim_silence(audio: np.ndarray) -> np.ndarray:
    audio = np.ascontiguousarray(audio, dtype=np.float32)
    if audio.size == 0:
        return audio

    frame_size = SILENCE_TRIM_FRAME_SAMPLES
    padded_size = int(np.ceil(audio.size / frame_size) * frame_size)
    if padded_size != audio.size:
        padded_audio = np.pad(audio, (0, padded_size - audio.size))
    else:
        padded_audio = audio

    frames = padded_audio.reshape(-1, frame_size)
    frame_rms = np.sqrt(np.mean(np.square(frames, dtype=np.float32), axis=1))
    peak_rms = float(frame_rms.max(initial=0.0))
    if peak_rms <= 0.0:
        return audio[:0]

    threshold = max(SILENCE_TRIM_MIN_RMS, peak_rms * SILENCE_TRIM_PEAK_RATIO)
    active_frames = np.flatnonzero(frame_rms >= threshold)
    if active_frames.size == 0:
        return audio[:0]

    start = max(0, int(active_frames[0] * frame_size) - SILENCE_TRIM_PAD_SAMPLES)
    end = min(audio.size, int((active_frames[-1] + 1) * frame_size) + SILENCE_TRIM_PAD_SAMPLES)
    return np.ascontiguousarray(audio[start:end], dtype=np.float32)


def load_vocab(path: Path, blank_token_id: int) -> list[str]:
    data = json.loads(path.read_text())

    if isinstance(data, dict) and "id_to_token" in data:
        token_data = data["id_to_token"]
    else:
        token_data = data

    if isinstance(token_data, list):
        vocab = list(token_data)
    elif isinstance(token_data, dict):
        vocab_size = max(int(key) for key in token_data) + 1
        vocab = [""] * vocab_size
        for key, value in token_data.items():
            vocab[int(key)] = value
    else:
        raise TranscriptionError(f"unsupported vocabulary format in {path}")

    if len(vocab) <= blank_token_id:
        vocab.extend([""] * (blank_token_id + 1 - len(vocab)))

    return vocab


def decode_tokens(token_ids: list[int], vocab: list[str], blank_token_id: int) -> str:
    pieces: list[str] = []
    first_piece = True

    for token_id in token_ids:
        if token_id == blank_token_id or token_id < 0 or token_id >= len(vocab):
            continue

        piece = vocab[token_id]
        if not piece:
            continue

        prepend_space = piece.startswith("▁")
        if prepend_space:
            piece = piece[1:]

        if not piece:
            continue

        if not first_piece and prepend_space and pieces:
            pieces.append(" ")

        pieces.append(piece)
        first_piece = False

    return "".join(pieces).strip()


def compile_models(model_dir: Path, device: str) -> LoadedParakeet:
    core = Core()
    preprocessor = core.compile_model(core.read_model(model_dir / "parakeet_melspectogram.xml"), device)
    encoder = core.compile_model(core.read_model(model_dir / "parakeet_encoder.xml"), device)
    decoder = core.compile_model(core.read_model(model_dir / "parakeet_decoder.xml"), device)
    joint = core.compile_model(core.read_model(model_dir / "parakeet_joint.xml"), device)
    vocab = load_vocab(model_dir / "parakeet_v3_vocab.json", BLANK_TOKEN_ID)
    return LoadedParakeet(
        preprocessor=preprocessor,
        encoder=encoder,
        decoder=decoder,
        joint=joint,
        vocab=vocab,
    )


def create_infer_bundle(loaded: LoadedParakeet) -> InferBundle:
    return InferBundle(
        preprocessor_request=loaded.preprocessor.create_infer_request(),
        encoder_request=loaded.encoder.create_infer_request(),
        decoder_request=loaded.decoder.create_infer_request(),
        joint_request=loaded.joint.create_infer_request(),
        vocab=loaded.vocab,
    )


def transcribe_short_audio(bundle: InferBundle, audio: np.ndarray, max_tokens: int) -> str:
    if audio.size > MAX_PREPROCESSOR_SAMPLES:
        max_seconds = MAX_PREPROCESSOR_SAMPLES / 16000.0
        raise AudioTooLongError(
            f"audio is too long for the current short-audio path "
            f"({audio.size} samples, max {MAX_PREPROCESSOR_SAMPLES} / {max_seconds:.1f}s)"
        )

    signal = np.zeros((1, MAX_PREPROCESSOR_SAMPLES), dtype=np.float32)
    signal[0, : audio.size] = audio
    signal_length = np.array([audio.size], dtype=np.int64)

    bundle.preprocessor_request.infer(
        {
            "input_signals": signal,
            "input_length": signal_length,
        }
    )
    mel = np.array(bundle.preprocessor_request.get_output_tensor(0).data[:], copy=True)
    mel_length = int(bundle.preprocessor_request.get_output_tensor(1).data[0])
    if mel.shape != (1, MEL_BINS, 1501):
        raise TranscriptionError(f"unexpected preprocessor output shape: {mel.shape}")

    bundle.encoder_request.infer(
        {
            "melspectogram": mel,
            "melspectogram_length": np.array([mel_length], dtype=np.int32),
        }
    )
    encoder_output = np.array(bundle.encoder_request.get_output_tensor(0).data[:], copy=True)
    valid_frames = int(bundle.encoder_request.get_output_tensor(1).data[0])

    if encoder_output.shape[1] != ENCODER_HIDDEN_SIZE:
        raise TranscriptionError(f"unexpected encoder hidden size: {encoder_output.shape}")
    if valid_frames <= 0:
        raise TranscriptionError("encoder produced no valid frames")

    hidden_state = np.zeros((2, 1, DECODER_HIDDEN_SIZE), dtype=np.float32)
    cell_state = np.zeros((2, 1, DECODER_HIDDEN_SIZE), dtype=np.float32)
    last_token = BLANK_TOKEN_ID
    cached_decoder_output: np.ndarray | None = None
    cached_token: int | None = None
    token_ids: list[int] = []
    frame_index = 0
    vocab_size = BLANK_TOKEN_ID + 1
    durations_offset = vocab_size

    while frame_index < valid_frames and len(token_ids) < max_tokens:
        if cached_decoder_output is None or cached_token != last_token:
            bundle.decoder_request.infer(
                {
                    "targets": np.array([[last_token]], dtype=np.int64),
                    "h_in": hidden_state,
                    "c_in": cell_state,
                }
            )
            decoder_output = np.array(bundle.decoder_request.get_output_tensor(0).data[:], copy=True)
            next_hidden = np.array(bundle.decoder_request.get_output_tensor(1).data[:], copy=True)
            next_cell = np.array(bundle.decoder_request.get_output_tensor(2).data[:], copy=True)
            cached_decoder_output = decoder_output
            cached_token = last_token
        else:
            decoder_output = cached_decoder_output
            next_hidden = hidden_state
            next_cell = cell_state

        advance_frame = True
        while advance_frame and frame_index < valid_frames and len(token_ids) < max_tokens:
            encoder_step = encoder_output[:, :, frame_index][:, None, :]
            bundle.joint_request.infer(
                {
                    "encoder_outputs": encoder_step,
                    "decoder_outputs": decoder_output,
                }
            )
            logits = np.array(bundle.joint_request.get_output_tensor(0).data[:], copy=False).reshape(-1)

            best_token = int(np.argmax(logits[:vocab_size]))
            best_duration_index = int(np.argmax(logits[durations_offset : durations_offset + len(DURATION_BINS)]))
            duration = DURATION_BINS[best_duration_index]
            if duration <= 0:
                duration = 1

            if best_token != BLANK_TOKEN_ID:
                token_ids.append(best_token)
                hidden_state = next_hidden.copy()
                cell_state = next_cell.copy()
                last_token = best_token
                cached_decoder_output = None
                advance_frame = False

            frame_index = min(frame_index + duration, valid_frames)

    last_frame = max(0, valid_frames - 1)
    additional_steps = 0
    consecutive_blanks = 0

    while additional_steps < 8 and consecutive_blanks < 1 and len(token_ids) < max_tokens:
        if cached_decoder_output is None or cached_token != last_token:
            bundle.decoder_request.infer(
                {
                    "targets": np.array([[last_token]], dtype=np.int64),
                    "h_in": hidden_state,
                    "c_in": cell_state,
                }
            )
            decoder_output = np.array(bundle.decoder_request.get_output_tensor(0).data[:], copy=True)
            next_hidden = np.array(bundle.decoder_request.get_output_tensor(1).data[:], copy=True)
            next_cell = np.array(bundle.decoder_request.get_output_tensor(2).data[:], copy=True)
            cached_decoder_output = decoder_output
            cached_token = last_token
        else:
            decoder_output = cached_decoder_output
            next_hidden = hidden_state
            next_cell = cell_state

        encoder_step = encoder_output[:, :, last_frame][:, None, :]
        bundle.joint_request.infer(
            {
                "encoder_outputs": encoder_step,
                "decoder_outputs": decoder_output,
            }
        )
        logits = np.array(bundle.joint_request.get_output_tensor(0).data[:], copy=False).reshape(-1)
        best_token = int(np.argmax(logits[:vocab_size]))

        if best_token != BLANK_TOKEN_ID:
            token_ids.append(best_token)
            hidden_state = next_hidden.copy()
            cell_state = next_cell.copy()
            last_token = best_token
            cached_decoder_output = None
            consecutive_blanks = 0
        else:
            consecutive_blanks += 1

        additional_steps += 1

    return decode_tokens(token_ids, bundle.vocab, BLANK_TOKEN_ID)


class ParakeetTranscriber:
    def __init__(self, model_dir: str | Path | None = None, device: str | None = None) -> None:
        self.model_dir = resolve_model_dir(str(model_dir) if model_dir else None)
        self.device = resolve_device(device)
        self._loaded: LoadedParakeet | None = None
        self._load_seconds = 0.0
        self._lock = threading.Lock()
        self._bundle_local = threading.local()

    @property
    def model_load_seconds(self) -> float:
        return self._load_seconds

    @property
    def is_loaded(self) -> bool:
        return self._loaded is not None

    def load(self) -> LoadedParakeet:
        if self._loaded is not None:
            return self._loaded

        with self._lock:
            if self._loaded is None:
                started = time.perf_counter()
                self._loaded = compile_models(self.model_dir, self.device)
                self._load_seconds = time.perf_counter() - started

        return self._loaded

    def _get_infer_bundle(self) -> InferBundle:
        loaded = self.load()
        bundle = getattr(self._bundle_local, "bundle", None)
        if bundle is None or getattr(self._bundle_local, "loaded_id", None) != id(loaded):
            bundle = create_infer_bundle(loaded)
            self._bundle_local.bundle = bundle
            self._bundle_local.loaded_id = id(loaded)
        return bundle

    def transcribe_audio(self, audio: np.ndarray, max_tokens: int = DEFAULT_MAX_TOKENS) -> TranscriptionResult:
        original_duration_seconds = audio.size / SAMPLE_RATE
        trimmed_audio = trim_silence(audio) if should_trim_silence() else np.ascontiguousarray(audio)
        if trimmed_audio.size == 0:
            return TranscriptionResult(
                text="",
                audio_duration_seconds=original_duration_seconds,
                inference_seconds=0.0,
                rtfx=0.0,
            )

        bundle = self._get_infer_bundle()
        started = time.perf_counter()
        text = transcribe_short_audio(bundle, trimmed_audio, max_tokens)
        inference_seconds = time.perf_counter() - started
        rtfx = original_duration_seconds / inference_seconds if inference_seconds > 0 else 0.0
        return TranscriptionResult(
            text=text,
            audio_duration_seconds=original_duration_seconds,
            inference_seconds=inference_seconds,
            rtfx=rtfx,
        )

    def transcribe_source(self, source: str, max_tokens: int = DEFAULT_MAX_TOKENS) -> TranscriptionResult:
        return self.transcribe_audio(load_audio_from_source(source), max_tokens=max_tokens)

    def transcribe_bytes(self, audio_bytes: bytes, max_tokens: int = DEFAULT_MAX_TOKENS) -> TranscriptionResult:
        return self.transcribe_audio(load_audio_from_bytes(audio_bytes), max_tokens=max_tokens)
