#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from openvino import Core

DEFAULT_MODEL_DIR = "/opt/models/parakeet-tdt-0.6b-v3-ov"
DEFAULT_DEVICE = "CPU"
BLANK_TOKEN_ID = 8192
DURATION_BINS = (0, 1, 2, 3, 4)
MAX_PREPROCESSOR_SAMPLES = 240000
MEL_BINS = 128
ENCODER_HIDDEN_SIZE = 1024
DECODER_HIDDEN_SIZE = 640


@dataclass
class CompiledParakeet:
    preprocessor_request: object
    encoder_request: object
    decoder_request: object
    joint_request: object
    vocab: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transcribe a short audio file with the OpenVINO Parakeet smoke-test pipeline.",
    )
    parser.add_argument("audio", help="Local audio path or URL supported by ffmpeg")
    parser.add_argument(
        "--device",
        default=os.environ.get("OPENVINO_DEVICE", DEFAULT_DEVICE),
        help=f"OpenVINO device name (default: {DEFAULT_DEVICE})",
    )
    parser.add_argument(
        "--model-dir",
        default=os.environ.get("MODEL_DIR", DEFAULT_MODEL_DIR),
        help=f"Directory containing the OpenVINO model files (default: {DEFAULT_MODEL_DIR})",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Maximum number of tokens to decode before stopping (default: 256)",
    )
    return parser.parse_args()


def fail(message: str) -> int:
    print(f"error: {message}", file=sys.stderr)
    return 1


def load_audio(source: str) -> np.ndarray:
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-loglevel",
        "error",
        "-i",
        source,
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-",
    ]

    try:
        result = subprocess.run(cmd, check=True, capture_output=True)
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg is required but was not found in PATH") from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg failed to decode audio: {stderr}") from exc

    audio = np.frombuffer(result.stdout, dtype=np.float32)
    if audio.size == 0:
        raise RuntimeError("decoded audio buffer is empty")

    return audio


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
        raise RuntimeError(f"unsupported vocabulary format in {path}")

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


def compile_models(model_dir: Path, device: str) -> CompiledParakeet:
    core = Core()

    preprocessor = core.compile_model(core.read_model(model_dir / "parakeet_melspectogram.xml"), device)
    encoder = core.compile_model(core.read_model(model_dir / "parakeet_encoder.xml"), device)
    decoder = core.compile_model(core.read_model(model_dir / "parakeet_decoder.xml"), device)
    joint = core.compile_model(core.read_model(model_dir / "parakeet_joint.xml"), device)
    vocab = load_vocab(model_dir / "parakeet_v3_vocab.json", BLANK_TOKEN_ID)

    return CompiledParakeet(
        preprocessor_request=preprocessor.create_infer_request(),
        encoder_request=encoder.create_infer_request(),
        decoder_request=decoder.create_infer_request(),
        joint_request=joint.create_infer_request(),
        vocab=vocab,
    )


def transcribe_short_audio(bundle: CompiledParakeet, audio: np.ndarray, max_tokens: int) -> str:
    if audio.size > MAX_PREPROCESSOR_SAMPLES:
        max_seconds = MAX_PREPROCESSOR_SAMPLES / 16000.0
        raise RuntimeError(
            f"audio is too long for the current smoke test "
            f"({audio.size} samples, max {MAX_PREPROCESSOR_SAMPLES} / {max_seconds:.1f}s)"
        )

    signal = np.zeros((1, MAX_PREPROCESSOR_SAMPLES), dtype=np.float32)
    signal[0, : audio.size] = audio
    signal_length = np.array([audio.size], dtype=np.int64)

    bundle.preprocessor_request.infer({
        "input_signals": signal,
        "input_length": signal_length,
    })
    mel = np.array(bundle.preprocessor_request.get_output_tensor(0).data[:], copy=True)
    mel_length = int(bundle.preprocessor_request.get_output_tensor(1).data[0])
    if mel.shape != (1, MEL_BINS, 1501):
        raise RuntimeError(f"unexpected preprocessor output shape: {mel.shape}")

    bundle.encoder_request.infer({
        "melspectogram": mel,
        "melspectogram_length": np.array([mel_length], dtype=np.int32),
    })
    encoder_output = np.array(bundle.encoder_request.get_output_tensor(0).data[:], copy=True)
    valid_frames = int(bundle.encoder_request.get_output_tensor(1).data[0])

    if encoder_output.shape[1] != ENCODER_HIDDEN_SIZE:
        raise RuntimeError(f"unexpected encoder hidden size: {encoder_output.shape}")
    if valid_frames <= 0:
        raise RuntimeError("encoder produced no valid frames")

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
            bundle.decoder_request.infer({
                "targets": np.array([[last_token]], dtype=np.int64),
                "h_in": hidden_state,
                "c_in": cell_state,
            })
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
            bundle.joint_request.infer({
                "encoder_outputs": encoder_step,
                "decoder_outputs": decoder_output,
            })
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
            bundle.decoder_request.infer({
                "targets": np.array([[last_token]], dtype=np.int64),
                "h_in": hidden_state,
                "c_in": cell_state,
            })
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
        bundle.joint_request.infer({
            "encoder_outputs": encoder_step,
            "decoder_outputs": decoder_output,
        })
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


def main() -> int:
    args = parse_args()
    model_dir = Path(args.model_dir)

    if not model_dir.is_dir():
        return fail(f"model directory does not exist: {model_dir}")

    load_started = time.perf_counter()
    try:
        bundle = compile_models(model_dir, args.device)
        model_load_seconds = time.perf_counter() - load_started

        audio = load_audio(args.audio)
        inference_started = time.perf_counter()
        transcript = transcribe_short_audio(bundle, audio, args.max_tokens)
        inference_seconds = time.perf_counter() - inference_started
    except Exception as exc:
        return fail(str(exc))

    audio_duration_seconds = audio.size / 16000.0
    rtfx = audio_duration_seconds / inference_seconds if inference_seconds > 0 else 0.0

    print(f"audio={args.audio}")
    print(f"device={args.device}")
    print(f"model_dir={model_dir}")
    print(f"audio_duration_seconds={audio_duration_seconds:.2f}")
    print(f"model_load_seconds={model_load_seconds:.3f}")
    print(f"inference_seconds={inference_seconds:.3f}")
    print(f"rtfx={rtfx:.2f}")
    print(f"transcript={transcript}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
