#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from home_dictation_api.engine import DEFAULT_DEVICE, DEFAULT_MODEL_DIR, ParakeetTranscriber, TranscriptionError


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


def main() -> int:
    args = parse_args()
    model_dir = Path(args.model_dir)

    if not model_dir.is_dir():
        return fail(f"model directory does not exist: {model_dir}")

    try:
        transcriber = ParakeetTranscriber(model_dir=model_dir, device=args.device)
        result = transcriber.transcribe_source(args.audio, max_tokens=args.max_tokens)
    except TranscriptionError as exc:
        return fail(str(exc))
    except Exception as exc:
        return fail(str(exc))

    print(f"audio={args.audio}")
    print(f"device={args.device}")
    print(f"model_dir={model_dir}")
    print(f"audio_duration_seconds={result.audio_duration_seconds:.2f}")
    print(f"model_load_seconds={transcriber.model_load_seconds:.3f}")
    print(f"inference_seconds={result.inference_seconds:.3f}")
    print(f"rtfx={result.rtfx:.2f}")
    print(f"transcript={result.text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
