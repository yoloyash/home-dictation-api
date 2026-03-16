#!/usr/bin/env python3

from __future__ import annotations

import os
import sys
from pathlib import Path

from huggingface_hub import snapshot_download
from openvino import Core

DEFAULT_MODEL_ID = "FluidInference/parakeet-tdt-0.6b-v3-ov"
DEFAULT_MODEL_REVISION = "dfd55eb6c85a9a8546a162bed84784245d5743c2"
REQUIRED_FILES = (
    "config.json",
    "parakeet_decoder.bin",
    "parakeet_decoder.xml",
    "parakeet_encoder.bin",
    "parakeet_encoder.xml",
    "parakeet_joint.bin",
    "parakeet_joint.xml",
    "parakeet_melspectogram.bin",
    "parakeet_melspectogram.xml",
    "parakeet_v3_vocab.json",
    "parakeet_vocab.json",
)
XML_FILES = (
    "parakeet_decoder.xml",
    "parakeet_encoder.xml",
    "parakeet_joint.xml",
    "parakeet_melspectogram.xml",
)


def resolve_model_dir(model_id: str) -> Path:
    model_dir = os.environ.get("MODEL_DIR")
    if model_dir:
        return Path(model_dir)

    model_root = Path(os.environ.get("MODEL_ROOT", "/opt/models"))
    model_name = model_id.rsplit("/", 1)[-1]
    return model_root / model_name


def fail(message: str) -> int:
    print(f"error: {message}", file=sys.stderr)
    return 1


def main() -> int:
    model_id = os.environ.get("MODEL_ID", DEFAULT_MODEL_ID)
    model_revision = os.environ.get("MODEL_REVISION", DEFAULT_MODEL_REVISION)
    model_dir = resolve_model_dir(model_id)

    print(f"model_id={model_id}")
    print(f"model_revision={model_revision}")
    print(f"model_dir={model_dir}")

    model_dir.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=model_id,
        revision=model_revision,
        local_dir=model_dir,
        allow_patterns=list(REQUIRED_FILES),
        token=os.environ.get("HF_TOKEN") or None,
    )

    missing_files = [name for name in REQUIRED_FILES if not (model_dir / name).is_file()]
    if missing_files:
        return fail(f"missing model files: {', '.join(missing_files)}")

    core = Core()
    devices = list(core.available_devices)
    if not devices:
        return fail("OpenVINO did not report any available devices")

    print(f"available_devices={devices}")
    for xml_name in XML_FILES:
        xml_path = model_dir / xml_name
        core.read_model(xml_path)
        print(f"loaded={xml_name}")

    print("bootstrap complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
