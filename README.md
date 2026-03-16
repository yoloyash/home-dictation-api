# Home Dictation API

Self-hosted dictation server with an OpenAI-style transcription API for devices around the house.

## Current Direction

- Host: mini PC with `Intel i5-9500T`, `15 GiB RAM`, `~399 GiB` free disk
- Model: `FluidInference/parakeet-tdt-0.6b-v3-ov`
- Client: `OpenWhispr`
- API shape: OpenAI-compatible `/v1/audio/transcriptions`
- Deploy first on LAN, later behind `api.<domain>` for remote access
- First client targets: macOS and Windows

## Docker Dev Runtime

The first milestone is a reproducible container runtime, not the API itself.

- Base image: `python:3.11-slim-bookworm`
- Runtime: `openvino` from PyPI
- System deps: `ffmpeg` only
- Persistent data: named volumes for Hugging Face cache and model storage

Quick start:

1. `cp .env.example .env`
2. `docker compose build`
3. `docker compose up -d`
4. `docker compose run --rm dev python -c "from openvino import Core; print(Core().available_devices)"`

The compose service is intentionally app-less for now. It exists to standardize the runtime before we add model bootstrap or API code.
