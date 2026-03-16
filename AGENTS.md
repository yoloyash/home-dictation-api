# AGENTS.md

This repo is intentionally small and early-stage. Keep changes simple and treat this file as the current project doc.

## Project Goal

Build a self-hosted home-lab dictation API with an OpenAI-style transcription endpoint for tools like `OpenWhispr`.

## Current Direction

- Host target: mini PC with `Intel i5-9500T`, `15 GiB RAM`, `~399 GiB` free disk
- Model: `FluidInference/parakeet-tdt-0.6b-v3-ov`
- Client: `OpenWhispr`
- API shape: OpenAI-compatible `POST /v1/audio/transcriptions`
- Model alias for compatibility: `whisper-1`
- Network plan: LAN first, remote access later
- First client targets: macOS and Windows

## Current Phase

- Focus on development and evaluation only
- Docker-first workflow
- CPU-only for now
- Prefer the smallest possible implementation that lets us test the model and end-to-end dictation flow

## What To Avoid For Now

- No GPU support work
- No production hardening unless explicitly requested
- No extra infrastructure like databases, queues, or multi-service setups
- No large abstractions or framework-heavy scaffolding

## Working Style

- Keep the repo minimal
- Prefer direct, readable code over premature architecture
- Use `docker compose` for local development
- Update `AGENTS.md` if the direction or setup changes
- Do not rely on `README.md` for current setup notes

## Current Dev Commands

- `cp .env.example .env`
- `docker compose build`
- `docker compose up`
- `curl http://localhost:8080/healthz`
- `curl http://localhost:8080/v1/models`
- `curl -X POST http://localhost:8080/v1/audio/transcriptions -F file=@/tmp/sample.wav -F model=whisper-1`
- `docker compose run --rm dev python -c "from openvino import Core; print(Core().available_devices)"`
- `docker compose run --rm dev python scripts/bootstrap_model.py`
- `docker compose run --rm dev python scripts/transcribe_file.py https://raw.githubusercontent.com/FluidInference/eddy-audio/main/assets/audio/first_10_seconds.wav`

## Current API Contract

- Base URL for OpenWhispr custom transcription should be `http://<host>:<port>/v1`
- Primary endpoint is `POST /v1/audio/transcriptions`
- Request fields aligned to OpenWhispr's custom endpoint flow: `file`, `model`, optional `language`, optional `prompt`
- Response shape is JSON with a top-level `text`
- `GET /v1/models` exposes `whisper-1` for compatibility
- If `API_KEY` is set, send it as `Authorization: Bearer <key>`

## Current Smoke-Test Limits

- `scripts/transcribe_file.py` and the v0 API both use the same short-audio path right now
- They support short audio only for now: up to `240000` samples after `ffmpeg` resampling, which is about `15` seconds at `16kHz` mono
- Long-audio chunking comes later
