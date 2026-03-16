# AGENTS.md

This repo is intentionally small and early-stage. Keep changes simple and directly aligned with the current README.

## Project Goal

Build a self-hosted home-lab dictation API with an OpenAI-style transcription endpoint for tools like `OpenWhispr`.

## Current Direction

- Host target: mini PC with `Intel i5-9500T`, `15 GiB RAM`, `~399 GiB` free disk
- Model: `FluidInference/parakeet-tdt-0.6b-v3-ov`
- Client: `OpenWhispr`
- API shape: OpenAI-compatible `POST /v1/audio/transcriptions`
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