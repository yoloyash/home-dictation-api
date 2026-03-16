# Home Dictation API

Self-hosted dictation server with an OpenAI-style transcription API for devices around the house.

## Current Direction

- Host: mini PC with `Intel i5-9500T`, `15 GiB RAM`, `Intel UHD 630`, `~399 GiB` free disk
- Model: `nvidia/parakeet-tdt-0.6b-v3`
- Client: `OpenWhispr`
- API shape: OpenAI-compatible `/v1/audio/transcriptions`
- Deploy first on LAN, later behind `api.<domain>` for remote access
- First client targets: macOS and Windows
