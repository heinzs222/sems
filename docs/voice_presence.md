# Voice Presence (Contextual TTS)

This project is a Twilio PSTN voice agent that answers inbound calls and speaks back over Twilio Media Streams (8kHz G.711 μ-law, 20ms frames).

The goal of “voice presence” here is practical: make the agent feel continuous and conversational (timing, prosody consistency, interruption), while keeping the ordering/checkout flow reliable.

## Architecture

High level request/response loop:

1. **Twilio → `/ws`**: inbound μ-law 8kHz audio frames (20ms).
2. **STT (Deepgram)**: streaming transcription + speech-start events for barge-in.
3. **LLM (Groq/OpenAI)**: generates the assistant reply text.
4. **TTS (pluggable)**: turns reply text into audio.
5. **Audio bridge**: convert to μ-law 8kHz, chunk to 20ms frames.
6. **Outbound pacer**: queues frames and sends at steady 20ms cadence + `mark` events.
7. **Barge-in**: if the caller starts speaking, send Twilio `clear`, cancel TTS, and listen.

## TTS Provider Selection

Set `TTS_PROVIDER`:

- `cartesia` (default): low-latency streaming TTS, CPU-friendly.
- `openai`: simple non-streaming WAV synthesis (then converted to μ-law).
- `csm`: **contextual** TTS via a GPU microservice, with **Cartesia fallback** for reliability.

Relevant env vars are in `.env.example`.

## Contextual TTS (CSM-style)

When `TTS_PROVIDER=csm`, the app maintains a rolling per-call buffer of recent turns:

- **User turn**: transcript text + a short caller audio snippet (optional).
- **Assistant turn**: reply text + a short synthesized audio snippet (optional).

The buffer is aggressively capped:

- Total window: `CSM_MAX_CONTEXT_SECONDS` (default ~24s).
- Per-snippet cap: derived from the window (3–6 seconds).

This context is sent to the microservice on each TTS request so the model can keep voice continuity and conversational rhythm across turns.

## GPU Microservice (`csm_service/`)

The microservice exposes a single endpoint:

- `POST /tts`
  - JSON body: `speaker_id`, `prompt_text`, `context[]`, plus optional generation params
  - Returns: `audio/wav` (24kHz mono PCM16) when `Accept: audio/wav` is set

The main app then converts:

`24kHz WAV → 8kHz PCM → μ-law 8kHz → 20ms frames`

### Run locally (GPU recommended)

From the repo root:

1. `cd csm_service`
2. `python -m venv .venv && .venv\\Scripts\\activate` (Windows) or `source .venv/bin/activate`
3. `pip install -r requirements.txt`
4. `uvicorn app:app --host 0.0.0.0 --port 8001`

Health check:

- `GET http://localhost:8001/health`

### Smoke test

From repo root:

- `python scripts/smoke_csm_service.py --endpoint http://localhost:8001 --out csm_smoke.wav`

### Deploy (RunPod/Modal/VM)

Use `csm_service/Dockerfile` on a GPU runtime and expose port `8001`.

Then set on Railway (main app):

- `TTS_PROVIDER=csm`
- `CSM_ENDPOINT=https://<your-csm-host>`
- Keep `CARTESIA_API_KEY` configured for fallback.

## Latency + Fallback Strategy

The app treats CSM as best-effort:

- If CSM is slow to respond, a short backchannel (“Mm-hm.” / “Hum-hum.”) can play quickly via Cartesia.
- If CSM errors or times out (`CSM_TIMEOUT_MS`), the turn falls back to Cartesia.

## Barge-in (Interruption)

When the caller speaks while the agent is talking:

1. Send Twilio `clear` (flush buffered audio).
2. Cancel in-flight TTS generation (including remote CSM request).
3. Stop queueing outbound audio frames immediately.

This keeps interruptions crisp and avoids “talking over” the caller.

## Tuning Tips

- If you hear choppiness: increase `JITTER_BUFFER_MS` slightly (e.g., 160 → 200).
- If the agent feels sluggish after interruptions: keep `JITTER_BUFFER_MIN_MS` modest (e.g., 80–120).
- If CSM timeouts happen: raise `CSM_TIMEOUT_MS` (but keep fallback enabled).
- If context feels “too sticky”: reduce `CSM_MAX_CONTEXT_SECONDS`.

## Troubleshooting

- **CSM returns 413**: context audio payload too large → lower `CSM_MAX_CONTEXT_SECONDS`.
- **CSM returns 503**: model still loading → wait for `/health` to show `status=ok`.
- **No audio on Twilio**: confirm your main app is sending μ-law 8kHz and 20ms frames; check Railway logs for pacer underflows.

