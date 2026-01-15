# Voice Presence (Contextual TTS)

This project is a Twilio PSTN voice agent that answers inbound calls and speaks back over Twilio Media Streams (8kHz G.711 μ-law, 20ms frames).

"Voice presence" here means practical, phone-friendly conversation: natural pacing, consistent prosody across turns, and crisp interruptions (barge-in), without breaking the ordering/checkout flow.

## Architecture

High-level loop:

1. **Twilio → `/ws`**: inbound μ-law 8kHz audio frames (20ms).
2. **STT (Deepgram)**: streaming transcription + VAD events.
3. **LLM (Groq/OpenAI)**: generates assistant reply text.
4. **TTS (pluggable)**: turns reply text into audio.
5. **Audio bridge**: convert to μ-law 8kHz, chunk to 20ms frames.
6. **Outbound pacer**: steady 20ms cadence + `mark` events.
7. **Barge-in**: when the caller speaks, send Twilio `clear`, cancel generation, and listen.

## TTS Provider Selection

Set `TTS_PROVIDER`:

- `cartesia` (default): low-latency streaming TTS, CPU-friendly.
- `openai`: non-streaming WAV synthesis (then converted to μ-law).
- `csm`: contextual TTS via a GPU microservice, with Cartesia fallback.

Relevant env vars are in `.env.example`.

## Contextual TTS (CSM-style)

When `TTS_PROVIDER=csm`, the app maintains a rolling per-call buffer of recent turns:

- **User turn**: transcript text + a short caller audio snippet (optional).
- **Assistant turn**: reply text + a short synthesized audio snippet (optional).

The buffer is aggressively capped:

- Total window: `CSM_MAX_CONTEXT_SECONDS` (default ~24s).
- Per-snippet cap: derived from the window (typically 3–6 seconds).

This context is sent to the microservice on each TTS request to encourage continuity in timing and prosody across turns.

## Turn-Taking (Thinking Pauses)

To avoid the agent "jumping in" while the caller is still thinking, the pipeline adds a short hold before acting on end-of-turn transcripts.

Tune:

- `TURN_END_GRACE_MS`: base hold after end-of-utterance.
- `TURN_END_SHORT_UTTERANCE_MS`: extra hold for short fragments (common mid-thought).
- `TURN_END_INCOMPLETE_MS`: extra hold when the last token looks unfinished (e.g., “and…”, “mais…”).
- `TURN_END_FALLBACK_MS`: hold used when Deepgram emits `is_final` without `speech_final`.

If the agent feels too slow, reduce these values. If it cuts callers off, increase them.

## GPU Microservice (`csm_service/`)

The microservice exposes:

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
- Keep `CARTESIA_API_KEY` configured for fallback

## Latency + Fallback Strategy

The app treats CSM as best-effort:

- If the microservice is slow to return, the main app can play a short backchannel via Cartesia.
- If CSM errors or times out (`CSM_TIMEOUT_MS`), the turn falls back to Cartesia.

## Barge-in (Interruption)

When the caller speaks while the agent is talking:

1. Send Twilio `clear` (flush buffered audio).
2. Cancel in-flight TTS generation (including remote CSM requests).
3. Stop queueing outbound audio frames immediately.

## Tuning Tips

- Choppy playback: increase `JITTER_BUFFER_MS` slightly (e.g., 160 → 200).
- Too many interruptions: raise `MIN_INTERRUPTION_WORDS`.
- Cuts callers off while thinking: increase `TURN_END_GRACE_MS` / `TURN_END_SHORT_UTTERANCE_MS`.
- Feels too slow: decrease `TURN_END_GRACE_MS` / `TURN_END_SHORT_UTTERANCE_MS`.
- Frequent CSM timeouts: raise `CSM_TIMEOUT_MS` (fallback still applies).
- Context feels “too sticky”: reduce `CSM_MAX_CONTEXT_SECONDS`.

## Troubleshooting

- **CSM returns 413**: context payload too large → lower `CSM_MAX_CONTEXT_SECONDS`.
- **CSM returns 503**: model still loading → wait for `/health` to show `status=ok`.
- **No audio on Twilio**: confirm outbound is μ-law 8kHz and 20ms frames; check logs for pacer underflows.
