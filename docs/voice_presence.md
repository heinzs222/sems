# Voice Presence (Contextual TTS + Speech-to-Speech)

This project is a Twilio PSTN voice agent that answers inbound calls and speaks back over Twilio Media Streams (8kHz G.711 μ-law, 20ms frames).

"Voice presence" here means phone-friendly conversation: natural pacing, consistent tone across turns, and crisp interruption (barge-in), without breaking audio pacing.

## Architecture

Two supported paths:

### A) Pipeline mode (classic)

`Twilio (μ-law/8k) -> STT (Deepgram) -> LLM (Groq/OpenAI) -> TTS (Cartesia/OpenAI/CSM) -> Twilio (μ-law/8k)`

### B) Speech-to-speech mode (OpenAI Realtime)

`Twilio (g711_ulaw/8k) -> OpenAI Realtime -> Twilio (g711_ulaw/8k)`

This mode bypasses Deepgram/Groq/Cartesia entirely and is the closest to a modern "voice mode" experience.

## TTS Provider Selection (Pipeline Mode)

Set `TTS_PROVIDER`:

- `cartesia` (default): low-latency streaming TTS, CPU-friendly.
- `openai`: non-streaming WAV synthesis (then converted to μ-law).
- `csm`: contextual TTS via a GPU microservice, with Cartesia fallback.

Relevant env vars are in `.env.example`.

## Speech-to-Speech (OpenAI Realtime)

Enable:

- `VOICE_MODE=openai_realtime` (or `VOICE_MODE=auto` with `OPENAI_API_KEY` set)
- `OPENAI_API_KEY`
- `OPENAI_REALTIME_MODEL` and `OPENAI_REALTIME_VOICE`

Optional:

- `OPENAI_REALTIME_INSTRUCTIONS_FILE=prompts/renewables_system_prompt.txt` (or `OPENAI_REALTIME_INSTRUCTIONS`)
- `OPENAI_REALTIME_TRANSCRIPTION_MODEL=whisper-1` to log transcripts
- `OPENAI_REALTIME_TOOLS=renewables` to enable lead-capture/scheduling tools

Stability / continuity tuning (if you hear split phrases, mid-answer silence, or false barge-in):

- `OPENAI_REALTIME_TURN_SILENCE_MS`: longer waits more for thinking pauses
- `OPENAI_REALTIME_VAD_THRESHOLD`: higher reduces false speech detection
- `OPENAI_REALTIME_PREFIX_PADDING_MS`: more context before speech start
- `OPENAI_REALTIME_BARGE_IN_DEBOUNCE_MS`: filters echo/noise barge-in
- `OPENAI_REALTIME_NOISE_REDUCTION`: `near_field` / `far_field` / empty
- `OPENAI_REALTIME_PACE_AHEAD_MS`: small prebuffer for smoother output

## Contextual TTS (CSM-style)

When `TTS_PROVIDER=csm`, the app keeps a rolling per-call buffer of recent turns:

- User turn: transcript text + short caller audio snippet (optional)
- Assistant turn: reply text + short synthesized audio snippet (optional)

The buffer is aggressively capped:

- Total window: `CSM_MAX_CONTEXT_SECONDS` (default ~24s)
- Per-snippet cap: derived from the window (typically 3–6 seconds)

This context is sent to the microservice on each TTS request to encourage continuity in timing and prosody across turns.

## GPU Microservice (`csm_service/`)

The microservice exposes:

- `POST /tts`
  - JSON body: `speaker_id`, `prompt_text`, `context[]`, plus optional generation params
  - Returns: `audio/wav` (24kHz mono PCM16) when `Accept: audio/wav` is set

The main app converts:

`24kHz WAV -> 8kHz PCM -> μ-law 8kHz -> 20ms frames`

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

Use `csm_service/Dockerfile` on a GPU runtime and expose port `8001`. Then set on Railway (main app):

- `TTS_PROVIDER=csm`
- `CSM_ENDPOINT=https://<your-csm-host>`
- Keep `CARTESIA_API_KEY` configured for fallback

## Turn-Taking (Pipeline Mode Thinking Pauses)

To avoid the agent "jumping in" while the caller is still thinking, the pipeline adds a short hold before acting on end-of-turn transcripts.

Tune:

- `TURN_END_GRACE_MS`: base hold after end-of-utterance
- `TURN_END_SHORT_UTTERANCE_MS`: extra hold for short fragments (often mid-thought)
- `TURN_END_INCOMPLETE_MS`: extra hold when the last token looks unfinished ("and...", "mais...")
- `TURN_END_FALLBACK_MS`: hold used when Deepgram emits `is_final` without `speech_final`

## Barge-in (Interruption)

When the caller speaks while the agent is talking:

1. Send Twilio `clear` (flush buffered audio)
2. Cancel in-flight generation (Realtime response or TTS request)
3. Stop queueing outbound audio frames immediately

## Tuning Tips

- Choppy playback (pipeline): increase `JITTER_BUFFER_MS` slightly (e.g., 160 -> 200)
- Too many interruptions (pipeline): raise `MIN_INTERRUPTION_WORDS`
- Cuts callers off while thinking (pipeline): increase `TURN_END_GRACE_MS` / `TURN_END_SHORT_UTTERANCE_MS`
- Frequent CSM timeouts: raise `CSM_TIMEOUT_MS` (fallback still applies)
- Context feels too "sticky": reduce `CSM_MAX_CONTEXT_SECONDS`

## Troubleshooting

- **CSM returns 413**: context payload too large -> lower `CSM_MAX_CONTEXT_SECONDS`
- **CSM returns 503**: model still loading -> wait for `/health` to show `status=ok`
- **No audio on Twilio**: confirm outbound is μ-law 8kHz and 20ms frames; check logs for pacer underflows or repeated barge-in events
