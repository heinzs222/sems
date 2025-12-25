# Twilio Voice Agent

A production-ready AI voice agent for Twilio phone calls, built with Python. Features real-time speech-to-text (Deepgram), LLM responses (Groq), text-to-speech (Cartesia), and intelligent routing.

## Features

- **Real-time Voice Conversations**: Full-duplex audio streaming via Twilio Media Streams
- **Speech-to-Text**: Deepgram streaming transcription
- **LLM Integration**: Groq with OpenAI-compatible API (fast inference)
- **Text-to-Speech**: Cartesia streaming synthesis
- **Barge-in Support**: Interrupt the agent mid-sentence (requires 3+ words)
- **Semantic Routing**: Fast cached responses for common queries
- **Structured Extraction**: Extract caller intent and information
- **Production Ready**: Health checks, metrics, structured logging

## Prerequisites

- **Python 3.10+**
- **ffmpeg** (for audio conversion)
- **ngrok** (for local development)
- **Twilio Account** with a phone number

### Installing Prerequisites

#### Python 3.10+
```bash
# Windows (winget)
winget install Python.Python.3.12

# macOS (Homebrew)
brew install python@3.12

# Ubuntu/Debian
sudo apt update && sudo apt install python3.12 python3.12-venv
```

#### ffmpeg
```bash
# Windows (winget)
winget install FFmpeg.FFmpeg

# macOS (Homebrew)
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg
```

#### ngrok
```bash
# Windows (winget)
winget install ngrok.ngrok

# macOS (Homebrew)
brew install ngrok

# Or download from https://ngrok.com/download
```

## Quick Start

### 1. Clone and Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd twilio-voice-agent

# Create virtual environment (using uv - recommended)
uv venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install dependencies
uv pip install -r requirements.txt

# Or using pip
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
# Copy example config
cp .env.example .env

# Edit .env with your credentials
# Required: TWILIO_*, DEEPGRAM_API_KEY, CARTESIA_API_KEY, GROQ_API_KEY
```

### 3. Build Cached Audio (Optional)

```bash
# Create placeholder WAV files in assets/audio/source/
# Then build the .mulaw files:
make audio

# Or manually:
bash scripts/build_audio.sh
```

### 4. Run the Server

```bash
# Using Make
make run

# Or directly
python -m server.app
```

Server starts at `http://localhost:7860`

### 5. Start ngrok

In a new terminal:

```bash
ngrok http 7860
```

Copy the ngrok URL (e.g., `abc123.ngrok-free.app`)

### 6. Update Configuration

Update your `.env` file:
```
PUBLIC_HOST=abc123.ngrok-free.app
```

Restart the server to apply changes.

### 7. Configure Twilio

#### Create TwiML Bin

1. Go to [Twilio Console](https://console.twilio.com/)
2. Navigate to **Explore Products** → **Developer Tools** → **TwiML Bins**
3. Click **Create new TwiML Bin**
4. Name it (e.g., "Voice Agent Stream")
5. Enter this TwiML (replace with your ngrok URL):

```xml
<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Connect>
        <Stream url="wss://abc123.ngrok-free.app/ws" />
    </Connect>
</Response>
```

6. Click **Create**

#### Assign to Phone Number

1. Go to **Phone Numbers** → **Manage** → **Active Numbers**
2. Click your phone number
3. Under **Voice Configuration**:
   - Set "A call comes in" to **TwiML Bin**
   - Select your TwiML Bin from the dropdown
4. Click **Save configuration**

### 8. Test the Agent

Call your Twilio phone number! The agent should:
1. Answer and greet you
2. Respond to your questions
3. Allow interruptions (say 3+ words while it's speaking)

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/metrics` | GET | JSON metrics |
| `/twiml` | POST | Generate TwiML (alternative to TwiML Bin) |
| `/ws` | WebSocket | Twilio Media Streams connection |

### Using `/twiml` Endpoint

Instead of a TwiML Bin, you can point your Twilio number directly to:
```
https://your-domain.com/twiml
```

Set this as the webhook URL for "A call comes in" (HTTP POST).

## Project Structure

```
twilio-voice-agent/
├── server/
│   └── app.py              # FastAPI server
├── src/
│   └── agent/
│       ├── config.py       # Configuration management
│       ├── audio.py        # Audio conversion utilities
│       ├── twilio_protocol.py  # Twilio WebSocket protocol
│       ├── routing.py      # Semantic router
│       ├── extract.py      # Structured extraction
│       ├── llm.py          # Groq LLM wrapper
│       ├── stt.py          # Deepgram STT
│       ├── tts.py          # Cartesia TTS
│       └── pipeline.py     # Voice pipeline orchestration
├── assets/
│   └── audio/
│       ├── source/         # WAV source files
│       └── *.mulaw         # Cached audio files
├── scripts/
│   ├── build_audio.sh      # Build cached audio
│   └── smoke_test.py       # Configuration test
├── tests/
│   └── ...                 # Test suite
├── .env.example            # Environment template
├── requirements.txt        # Pinned dependencies
├── Makefile               # Build commands
└── README.md              # This file
```

## Configuration Reference

See `.env.example` for all configuration options.

### Required Variables

| Variable | Description |
|----------|-------------|
| `PUBLIC_HOST` | Your public hostname (ngrok URL without https://) |
| `TWILIO_ACCOUNT_SID` | Twilio Account SID |
| `TWILIO_AUTH_TOKEN` | Twilio Auth Token |
| `DEEPGRAM_API_KEY` | Deepgram API Key |
| `CARTESIA_API_KEY` | Cartesia API Key |
| `GROQ_API_KEY` | Groq API Key |
| `GROQ_MODEL` | Groq model name (e.g., `llama-3.3-70b-versatile`) |

### Feature Flags

| Variable | Default | Description |
|----------|---------|-------------|
| `ROUTER_ENABLED` | `true` | Enable semantic routing |
| `OUTLINES_ENABLED` | `false` | Use Outlines for extraction |

## Creating Cached Audio Files

The semantic router uses pre-rendered audio for fast responses.

### 1. Create Source WAV Files

Create WAV files in `assets/audio/source/`:
- `pricing.wav` - Response about pricing
- `who_are_you.wav` - Self-introduction
- `stop.wav` - Goodbye message

Requirements:
- Format: WAV
- Sample rate: Any (will be converted)
- Channels: Mono preferred

You can record these yourself or use a TTS service.

### 2. Build .mulaw Files

```bash
make audio
# Or: bash scripts/build_audio.sh
```

This converts WAV files to raw mu-law 8kHz format for Twilio.

## Troubleshooting

### No Audio

**Symptoms**: Call connects but no audio in either direction.

**Solutions**:
1. Verify ngrok is running and URL is correct in TwiML
2. Check WebSocket connection in server logs
3. Ensure `PUBLIC_HOST` in `.env` matches ngrok URL (no `https://` prefix)
4. Verify Twilio phone number is configured correctly

### One-Way Audio

**Symptoms**: You can hear the agent but it can't hear you (or vice versa).

**Solutions**:
1. Check Deepgram API key is valid
2. Check Cartesia API key is valid
3. Verify audio conversion is working (check logs for errors)
4. Test with `make smoke` to verify API connections

### Choppy Audio

**Symptoms**: Audio is choppy, robotic, or has gaps.

**Solutions**:
1. Check your internet connection
2. Increase buffer sizes if on slow connection
3. Monitor server CPU usage
4. Check for errors in TTS streaming

### Agent Keeps Talking After Interruption

**Symptoms**: Agent doesn't stop when you interrupt.

**Solutions**:
1. Speak at least 3 words clearly (barge-in threshold)
2. Check `MIN_INTERRUPTION_WORDS` setting
3. Verify STT is receiving audio (check transcription logs)
4. Ensure `allow_interruptions` is enabled in pipeline

### Groq Model Missing

**Symptoms**: Server exits with "Model not found" error.

**Solutions**:
1. Verify `GROQ_MODEL` is spelled correctly
2. Check available models at [Groq Console](https://console.groq.com/)
3. Common models: `llama-3.3-70b-versatile`, `llama-3.1-8b-instant`, `mixtral-8x7b-32768`

### Slow First Token

**Symptoms**: Long delay before agent starts responding.

**Solutions**:
1. Use a smaller/faster model (`llama-3.1-8b-instant`)
2. Check Groq API status
3. Reduce system prompt length
4. Enable routing for common queries

### Deepgram Format Errors

**Symptoms**: STT errors about audio format.

**Solutions**:
1. Verify audio conversion is producing 16kHz linear PCM
2. Check Deepgram encoding settings match audio format
3. Review audio.py conversion code

### Connection Drops

**Symptoms**: Calls disconnect unexpectedly.

**Solutions**:
1. Check server logs for exceptions
2. Verify ngrok tunnel is stable
3. Increase Twilio timeout settings
4. Check for unhandled exceptions in pipeline

## Development

### Running Tests

```bash
# All tests
make test

# With coverage
make test-cov

# Specific test file
pytest tests/test_audio.py -v
```

### Smoke Test

Verify configuration without making a call:

```bash
make smoke
# Or: python scripts/smoke_test.py
```

### Code Formatting

```bash
make format  # Format code
make lint    # Check linting
```

## Deployment

### Docker

```bash
# Build image
docker build -t twilio-voice-agent .

# Run container
docker run -p 7860:7860 --env-file .env twilio-voice-agent
```

### Production Considerations

1. **TLS**: Use HTTPS/WSS (reverse proxy with SSL termination)
2. **Secrets**: Use a secrets manager, not .env files
3. **Monitoring**: Set up alerting on `/health` and `/metrics`
4. **Scaling**: Consider load balancing for multiple instances
5. **Logging**: Ship logs to centralized logging system

See `SECURITY.md` for security guidelines.

## License

[Your License Here]

## Support

For issues and feature requests, please open a GitHub issue.
