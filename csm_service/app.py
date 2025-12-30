from __future__ import annotations

import asyncio
import base64
import io
import os
import time
import wave
from typing import Any, Optional

import numpy as np
import torch
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field
from transformers import AutoProcessor, CsmForConditionalGeneration


SAMPLE_RATE_HZ = 24000

DEFAULT_MODEL_ID = os.getenv("CSM_MODEL_ID", "sesame/csm-1b")
DEFAULT_DEVICE = os.getenv("CSM_DEVICE", "").strip().lower()

MAX_CONTEXT_TURNS = int(os.getenv("CSM_MAX_CONTEXT_TURNS", "12"))
MAX_TOTAL_AUDIO_SECONDS = float(os.getenv("CSM_MAX_CONTEXT_SECONDS", "30"))
MAX_AUDIO_B64_BYTES = int(os.getenv("CSM_MAX_AUDIO_B64_BYTES", str(1_800_000)))


def _wav_bytes_to_float32_mono_24k(wav_bytes: bytes) -> np.ndarray:
    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            sample_rate = wf.getframerate()
            pcm = wf.readframes(wf.getnframes())
    except Exception as e:
        raise ValueError(f"Invalid WAV: {e}") from e

    if sampwidth != 2:
        raise ValueError("WAV must be 16-bit PCM")
    if sample_rate != SAMPLE_RATE_HZ:
        raise ValueError(f"WAV must be {SAMPLE_RATE_HZ} Hz, got {sample_rate}")

    audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    if channels == 1:
        return audio
    if channels == 2:
        audio = audio.reshape(-1, 2).mean(axis=1)
        return audio
    raise ValueError(f"Unsupported channel count: {channels}")


def _float32_to_wav_bytes(audio: np.ndarray, *, sample_rate: int = SAMPLE_RATE_HZ) -> bytes:
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        pcm = b""
    else:
        audio = np.clip(audio, -1.0, 1.0)
        pcm_i16 = (audio * 32767.0).astype(np.int16)
        pcm = pcm_i16.tobytes()

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(pcm)
    return buf.getvalue()


def _normalize_role(role: str) -> str:
    r = (role or "").strip().lower()
    if r in ("assistant", "agent", "bot", "0"):
        return "0"
    return "1"


class ContextTurn(BaseModel):
    role: str = Field(..., description="Speaker id ('0' for agent, '1' for user) or a friendly role string.")
    text: str = Field("", description="Text for this turn.")
    audio_base64_wav_24khz: Optional[str] = Field(
        default=None, description="Optional mono PCM16 WAV base64 at 24kHz."
    )


class TTSRequest(BaseModel):
    speaker_id: int = Field(0, description="Speaker id for the agent; typically 0.")
    prompt_text: str = Field(..., min_length=1)
    context: list[ContextTurn] = Field(default_factory=list)
    voice_style: str = Field("default", description="Soft style tag; model-dependent.")

    deterministic: bool = Field(True)
    temperature: float = Field(1.0, ge=0.0, le=2.0)
    max_length: Optional[int] = Field(default=250, ge=16, le=1024)


class TTSJSONResponse(BaseModel):
    audio_base64_wav_24khz: str
    sample_rate_hz: int
    latency_ms: float
    model_id: str
    device: str


app = FastAPI(title="CSM Contextual TTS", version="0.1.0")


class _ModelState:
    def __init__(self) -> None:
        self.model: Optional[CsmForConditionalGeneration] = None
        self.processor: Optional[Any] = None
        self.device: str = "cpu"
        self.lock = asyncio.Lock()
        self.model_id: str = DEFAULT_MODEL_ID


STATE = _ModelState()


@app.on_event("startup")
def _load_model() -> None:
    device_str = DEFAULT_DEVICE or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)

    STATE.device = device_str
    STATE.model_id = DEFAULT_MODEL_ID

    torch_dtype = torch.float16 if device_str.startswith("cuda") else None

    processor = AutoProcessor.from_pretrained(STATE.model_id)
    model = CsmForConditionalGeneration.from_pretrained(
        STATE.model_id,
        torch_dtype=torch_dtype,
    )
    model.to(device)
    model.eval()

    STATE.processor = processor
    STATE.model = model


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok" if STATE.model is not None else "loading",
        "model_id": STATE.model_id,
        "device": STATE.device,
    }


@app.post("/tts")
async def tts(
    req: TTSRequest,
    accept: Optional[str] = Header(default=None),
) -> Response:
    if STATE.model is None or STATE.processor is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet")

    context = req.context[:MAX_CONTEXT_TURNS]
    total_b64 = sum(len(t.audio_base64_wav_24khz or "") for t in context)
    if total_b64 > MAX_AUDIO_B64_BYTES:
        raise HTTPException(status_code=413, detail="Context audio payload too large")

    conversation: list[dict[str, Any]] = []
    total_audio_seconds = 0.0

    for turn in context:
        role = _normalize_role(turn.role)
        content: list[dict[str, Any]] = []
        if turn.text:
            content.append({"type": "text", "text": turn.text})
        if turn.audio_base64_wav_24khz:
            wav_bytes = base64.b64decode(turn.audio_base64_wav_24khz)
            audio = _wav_bytes_to_float32_mono_24k(wav_bytes)
            total_audio_seconds += float(audio.shape[0]) / float(SAMPLE_RATE_HZ)
            if total_audio_seconds > MAX_TOTAL_AUDIO_SECONDS:
                break
            content.append({"type": "audio", "path": audio})
        conversation.append({"role": role, "content": content})

    # Prompt turn
    conversation.append(
        {
            "role": str(int(req.speaker_id)),
            "content": [{"type": "text", "text": req.prompt_text}],
        }
    )

    processor = STATE.processor
    model = STATE.model

    started = time.time()
    async with STATE.lock:
        inputs = processor.apply_chat_template(
            conversation,
            tokenize=True,
            return_dict=True,
        ).to(model.device)

        do_sample = not bool(req.deterministic)
        gen_kwargs = {
            "do_sample": do_sample,
            "depth_decoder_do_sample": do_sample,
            "temperature": float(req.temperature),
            "depth_decoder_temperature": float(req.temperature),
        }
        if req.max_length:
            gen_kwargs["max_length"] = int(req.max_length)

        audio_out = model.generate(**inputs, output_audio=True, **gen_kwargs)

    latency_ms = (time.time() - started) * 1000

    # `audio_out` can be a list of tensors, a tensor, or a nested structure.
    if isinstance(audio_out, (list, tuple)):
        audio_tensor = audio_out[0]
    else:
        audio_tensor = audio_out

    if isinstance(audio_tensor, torch.Tensor):
        audio_np = audio_tensor.detach().float().cpu().numpy()
    else:
        audio_np = np.asarray(audio_tensor, dtype=np.float32)

    wav_bytes = _float32_to_wav_bytes(audio_np, sample_rate=SAMPLE_RATE_HZ)

    wants_wav = bool(accept and "audio/wav" in accept.lower())
    if wants_wav:
        return Response(
            content=wav_bytes,
            media_type="audio/wav",
            headers={
                "X-Latency-Ms": f"{latency_ms:.2f}",
                "X-Model-Id": STATE.model_id,
                "X-Device": STATE.device,
            },
        )

    payload = TTSJSONResponse(
        audio_base64_wav_24khz=base64.b64encode(wav_bytes).decode("ascii"),
        sample_rate_hz=SAMPLE_RATE_HZ,
        latency_ms=latency_ms,
        model_id=STATE.model_id,
        device=STATE.device,
    )
    return JSONResponse(content=payload.model_dump())
