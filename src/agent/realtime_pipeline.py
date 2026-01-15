"""
OpenAI Realtime (speech-to-speech) pipeline for Twilio Media Streams.

This mode bypasses the classic STT -> LLM -> TTS pipeline and instead streams:

Twilio (g711_ulaw 8kHz) -> OpenAI Realtime -> Twilio (g711_ulaw 8kHz)

Key goals:
- Low latency (no extra turn-end holds).
- Clean barge-in: if caller speaks, stop assistant audio immediately.
- No scripted / deterministic checkout flow.
"""

from __future__ import annotations

import asyncio
import base64
import json
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Awaitable, Callable, Optional

import structlog
import websockets

from src.agent.audio import TWILIO_FRAME_SIZE
from src.agent.config import Config, get_config
from src.agent.twilio_protocol import (
    TwilioProtocolHandler,
    TwilioEventType,
    TwilioMediaEvent,
    TwilioStartEvent,
    create_clear_message,
    create_media_message,
    parse_twilio_message,
)

logger = structlog.get_logger(__name__)


class RealtimeState(str, Enum):
    IDLE = "idle"
    LISTENING = "listening"
    SPEAKING = "speaking"
    STOPPED = "stopped"


@dataclass(frozen=True)
class OutboundMessage:
    message: str
    pace: bool = True


def _b64encode(data: bytes) -> str:
    return base64.b64encode(data).decode("utf-8")


def _b64decode(data: str) -> bytes:
    try:
        return base64.b64decode(data)
    except Exception:
        return b""


def _default_realtime_instructions(config: Config) -> str:
    # Keep this intentionally short and non-scripted. The goal is natural phone conversation,
    # not a deterministic flow.
    return (
        f"You are {config.agent_name}, a friendly voice assistant on a phone call for {config.company_name}. "
        "Sound calm and human. Keep responses short and phone-friendly. "
        "Match the caller's language (English or French) and stick with it unless the caller switches. "
        "If the caller pauses, wait quietly—do not repeatedly ask if they are still there. "
        "If interrupted, stop speaking immediately and listen. "
        "Never claim to be a human."
    )


class OpenAIRealtimePipeline:
    """
    Twilio Media Streams pipeline powered by OpenAI Realtime speech-to-speech.

    Interface is compatible with `server/app.py`:
    - `start()`
    - `stop()`
    - `handle_message(raw_message)`
    """

    def __init__(
        self,
        send_message: Callable[[str], Awaitable[None]],
        *,
        config: Optional[Config] = None,
    ):
        self.config = config or get_config()
        self._send_message = send_message

        self._protocol = TwilioProtocolHandler()
        self._state: RealtimeState = RealtimeState.IDLE
        self._is_running: bool = False

        # Twilio outbound pacing (keeps buffer small and barge-in crisp).
        self._outbound_queue: asyncio.Queue[Optional[OutboundMessage]] = asyncio.Queue()
        self._pacer_task: Optional[asyncio.Task] = None
        self._pacer_running: bool = False

        # OpenAI Realtime WS
        self._openai_ws: Optional[Any] = None
        self._openai_recv_task: Optional[asyncio.Task] = None
        self._openai_send_task: Optional[asyncio.Task] = None
        self._openai_send_queue: asyncio.Queue[Optional[dict]] = asyncio.Queue(maxsize=2000)
        self._openai_connected: asyncio.Event = asyncio.Event()

        # Audio buffering (avoid padding between deltas -> no "ticking").
        self._outbound_ulaw_remainder: bytes = b""

        # Response tracking / ignore late audio after barge-in.
        self._active_response_id: Optional[str] = None
        self._ignore_audio: bool = False

    @property
    def call_sid(self) -> str:
        return self._protocol.call_sid

    @property
    def stream_sid(self) -> str:
        return self._protocol.stream_sid

    async def start(self) -> None:
        self._is_running = True
        self._state = RealtimeState.IDLE
        logger.info("OpenAI Realtime pipeline started")

    async def stop(self) -> None:
        if not self._is_running:
            return

        self._is_running = False
        self._state = RealtimeState.STOPPED

        for task in (self._openai_recv_task, self._openai_send_task, self._pacer_task):
            if task and not task.done():
                task.cancel()

        # Unblock queues
        try:
            self._openai_send_queue.put_nowait(None)
        except Exception:
            pass
        try:
            self._outbound_queue.put_nowait(None)
        except Exception:
            pass

        await asyncio.gather(
            *[t for t in (self._openai_recv_task, self._openai_send_task, self._pacer_task) if t],
            return_exceptions=True,
        )

        if self._openai_ws:
            try:
                await self._openai_ws.close()
            except Exception:
                pass

        self._openai_ws = None
        self._openai_recv_task = None
        self._openai_send_task = None

        self._pacer_task = None
        self._pacer_running = False
        self._outbound_ulaw_remainder = b""

        logger.info("OpenAI Realtime pipeline stopped")

    async def handle_message(self, raw_message: str) -> None:
        try:
            event_type, event = parse_twilio_message(raw_message)
        except ValueError as e:
            logger.warning("Failed to parse Twilio message", error=str(e))
            return

        if event_type == TwilioEventType.START:
            await self._handle_start(event)
            return

        if event_type == TwilioEventType.MEDIA:
            await self._handle_media(event)
            return

        if event_type == TwilioEventType.STOP:
            self._protocol.handle_stop()
            await self.stop()
            return

        # Other events are ignored in this mode (marks/dtmf/etc.).

    async def _handle_start(self, event: TwilioStartEvent) -> None:
        self._protocol.handle_start(event)
        self._state = RealtimeState.LISTENING

        logger.info("Call started (realtime)", call_sid=event.call_sid, stream_sid=event.stream_sid)

        await self._start_pacer()
        await self._connect_openai()

        # Optional initial greeting to avoid "dead air" at call start.
        await self._openai_send(
            {
                "type": "response.create",
                "response": {
                    "modalities": ["audio"],
                    "instructions": "Greet the caller briefly and ask how you can help.",
                },
            }
        )

    async def _handle_media(self, event: TwilioMediaEvent) -> None:
        if not self._is_running:
            return

        if not event.payload:
            return

        # Send Twilio audio directly (g711_ulaw 8kHz).
        await self._openai_send_audio(event.payload)

    async def _start_pacer(self) -> None:
        if self._pacer_running and self._pacer_task and not self._pacer_task.done():
            return
        # Reset the queue on each start to ensure no stale poison-pill remains.
        self._outbound_queue = asyncio.Queue()
        self._pacer_running = True
        self._pacer_task = asyncio.create_task(self._audio_pacer())

    async def _stop_pacer(self) -> None:
        self._pacer_running = False
        # Drain queue
        while not self._outbound_queue.empty():
            try:
                self._outbound_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        await self._outbound_queue.put(None)
        if self._pacer_task and not self._pacer_task.done():
            self._pacer_task.cancel()
            try:
                await self._pacer_task
            except asyncio.CancelledError:
                pass

    async def _enqueue_twilio(self, message: str, *, pace: bool) -> None:
        await self._outbound_queue.put(OutboundMessage(message=message, pace=pace))

    async def _audio_pacer(self) -> None:
        frame_duration = 0.020
        next_send_time = time.time()
        try:
            while self._pacer_running and self._is_running:
                outbound = await self._outbound_queue.get()
                if outbound is None:
                    break

                if not outbound.pace:
                    await self._send_message(outbound.message)
                    continue

                now = time.time()
                if now < next_send_time:
                    await asyncio.sleep(next_send_time - now)
                await self._send_message(outbound.message)
                next_send_time = max(next_send_time + frame_duration, time.time())
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error("Realtime pacer failed", error=str(e))

    async def _connect_openai(self) -> None:
        if self._openai_ws:
            return

        api_key = (self.config.openai_api_key or "").strip()
        model = (self.config.openai_realtime_model or "").strip()
        if not api_key or not model:
            raise RuntimeError("OpenAI Realtime requires OPENAI_API_KEY and OPENAI_REALTIME_MODEL")

        url = f"wss://api.openai.com/v1/realtime?model={model}"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "OpenAI-Beta": "realtime=v1",
        }

        self._openai_ws = await websockets.connect(url, additional_headers=headers, open_timeout=10)
        self._openai_connected.set()

        self._openai_send_task = asyncio.create_task(self._openai_send_loop())
        self._openai_recv_task = asyncio.create_task(self._openai_receive_loop())

        instructions = (self.config.openai_realtime_instructions or "").strip() or _default_realtime_instructions(
            self.config
        )

        await self._openai_send(
            {
                "type": "session.update",
                "session": {
                    "modalities": ["audio", "text"],
                    "instructions": instructions,
                    "voice": self.config.openai_realtime_voice,
                    "input_audio_format": "g711_ulaw",
                    "output_audio_format": "g711_ulaw",
                    "turn_detection": {
                        "type": "server_vad",
                        "silence_duration_ms": int(self.config.openai_realtime_turn_silence_ms),
                        "prefix_padding_ms": 100,
                    },
                },
            }
        )

        logger.info(
            "OpenAI Realtime connected",
            model=model,
            voice=self.config.openai_realtime_voice,
            turn_silence_ms=self.config.openai_realtime_turn_silence_ms,
        )

    async def _openai_send(self, message: dict) -> None:
        if not self._is_running:
            return

        # Avoid blocking the Twilio receiver on OpenAI backpressure.
        try:
            self._openai_send_queue.put_nowait(message)
        except asyncio.QueueFull:
            # Drop non-audio control if we're saturated; audio input will naturally apply backpressure.
            logger.warning("OpenAI send queue full; dropping event", type=message.get("type"))

    async def _openai_send_audio(self, ulaw_bytes: bytes) -> None:
        if not ulaw_bytes or not self._is_running:
            return
        await self._openai_send(
            {
                "type": "input_audio_buffer.append",
                "audio": _b64encode(ulaw_bytes),
            }
        )

    async def _openai_send_loop(self) -> None:
        ws = self._openai_ws
        if not ws:
            return

        try:
            while self._is_running:
                item = await self._openai_send_queue.get()
                if item is None:
                    break
                try:
                    await ws.send(json.dumps(item))
                except Exception as e:
                    logger.error("OpenAI send failed", error=str(e), type=item.get("type"))
                    break
        except asyncio.CancelledError:
            pass

    async def _openai_receive_loop(self) -> None:
        ws = self._openai_ws
        if not ws:
            return

        try:
            async for raw in ws:
                if not self._is_running:
                    break
                try:
                    event = json.loads(raw)
                except Exception:
                    continue

                event_type = event.get("type")

                if event_type == "error":
                    logger.error("OpenAI Realtime error", details=event)
                    continue

                if event_type == "input_audio_buffer.speech_started":
                    await self._handle_user_speech_started()
                    continue

                if event_type == "input_audio_buffer.speech_stopped":
                    await self._handle_user_speech_stopped()
                    continue

                if event_type == "response.created":
                    self._active_response_id = event.get("response", {}).get("id") or event.get("response_id")
                    self._ignore_audio = False
                    self._state = RealtimeState.SPEAKING
                    continue

                if event_type in ("response.done", "response.completed", "response.cancelled"):
                    self._active_response_id = None
                    self._state = RealtimeState.LISTENING
                    # Flush any partial audio at end of response.
                    await self._flush_outbound_remainder()
                    continue

                if event_type in ("response.audio.delta", "response.output_audio.delta"):
                    if self._ignore_audio:
                        continue
                    response_id = event.get("response_id")
                    if response_id and self._active_response_id and response_id != self._active_response_id:
                        continue
                    delta = event.get("delta") or event.get("audio")
                    if isinstance(delta, str) and delta:
                        await self._enqueue_openai_audio(_b64decode(delta))
                    continue

                # Optional: log transcripts for debugging (no user-facing text in PSTN).
                if event_type in (
                    "conversation.item.input_audio_transcription.completed",
                    "response.audio_transcript.done",
                ):
                    try:
                        logger.info("Realtime transcript", text=str(event.get("transcript") or event)[:200])
                    except Exception:
                        pass

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error("OpenAI receive loop failed", error=str(e))

    async def _enqueue_openai_audio(self, ulaw_bytes: bytes) -> None:
        if not ulaw_bytes or not self._is_running:
            return

        # Accumulate and frame to 20ms without padding between deltas.
        self._outbound_ulaw_remainder += ulaw_bytes
        while len(self._outbound_ulaw_remainder) >= TWILIO_FRAME_SIZE:
            frame = self._outbound_ulaw_remainder[:TWILIO_FRAME_SIZE]
            self._outbound_ulaw_remainder = self._outbound_ulaw_remainder[TWILIO_FRAME_SIZE:]
            msg = create_media_message(self.stream_sid, frame)
            await self._enqueue_twilio(msg, pace=True)

    async def _flush_outbound_remainder(self) -> None:
        if not self._outbound_ulaw_remainder:
            return
        frame = self._outbound_ulaw_remainder.ljust(TWILIO_FRAME_SIZE, b"\xff")
        self._outbound_ulaw_remainder = b""
        msg = create_media_message(self.stream_sid, frame)
        await self._enqueue_twilio(msg, pace=True)

    async def _handle_user_speech_started(self) -> None:
        # Caller started speaking: stop assistant audio immediately.
        if not self.stream_sid:
            return

        self._ignore_audio = True
        self._outbound_ulaw_remainder = b""

        # Hard flush: stop our pacer + drop queued frames, then clear Twilio's buffer.
        # This keeps barge-in crisp (no "talking over" the caller).
        await self._stop_pacer()
        try:
            await self._send_message(create_clear_message(self.stream_sid))
        except Exception as e:
            logger.warning("Failed to send Twilio clear", error=str(e))
        await self._start_pacer()

        # Cancel OpenAI response (best effort).
        await self._openai_send({"type": "response.cancel"})

        self._state = RealtimeState.LISTENING

    async def _handle_user_speech_stopped(self) -> None:
        # Commit the audio buffer and request a response.
        await self._openai_send({"type": "input_audio_buffer.commit"})
        await self._openai_send({"type": "response.create", "response": {"modalities": ["audio"]}})
