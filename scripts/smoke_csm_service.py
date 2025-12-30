from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import httpx


def _build_payload(prompt_text: str) -> dict:
    # Minimal context: text-only turns. Audio context is optional.
    return {
        "speaker_id": 0,
        "prompt_text": prompt_text,
        "context": [
            {"role": "1", "text": "Hi there."},
            {"role": "0", "text": "Hey! How can I help?"},
        ],
        "voice_style": "default",
        "deterministic": True,
        "temperature": 1.0,
        "max_length": 200,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test for the CSM TTS microservice.")
    parser.add_argument(
        "--endpoint",
        default=os.getenv("CSM_ENDPOINT", "http://localhost:8001"),
        help="Base URL for the microservice (default: $CSM_ENDPOINT or http://localhost:8001)",
    )
    parser.add_argument(
        "--text",
        default="Got it. What name should I put on the order?",
        help="Prompt text to synthesize",
    )
    parser.add_argument(
        "--out",
        default="csm_smoke.wav",
        help="Output WAV path",
    )
    args = parser.parse_args()

    endpoint = (args.endpoint or "").rstrip("/")
    url = endpoint + "/tts"

    payload = _build_payload(args.text)

    print(f"POST {url}")
    with httpx.Client(timeout=60.0) as client:
        resp = client.post(url, json=payload, headers={"Accept": "audio/wav"})
    resp.raise_for_status()

    out_path = Path(args.out)
    out_path.write_bytes(resp.content)

    latency = resp.headers.get("x-latency-ms") or resp.headers.get("X-Latency-Ms")
    model_id = resp.headers.get("x-model-id") or resp.headers.get("X-Model-Id")
    device = resp.headers.get("x-device") or resp.headers.get("X-Device")

    print(f"Saved: {out_path.resolve()}")
    if latency:
        print(f"Latency: {latency} ms")
    if model_id or device:
        print(f"Model: {model_id or '?'}  Device: {device or '?'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

