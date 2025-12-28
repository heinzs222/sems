"""
Quick language switching harness.

Runs a few deterministic assertions for English <-> French stabilization rules.

Usage:
  python scripts/language_switch_harness.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent.language import LangState, detect_language_override


def main() -> None:
    state = LangState(current="en")
    now = 0.0

    def feed(
        text: str,
        *,
        detected_language: str | None,
        language_confidence: float | None,
        dt: float = 1.0,
    ) -> tuple[bool, str]:
        nonlocal now
        now += dt

        override = detect_language_override(text)
        if override:
            previous = state.current
            state.force(override, now)
            assert state.current == override
            return (override != previous), "override"

        switched, reason = state.update_from_detection(
            text=text,
            detected_language=detected_language,
            language_confidence=language_confidence,
            now=now,
        )
        return switched, reason

    # 1) Ignore very short transcripts (<12 chars)
    switched, reason = feed("bonjour", detected_language="fr", language_confidence=0.99)
    assert switched is False and reason == ""
    assert state.current == "en"

    # 2) Require 2 consecutive finals to switch
    feed(
        "Bonjour, je voudrais connaître le prix.",
        detected_language="fr",
        language_confidence=0.9,
    )
    assert state.current == "en"

    switched, reason = feed(
        "Je veux connaître le prix, s'il vous plaît.",
        detected_language="fr",
        language_confidence=0.9,
    )
    assert switched is True and reason == "stabilized"
    assert state.current == "fr"

    # 3) Hold time (20s) prevents immediate switch back
    feed(
        "Hello, can you help me with pricing?",
        detected_language="en",
        language_confidence=0.9,
        dt=1.0,
    )
    assert state.current == "fr"

    feed(
        "Hello again, I prefer English.",
        detected_language="en",
        language_confidence=0.9,
        dt=1.0,
    )
    assert state.current == "fr"

    # After hold, still needs 2 consecutive finals.
    feed(
        "Hello, I'd like to continue in English.",
        detected_language="en",
        language_confidence=0.9,
        dt=21.0,
    )
    assert state.current == "fr"

    switched, reason = feed(
        "I have a question about the store.",
        detected_language="en",
        language_confidence=0.9,
        dt=1.0,
    )
    assert switched is True and reason == "stabilized"
    assert state.current == "en"

    # 4) Explicit overrides always win
    switched, reason = feed(
        "parle français",
        detected_language="en",
        language_confidence=0.9,
        dt=1.0,
    )
    assert switched is True and reason == "override"
    assert state.current == "fr"

    print("OK")


if __name__ == "__main__":
    main()
