"""
Unit tests for stabilized language switching.
"""

from src.agent.language import LangState, detect_language_override


class TestLanguageOverridePhrases:
    def test_french_overrides(self):
        assert detect_language_override("parle français") == "fr"
        assert detect_language_override("en français") == "fr"
        assert detect_language_override("français s’il te plaît") == "fr"

    def test_english_overrides(self):
        assert detect_language_override("speak English") == "en"
        assert detect_language_override("in English please") == "en"
        assert detect_language_override("English please") == "en"


class TestLangStateStabilization:
    def test_ignore_short_transcripts(self):
        state = LangState(current="en")
        switched, reason = state.update_from_detection(
            text="bonjour",
            detected_language="fr",
            language_confidence=0.99,
            now=1.0,
        )
        assert switched is False
        assert reason == ""
        assert state.current == "en"

    def test_requires_two_consecutive_finals_to_switch(self):
        state = LangState(current="en")
        switched, _ = state.update_from_detection(
            text="Bonjour, je voudrais parler en français.",
            detected_language="fr",
            language_confidence=0.9,
            now=1.0,
        )
        assert switched is False
        assert state.current == "en"

        switched, reason = state.update_from_detection(
            text="Je veux connaître le prix, s'il vous plaît.",
            detected_language="fr",
            language_confidence=0.9,
            now=2.0,
        )
        assert switched is True
        assert reason == "stabilized"
        assert state.current == "fr"

    def test_hold_time_prevents_flapping(self):
        state = LangState(current="en")

        # Switch to French (2 consecutive eligible finals).
        state.update_from_detection(
            text="Bonjour, je voudrais parler en français.",
            detected_language="fr",
            language_confidence=0.9,
            now=1.0,
        )
        switched, _ = state.update_from_detection(
            text="Je veux connaître le prix, s'il vous plaît.",
            detected_language="fr",
            language_confidence=0.9,
            now=2.0,
        )
        assert switched is True
        assert state.current == "fr"

        # Within hold time (20s), do not switch back to English.
        switched, _ = state.update_from_detection(
            text="Hello, I want to continue in English.",
            detected_language="en",
            language_confidence=0.9,
            now=10.0,
        )
        assert switched is False
        assert state.current == "fr"

        # After hold, still needs two consecutive eligible finals.
        switched, _ = state.update_from_detection(
            text="Hello, I'd like to continue in English.",
            detected_language="en",
            language_confidence=0.9,
            now=23.0,
        )
        assert switched is False
        assert state.current == "fr"

        switched, reason = state.update_from_detection(
            text="I have a question about the store.",
            detected_language="en",
            language_confidence=0.9,
            now=24.0,
        )
        assert switched is True
        assert reason == "stabilized"
        assert state.current == "en"

