"""
Tests for lightweight language inference from transcript text.
"""

from src.agent.language import infer_language_from_text


class TestInferLanguageFromText:
    def test_detects_french(self):
        lang, confidence = infer_language_from_text(
            "Bonjour, je voudrais commander un tiramisu classique s'il vous plaît."
        )
        assert lang == "fr"
        assert confidence is not None and confidence >= 0.75

    def test_detects_english(self):
        lang, confidence = infer_language_from_text(
            "Hello, I want to order a matcha milk tea please."
        )
        assert lang == "en"
        assert confidence is not None and confidence >= 0.75

    def test_returns_none_for_ambiguous_short_content(self):
        lang, confidence = infer_language_from_text("Matcha taro")
        assert lang is None
        assert confidence is None

