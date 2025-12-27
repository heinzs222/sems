"""
Tests for language switching utilities.
"""

from src.agent.language import detect_language_switch_command


class TestLanguageSwitchDetection:
    def test_detect_switch_to_french_english_phrase(self):
        assert detect_language_switch_command("Can you speak French?") == "fr"
        assert detect_language_switch_command("switch to french please") == "fr"
        assert detect_language_switch_command("talk in french") == "fr"
        assert detect_language_switch_command("in french") == "fr"

    def test_detect_switch_to_french_french_phrase(self):
        assert detect_language_switch_command("Parle français") == "fr"
        assert detect_language_switch_command("Parlez francais s'il vous plaît") == "fr"
        assert detect_language_switch_command("Passe en français") == "fr"
        assert detect_language_switch_command("en français") == "fr"

    def test_detect_switch_to_english_english_phrase(self):
        assert detect_language_switch_command("Can you speak English?") == "en"
        assert detect_language_switch_command("switch to english please") == "en"
        assert detect_language_switch_command("talk in english") == "en"
        assert detect_language_switch_command("in english") == "en"

    def test_detect_switch_to_english_french_phrase(self):
        assert detect_language_switch_command("Parle anglais") == "en"
        assert detect_language_switch_command("Parlez anglais s'il vous plaît") == "en"
        assert detect_language_switch_command("Passe en anglais") == "en"
        assert detect_language_switch_command("en anglais") == "en"

    def test_no_switch_detected(self):
        assert detect_language_switch_command("") is None
        assert detect_language_switch_command("hello there") is None
        assert detect_language_switch_command("bonjour") is None

