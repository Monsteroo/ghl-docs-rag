from unittest.mock import MagicMock, patch

from translate import translate_to_english


def test_ascii_text_is_returned_unchanged_without_calling_the_api():
    with patch("translate.client.messages.create") as mock_create:
        result = translate_to_english("how does it work?")
    mock_create.assert_not_called()
    assert result == "how does it work?"


def test_non_ascii_text_is_translated_via_the_api():
    fake_response = MagicMock()
    fake_response.content = [MagicMock(type="text", text="How do I add tags to a contact?")]
    with patch("translate.client.messages.create", return_value=fake_response) as mock_create:
        result = translate_to_english("Як додати теги до контакта?")
    mock_create.assert_called_once()
    assert result == "How do I add tags to a contact?"


def test_a_leading_thinking_block_is_skipped_in_favor_of_the_text_block():
    fake_response = MagicMock()
    fake_response.content = [
        MagicMock(type="thinking", text=None, thinking="..."),
        MagicMock(type="text", text="How do I add tags to a contact?"),
    ]
    with patch("translate.client.messages.create", return_value=fake_response):
        result = translate_to_english("Як додати теги до контакта?")
    assert result == "How do I add tags to a contact?"


def test_falls_back_to_the_original_text_if_no_text_block_is_returned():
    fake_response = MagicMock()
    fake_response.content = [MagicMock(type="thinking", text=None, thinking="...")]
    with patch("translate.client.messages.create", return_value=fake_response):
        result = translate_to_english("Як додати теги до контакта?")
    assert result == "Як додати теги до контакта?"
