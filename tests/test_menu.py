"""
Tests for menu parsing and matching.
"""

from __future__ import annotations

from src.agent.menu import (
    load_menu_from_html,
    resolve_menu_path,
    looks_like_menu_request,
    find_menu_items,
)


def test_menu_parses_expected_sections_and_items() -> None:
    path = resolve_menu_path("menu.html")
    catalog = load_menu_from_html(path)

    assert len(catalog.items) > 0
    assert "Thé Au Lait" in catalog.sections

    # Spot-check a known item from the committed menu snapshot.
    assert any(
        item.section == "Thé Au Lait" and item.name == "Noix De Coco" and item.price_text == "$8.55"
        for item in catalog.items
    )


def test_looks_like_menu_request_keywords() -> None:
    assert looks_like_menu_request("what's on the menu?", language="en") is True
    assert looks_like_menu_request("can I order?", language="en") is True
    assert looks_like_menu_request("je veux commander", language="fr") is True
    assert looks_like_menu_request("en français s'il te plaît", language="fr") is False


def test_find_menu_items_matches_name_tokens() -> None:
    catalog = load_menu_from_html(resolve_menu_path("menu.html"))

    matches = find_menu_items(catalog, "I want a taro milk tea", limit=5)
    assert any(item.name == "Taro" for item in matches)

