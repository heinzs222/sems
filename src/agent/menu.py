"""
Menu parsing and retrieval.

Loads `menu.html` and extracts sections, item names, and prices so the agent can
answer menu questions without hallucinating.

The menu source is assumed to be an Uber Eats-style HTML export, where:
- Section titles appear under `data-testid="catalog-section-title"` (h3 text)
- Items appear under `li[data-testid^="store-item-"]`
- Item name/price appear inside `span[data-testid="rich-text"]`
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from functools import lru_cache
from html.parser import HTMLParser
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import unicodedata

import structlog

logger = structlog.get_logger(__name__)

_PRICE_RE = re.compile(r"\$\s*([0-9]+(?:\.[0-9]{1,2})?)")


@dataclass(frozen=True)
class MenuItem:
    section: str
    name: str
    price_text: str
    price_cents: int


@dataclass(frozen=True)
class MenuCatalog:
    sections: Dict[str, Tuple[MenuItem, ...]]

    @property
    def items(self) -> Tuple[MenuItem, ...]:
        items: List[MenuItem] = []
        for section_items in self.sections.values():
            items.extend(section_items)
        return tuple(items)

    def to_prompt_lines(self, *, max_items: Optional[int] = None) -> List[str]:
        """
        Render a compact, deterministic representation for prompting.
        """
        lines: List[str] = []
        count = 0
        for section_name in sorted(self.sections.keys()):
            lines.append(f"[SECTION] {section_name}")
            for item in self.sections[section_name]:
                lines.append(f"- {item.name} — {item.price_text}")
                count += 1
                if max_items is not None and count >= max_items:
                    return lines
        return lines


def _project_root() -> Path:
    # src/agent/menu.py -> src/agent -> src -> project root
    return Path(__file__).resolve().parent.parent.parent


def resolve_menu_path(menu_path: Optional[str] = None) -> Path:
    """
    Resolve a menu path.

    If menu_path is relative, it is interpreted relative to the project root.
    Defaults to `menu.html` in the project root.
    """
    if not menu_path:
        return _project_root() / "menu.html"

    path = Path(menu_path)
    if path.is_absolute():
        return path
    return _project_root() / path


def _normalize(text: str) -> str:
    """
    Normalize text for matching: casefold + strip accents + collapse whitespace.
    """
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = " ".join(text.split())
    return text.casefold()


def parse_price_to_cents(price_text: str) -> Optional[int]:
    match = _PRICE_RE.search(price_text or "")
    if not match:
        return None

    try:
        amount = Decimal(match.group(1))
        cents = int((amount * 100).quantize(Decimal("1")))
        return cents
    except (InvalidOperation, ValueError):
        return None


class _MenuHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._current_section: str = "Uncategorized"
        self._in_section_title_div = False
        self._capture_section_text = False

        self._in_item = False
        self._current_item_name: Optional[str] = None
        self._current_item_price_text: Optional[str] = None
        self._capture_rich_text = False

        self._items: List[MenuItem] = []

    @property
    def items(self) -> Tuple[MenuItem, ...]:
        return tuple(self._items)

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        attrs_dict = {k: v for k, v in attrs}

        if tag == "div" and attrs_dict.get("data-testid") == "catalog-section-title":
            self._in_section_title_div = True
            return

        if self._in_section_title_div and tag == "h3":
            self._capture_section_text = True
            return

        data_testid = attrs_dict.get("data-testid") or ""
        if tag == "li" and data_testid.startswith("store-item-"):
            self._in_item = True
            self._current_item_name = None
            self._current_item_price_text = None
            return

        if self._in_item and tag == "span" and attrs_dict.get("data-testid") == "rich-text":
            self._capture_rich_text = True
            return

    def handle_endtag(self, tag: str) -> None:
        if tag == "div" and self._in_section_title_div:
            self._in_section_title_div = False
            self._capture_section_text = False
            return

        if tag == "h3" and self._capture_section_text:
            self._capture_section_text = False
            return

        if tag == "span" and self._capture_rich_text:
            self._capture_rich_text = False
            return

        if tag == "li" and self._in_item:
            self._finalize_item()
            self._in_item = False
            self._current_item_name = None
            self._current_item_price_text = None
            self._capture_rich_text = False
            return

    def handle_data(self, data: str) -> None:
        text = (data or "").strip()
        if not text:
            return

        if self._capture_section_text:
            # Section titles are short. Ignore any weird long text blocks.
            if len(text) <= 80:
                self._current_section = text
            return

        if not (self._in_item and self._capture_rich_text):
            return

        # Within a store item, the first rich-text span is usually the item name,
        # and the next rich-text span starting with $ is the price.
        if self._current_item_name is None and not text.startswith("$"):
            self._current_item_name = text
            return

        if self._current_item_price_text is None and text.startswith("$"):
            self._current_item_price_text = text
            return

    def _finalize_item(self) -> None:
        if not self._current_item_name or not self._current_item_price_text:
            return

        cents = parse_price_to_cents(self._current_item_price_text)
        if cents is None:
            return

        self._items.append(
            MenuItem(
                section=self._current_section or "Uncategorized",
                name=self._current_item_name,
                price_text=self._current_item_price_text,
                price_cents=cents,
            )
        )


def load_menu_from_html(path: Path) -> MenuCatalog:
    html = path.read_bytes().decode("utf-8", errors="replace")

    parser = _MenuHTMLParser()
    parser.feed(html)

    sections: Dict[str, List[MenuItem]] = {}
    for item in parser.items:
        sections.setdefault(item.section, []).append(item)

    # Freeze
    frozen_sections: Dict[str, Tuple[MenuItem, ...]] = {
        section: tuple(items) for section, items in sections.items()
    }

    return MenuCatalog(sections=frozen_sections)


@lru_cache(maxsize=1)
def get_menu_catalog(menu_path: Optional[str] = None) -> Optional[MenuCatalog]:
    """
    Load and cache the menu catalog.

    Returns None if the menu file is missing or cannot be parsed.
    """
    path = resolve_menu_path(menu_path)
    if not path.exists():
        logger.warning("Menu file not found", menu_path=str(path))
        return None

    try:
        catalog = load_menu_from_html(path)
        logger.info(
            "Menu loaded",
            menu_path=str(path),
            num_sections=len(catalog.sections),
            num_items=len(catalog.items),
        )
        return catalog
    except Exception as e:
        logger.error("Failed to load menu", menu_path=str(path), error=str(e))
        return None


def find_menu_items(catalog: MenuCatalog, query: str, *, limit: int = 12) -> List[MenuItem]:
    """
    Find menu items by fuzzy substring match against the item name and section.
    """
    q = _normalize(query)
    if not q:
        return []

    scored: List[Tuple[int, MenuItem]] = []
    for item in catalog.items:
        name_n = _normalize(item.name)
        section_n = _normalize(item.section)

        score = 0
        if q in name_n:
            score += 3
        if q in section_n:
            score += 1
        # Light token overlap boost.
        q_tokens = set(q.split())
        if q_tokens:
            name_tokens = set(name_n.split())
            overlap = len(q_tokens & name_tokens)
            score += min(2, overlap)

        if score > 0:
            scored.append((score, item))

    scored.sort(key=lambda t: (-t[0], t[1].section, t[1].name))
    return [item for _, item in scored[:limit]]


def find_menu_sections(catalog: MenuCatalog, query: str, *, limit: int = 3) -> List[str]:
    """
    Find likely menu sections mentioned in the user's query.

    Returns section names ordered by match strength.
    """
    q = _normalize(query)
    if not q:
        return []

    q_tokens = set(q.split())
    if not q_tokens:
        return []

    scored: List[Tuple[int, str]] = []
    for section_name in catalog.sections.keys():
        section_n = _normalize(section_name)
        section_tokens = set(section_n.split())
        overlap = len(q_tokens & section_tokens)
        if overlap <= 0:
            continue

        score = overlap
        if section_n and section_n in q:
            score += 3
        scored.append((score, section_name))

    scored.sort(key=lambda t: (-t[0], t[1]))
    return [name for _, name in scored[:limit]]


def looks_like_full_menu_request(text: str, *, language: str = "en") -> bool:
    """
    Heuristic: user wants the full menu listing (not just a section or a few items).
    """
    t = _normalize(text)
    if not t:
        return False

    if language.strip().lower().startswith("fr"):
        keywords = [
            "tout le menu",
            "menu complet",
            "liste complete",
            "liste complète",
            "toute la liste",
            "tous les articles",
        ]
    else:
        keywords = [
            "full menu",
            "the full menu",
            "entire menu",
            "whole menu",
            "all items",
            "everything on the menu",
            "list everything",
        ]

    return any(k in t for k in keywords)


def looks_like_menu_request(text: str, *, language: str = "en") -> bool:
    """
    Lightweight heuristic to decide whether we should include menu context for the LLM.
    """
    t = _normalize(text)
    if not t:
        return False

    if "menu" in t:
        return True

    if language.strip().lower().startswith("fr"):
        keywords = [
            "commander",
            "commande",
            "qu'est ce que vous avez",
            "qu est ce que vous avez",
            "qu est-ce que vous avez",
            "prix",
            "boisson",
            "dessert",
            "je vais prendre",
            "je veux",
            "recommande",
            "suggestion",
            "populaire",
        ]
    else:
        keywords = [
            "what do you have",
            "what's on the menu",
            "whats on the menu",
            "order",
            "i want",
            "i'll take",
            "price",
            "how much is",
            "drink",
            "dessert",
            "recommend",
            "suggest",
            "popular",
        ]

    return any(k in t for k in keywords)
