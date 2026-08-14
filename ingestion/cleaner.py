"""HTML and text cleaning for crawled pages."""

from __future__ import annotations

import hashlib
import re
from typing import Optional

from bs4 import BeautifulSoup, NavigableString, Tag

_DROP_TAGS = {
    "script",
    "style",
    "noscript",
    "iframe",
    "svg",
    "canvas",
    "form",
    "button",
    "nav",
    "footer",
    "header",
    "aside",
    "input",
    "select",
    "textarea",
}

_DROP_ATTR_HINTS = ("cookie", "consent", "newsletter", "social", "share", "menu")

_WHITESPACE_RE = re.compile(r"[ \t]+")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")


def normalize_text(text: str) -> str:
    """Collapse noisy whitespace while preserving paragraph breaks."""
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = _WHITESPACE_RE.sub(" ", cleaned)
    cleaned = "\n".join(line.strip() for line in cleaned.split("\n"))
    cleaned = _MULTI_NEWLINE_RE.sub("\n\n", cleaned)
    return cleaned.strip()


def extract_title(soup: BeautifulSoup, fallback: str = "") -> str:
    if soup.title and soup.title.string:
        title = soup.title.string.strip()
        if title:
            return title
    heading = soup.find(["h1", "h2"])
    if heading:
        heading_text = heading.get_text(" ", strip=True)
        if heading_text:
            return heading_text
    return fallback


def html_to_text(html: str, fallback_title: str = "") -> tuple[str, str]:
    """Return (title, cleaned_text) from raw HTML."""
    soup = BeautifulSoup(html, "lxml")
    title = extract_title(soup, fallback=fallback_title)

    for tag in soup.find_all(_DROP_TAGS):
        tag.decompose()

    for tag in soup.find_all(True):
        classes = " ".join(tag.get("class", [])).lower()
        element_id = str(tag.get("id", "")).lower()
        if any(hint in classes or hint in element_id for hint in _DROP_ATTR_HINTS):
            tag.decompose()

    main = soup.find("main") or soup.find("article") or soup.body or soup
    blocks: list[str] = []
    for element in main.find_all(["h1", "h2", "h3", "h4", "p", "li"]):
        if not isinstance(element, Tag):
            continue
        text = element.get_text(" ", strip=True)
        if not text:
            continue
        if element.name in {"h1", "h2", "h3", "h4"}:
            blocks.append(f"\n## {text}\n")
        else:
            blocks.append(text)

    if not blocks:
        raw = main.get_text("\n", strip=True) if main else ""
        if not raw:
            raw = " ".join(
                token.strip()
                for token in soup.stripped_strings
                if isinstance(token, (str, NavigableString))
            )
        return title, normalize_text(raw)

    return title, normalize_text("\n\n".join(blocks))


def content_sha256(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def markdown_to_text(markdown: str, fallback_title: str = "") -> tuple[str, str]:
    """Normalize local Markdown into title + body text."""
    lines = markdown.replace("\r\n", "\n").split("\n")
    title: Optional[str] = None
    body_lines: list[str] = []
    for line in lines:
        stripped = line.strip()
        if title is None and stripped.startswith("# "):
            title = stripped[2:].strip()
            continue
        body_lines.append(line)
    resolved_title = title or fallback_title
    return resolved_title, normalize_text("\n".join(body_lines))
