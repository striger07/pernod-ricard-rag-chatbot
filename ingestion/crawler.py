"""Crawl4AI + BeautifulSoup ingestion with retries and local fallback."""

from __future__ import annotations

import asyncio
import ipaddress
import socket
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional, Sequence
from urllib.parse import urljoin, urlparse

import httpx
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from config.settings import Settings, get_settings
from config.sources import allowed_hostnames, seed_urls, source_by_hostname
from ingestion.cleaner import content_sha256, html_to_text, markdown_to_text
from ingestion.models import CrawledDocument
from utils.logging import get_logger

logger = get_logger(__name__)

_PRIVATE_NETWORKS = (
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fc00::/7"),
)


class CrawlError(Exception):
    """Raised when a page cannot be crawled after retries."""


def _hostname_allowed(hostname: str, allowlist: Sequence[str]) -> bool:
    host = hostname.lower().lstrip(".")
    if host.startswith("www."):
        bare = host[4:]
    else:
        bare = host
    return host in allowlist or bare in allowlist or f"www.{bare}" in allowlist


def _is_private_ip(address: str) -> bool:
    try:
        ip = ipaddress.ip_address(address)
    except ValueError:
        return True
    return any(ip in network for network in _PRIVATE_NETWORKS) or ip.is_private or ip.is_loopback


def validate_public_url(url: str, allowlist: Sequence[str]) -> str:
    """Reject non-http(s), off-allowlist hosts, and private IPs (SSRF control)."""
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise CrawlError(f"Unsupported URL scheme: {parsed.scheme!r}")
    hostname = (parsed.hostname or "").lower()
    if not hostname:
        raise CrawlError("URL is missing a hostname")
    if not _hostname_allowed(hostname, allowlist):
        raise CrawlError(f"Host {hostname!r} is not in the crawl allowlist")
    try:
        infos = socket.getaddrinfo(hostname, None)
    except socket.gaierror as exc:
        raise CrawlError(f"DNS resolution failed for {hostname}") from exc
    for info in infos:
        sockaddr = info[4]
        if sockaddr and _is_private_ip(str(sockaddr[0])):
            raise CrawlError(f"Refusing to crawl private address for {hostname}")
    return url


def _source_name_for_url(url: str) -> str:
    hostname = (urlparse(url).hostname or "").lower()
    mapping = source_by_hostname()
    bare = hostname[4:] if hostname.startswith("www.") else hostname
    source = mapping.get(hostname) or mapping.get(bare)
    if source:
        return source.name
    return hostname or "unknown"


async def _fetch_with_httpx(url: str, settings: Settings) -> tuple[str, str]:
    headers = {"User-Agent": settings.crawl_user_agent, "Accept": "text/html,application/xhtml+xml"}
    timeout = httpx.Timeout(settings.crawl_request_timeout_seconds)
    async with httpx.AsyncClient(follow_redirects=True, timeout=timeout, headers=headers) as client:
        response = await client.get(url)
        response.raise_for_status()
        content_type = response.headers.get("content-type", "")
        if "html" not in content_type and "text" not in content_type:
            raise CrawlError(f"Unsupported content type {content_type!r} for {url}")
        return str(response.url), response.text


async def _fetch_with_crawl4ai(url: str, settings: Settings) -> Optional[tuple[str, str]]:
    try:
        from crawl4ai import AsyncWebCrawler
    except Exception as exc:  # pragma: no cover - optional runtime dependency failure
        logger.warning("crawl4ai_unavailable", error=str(exc))
        return None

    try:
        async with AsyncWebCrawler(verbose=False) as crawler:
            result = await crawler.arun(url=url)
    except TypeError:
        crawler = AsyncWebCrawler()
        try:
            result = await crawler.arun(url=url)
        finally:
            closer = getattr(crawler, "close", None)
            if closer is not None:
                maybe = closer()
                if asyncio.iscoroutine(maybe):
                    await maybe
    except Exception as exc:
        logger.warning("crawl4ai_failed", url=url, error=str(exc))
        return None

    success = getattr(result, "success", True)
    if success is False:
        logger.warning("crawl4ai_unsuccessful", url=url, error=getattr(result, "error_message", ""))
        return None

    html = getattr(result, "cleaned_html", None) or getattr(result, "html", None) or ""
    if not html:
        markdown = getattr(result, "markdown", "") or ""
        if markdown:
            return url, f"<html><body><pre>{markdown}</pre></body></html>"
        return None
    final_url = getattr(result, "url", None) or url
    return str(final_url), str(html)


async def fetch_page(url: str, settings: Settings) -> CrawledDocument:
    """Fetch one URL with Crawl4AI, falling back to HTTPX + BeautifulSoup."""
    allowlist = allowed_hostnames()
    validate_public_url(url, allowlist)

    async for attempt in AsyncRetrying(
        stop=stop_after_attempt(settings.crawl_max_retries),
        wait=wait_exponential(
            multiplier=settings.crawl_retry_backoff_seconds,
            min=settings.crawl_retry_backoff_seconds,
            max=20,
        ),
        retry=retry_if_exception_type((httpx.HTTPError, CrawlError, TimeoutError, OSError)),
        reraise=True,
    ):
        with attempt:
            fetched = await _fetch_with_crawl4ai(url, settings)
            if fetched is None:
                fetched = await _fetch_with_httpx(url, settings)
            final_url, html = fetched
            title, text = html_to_text(html, fallback_title=url)
            if len(text) < 40:
                raise CrawlError(f"Extracted content too short for {url}")
            timestamp = datetime.now(timezone.utc)
            outbound = extract_same_domain_links(html, final_url, allowlist)
            document = CrawledDocument(
                content=text,
                title=title,
                url=final_url,
                source=_source_name_for_url(final_url),
                timestamp=timestamp,
                content_hash=content_sha256(text),
                metadata={
                    "crawl_method": "crawl4ai_or_httpx",
                    "original_url": url,
                    "ingested_at": timestamp.isoformat(),
                    "outbound_links": outbound[:50],
                },
            )
            logger.info("crawl_success", url=final_url, title=title, chars=len(text))
            return document

    raise CrawlError(f"Exhausted retries for {url}")


def load_synthetic_documents(directory: str | Path) -> list[CrawledDocument]:
    """Load local Markdown files when live crawling is blocked."""
    root = Path(directory)
    if not root.exists():
        logger.warning("synthetic_dir_missing", path=str(root))
        return []

    documents: list[CrawledDocument] = []
    for path in sorted(root.rglob("*.md")):
        raw = path.read_text(encoding="utf-8")
        title, text = markdown_to_text(raw, fallback_title=path.stem.replace("_", " ").title())
        source_name = path.parent.name if path.parent.name != root.name else "synthetic"
        relative = path.as_posix()
        url = f"synthetic://{relative}"
        timestamp = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        documents.append(
            CrawledDocument(
                content=text,
                title=title,
                url=url,
                source=source_name,
                timestamp=timestamp,
                content_hash=content_sha256(text),
                metadata={
                    "crawl_method": "synthetic_fallback",
                    "local_path": str(path),
                    "ingested_at": datetime.now(timezone.utc).isoformat(),
                },
            )
        )
    logger.info("synthetic_loaded", count=len(documents), directory=str(root))
    return documents


def extract_same_domain_links(html: str, base_url: str, allowlist: Sequence[str]) -> list[str]:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "lxml")
    found: list[str] = []
    for anchor in soup.find_all("a", href=True):
        href = str(anchor["href"]).strip()
        if not href or href.startswith(("#", "mailto:", "javascript:")):
            continue
        absolute = urljoin(base_url, href)
        parsed = urlparse(absolute)
        if parsed.scheme not in {"http", "https"}:
            continue
        hostname = (parsed.hostname or "").lower()
        if not _hostname_allowed(hostname, allowlist):
            continue
        cleaned = parsed._replace(fragment="").geturl()
        found.append(cleaned)
    return found


class KnowledgeCrawler:
    """Polite, allowlisted crawler with synthetic fallback."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()

    async def crawl(
        self,
        urls: Optional[Iterable[str]] = None,
        *,
        use_fallback_on_failure: Optional[bool] = None,
    ) -> list[CrawledDocument]:
        seeds = list(urls) if urls is not None else list(seed_urls(self.settings.crawl_max_pages))
        allowlist = allowed_hostnames()
        fallback = (
            self.settings.crawl_use_synthetic_fallback
            if use_fallback_on_failure is None
            else use_fallback_on_failure
        )

        seen: set[str] = set()
        queued: list[tuple[str, int]] = []
        for seed in seeds:
            try:
                validate_public_url(seed, allowlist)
            except CrawlError as exc:
                logger.warning("seed_rejected", url=seed, error=str(exc))
                continue
            if seed not in seen:
                seen.add(seed)
                queued.append((seed, 0))

        documents: list[CrawledDocument] = []
        hashes: set[str] = set()

        while queued and len(documents) < self.settings.crawl_max_pages:
            url, depth = queued.pop(0)
            try:
                document = await fetch_page(url, self.settings)
            except Exception as exc:
                logger.error("crawl_failure", url=url, error=str(exc))
                continue
            if document.content_hash in hashes:
                logger.info("duplicate_skipped", url=document.url)
                continue
            hashes.add(document.content_hash)
            documents.append(document)
            if depth < self.settings.crawl_max_depth:
                for link in document.metadata.get("outbound_links", []):
                    if link in seen:
                        continue
                    try:
                        validate_public_url(link, allowlist)
                    except CrawlError:
                        continue
                    seen.add(link)
                    queued.append((link, depth + 1))
            await asyncio.sleep(self.settings.crawl_rate_limit_seconds)

        if documents:
            return documents

        if fallback:
            logger.warning("crawl_blocked_using_synthetic_fallback")
            synthetic = load_synthetic_documents(self.settings.synthetic_data_dir)
            unique: list[CrawledDocument] = []
            seen_hashes: set[str] = set()
            for document in synthetic:
                if document.content_hash in seen_hashes:
                    continue
                seen_hashes.add(document.content_hash)
                unique.append(document)
            return unique

        logger.error("crawl_empty_no_fallback")
        return []
