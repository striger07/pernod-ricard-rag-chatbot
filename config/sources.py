"""Central knowledge-source configuration for crawling and citations."""

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple


@dataclass(frozen=True)
class KnowledgeSource:
    """A crawlable knowledge source with an allowlisted hostname."""

    name: str
    base_url: str
    category: str
    seed_urls: Tuple[str, ...] = field(default_factory=tuple)
    official_site: str = ""

    @property
    def hostname(self) -> str:
        from urllib.parse import urlparse

        host = (urlparse(self.base_url).hostname or "").lower()
        return host[4:] if host.startswith("www.") else host


PERNOD_RICARD_SOURCES: Tuple[KnowledgeSource, ...] = (
    KnowledgeSource(
        name="Pernod Ricard",
        base_url="https://www.pernod-ricard.com",
        category="corporate",
        seed_urls=(
            "https://www.pernod-ricard.com/en",
            "https://www.pernod-ricard.com/en/our-brands",
            "https://www.pernod-ricard.com/en/our-group",
        ),
        official_site="https://www.pernod-ricard.com",
    ),
)

BRAND_SOURCES: Tuple[KnowledgeSource, ...] = (
    KnowledgeSource(
        name="Absolut",
        base_url="https://www.absolut.com",
        category="brand",
        seed_urls=("https://www.absolut.com/en/",),
        official_site="https://www.absolut.com",
    ),
    KnowledgeSource(
        name="Chivas Regal",
        base_url="https://www.chivas.com",
        category="brand",
        seed_urls=("https://www.chivas.com/",),
        official_site="https://www.chivas.com",
    ),
    KnowledgeSource(
        name="Jameson",
        base_url="https://www.jamesonwhiskey.com",
        category="brand",
        seed_urls=("https://www.jamesonwhiskey.com/",),
        official_site="https://www.jamesonwhiskey.com",
    ),
    KnowledgeSource(
        name="The Glenlivet",
        base_url="https://www.theglenlivet.com",
        category="brand",
        seed_urls=("https://www.theglenlivet.com/",),
        official_site="https://www.theglenlivet.com",
    ),
    KnowledgeSource(
        name="Beefeater",
        base_url="https://www.beefeatergin.com",
        category="brand",
        seed_urls=("https://www.beefeatergin.com/",),
        official_site="https://www.beefeatergin.com",
    ),
    KnowledgeSource(
        name="Ballantine's",
        base_url="https://www.ballantines.com",
        category="brand",
        seed_urls=("https://www.ballantines.com/",),
        official_site="https://www.ballantines.com",
    ),
    KnowledgeSource(
        name="Royal Salute",
        base_url="https://www.royalsalute.com",
        category="brand",
        seed_urls=("https://www.royalsalute.com/",),
        official_site="https://www.royalsalute.com",
    ),
    KnowledgeSource(
        name="Malibu",
        base_url="https://www.maliburumdrinks.com",
        category="brand",
        seed_urls=("https://www.maliburumdrinks.com/",),
        official_site="https://www.maliburumdrinks.com",
    ),
    KnowledgeSource(
        name="Kahlúa",
        base_url="https://www.kahlua.com",
        category="brand",
        seed_urls=("https://www.kahlua.com/",),
        official_site="https://www.kahlua.com",
    ),
    KnowledgeSource(
        name="G.H. Mumm",
        base_url="https://www.ghmumm.com",
        category="brand",
        seed_urls=("https://www.ghmumm.com/",),
        official_site="https://www.ghmumm.com",
    ),
    KnowledgeSource(
        name="Perrier-Jouët",
        base_url="https://www.perrier-jouet.com",
        category="brand",
        seed_urls=("https://www.perrier-jouet.com/",),
        official_site="https://www.perrier-jouet.com",
    ),
)

EXTERNAL_SOURCES: Tuple[KnowledgeSource, ...] = (
    KnowledgeSource(
        name="Wikipedia",
        base_url="https://en.wikipedia.org",
        category="external",
        seed_urls=(
            "https://en.wikipedia.org/wiki/Pernod_Ricard",
            "https://en.wikipedia.org/wiki/Absolut_Vodka",
            "https://en.wikipedia.org/wiki/Chivas_Regal",
            "https://en.wikipedia.org/wiki/Jameson_Irish_Whiskey",
            "https://en.wikipedia.org/wiki/The_Glenlivet",
        ),
        official_site="https://en.wikipedia.org",
    ),
    KnowledgeSource(
        name="Reuters",
        base_url="https://www.reuters.com",
        category="external",
        seed_urls=("https://www.reuters.com/companies/RI.PA",),
        official_site="https://www.reuters.com",
    ),
    KnowledgeSource(
        name="Financial Times",
        base_url="https://www.ft.com",
        category="external",
        seed_urls=("https://www.ft.com/stream/pernod-ricard",),
        official_site="https://www.ft.com",
    ),
    KnowledgeSource(
        name="Difford's Guide",
        base_url="https://www.diffordsguide.com",
        category="external",
        seed_urls=("https://www.diffordsguide.com/brands/pernod-ricard",),
        official_site="https://www.diffordsguide.com",
    ),
    KnowledgeSource(
        name="Decanter",
        base_url="https://www.decanter.com",
        category="external",
        seed_urls=("https://www.decanter.com/",),
        official_site="https://www.decanter.com",
    ),
)


def all_sources() -> Tuple[KnowledgeSource, ...]:
    return PERNOD_RICARD_SOURCES + BRAND_SOURCES + EXTERNAL_SOURCES


def allowed_hostnames() -> Tuple[str, ...]:
    hosts: List[str] = []
    for source in all_sources():
        host = source.hostname
        if host:
            hosts.append(host)
            hosts.append(f"www.{host}")
    return tuple(sorted(set(hosts)))


def source_by_hostname() -> Dict[str, KnowledgeSource]:
    mapping: Dict[str, KnowledgeSource] = {}
    for source in all_sources():
        mapping[source.hostname] = source
        mapping[f"www.{source.hostname}"] = source
    return mapping


def official_url_for_source_name(name: str) -> str:
    needle = name.strip().lower()
    for source in all_sources():
        if source.name.lower() == needle:
            return source.official_site or source.base_url
    return "https://www.pernod-ricard.com"


def seed_urls(limit: int | None = None) -> Sequence[str]:
    urls: List[str] = []
    for source in all_sources():
        urls.extend(source.seed_urls)
    if limit is not None:
        return urls[:limit]
    return urls
