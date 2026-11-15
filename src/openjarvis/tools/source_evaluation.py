"""Fast, evidence-oriented source profiles for web-search results.

This module deliberately does *not* try to maintain a worldwide allow-list of
government websites.  A URL alone cannot prove who operates a site.  Instead,
it produces conservative categories and records whether the category comes
from a verified seed or merely a domain convention.  Callers can then require
stronger evidence for sensitive claims without treating a commercial page as
official by accident.
"""

from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlsplit

from openjarvis.core.paths import get_cache_dir

_CACHE_TTL = timedelta(days=30)
_CACHE_LOCK = threading.RLock()
# Small, auditable seeds for public institutions whose national domains do not
# follow a generic government suffix.  This is not a country-by-country list.
_VERIFIED_PUBLIC_HOSTS = frozenset(
    {"canada.ca", "who.int", "oecd.org", "worldbank.org"}
)


@dataclass(frozen=True)
class SourceProfile:
    """A conservative description of a search-result publisher."""

    host: str
    category: str
    confidence: str
    label: str
    institutional: bool
    primary_candidate: bool
    checked_at: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class SourceProfileCache:
    """Small on-disk cache for deterministic domain profiles.

    The profile calculation is cheap today, but caching creates one stable
    place to retain future page-identity verification without repeating work on
    every chat turn.  A cache failure never blocks a web answer.
    """

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or get_cache_dir() / "source_profiles.json"

    def get(self, host: str) -> SourceProfile | None:
        try:
            with _CACHE_LOCK:
                raw = json.loads(self.path.read_text(encoding="utf-8"))
            item = raw.get(host)
            if not isinstance(item, dict):
                return None
            checked_at = datetime.fromisoformat(str(item["checked_at"]))
            if datetime.now(timezone.utc) - checked_at > _CACHE_TTL:
                return None
            return SourceProfile(**item)
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            return None

    def put(self, profile: SourceProfile) -> None:
        try:
            with _CACHE_LOCK:
                data: dict[str, object] = {}
                if self.path.exists():
                    loaded = json.loads(self.path.read_text(encoding="utf-8"))
                    if isinstance(loaded, dict):
                        data = loaded
                self.path.parent.mkdir(parents=True, exist_ok=True)
                data[profile.host] = profile.to_dict()
                temporary = self.path.with_suffix(".tmp")
                temporary.write_text(
                    json.dumps(data, ensure_ascii=False, sort_keys=True),
                    encoding="utf-8",
                )
                temporary.replace(self.path)
        except (OSError, TypeError, json.JSONDecodeError):
            return


def _matches_government_convention(host: str) -> bool:
    labels = host.split(".")
    return (
        "gov" in labels
        or "govt" in labels
        or (len(labels) >= 2 and labels[-2:] == ["gc", "ca"])
    )


def _matches_academic_convention(host: str) -> bool:
    return "edu" in host.split(".") or ".ac." in f".{host}."


def evaluate_source(
    url: str, *, cache: SourceProfileCache | None = None
) -> SourceProfile:
    """Classify a URL without claiming ownership that the URL cannot prove."""
    host = (urlsplit(url).hostname or "").lower().strip(".")
    now = datetime.now(timezone.utc).isoformat()
    if not host:
        return SourceProfile(
            host="",
            category="unknown",
            confidence="none",
            label="não verificada",
            institutional=False,
            primary_candidate=False,
            checked_at=now,
        )

    if cache is None:
        try:
            cache = SourceProfileCache()
        except Exception:  # Cache availability must never break web search.
            cache = None
    if cache is not None:
        cached = cache.get(host)
        if cached is not None:
            return cached

    registered_public = any(
        host == trusted or host.endswith(f".{trusted}")
        for trusted in _VERIFIED_PUBLIC_HOSTS
    )
    if registered_public:
        profile = SourceProfile(
            host=host,
            category="public_institution",
            confidence="verified_seed",
            label="instituição pública reconhecida",
            institutional=True,
            primary_candidate=True,
            checked_at=now,
        )
    elif _matches_government_convention(host):
        profile = SourceProfile(
            host=host,
            category="government_domain",
            confidence="domain_pattern",
            label="domínio governamental provável",
            institutional=True,
            primary_candidate=True,
            checked_at=now,
        )
    elif _matches_academic_convention(host):
        profile = SourceProfile(
            host=host,
            category="academic",
            confidence="domain_pattern",
            label="fonte acadêmica provável",
            institutional=True,
            primary_candidate=False,
            checked_at=now,
        )
    elif host.endswith(".int") or host.endswith(".un.org"):
        profile = SourceProfile(
            host=host,
            category="international_organization",
            confidence="domain_pattern",
            label="organização internacional provável",
            institutional=True,
            primary_candidate=True,
            checked_at=now,
        )
    else:
        profile = SourceProfile(
            host=host,
            category="secondary_or_commercial",
            confidence="unverified",
            label="fonte secundária ou comercial",
            institutional=False,
            primary_candidate=False,
            checked_at=now,
        )
    if cache is not None:
        cache.put(profile)
    return profile


def is_sensitive_query(query: str) -> bool:
    """Identify subjects where unsupported facts need a stronger source."""
    lowered = query.casefold()
    terms = (
        "visto", "visa", "imigra", "lei", "legal", "regula", "tribut",
        "imposto", "tax", "medic", "saúde", "saude", "doença", "doenca",
        "remédio", "remedio", "tratamento", "financeir", "investimento",
        "juros", "crédito", "credito", "bolsa de valores",
    )
    return any(term in lowered for term in terms)

