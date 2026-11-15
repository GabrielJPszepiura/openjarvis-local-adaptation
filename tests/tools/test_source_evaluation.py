"""Tests for deterministic web-source profiling."""

from __future__ import annotations

from pathlib import Path

from openjarvis.tools.source_evaluation import SourceProfileCache, evaluate_source


def test_public_institution_seed_covers_subdomains(tmp_path: Path):
    profile = evaluate_source(
        "https://www.canada.ca/en/immigration-refugees-citizenship.html",
        cache=SourceProfileCache(tmp_path / "profiles.json"),
    )

    assert profile.category == "public_institution"
    assert profile.confidence == "verified_seed"
    assert profile.primary_candidate is True


def test_domain_convention_is_not_presented_as_verified_official(tmp_path: Path):
    profile = evaluate_source(
        "https://example.gov.br/page",
        cache=SourceProfileCache(tmp_path / "profiles.json"),
    )

    assert profile.category == "government_domain"
    assert profile.confidence == "domain_pattern"
    assert "provável" in profile.label


def test_secondary_source_is_not_an_institutional_candidate(tmp_path: Path):
    profile = evaluate_source(
        "https://agency.example/guide",
        cache=SourceProfileCache(tmp_path / "profiles.json"),
    )

    assert profile.category == "secondary_or_commercial"
    assert profile.institutional is False


def test_profile_is_reused_from_cache(tmp_path: Path):
    cache = SourceProfileCache(tmp_path / "profiles.json")
    first = evaluate_source("https://example.edu.br/course", cache=cache)
    second = evaluate_source("https://example.edu.br/another-course", cache=cache)

    assert first == second
