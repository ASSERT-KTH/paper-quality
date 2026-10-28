"""Selection rules of scripts/build_refsets.py: they decide what counts as good and bad."""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import build_refsets as b  # noqa: E402


@pytest.mark.parametrize(
    "ratings, negative",
    [([2, 2, 0], True), ([2, 2, 4], False), ([4, 4, 4], False), ([0, 2, 6], False), ([2, 2, 2, 2], True)],
)
def test_is_negative(ratings: list[int], negative: bool) -> None:
    assert b.is_negative(ratings) is negative


@pytest.mark.parametrize(
    "ratings, boundary",
    [([2, 6, 6], True), ([4, 4, 6], False), ([2, 8, 4], True), ([0, 2, 2], False), ([8, 8, 10], False)],
)
def test_is_boundary(ratings: list[int], boundary: bool) -> None:
    assert b.is_boundary(ratings) is boundary


def test_norm_ignores_case_punctuation_and_accents() -> None:
    assert b._norm("Scientific-Judgment Collapse: Évaluation!") == b._norm("scientific judgment collapse evaluation")


def test_pin_before(monkeypatch: pytest.MonkeyPatch) -> None:
    versions = [("v1", dt.date(2025, 9, 20)), ("v2", dt.date(2025, 11, 30)), ("v3", dt.date(2026, 3, 1))]
    monkeypatch.setattr(b, "arxiv_versions", lambda _id: versions)
    assert b.pin_before("x", dt.date(2026, 1, 26)) == ("v2", "2025-11-30")
    assert b.pin_before("x", dt.date(2025, 1, 1)) == ("v1", "2025-09-20")  # posted after the cutoff: v1


def test_pin_before_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(b, "arxiv_versions", lambda _id: [])
    assert b.pin_before("x", dt.date(2026, 1, 26)) is None
