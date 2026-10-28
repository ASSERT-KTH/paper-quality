"""Fetching: which full text is used, and what is cached."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from paperq import _fetch

FIXTURES = Path(__file__).parent / "fixtures"
ATOM = (FIXTURES / "atom.xml").read_bytes()
HTML = (FIXTURES / "paper.html").read_bytes()


def _serve(monkeypatch: pytest.MonkeyPatch, pages: dict[str, bytes | None]) -> list[str]:
    asked: list[str] = []

    def cached(url: str) -> bytes | None:
        asked.append(url)
        return next((body for prefix, body in pages.items() if url.startswith(prefix)), None)

    monkeypatch.setattr(_fetch, "_cached", cached)
    monkeypatch.setattr(_fetch, "_get", lambda url: SimpleNamespace(content=cached(url)))
    return asked


def test_html_is_preferred(monkeypatch: pytest.MonkeyPatch) -> None:
    asked = _serve(monkeypatch, {"http://export.arxiv.org": ATOM, "https://arxiv.org/html/": HTML})
    paper = _fetch.fetch("2610.19339")
    assert paper.version == "v2" and paper.url == "https://arxiv.org/abs/2610.19339v2"
    assert paper.text_source == "html" and "Wilcoxon" in paper.text
    assert "https://arxiv.org/html/2610.19339v2" in asked and not any("/pdf/" in u for u in asked)


def test_pdf_when_no_html(monkeypatch: pytest.MonkeyPatch) -> None:
    _serve(monkeypatch, {"http://export.arxiv.org": ATOM, "https://arxiv.org/pdf/": b"%PDF"})
    monkeypatch.setattr(_fetch, "_pdf_text", lambda pdf: "1 Introduction\nprose")
    paper = _fetch.fetch("2610.19339v2")
    assert paper.text_source == "pdf" and paper.html is None and paper.text.startswith("1 Introduction")


def test_no_full_text(monkeypatch: pytest.MonkeyPatch) -> None:
    _serve(monkeypatch, {"http://export.arxiv.org": ATOM})
    paper = _fetch.fetch("2610.19339v2")
    assert paper.text_source is None and paper.text == ""


def test_unknown_paper(monkeypatch: pytest.MonkeyPatch) -> None:
    _serve(monkeypatch, {})
    with pytest.raises(_fetch.FetchError):
        _fetch.fetch("2610.19339v1")


def test_cache_is_used(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    calls: list[str] = []

    def get(url: str) -> SimpleNamespace:
        calls.append(url)
        return SimpleNamespace(status_code=200, content=b"body", raise_for_status=lambda: None)

    monkeypatch.setattr(_fetch, "_get", get)
    assert _fetch._cached("https://example.org/a") == b"body"
    assert _fetch._cached("https://example.org/a") == b"body"
    assert calls == ["https://example.org/a"]


def test_404_is_none_and_not_cached(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setattr(_fetch, "_get", lambda url: SimpleNamespace(status_code=404))
    assert _fetch._cached("https://example.org/missing") is None
    assert not (tmp_path / "paperq").exists()
