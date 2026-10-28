"""Fetching: which full text is used, and what is cached."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from paperq import _fetch

FIXTURES = Path(__file__).parent / "fixtures"
ATOM = (FIXTURES / "atom.xml").read_bytes()
HTML = (FIXTURES / "paper.html").read_bytes()


def _serve(monkeypatch: pytest.MonkeyPatch, pages: dict[str, bytes | None], tmp: Path | None = None) -> list[str]:
    asked: list[str] = []

    def cached(url: str) -> bytes | None:
        asked.append(url)
        return next((body for prefix, body in pages.items() if url.startswith(prefix)), None)

    def get(url: str) -> SimpleNamespace:
        body = cached(url)
        return SimpleNamespace(status_code=200 if body is not None else 404, content=body or b"")

    monkeypatch.setattr(_fetch, "_cached", cached)
    monkeypatch.setattr(_fetch, "_get", get)
    monkeypatch.setattr(_fetch, "API_RETRY_WAITS", (0,))
    monkeypatch.setattr(_fetch, "cache_dir", lambda: tmp or Path("/nonexistent-paperq-test"))
    return asked


def test_html_is_preferred(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    asked = _serve(monkeypatch, {"http://export.arxiv.org": ATOM, "https://arxiv.org/html/": HTML}, tmp_path)
    paper = _fetch.fetch("2610.19339")
    assert paper.version == "v2" and paper.url == "https://arxiv.org/abs/2610.19339v2"
    assert paper.text_source == "html" and "Wilcoxon" in paper.text
    assert "https://arxiv.org/html/2610.19339v2" in asked and not any("/pdf/" in u for u in asked)


def test_pdf_when_no_html(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _serve(monkeypatch, {"http://export.arxiv.org": ATOM, "https://arxiv.org/pdf/": b"%PDF"}, tmp_path)
    monkeypatch.setattr(_fetch, "_pdf_text", lambda pdf: "1 Introduction\nprose")
    paper = _fetch.fetch("2610.19339v2")
    assert paper.text_source == "pdf" and paper.html is None and paper.text.startswith("1 Introduction")


def test_no_full_text(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _serve(monkeypatch, {"http://export.arxiv.org": ATOM}, tmp_path)
    paper = _fetch.fetch("2610.19339v2")
    assert paper.text_source is None and paper.text == ""


def test_unknown_paper(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _serve(monkeypatch, {}, tmp_path)
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


def test_throttled_api_is_retried_and_never_cached(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    answers = iter([b"", b"", ATOM])  # arXiv throttles with an empty 200
    monkeypatch.setattr(_fetch, "_get", lambda url: SimpleNamespace(status_code=200, content=next(answers)))
    monkeypatch.setattr(_fetch, "cache_dir", lambda: tmp_path)
    monkeypatch.setattr(_fetch, "API_RETRY_WAITS", (0, 0, 0))
    assert _fetch._metadata("q", cacheable=True)["version"] == "v2"
    cached = list((tmp_path / "http").iterdir())
    assert len(cached) == 1 and b"<entry>" in cached[0].read_bytes()


def test_throttled_api_gives_up(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(_fetch, "_get", lambda url: SimpleNamespace(status_code=200, content=b""))
    monkeypatch.setattr(_fetch, "cache_dir", lambda: tmp_path)
    monkeypatch.setattr(_fetch, "API_RETRY_WAITS", (0,))
    with pytest.raises(_fetch.FetchError, match="empty answer"):
        _fetch._metadata("q", cacheable=True)
    assert not (tmp_path / "http").exists()
