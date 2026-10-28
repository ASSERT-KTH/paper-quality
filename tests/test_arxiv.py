"""The shared arXiv client: throttling is waited out, never mistaken for an answer."""

from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from paperq import _arxiv

FEED = b'<feed xmlns="http://www.w3.org/2005/Atom"><entry></entry></feed>'


def _answers(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *answers: Any) -> list[str]:
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setattr(_arxiv, "INTERVAL", 0)
    replies, asked = iter(answers), []

    def get(url: str, **_kwargs: object) -> SimpleNamespace:
        asked.append(url)
        status, body, *headers = next(replies)
        return SimpleNamespace(status_code=status, content=body, headers=headers[0] if headers else {})

    monkeypatch.setattr(_arxiv.requests, "get", get)
    return asked


def test_throttling_is_retried(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    asked = _answers(monkeypatch, tmp_path, (200, b""), (429, b"", {"Retry-After": "0"}), (200, FEED))
    assert _arxiv.get("u", _arxiv.api_ok, waits=(0, 0, 0)).content == FEED
    assert len(asked) == 3


def test_gives_up(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _answers(monkeypatch, tmp_path, (429, b""), (429, b""))
    with pytest.raises(_arxiv.ArxivError, match="HTTP 429"):
        _arxiv.get("u", _arxiv.api_ok, waits=(0,))


def test_404_is_an_answer(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _answers(monkeypatch, tmp_path, (404, b"not found"))
    assert _arxiv.get("u", _arxiv.api_ok, waits=(0,)).status_code == 404


def test_page_ok() -> None:
    assert _arxiv.page_ok(SimpleNamespace(status_code=200, content=b"<!DOCTYPE html><HTML>"))  # type: ignore[arg-type]
    assert not _arxiv.page_ok(SimpleNamespace(status_code=200, content=b""))  # type: ignore[arg-type]


def test_pace_is_shared_through_the_lock_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setattr(_arxiv, "INTERVAL", 0.2)
    _arxiv._paced()
    start = time.time()
    _arxiv._paced()  # a second caller (here the same process) waits for the interval
    assert time.time() - start >= 0.15
    assert float((tmp_path / "arxiv.lock").read_text()) > 0
