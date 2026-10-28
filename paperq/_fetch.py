"""Fetch a paper's metadata and full text from arXiv, with an on-disk cache.

Full text comes from arXiv's HTML rendering when there is one, else from the PDF through
`pdftotext`. A versioned id (2610.19339v2) is immutable, so its downloads are cached forever.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import feedparser
import requests

from paperq._facts import html_body_text

USER_AGENT = "paperq/0.1 (https://github.com/ASSERT-KTH/paper-quality)"
TIMEOUT = 60
ARXIV_ID = re.compile(r"(?:arxiv\.org/(?:abs|pdf|html)/|arxiv:)?(\d{4}\.\d{4,5})(v\d+)?", re.I)


class FetchError(Exception):
    """The paper could not be fetched at all."""


@dataclass
class Paper:
    id: str
    version: str | None
    title: str
    authors: list[str]
    url: str
    published: str | None
    journal_ref: str | None
    comments: str | None
    html: str | None
    text: str
    text_source: str | None  # "html", "pdf" or None when no full text was found


def cache_dir() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
    return Path(base) / "paperq"


def parse_arxiv_id(ref: str) -> tuple[str, str | None]:
    """`2610.19339v2`, `arxiv:2610.19339` or an arXiv URL → (`2610.19339`, `v2`)."""
    found = ARXIV_ID.search(ref.strip())
    if not found:
        raise FetchError(f"not an arXiv id or URL: {ref}")
    return found.group(1), found.group(2)


def _get(url: str) -> requests.Response:
    return requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT)


def _cached(url: str) -> bytes | None:
    """GET through the cache; None on 404. Other HTTP errors raise."""
    path = cache_dir() / "http" / hashlib.sha256(url.encode()).hexdigest()
    if path.exists():
        return path.read_bytes()
    response = _get(url)
    if response.status_code == 404:
        return None
    response.raise_for_status()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(response.content)
    return response.content


def parse_atom(atom: str) -> dict[str, Any]:
    """The metadata of the single entry of an arXiv API answer."""
    feed = feedparser.parse(atom)
    if not feed.entries or "id" not in feed.entries[0]:
        raise FetchError("arXiv returned no entry")
    entry = feed.entries[0]
    _, version = parse_arxiv_id(entry["id"])
    return {
        "version": version,
        "title": re.sub(r"\s+", " ", entry.get("title", "")).strip(),
        "authors": [a.get("name", "") for a in entry.get("authors", [])],
        "published": (entry.get("published") or "")[:10] or None,
        "journal_ref": entry.get("arxiv_journal_ref"),
        "comments": entry.get("arxiv_comment"),
    }


def _pdf_text(pdf: bytes) -> str:
    if not shutil.which("pdftotext"):
        raise FetchError("no HTML rendering, and pdftotext is not installed")
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "paper.pdf"
        src.write_bytes(pdf)
        out = subprocess.run(["pdftotext", "-layout", str(src), "-"], capture_output=True, text=True, timeout=120)
    if out.returncode != 0:
        raise FetchError(f"pdftotext failed: {out.stderr.strip()[:200]}")
    return out.stdout


def fetch(ref: str) -> Paper:
    """Metadata and full text of an arXiv paper. Raises FetchError when even metadata fails."""
    arxiv_id, version = parse_arxiv_id(ref)
    query = f"http://export.arxiv.org/api/query?id_list={arxiv_id}{version or ''}"
    try:
        # an unversioned query names the latest version, which changes: never cache it
        atom = _cached(query) if version else _get(query).content
    except requests.RequestException as exc:
        raise FetchError(f"arXiv API: {exc}") from exc
    if not atom:
        raise FetchError(f"arXiv API has no {arxiv_id}")
    meta = parse_atom(atom.decode("utf-8", "replace"))
    version = version or meta["version"]
    vid = f"{arxiv_id}{version or ''}"

    html: str | None = None
    text, source = "", None
    try:
        raw = _cached(f"https://arxiv.org/html/{vid}")
        if raw:
            html = raw.decode("utf-8", "replace")
            source = "html"
        else:
            pdf = _cached(f"https://arxiv.org/pdf/{vid}")
            if pdf:
                text, source = _pdf_text(pdf), "pdf"
    except requests.RequestException as exc:
        raise FetchError(f"full text of {vid}: {exc}") from exc

    if html is not None:
        text = html_body_text(html)
    return Paper(
        id=arxiv_id,
        version=version,
        title=meta["title"],
        authors=meta["authors"],
        url=f"https://arxiv.org/abs/{vid}",
        published=meta["published"],
        journal_ref=meta["journal_ref"],
        comments=meta["comments"],
        html=html,
        text=text,
        text_source=source,
    )
