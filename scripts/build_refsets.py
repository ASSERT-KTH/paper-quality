#!/usr/bin/env python3
"""Build the reference sets that paperq is validated against.

Every set is a JSONL file in refsets/, one paper per line, pinned to an arXiv version,
with the evidence that put it in the set. Selection is deterministic: the rules are the
constants below, and sampling uses SEED.

  positives  ICLR 2026 orals
  negatives  ICLR 2026 rejected or withdrawn papers that every reviewer rejected,
             plus CS papers retracted for unreliable content (Retraction Watch)
  boundary   ICLR 2026 papers with a borderline mean and reviewers who disagree
  injected   positives chosen to receive planted errors (the errors are made later)
  rewrites   positives and negatives chosen for meaning-preserving rewrites (made later)

ICLR papers are pinned to their last arXiv version dated before the decisions, so a grade
judges what the reviewers judged. Retracted papers are pinned to v1: their last version is
often the withdrawal notice.

OpenReview needs a login: credentials come from the keyring (`openreview` service,
`username` and `password` entries). Downloads are cached in ~/.cache/paperq/refsets.

usage: scripts/build_refsets.py [--target N]
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import random
import re
import statistics
import sys
import unicodedata
import urllib.parse
from pathlib import Path
from typing import Any

import feedparser
import requests

from paperq import _arxiv

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "refsets"
CACHE = Path.home() / ".cache" / "paperq" / "refsets"
SEED = 20261028
USER_AGENT = "paperq-refsets (https://github.com/ASSERT-KTH/paper-quality)"

ICLR = "ICLR.cc/2026/Conference"
ICLR_DECISIONS = dt.date(2026, 1, 26)  # pin each paper to the arXiv version the reviewers could see
# ICLR 2026 ratings: 0 strong reject, 2 reject, 4 marginally below, 6 marginally above, 8 accept, 10 strong accept
NEGATIVE_MAX_RATING = 4  # nobody above "marginally below"
NEGATIVE_MAX_MEAN = 2.5
BOUNDARY_MEAN = (4.0, 6.0)
BOUNDARY_MIN_SPREAD = 4  # e.g. a 2 next to a 6
MIN_REVIEWS = 3

RW_CSV = "https://gitlab.com/crossref/retraction-watch-data/-/raw/main/retraction_watch.csv"
# Retraction reasons that say the content is unreliable, not only that the process was abused
RW_CONTENT_REASONS = (
    "Error in", "Unreliable Results", "Unreliable Data", "Unreliable Image", "Concerns/Issues about Data",
    "Concerns/Issues about Results", "Falsification", "Fabrication", "Computer-Aided Content", "Paper Mill",
)
RW_MIN_YEAR = 2015
RW_SHARE = 0.5  # at most this share of the negatives comes from retractions
RW_MAX_CANDIDATES = 600  # few retracted papers are on arXiv, and each lookup costs 3 seconds

INJECTED = 30
REWRITES = 30


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _norm(title: str) -> str:
    title = unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()


def _cached_json(name: str, produce: Any) -> Any:
    path = CACHE / name
    if path.exists():
        return json.loads(path.read_text())
    value = produce()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return value


# --------------------------------------------------------------------------- arXiv

def _arxiv_get(url: str) -> str:
    """Through paperq's paced client, which every paperq process shares."""
    ok = _arxiv.api_ok if "export.arxiv.org" in url else _arxiv.page_ok
    return _arxiv.get(url, ok).text


STOPWORDS = {"and", "not", "the", "for", "with", "from", "into", "via", "andnot", "are", "can", "its"}


def arxiv_match(title: str, first_author: str | None) -> str | None:
    """The arXiv id of the paper with exactly this title (and this first author's last name)."""
    key = _norm(title)

    def search() -> list[dict[str, Any]]:
        # "and", "or", "not" are query operators to arXiv, and stopwords only dilute the ranking
        words = " AND ".join(f"ti:{w}" for w in [w for w in key.split() if len(w) > 2 and w not in STOPWORDS][:10])
        query = urllib.parse.urlencode({"search_query": words, "max_results": "10"})
        feed = feedparser.parse(_arxiv_get(f"http://export.arxiv.org/api/query?{query}"))
        return [{"id": e.id, "title": e.title, "authors": [a.name for a in e.get("authors", [])]} for e in feed.entries]

    for entry in _cached_json(f"arxiv-search/{hashlib.sha256(key.encode()).hexdigest()}.json", search):
        if _norm(entry["title"]) != key:
            continue
        if first_author and entry["authors"]:
            last = _norm(first_author).split()[-1:]
            if last and last[0] not in _norm(entry["authors"][0]).split():
                continue
        found = re.search(r"(\d{4}\.\d{4,5})", entry["id"])
        if found:
            return found.group(1)
    return None


def arxiv_versions(arxiv_id: str) -> list[tuple[str, dt.date]]:
    """[(v1, date), (v2, date), ...] from the abstract page's submission history."""

    def history() -> list[list[str]]:
        page = _arxiv_get(f"https://arxiv.org/abs/{arxiv_id}")
        block = page.split('class="submission-history"', 1)[-1].split("</div>", 1)[0]
        found = re.findall(r"\[(v\d+)\](?:</a>)?(?:</strong>)?\s*\w{3}, (\d{1,2} \w{3} \d{4})", block)
        return [[v, dt.datetime.strptime(d, "%d %b %Y").date().isoformat()] for v, d in found]

    return [(v, dt.date.fromisoformat(d)) for v, d in _cached_json(f"arxiv-versions/{arxiv_id}.json", history)]


def pin_before(arxiv_id: str, cutoff: dt.date) -> tuple[str, str] | None:
    """The last version dated on or before the cutoff, else v1: (version, date)."""
    versions = arxiv_versions(arxiv_id)
    if not versions:
        return None
    before = [v for v in versions if v[1] <= cutoff]
    version, date = before[-1] if before else versions[0]
    return version, date.isoformat()


# --------------------------------------------------------------------------- OpenReview


def _client() -> Any:
    import keyring
    import openreview

    user = keyring.get_password("openreview", "username")
    password = keyring.get_password("openreview", "password") or keyring.get_password("openreview", user or "")
    return openreview.api.OpenReviewClient(baseurl="https://api2.openreview.net", username=user, password=password)


def _value(content: dict[str, Any], key: str) -> Any:
    field = content.get(key)
    return field.get("value") if isinstance(field, dict) else field


def iclr_submissions(client: Any) -> list[dict[str, Any]]:
    """Every ICLR 2026 submission that reached a decision or was withdrawn, without reviews."""

    def pull() -> list[dict[str, Any]]:
        out = []
        for venueid in (ICLR, f"{ICLR}/Rejected_Submission", f"{ICLR}/Withdrawn_Submission"):
            notes = client.get_all_notes(content={"venueid": venueid})
            log(f"  {venueid}: {len(notes)}")
            for note in notes:
                out.append({
                    "forum": note.id,
                    "title": _value(note.content, "title"),
                    "authors": _value(note.content, "authors") or [],
                    "venue": _value(note.content, "venue"),
                    "venueid": venueid,
                })
        return out

    return _cached_json("iclr2026-submissions.json", pull)


def iclr_ratings(client: Any, forum: str) -> list[int]:
    def pull() -> list[int]:
        ratings = []
        for note in client.get_notes(forum=forum):
            if any(inv.endswith("Official_Review") for inv in note.invitations):
                rating = _value(note.content, "rating")
                if rating is not None and (found := re.match(r"\s*(\d+)", str(rating))):
                    ratings.append(int(found.group(1)))
        return ratings

    return _cached_json(f"iclr2026-ratings/{forum}.json", pull)


def _iclr_record(sub: dict[str, Any], arxiv_id: str, ratings: list[int], set_name: str) -> dict[str, Any] | None:
    pinned = pin_before(arxiv_id, ICLR_DECISIONS)
    if not pinned:
        return None
    return {
        "set": set_name,
        "arxiv_id": arxiv_id,
        "version": pinned[0],
        "version_date": pinned[1],
        "title": sub["title"],
        "source": "openreview",
        "evidence": {
            "venue": sub["venue"],
            "venueid": sub["venueid"],
            "forum": f"https://openreview.net/forum?id={sub['forum']}",
            "ratings": ratings,
        },
    }


def select_iclr(client: Any, pool: list[dict[str, Any]], want: int, keep: Any, set_name: str) -> list[dict[str, Any]]:
    """Walk a shuffled pool until `want` papers pass `keep(ratings)` and are on arXiv."""
    chosen: list[dict[str, Any]] = []
    for n, sub in enumerate(pool):
        if len(chosen) >= want:
            break
        ratings = iclr_ratings(client, sub["forum"])
        if len(ratings) < MIN_REVIEWS or not keep(ratings):
            continue
        arxiv_id = arxiv_match(sub["title"], (sub["authors"] or [None])[0])
        if arxiv_id and (record := _iclr_record(sub, arxiv_id, ratings, set_name)):
            chosen.append(record)
            log(f"  {set_name}: {len(chosen)}/{want} after {n + 1} candidates")
    return chosen


def is_negative(ratings: list[int]) -> bool:
    return max(ratings) <= NEGATIVE_MAX_RATING and statistics.mean(ratings) <= NEGATIVE_MAX_MEAN


def is_boundary(ratings: list[int]) -> bool:
    low, high = BOUNDARY_MEAN
    return low <= statistics.mean(ratings) <= high and max(ratings) - min(ratings) >= BOUNDARY_MIN_SPREAD


# --------------------------------------------------------------------------- Retraction Watch


def retractions(want: int, rng: random.Random) -> list[dict[str, Any]]:
    path = CACHE / "retraction_watch.csv"
    if not path.exists():
        log("  downloading Retraction Watch")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(requests.get(RW_CSV, headers={"User-Agent": USER_AGENT}, timeout=300).content)
    rows = list(csv.DictReader(path.open(encoding="utf-8", errors="replace")))
    candidates = [
        r for r in rows
        if "Computer Science" in r["Subject"]
        and r["RetractionNature"] == "Retraction"
        and any(reason in r["Reason"] for reason in RW_CONTENT_REASONS)
        and (m := re.search(r"(\d{4})", r["OriginalPaperDate"])) and int(m.group(1)) >= RW_MIN_YEAR
    ]
    rng.shuffle(candidates)
    log(f"  retraction candidates: {len(candidates)}")
    chosen: list[dict[str, Any]] = []
    for n, row in enumerate(candidates[:RW_MAX_CANDIDATES]):
        if len(chosen) >= want:
            break
        first_author = row["Author"].split(";")[0].strip() or None
        arxiv_id = arxiv_match(row["Title"], first_author)
        if not arxiv_id or not (versions := arxiv_versions(arxiv_id)):
            continue
        chosen.append({
            "set": "negatives",
            "arxiv_id": arxiv_id,
            "version": versions[0][0],
            "version_date": versions[0][1].isoformat(),
            "title": row["Title"],
            "source": "retraction-watch",
            "evidence": {
                "record_id": row["Record ID"],
                "journal": row["Journal"],
                "doi": row["OriginalPaperDOI"],
                "retraction_doi": row["RetractionDOI"],
                "retraction_date": row["RetractionDate"],
                "reasons": [r for r in row["Reason"].split(";") if r.strip()],
            },
        })
        log(f"  retractions: {len(chosen)}/{want} after {n + 1} candidates")
    return chosen


# --------------------------------------------------------------------------- main


def write(name: str, records: list[dict[str, Any]]) -> None:
    OUT.mkdir(exist_ok=True)
    with (OUT / f"{name}.jsonl").open("w") as out:
        for record in records:
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
    log(f"wrote refsets/{name}.jsonl: {len(records)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target", type=int, default=60, help="papers per set (validation needs ≥50)")
    args = parser.parse_args(argv)

    client = _client()
    subs = iclr_submissions(client)
    orals = [s for s in subs if s["venue"] == "ICLR 2026 Oral"]
    declined = [s for s in subs if s["venueid"] != ICLR]
    log(f"ICLR 2026: {len(subs)} submissions, {len(orals)} orals, {len(declined)} rejected or withdrawn")

    rng = random.Random(SEED)
    rng.shuffle(orals)
    positives = select_iclr(client, orals, args.target, lambda _r: True, "positives")

    from_rw = retractions(int(args.target * RW_SHARE), random.Random(SEED))
    pool = declined[:]
    random.Random(SEED + 1).shuffle(pool)
    negatives = from_rw + select_iclr(client, pool, args.target - len(from_rw), is_negative, "negatives")

    pool = subs[:]
    random.Random(SEED + 2).shuffle(pool)
    taken = {r["arxiv_id"] for r in positives + negatives}
    boundary = [r for r in select_iclr(client, pool, args.target + 10, is_boundary, "boundary")
                if r["arxiv_id"] not in taken][: args.target]

    pick = random.Random(SEED + 3)
    injected = [dict(r, set="injected") for r in pick.sample(positives, min(INJECTED, len(positives)))]
    half = REWRITES // 2
    rewrites = [dict(r, set="rewrites") for r in pick.sample(positives, min(half, len(positives)))
                + pick.sample(negatives, min(REWRITES - half, len(negatives)))]

    for name, records in (("positives", positives), ("negatives", negatives), ("boundary", boundary),
                          ("injected", injected), ("rewrites", rewrites)):
        write(name, records)
    return 0


if __name__ == "__main__":
    sys.exit(main())
