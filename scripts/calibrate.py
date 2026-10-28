#!/usr/bin/env python3
"""Calibrate the rubric judges on the development set and write paperq/calibration.json.

Judges differ by a constant offset and scale. Each judge's mean and standard deviation of
sample scores over the development papers map its scores onto the pooled scale (see
paperq._rubric.calibrate). Only development papers are used, never a reference-set paper.

usage: scripts/calibrate.py [dev/ids.txt]
"""

from __future__ import annotations

import datetime as dt
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from paperq import _rubric
from paperq._claims import CLASSIFIER, classify_and_check
from paperq._fetch import FetchError, fetch
from paperq._llm import LLMError

ROOT = Path(__file__).resolve().parent.parent


def refset_ids() -> set[str]:
    return {json.loads(line)["arxiv_id"] for path in (ROOT / "refsets").glob("*.jsonl")
            for line in path.read_text().splitlines() if line.strip()}


def main(argv: list[str]) -> int:
    ids_file = Path(argv[0]) if argv else ROOT / "dev" / "ids.txt"
    ids = [line.split("#")[0].strip() for line in ids_file.read_text().splitlines() if line.split("#")[0].strip()]
    leaked = {i.split("v")[0] for i in ids} & refset_ids()
    if leaked:
        print(f"calibrate: development ids in the reference sets: {sorted(leaked)}", file=sys.stderr)
        return 2

    scores: dict[str, list[float]] = defaultdict(list)
    usage: dict[str, dict[str, int]] = {}
    for ref in ids:
        try:
            paper = fetch(ref)
        except FetchError as exc:
            print(f"  ! {ref}: {exc}", file=sys.stderr)
            continue
        try:
            classification, _claims = classify_and_check(paper.title, paper.text, CLASSIFIER, usage)
        except LLMError as exc:
            print(f"  ! {ref}: {exc}", file=sys.stderr)
            continue
        rubric, _verdict = _rubric.grade(paper.title, paper.text, classification["evidence_type"], usage,
                                         calibration={})
        for run in rubric["runs"]:
            scores[run["model"]].append(run["score"])
        print(f"  {ref}: {classification['evidence_type']}, "
              + ", ".join(f"{m}={statistics.mean(r['score'] for r in rubric['runs'] if r['model'] == m):.2f}"
                          for m in rubric["judges"] if any(r["model"] == m for r in rubric["runs"])),
              file=sys.stderr)

    pooled = [s for values in scores.values() for s in values]
    calibration = {
        "date": dt.date.today().isoformat(),
        "dev_set": str(ids_file.relative_to(ROOT)) if ids_file.is_relative_to(ROOT) else str(ids_file),
        "papers": len(ids),
        "judges": {m: {"mean": round(statistics.mean(v), 4), "sd": round(statistics.stdev(v), 4), "n": len(v)}
                   for m, v in sorted(scores.items())},
        "pooled": {"mean": round(statistics.mean(pooled), 4), "sd": round(statistics.stdev(pooled), 4),
                   "n": len(pooled)},
        "usage": usage,
    }
    _rubric.CALIBRATION_FILE.write_text(json.dumps(calibration, indent=2) + "\n")
    print(json.dumps({k: calibration[k] for k in ("judges", "pooled")}))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
