#!/usr/bin/env python3
"""Run the pre-registered validation of paperq (prereg/2026-10-28-paperq-validation.md).

  scripts/validate.py run     assess every reference-set paper and its variants (spends tokens)
  scripts/validate.py report  compute the pre-registered statistics from the stored records

`run` refuses unless HEAD is tagged `validation-freeze`: no reference-set paper is assessed
by an instrument that can still change. Records are stored in results/records/<set>/.

  report --set positives,negatives --json   {"auroc": ...}
  report --set rewrites --json              {"median_abs_grade_delta": ...}
  report --set injected --json              {"detection_rate": ...}
  report --set boundary --json              {"uncertain_rate": ...}
  report --set refs-fabricated --json       {"recall": ...}
  report --json                             everything, with the decision
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import make_variants  # noqa: E402

from paperq import __version__, main  # noqa: E402
from paperq._fetch import FetchError, Paper, fetch  # noqa: E402
from paperq._llm import LLMError, ask  # noqa: E402
from paperq._rubric import JUDGES  # noqa: E402
from paperq._schema import new_record  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
REFSETS = ROOT / "refsets"
RECORDS = ROOT / "results" / "records"
SEED = 20261028
FREEZE_TAG = "validation-freeze"
UNCERTAIN_AS = 2.5  # pre-registered: an uncertain verdict counts as this grade in H1
STABILITY = 10  # per class

THRESHOLDS = {"H1": 0.75, "H2": 0.25, "H3": 0.6, "H5": 1.0, "C1_auroc": 0.70, "C1_drop": 0.05,
              "H6": 0.25, "H7": 0.6}


def load(name: str) -> list[dict[str, Any]]:
    path = REFSETS / f"{name}.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _frozen() -> bool:
    out = subprocess.run(["git", "-C", str(ROOT), "tag", "--points-at", "HEAD"], capture_output=True, text=True)
    return FREEZE_TAG in out.stdout.split()


def _record(paper: Paper, use_cache: bool = True, network: bool = True, llm: bool = True) -> dict[str, Any]:
    record = new_record(
        {"id": paper.id, "version": paper.version, "title": paper.title, "authors": paper.authors, "url": paper.url,
         "published": paper.published, "journal_ref": paper.journal_ref, "comments": paper.comments,
         "text_source": paper.text_source},
        {"name": "paperq", "version": __version__, "commit": main._commit()},
    )
    main.facts_layer(paper, record, network=network)
    if llm:
        main.llm_layers(paper, record, use_cache=use_cache)
    return record


def _store(set_name: str, key: str, record: dict[str, Any]) -> None:
    path = RECORDS / set_name / f"{key}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, ensure_ascii=False))


def _stored(set_name: str, key: str) -> dict[str, Any] | None:
    path = RECORDS / set_name / f"{key}.json"
    return json.loads(path.read_text()) if path.exists() else None


def _key(entry: dict[str, Any]) -> str:
    return f"{entry['arxiv_id']}{entry['version']}"


def _assess(set_name: str, entry: dict[str, Any], make: Any = None, use_cache: bool = True,
            store_as: str | None = None) -> dict[str, Any] | None:
    """Assess one entry (or the variant `make(paper)` builds), twice at most, and store it.

    A stored record is returned as it is, so an interrupted run resumes; `use_cache` only
    decides whether the models are asked again.
    """
    key = _key(entry)
    target = store_as or set_name
    if done := _stored(target, key):
        return done
    references_only = set_name == "refs-fabricated"
    for attempt in range(2):
        try:
            paper, variant = fetch(key), None
            if make:
                paper, variant = make(paper)
            record = _record(paper, use_cache=use_cache, network=references_only, llm=not references_only)
            if record["verdict"] != "not-assessed" or references_only:
                record["refset"], record["variant"] = entry, variant
                _store(target, key, record)
                return record
            reasons = record["errors"] + (record["rubric"] or {}).get("failures", [])
            print(f"  ! {target} {key}: not assessed: {reasons}", file=sys.stderr)
        except (FetchError, LLMError, make_variants.VariantError) as exc:
            print(f"  ! {target} {key} (attempt {attempt + 1}): {exc}", file=sys.stderr)
    _store(target, key, {"excluded": True, "refset": entry})
    return None


PROBE = """Without looking anything up: do you know the scientific paper titled
"{title}"? If you do, say where it was submitted or published and what happened to it.

Reply with: {{"known": true | false, "venue": "<venue or null>",
"outcome": "oral" | "accepted" | "rejected" | "withdrawn" | "retracted" | "unknown"}}"""


def _truth(entry: dict[str, Any]) -> set[str]:
    if entry["source"] == "retraction-watch":
        return {"retracted"}
    venue = entry["evidence"]["venue"] or ""
    if "Oral" in venue:
        return {"oral", "accepted"}
    if "Withdrawn" in venue:
        return {"withdrawn", "rejected"}
    return {"rejected"} if entry["evidence"]["venueid"].endswith("Rejected_Submission") else {"accepted"}


def probe(entry: dict[str, Any]) -> dict[str, Any]:
    """C1: does either judge know this paper's outcome without seeing it?"""
    answers = {}
    for model in JUDGES:
        try:
            answers[model] = ask(model, PROBE.format(title=entry["title"])).data
        except LLMError as exc:
            answers[model] = {"error": str(exc)}
    recognized = any(a.get("known") and a.get("outcome") in _truth(entry) for a in answers.values())
    return {"answers": answers, "recognized": recognized}


def run(_args: argparse.Namespace) -> int:
    if not _frozen():
        print(f"validate: HEAD is not tagged {FREEZE_TAG}; refusing to assess reference-set papers", file=sys.stderr)
        return 2
    sets = {name: load(name) for name in ("positives", "negatives", "boundary", "injected", "rewrites")}
    for name in ("positives", "negatives", "boundary"):
        for n, entry in enumerate(sets[name], 1):
            _assess(name, entry)
            print(f"{name}: {n}/{len(sets[name])}", file=sys.stderr)
    for entry in sets["positives"] + sets["negatives"]:
        path = RECORDS / "probe" / f"{_key(entry)}.json"
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(probe(entry)))
    for entry in sets["injected"]:
        _assess("injected", entry, make=make_variants.injected)
    for entry in sets["rewrites"]:
        _assess("rewrites", entry, make=make_variants.rewritten)
    for entry in random.Random(SEED + 4).sample(sets["positives"], 10):
        _assess("refs-fabricated", entry, make=make_variants.fabricated_refs)
    rng = random.Random(SEED + 5)
    for entry in rng.sample(sets["positives"], STABILITY) + rng.sample(sets["negatives"], STABILITY):
        _assess("stability", entry, use_cache=False, store_as="stability")
    return 0


# --------------------------------------------------------------------------- statistics


def auroc(positive: list[float], negative: list[float]) -> float:
    """P(a random positive grades above a random negative), ties counting half."""
    wins = sum((p > n) + 0.5 * (p == n) for p in positive for n in negative)
    return wins / (len(positive) * len(negative))


def bootstrap(stat: Any, *samples: list[Any], n: int = 2000) -> list[float]:
    rng = random.Random(SEED)
    values = sorted(stat(*[[rng.choice(s) for _ in s] for s in samples]) for _ in range(n))
    return [round(values[int(0.025 * n)], 3), round(values[int(0.975 * n)], 3)]


def grade_of(record: dict[str, Any]) -> float | None:
    if record.get("excluded") or record.get("verdict") == "not-assessed":
        return None
    return UNCERTAIN_AS if record["verdict"] == "uncertain" else record["rubric"]["grade"]


def _records(name: str) -> list[dict[str, Any]]:
    return [json.loads(p.read_text()) for p in sorted((RECORDS / name).glob("*.json"))]


def _exclusion(name: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    excluded = [r["refset"]["arxiv_id"] for r in records if r.get("excluded")]
    total = len(load(name)) if (REFSETS / f"{name}.jsonl").exists() else len(records)
    return {"n": total, "excluded": excluded, "inconclusive": len(excluded) > 0.1 * max(total, 1)}


def _tokens(s: str) -> set[str]:
    return set(re.findall(r"\w+", s.lower()))


def jaccard(a: str, b: str) -> float:
    x, y = _tokens(a), _tokens(b)
    return len(x & y) / len(x | y) if x | y else 0.0


def report(args: argparse.Namespace) -> int:
    wanted = set(args.set.split(",")) if args.set else None
    out: dict[str, Any] = {}

    def want(*names: str) -> bool:
        return wanted is None or set(names) <= wanted

    pos, neg = _records("positives"), _records("negatives")
    if want("positives", "negatives") and pos and neg:
        gp = [g for r in pos if (g := grade_of(r)) is not None]
        gn = [g for r in neg if (g := grade_of(r)) is not None]
        out["auroc"] = round(auroc(gp, gn), 3)
        out["auroc_ci"] = bootstrap(auroc, gp, gn)
        for source in ("openreview", "retraction-watch"):
            sub = [g for r in neg if r["refset"]["source"] == source and (g := grade_of(r)) is not None]
            if sub:
                out[f"auroc_{source}"] = round(auroc(gp, sub), 3)
        probes = {p.stem: json.loads(p.read_text()) for p in (RECORDS / "probe").glob("*.json")}
        if probes:
            unknown = {k for k, v in probes.items() if not v["recognized"]}
            up = [g for r in pos if _key(r["refset"]) in unknown and (g := grade_of(r)) is not None]
            un = [g for r in neg if _key(r["refset"]) in unknown and (g := grade_of(r)) is not None]
            out["recognized"] = len(probes) - len(unknown)
            out["auroc_unrecognized"] = round(auroc(up, un), 3) if up and un else None
        out["exclusions"] = {"positives": _exclusion("positives", pos), "negatives": _exclusion("negatives", neg)}

    if want("rewrites"):
        deltas = []
        for r in _records("rewrites"):
            original = _stored("positives", _key(r["refset"])) or _stored("negatives", _key(r["refset"]))
            if original and (a := grade_of(original)) is not None and (b := grade_of(r)) is not None:
                deltas.append(abs(a - b))
        if deltas:
            out["median_abs_grade_delta"] = round(statistics.median(deltas), 3)
            out["rewrite_pairs"] = len(deltas)

    if want("injected"):
        detected = planted = drops = pairs = 0
        for r in _records("injected"):
            if r.get("excluded") or not r.get("variant"):
                continue
            quotes = [f["quote"] for run in (r["rubric"] or {}).get("runs", []) for f in run["findings"]]
            quotes += [c["evidence"] or "" for c in r["claims"]]
            for error in r.get("variant", {}).get("errors", []):
                planted += 1
                detected += any(jaccard(q, error["replacement"]) >= 0.5 for q in quotes)
            original = _stored("positives", _key(r["refset"]))
            if original and (a := grade_of(original)) is not None and (b := grade_of(r)) is not None:
                pairs += 1
                drops += b < a
        if planted:
            out["detection_rate"] = round(detected / planted, 3)
            out["planted_errors"] = planted
        if pairs:
            out["grade_drop_rate"] = round(drops / pairs, 3)

    if want("boundary"):
        def rate(records: list[dict[str, Any]]) -> float | None:
            done = [r for r in records if grade_of(r) is not None]
            return round(sum(r["verdict"] == "uncertain" for r in done) / len(done), 3) if done else None

        out["uncertain_rate"] = rate(_records("boundary"))
        out["uncertain_rate_positives"], out["uncertain_rate_negatives"] = rate(pos), rate(neg)

    if want("refs-fabricated"):
        flagged = planted = 0
        for r in _records("refs-fabricated"):
            fakes = {f["title"] for f in r.get("variant", {}).get("references", [])}
            planted += len(fakes)
            found = [f["ref"].lower() for f in r["facts"].get("references", {}).get("findings", [])]
            flagged += sum(any(t[:60].lower() in ref for ref in found) for t in fakes)
        if planted:
            out["recall"] = round(flagged / planted, 3)
            out["planted_references"] = planted

    if want("stability"):
        deltas = []
        for r in _records("stability"):
            original = _stored("positives", _key(r["refset"])) or _stored("negatives", _key(r["refset"]))
            if original and (a := grade_of(original)) is not None and (b := grade_of(r)) is not None:
                deltas.append(abs(a - b))
        if deltas:
            out["stability_median_abs_delta"] = round(statistics.median(deltas), 3)

    if wanted is None:
        out["decision"] = decide(out)
    if args.json:
        print(json.dumps(out))
    else:
        for key, value in out.items():
            print(f"{key}: {value}")
    return 0


def decide(stats: dict[str, Any]) -> dict[str, Any]:
    """The pre-registered decision rule: H1, H2, H3, H5 and C1 must all pass."""
    checks = {
        "H1": stats.get("auroc") is not None and stats["auroc"] >= THRESHOLDS["H1"],
        "H2": stats.get("median_abs_grade_delta") is not None
        and stats["median_abs_grade_delta"] <= THRESHOLDS["H2"],
        "H3": stats.get("detection_rate") is not None and stats["detection_rate"] >= THRESHOLDS["H3"],
        "H5": stats.get("recall") is not None and stats["recall"] >= THRESHOLDS["H5"],
        "C1": stats.get("auroc_unrecognized") is not None
        and stats["auroc_unrecognized"] >= THRESHOLDS["C1_auroc"]
        and stats["auroc"] - stats["auroc_unrecognized"] <= THRESHOLDS["C1_drop"],
    }
    inconclusive = any(e["inconclusive"] for e in stats.get("exclusions", {}).values())
    return {"checks": checks, "inconclusive": inconclusive, "validated": all(checks.values()) and not inconclusive}


def main_(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("run").set_defaults(func=run)
    p_report = sub.add_parser("report")
    p_report.add_argument("--set", help="comma-separated sets; default all, with the decision")
    p_report.add_argument("--json", action="store_true")
    p_report.set_defaults(func=report)
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main_())
