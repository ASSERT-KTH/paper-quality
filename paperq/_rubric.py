"""Rubric layer: anchored checklist scores from several samples of several judge models.

The checklist depends on the evidence type: a position paper is not marked down for having
no baselines. Each item is scored 1-4 against written anchors that describe substance, never
style. One sample's score is the mean of its items; the grade is the median over every
sample of every judge, and the spread decides whether a grade is given at all.
"""

from __future__ import annotations

import concurrent.futures
import json
import statistics
from pathlib import Path
from typing import Any

from paperq._claims import _add_usage, is_verbatim, paper_text
from paperq._llm import LLMError, ask

JUDGES = ("claude-sonnet-5-5", "gpt-5.6-sol")
SAMPLES = 5
UNCERTAIN_IQR = 1.0  # pre-registered: above this spread the verdict is "uncertain"
# Each judge's mean and standard deviation of sample scores on the development set
# (scripts/calibrate.py). Judges differ by a constant: on 2610.19339 one scored ~2.7, the
# other ~3.7. Uncalibrated, the spread would measure which judge spoke, not the paper.
CALIBRATION_FILE = Path(__file__).with_name("calibration.json")


def load_calibration(path: Path = CALIBRATION_FILE) -> dict[str, Any] | None:
    return json.loads(path.read_text()) if path.exists() else None


def calibrate(score: float, model: str, calibration: dict[str, Any] | None) -> float:
    """A judge's score mapped onto the pooled scale: z-scored with the judge's own statistics."""
    if not calibration or model not in calibration["judges"]:
        return score
    judge, pooled = calibration["judges"][model], calibration["pooled"]
    z = (score - judge["mean"]) / judge["sd"] if judge["sd"] else 0.0
    return min(4.0, max(1.0, pooled["mean"] + z * pooled["sd"]))

ANCHORS = "4 = sound; 3 = minor issues that leave the conclusions standing; " \
    "2 = a major weakness that undermines part of the conclusions; 1 = a fatal flaw: the conclusions do not follow"

COMMON = {
    "claims_supported": "Are the central claims backed by evidence or argument shown in the paper itself?",
    "internal_consistency": "Do numbers, claims and definitions agree across abstract, text, tables and figures?",
    "scope": "Do the conclusions stay within what the evidence covers, and are the limits stated?",
    "positioning": "Is novelty argued against the prior work the paper cites, rather than asserted?",
}
EMPIRICAL = {
    "comparisons": "Are there adequate baselines, ablations or controls for the claims made?",
    "statistics": "Is variability reported (runs, seeds, intervals, tests) where conclusions depend on differences?",
    "validity": "Do the measures and data actually measure the claim (no leakage, no proxy confusion)?",
    "reproducibility": "Are method, data and setup described (or released) well enough to redo the work?",
}
ARGUMENT = {
    "argument_validity": "Do the conclusions follow from the premises, and are the premises supported?",
    "counterarguments": "Are the strongest objections and alternative views engaged with?",
}
SURVEY = {
    "coverage": "Is the selection of work systematic and stated, and is the coverage adequate for the claims?",
    "synthesis": "Does it organize and compare the work, rather than list it?",
}
CHECKLISTS: dict[str, dict[str, str]] = {
    "empirical": COMMON | EMPIRICAL,
    "benchmark": COMMON | EMPIRICAL,
    "replication": COMMON | EMPIRICAL,
    "system-demo": COMMON | {k: EMPIRICAL[k] for k in ("comparisons", "validity", "reproducibility")},
    "case-study": COMMON | {k: EMPIRICAL[k] for k in ("validity", "reproducibility")},
    "position": COMMON | ARGUMENT,
    "survey": COMMON | SURVEY,
}

RUBRIC_PROMPT = """You are assessing the scientific quality of a paper: whether its conclusions
are warranted by what it shows. You are not reviewing its writing.

Judge the substance only. Ignore writing quality, tone, confidence, formatting, the authors'
or venue's reputation, and anything you may remember about this paper or how it was received.

The paper is a {evidence_type} paper. Score each item 1 to 4: {anchors}.

{items}

For every score below 4, report the findings behind it. A finding quotes the paper verbatim
(at most 300 characters): the passage that shows the problem. A finding without a quote is
not accepted.

Reply with: {{"items": {{"<item>": <score>, ...}}, "findings": [{{"item": "<item>",
"severity": "fatal" | "major" | "minor", "quote": "...", "explanation": "<one sentence>"}}]}}

PAPER
Title: {title}

{text}"""


def _iqr(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    q = statistics.quantiles(values, n=4, method="inclusive")
    return q[2] - q[0]


def _one(model: str, sample: int, prompt: str, checklist: dict[str, str], text: str,
         use_cache: bool) -> tuple[dict[str, Any], Any]:
    answer = ask(model, prompt, sample=sample, use_cache=use_cache)
    raw = answer.data.get("items") or {}
    items = {k: int(v) for k, v in raw.items() if k in checklist and isinstance(v, int | float) and 1 <= v <= 4}
    if len(items) != len(checklist):
        raise LLMError(f"{model} scored {sorted(items)} of {sorted(checklist)}")
    findings = []
    for f in answer.data.get("findings", []):
        if isinstance(f, dict) and f.get("quote"):
            findings.append({
                "item": f.get("item"), "severity": f.get("severity"), "quote": str(f["quote"]),
                "explanation": str(f.get("explanation", "")), "verbatim": is_verbatim(str(f["quote"]), text),
            })
    score = statistics.mean(items.values())
    return {"model": model, "sample": sample, "score": round(score, 3), "items": items, "findings": findings}, answer


def grade(title: str, text: str, evidence_type: str, usage: dict[str, dict[str, int]], judges: tuple[str, ...] = JUDGES,
          samples: int = SAMPLES, use_cache: bool = True, calibration: dict[str, Any] | None = None
          ) -> tuple[dict[str, Any], str]:
    """(rubric, verdict). A failed sample is recorded and left out; the grade needs half of them.

    `calibration` defaults to the shipped calibration.json; pass {} for raw scores.
    """
    if calibration is None:
        calibration = load_calibration()
    checklist = CHECKLISTS[evidence_type]
    prompt = RUBRIC_PROMPT.format(
        evidence_type=evidence_type, anchors=ANCHORS, title=title, text=paper_text(text),
        items="\n".join(f"- {name}: {question}" for name, question in checklist.items()),
    )
    runs: list[dict[str, Any]] = []
    failures: list[str] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        # sample 0 of each judge first: the provider caches the prompt, which the other samples then reuse
        waves = [[(m, 0) for m in judges], [(m, s) for m in judges for s in range(1, samples)]]
        for wave in waves:
            futures = {pool.submit(_one, m, s, prompt, checklist, text, use_cache): (m, s) for m, s in wave}
            for future in concurrent.futures.as_completed(futures):
                model, sample = futures[future]
                try:
                    run, answer = future.result()
                except LLMError as exc:
                    failures.append(f"{model} sample {sample}: {exc}")
                    continue
                _add_usage(usage, f"rubric:{model}", answer)
                run["calibrated"] = round(calibrate(run["score"], model, calibration), 3)
                runs.append(run)
    runs.sort(key=lambda r: (r["model"], r["sample"]))
    scores = [r["calibrated"] for r in runs]
    rubric: dict[str, Any] = {"checklist": checklist, "runs": runs, "failures": failures,
                              "grade": None, "iqr": None, "judges": list(judges), "samples": samples,
                              "calibration": calibration or None}
    if len(runs) * 2 < len(judges) * samples:
        return rubric, "not-assessed"
    rubric["grade"] = round(statistics.median(scores), 3)
    rubric["iqr"] = round(_iqr(scores), 3)
    return rubric, "uncertain" if rubric["iqr"] > UNCERTAIN_IQR else "graded"


def findings(rubric: dict[str, Any]) -> list[dict[str, Any]]:
    """Every verbatim finding of every run, most severe first, for `paperq explain`."""
    order = {"fatal": 0, "major": 1, "minor": 2}
    out = [dict(f, model=r["model"], sample=r["sample"]) for r in rubric.get("runs", []) for f in r["findings"]
           if f["verbatim"]]
    return sorted(out, key=lambda f: (order.get(f["severity"], 3), f["item"] or ""))

