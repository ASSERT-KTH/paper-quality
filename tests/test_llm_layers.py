"""The LLM layers, with a scripted model in place of agent.py."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from paperq import _claims, _llm, _rubric
from paperq._llm import Answer, LLMError, parse_json

TEXT = "We evaluate on 3 datasets. Table 2 shows a 4.1 point gain over the baseline. We did not run seeds."


def test_parse_json_fenced_and_bare() -> None:
    assert parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json('Sure: {"a": {"b": 2}} done') == {"a": {"b": 2}}
    with pytest.raises(LLMError):
        parse_json("no json here")
    with pytest.raises(LLMError):
        parse_json("[1, 2]")


def test_ask_caches_by_model_prompt_and_sample(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    calls: list[str] = []

    def run(model: str, prompt: str) -> dict[str, Any]:
        calls.append(model)
        return {"final_reply": '{"ok": true}', "usage": {"prompt": 100, "completion": 5}}

    monkeypatch.setattr(_llm, "_run", run)
    first = _llm.ask("m", "p")
    again = _llm.ask("m", "p")
    other = _llm.ask("m", "p", sample=1)
    assert (first.cached, again.cached, other.cached) == (False, True, False)
    assert again.data == {"ok": True} and again.input_tokens == 100
    assert len(calls) == 2
    _llm.ask("m", "p", use_cache=False)
    assert len(calls) == 3


def test_ask_retries_once_then_fails(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    replies = iter(["not json", "still not json"])
    monkeypatch.setattr(_llm, "_run", lambda m, p: {"final_reply": next(replies), "usage": {}})
    with pytest.raises(LLMError):
        _llm.ask("m", "p")


def test_run_reports_agent_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_llm.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=75, stderr="no quota",
                                                                                 stdout=""))
    with pytest.raises(LLMError, match="no quota"):
        _llm._run("m", "p")


def _scripted(monkeypatch: pytest.MonkeyPatch, module: Any, reply: Any) -> list[tuple[str, str, int]]:
    asked: list[tuple[str, str, int]] = []

    def ask(model: str, prompt: str, sample: int = 0, use_cache: bool = True) -> Answer:
        asked.append((model, prompt, sample))
        data = reply(model, prompt, sample)
        if isinstance(data, Exception):
            raise data
        return Answer(data, 1000, 50, cached=False)

    monkeypatch.setattr(module, "ask", ask)
    return asked


def test_classify_and_check(monkeypatch: pytest.MonkeyPatch) -> None:
    def reply(model: str, prompt: str, sample: int) -> dict[str, Any]:
        if "classifying" in prompt:
            return {"evidence_type": "empirical", "rationale": "measures",
                    "claims": [{"claim": "4.1 point gain", "location": "4"}]}
        return {"claims": [
            {"claim": "4.1 point gain", "status": "supported", "quote": "Table 2 shows a 4.1 point gain"},
            {"claim": "invented", "status": "partially", "quote": "a sentence that is not in the paper"},
            {"claim": "bad status", "status": "maybe"},
        ]}

    usage: dict[str, dict[str, int]] = {}
    _scripted(monkeypatch, _claims, reply)
    classification, claims = _claims.classify_and_check("T", TEXT, "cheap", usage)
    assert classification["evidence_type"] == "empirical"
    assert [c["status"] for c in claims] == ["supported", "partially"]  # the invalid status is dropped
    assert [c["quote_verbatim"] for c in claims] == [True, False]
    spent = {"input_tokens": 1000, "cached_input_tokens": 0, "output_tokens": 50}
    assert usage == {"classification": spent, "claims": spent}


def test_classify_rejects_unknown_type(monkeypatch: pytest.MonkeyPatch) -> None:
    _scripted(monkeypatch, _claims, lambda m, p, s: {"evidence_type": "opinion", "claims": []})
    with pytest.raises(LLMError):
        _claims.classify_and_check("T", TEXT, "cheap", {})


def _items(score: int, kind: str = "empirical") -> dict[str, int]:
    return dict.fromkeys(_rubric.CHECKLISTS[kind], score)


def test_grade_median_and_verbatim_findings(monkeypatch: pytest.MonkeyPatch) -> None:
    def reply(model: str, prompt: str, sample: int) -> dict[str, Any]:
        score = 3 if model == "a" else 2 + (sample % 2)  # a: 3,3,3  b: 2,3,2
        return {"items": _items(score), "findings": [
            {"item": "statistics", "severity": "major", "quote": "We did not run seeds.", "explanation": "no seeds"},
            {"item": "validity", "severity": "fatal", "quote": "made up quote", "explanation": "hallucinated"},
            {"item": "scope", "severity": "minor", "explanation": "no quote: not accepted"},
        ]}

    _scripted(monkeypatch, _rubric, reply)
    usage: dict[str, dict[str, int]] = {}
    rubric, verdict = _rubric.grade("T", TEXT, "empirical", usage, judges=("a", "b"), samples=3, calibration={})
    assert verdict == "graded" and rubric["grade"] == 3.0 and rubric["iqr"] <= 1.0
    assert len(rubric["runs"]) == 6 and set(usage) == {"rubric:a", "rubric:b"}
    shown = _rubric.findings(rubric)
    assert {f["quote"] for f in shown} == {"We did not run seeds."}  # unverifiable quotes are not shown


def test_grade_uncertain_when_judges_disagree(monkeypatch: pytest.MonkeyPatch) -> None:
    _scripted(monkeypatch, _rubric, lambda m, p, s: {"items": _items(4 if m == "a" else 1), "findings": []})
    rubric, verdict = _rubric.grade("T", TEXT, "empirical", {}, judges=("a", "b"), samples=2, calibration={})
    assert verdict == "uncertain" and rubric["iqr"] > _rubric.UNCERTAIN_IQR


def test_grade_needs_half_the_samples(monkeypatch: pytest.MonkeyPatch) -> None:
    _scripted(monkeypatch, _rubric, lambda m, p, s: LLMError("quota") if m == "b" or s else {"items": _items(3)})
    rubric, verdict = _rubric.grade("T", TEXT, "empirical", {}, judges=("a", "b"), samples=2, calibration={})
    assert verdict == "not-assessed" and rubric["grade"] is None and len(rubric["failures"]) == 3


def test_grade_rejects_incomplete_scores(monkeypatch: pytest.MonkeyPatch) -> None:
    _scripted(monkeypatch, _rubric, lambda m, p, s: {"items": {"scope": 3}})
    rubric, verdict = _rubric.grade("T", TEXT, "position", {}, judges=("a",), samples=2, calibration={})
    assert verdict == "not-assessed" and len(rubric["failures"]) == 2


def test_position_papers_are_not_asked_for_baselines() -> None:
    assert "comparisons" not in _rubric.CHECKLISTS["position"]
    assert "argument_validity" in _rubric.CHECKLISTS["position"]
    assert set(_rubric.CHECKLISTS) == set(_claims.EVIDENCE_DEFINITIONS)


def test_prompt_forbids_style_and_memory() -> None:
    assert "Ignore writing quality, tone" in _rubric.RUBRIC_PROMPT
    assert "anything you may remember about this paper" in _rubric.RUBRIC_PROMPT
    json.dumps(_rubric.CHECKLISTS)  # serializable into the record


CALIBRATION = {"judges": {"harsh": {"mean": 2.5, "sd": 0.5}, "lenient": {"mean": 3.5, "sd": 0.25}},
               "pooled": {"mean": 3.0, "sd": 0.5}}


def test_calibrate_removes_a_judges_offset() -> None:
    assert _rubric.calibrate(2.5, "harsh", CALIBRATION) == 3.0  # each judge's own mean maps to the pooled mean
    assert _rubric.calibrate(3.5, "lenient", CALIBRATION) == 3.0
    assert _rubric.calibrate(3.0, "harsh", CALIBRATION) == 3.5  # one of its sd above
    assert _rubric.calibrate(4.0, "lenient", CALIBRATION) == 4.0  # clipped to the scale
    assert _rubric.calibrate(2.0, "unknown", CALIBRATION) == 2.0
    assert _rubric.calibrate(2.0, "harsh", None) == 2.0


def test_calibrated_judges_agree(monkeypatch: pytest.MonkeyPatch) -> None:
    _scripted(monkeypatch, _rubric,
              lambda m, p, s: {"items": _items(2 if m == "harsh" else 4), "findings": []})
    raw, raw_verdict = _rubric.grade("T", TEXT, "empirical", {}, judges=("harsh", "lenient"), samples=2,
                                     calibration={})
    fixed, fixed_verdict = _rubric.grade("T", TEXT, "empirical", {}, judges=("harsh", "lenient"), samples=2,
                                         calibration={"judges": {"harsh": {"mean": 2, "sd": 0.5},
                                                                 "lenient": {"mean": 4, "sd": 0.5}},
                                                      "pooled": {"mean": 3, "sd": 0.5}})
    assert raw_verdict == "uncertain" and fixed_verdict == "graded" and fixed["grade"] == 3.0
    assert [r["score"] for r in fixed["runs"]] == [2, 2, 4, 4]  # raw scores stay in the record
