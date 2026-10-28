"""The validation instruments: statistics on known cases, and the variants they are fed."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import make_variants as mv  # noqa: E402
import validate as v  # noqa: E402

from paperq import _facts  # noqa: E402
from paperq._fetch import Paper  # noqa: E402
from paperq._llm import Answer  # noqa: E402

HTML = (Path(__file__).parent / "fixtures" / "paper.html").read_text()


def _paper(html: str | None = HTML, text: str | None = None) -> Paper:
    return Paper(id="2501.00001", version="v2", title="T", authors=[], url="u", published=None, journal_ref=None,
                 comments=None, html=html, text=text if text is not None else _facts.html_body_text(HTML),
                 text_source="html")


def test_auroc_known_cases() -> None:
    assert v.auroc([3, 4], [1, 2]) == 1.0  # perfect separation
    assert v.auroc([1, 2], [3, 4]) == 0.0  # reversed
    assert v.auroc([2, 2], [2, 2]) == 0.5  # all ties
    assert v.auroc([3, 1], [2]) == 0.5


def test_bootstrap_brackets_the_estimate() -> None:
    low, high = v.bootstrap(v.auroc, [3, 4, 3.5, 2], [1, 2, 2.5, 3])
    assert low <= v.auroc([3, 4, 3.5, 2], [1, 2, 2.5, 3]) <= high


def test_jaccard() -> None:
    assert v.jaccard("the gain is 4.1 points", "The gain is 4.1 points") == 1.0
    assert v.jaccard("completely different", "nothing shared") == 0.0


def test_grade_of_uncertain_and_excluded() -> None:
    assert v.grade_of({"verdict": "uncertain", "rubric": {"grade": 3.9}}) == v.UNCERTAIN_AS
    assert v.grade_of({"verdict": "graded", "rubric": {"grade": 3.1}}) == 3.1
    assert v.grade_of({"excluded": True}) is None


def test_decide() -> None:
    passing = {"auroc": 0.8, "median_abs_grade_delta": 0.2, "detection_rate": 0.7, "recall": 1.0,
               "auroc_unrecognized": 0.78, "exclusions": {"positives": {"inconclusive": False}}}
    assert v.decide(passing)["validated"] is True
    assert v.decide(dict(passing, auroc_unrecognized=0.72))["checks"]["C1"] is False  # dropped 0.08
    assert v.decide(dict(passing, detection_rate=0.5))["validated"] is False
    assert v.decide(dict(passing, exclusions={"x": {"inconclusive": True}}))["validated"] is False
    assert v.decide({})["validated"] is False


def test_run_refuses_without_freeze_tag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(v, "_frozen", lambda: False)
    assert v.run(argparse.Namespace()) == 2


def test_fabricated_references_are_parsed_as_references() -> None:
    paper, variant = mv.fabricated_refs(_paper())
    titles = [r.get("title") for r in _facts.references(paper.html)]
    assert all(f["title"] in titles for f in variant["references"])
    assert len(titles) == 3 + 3
    assert mv.fake_references("a") == mv.fake_references("a") != mv.fake_references("b")


def test_fabricated_references_need_html() -> None:
    with pytest.raises(mv.VariantError):
        mv.fabricated_refs(_paper(html=None))


def _generator(monkeypatch: pytest.MonkeyPatch, reply: Any) -> None:
    monkeypatch.setattr(mv, "ask", lambda model, prompt, sample=0: Answer(reply(prompt, sample), 1, 1, cached=False))


def test_injected_places_errors_verbatim(monkeypatch: pytest.MonkeyPatch) -> None:
    text = "Intro text. Table 2 shows a 4.1 point gain over the baseline. We ran 5 seeds in total here. End of it."
    errors = [{"original": "Table 2 shows a 4.1 point gain over the baseline.",
               "replacement": "Table 2 shows a 9.9 gain."},
              {"original": "We ran 5   seeds in total here.", "replacement": "We ran 50 seeds in total."},
              {"original": "End of it.", "replacement": "Too short to place."}]
    _generator(monkeypatch, lambda p, s: {"errors": errors})
    with pytest.raises(mv.VariantError, match="only 2 of 3"):
        mv.injected(_paper(text=text))
    monkeypatch.setattr(mv, "ERRORS", 2)
    paper, variant = mv.injected(_paper(text=text))
    assert "9.9 gain" in paper.text and "50 seeds" in paper.text and paper.html is None
    assert len(variant["errors"]) == 2


def test_rewrite_keeps_numbers_or_keeps_the_chunk(monkeypatch: pytest.MonkeyPatch) -> None:
    text = "We measure 3 systems on 12 tasks."
    _generator(monkeypatch, lambda p, s: {"text": "On 12 tasks, 3 systems are measured."})
    paper, variant = mv.rewritten(_paper(text=text))
    assert variant["kept_original"] == 0 and "are measured" in paper.text
    _generator(monkeypatch, lambda p, s: {"text": "On 13 tasks, 3 systems are measured."})  # a number changed
    with pytest.raises(mv.VariantError):
        mv.rewritten(_paper(text=text))


def test_chunks_cover_the_text() -> None:
    text = "A sentence here. " * 3000
    chunks = mv._chunks(text)
    assert "".join(chunks) == text and all(len(c) <= mv.CHUNK for c in chunks) and len(chunks) > 1
