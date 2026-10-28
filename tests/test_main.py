"""The command line and the record it prints."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from paperq import _facts, main
from paperq._fetch import FetchError, Paper
from paperq._schema import SCHEMA, check, new_record

HTML = (Path(__file__).parent / "fixtures" / "paper.html").read_text()


def _paper() -> Paper:
    return Paper(
        id="2610.19339", version="v1", title="A Fixture Paper", authors=["A. Author"],
        url="https://arxiv.org/abs/2610.19339v1", published="2026-10-14", journal_ref=None,
        comments="12 pages, accepted at ICLR 2027", html=HTML, text=_facts.html_body_text(HTML), text_source="html",
    )


@pytest.fixture
def fixture_paper(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(main, "fetch", lambda _ref: _paper())


def test_new_record_conforms() -> None:
    assert check(new_record({"id": "x"}, {"name": "paperq", "version": "0", "commit": "c"})) == []


def test_check_flags_bad_enums() -> None:
    record = new_record({}, {})
    record.update(verdict="great", classification={"evidence_type": "vibes"}, claims=[{"claim": "c", "status": "?"}])
    assert len(check(record)) == 3


def test_schema_command(capsys: pytest.CaptureFixture[str]) -> None:
    assert main.main(["schema"]) == 0
    assert json.loads(capsys.readouterr().out)["title"] == SCHEMA["title"]


def test_llm_layers_not_built_yet(capsys: pytest.CaptureFixture[str]) -> None:
    assert main.main(["assess", "2610.19339"]) == 2
    assert "--facts-only" in capsys.readouterr().err


def test_assess_json(fixture_paper: None, capsys: pytest.CaptureFixture[str]) -> None:
    assert main.main(["assess", "2610.19339", "--facts-only", "--offline", "--json"]) == 0
    record = json.loads(capsys.readouterr().out)
    assert check(record) == []
    facts = record["facts"]
    assert facts["venue"] == "ICLR 2027" and facts["pages"] == 12
    assert facts["has_evaluation_section"] and facts["has_limitations_section"]
    assert facts["references"] == {"count": 3, "structured": 2, "checked": 0, "findings": []}
    assert [link["alive"] for link in facts["artifact_links"]] == [None, None]  # offline: unknown
    assert record["verdict"] == "not-assessed" and record["reputation"] is None


def test_assess_summary_is_five_lines(fixture_paper: None, capsys: pytest.CaptureFixture[str]) -> None:
    assert main.main(["assess", "2610.19339", "--facts-only", "--offline"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 5 and lines[0].startswith("2610.19339v1")
    assert "(liveness unchecked)" in lines[3]


def test_pdf_reference_count_is_unknown(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    pdf = _paper()
    pdf.html, pdf.text_source, pdf.text = None, "pdf", "1 Introduction\nWe argue.\n2 Discussion\n"
    monkeypatch.setattr(main, "fetch", lambda _ref: pdf)
    assert main.main(["assess", "x", "--facts-only", "--offline", "--json"]) == 0
    facts = json.loads(capsys.readouterr().out)["facts"]
    assert facts["references"]["count"] is None
    assert facts["sections"] == ["Introduction", "Discussion"] and not facts["has_evaluation_section"]


def test_fetch_failure_exits_1(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    def fail(_ref: str) -> Paper:
        raise FetchError("arXiv API has no 9999.99999")

    monkeypatch.setattr(main, "fetch", fail)
    assert main.main(["assess", "9999.99999", "--facts-only"]) == 1
    assert "9999.99999" in capsys.readouterr().err
