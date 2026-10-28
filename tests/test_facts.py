"""The deterministic facts layer, on a fixture written in arXiv's LaTeXML markup."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import requests

from paperq import _facts
from paperq._fetch import FetchError, parse_arxiv_id, parse_atom

FIXTURES = Path(__file__).parent / "fixtures"
HTML = (FIXTURES / "paper.html").read_text()


def test_sections_drop_numbers_and_keep_order() -> None:
    assert _facts.sections(HTML) == ["Introduction", "Evaluation", "Setup"]


def test_paragraph_titles_only_on_request() -> None:
    assert "Limitations." in _facts.sections(HTML, paragraphs=True)
    assert "Limitations." not in _facts.sections(HTML)


def test_artifacts_are_the_papers_own_links() -> None:
    links = _facts.artifact_links(HTML, "")
    assert links == ["https://github.com/acme/fixture", "https://acme.github.io/demo/"]  # deduplicated, in order


def test_artifacts_ignore_arxiv_chrome_and_cited_code() -> None:
    links = _facts.artifact_links(HTML, "")
    assert not any("html_feedback" in u or "other/tool" in u for u in links)


def test_artifacts_from_pdf_text() -> None:
    assert _facts.artifact_links(None, "see https://zenodo.org/records/123.") == ["https://zenodo.org/records/123"]


def test_body_text_keeps_formulas_and_drops_bibliography() -> None:
    text = _facts.html_body_text(HTML)
    assert "p<0.01" in text and "\\pm" in text
    assert "closer look" not in text and "Report issue" not in text


def test_statistics_counts() -> None:
    stats = _facts.statistics(_facts.html_body_text(HTML))
    assert stats["p_value"] == 1
    assert stats["dispersion"] >= 1
    assert stats["seeds_or_runs"] == 1
    assert stats["sample_size"] == 1
    assert stats["significance_test"] == 1
    assert stats["correlation_or_agreement"] == 1
    assert stats["effect_size"] == 0


def test_statistics_none_in_prose() -> None:
    assert not any(_facts.statistics("We argue that disclosure is not enough.").values())


@pytest.mark.parametrize(
    "comments, journal_ref, expected",
    [
        ("12 pages, accepted at ICLR 2027 (oral)", None, "ICLR 2027"),
        ("To appear in the Proceedings of ICSE 2027", None, "Proceedings of ICSE 2027"),
        ("12 pages", "Ethics Inf. Technol. 28, 54 (2026)", "Ethics Inf. Technol. 28, 54 (2026)"),
        ("12 pages, under review", None, None),
        (None, None, None),
    ],
)
def test_venue(comments: str | None, journal_ref: str | None, expected: str | None) -> None:
    assert _facts.venue(comments, journal_ref) == expected


def test_pages() -> None:
    assert _facts.pages("12 pages, 3 figures") == 12
    assert _facts.pages("work in progress") is None


def test_references_structured() -> None:
    first, second, third = _facts.references(HTML)
    assert first["title"] == "A closer look into using large language models for automatic evaluation"
    assert first["author"] == "C. Chiang and H. Lee"
    assert first["year"] == "2023"
    assert first["doi"] == "10.18653/v1/2023.findings-emnlp.599"
    assert first["url"] == "https://aclanthology.org/2023.findings-emnlp.599/"
    assert first["ENTRYTYPE"] == "inproceedings"
    assert not first["raw"].startswith("class=")
    assert second["author"] == "A. Other and B. Second and C. Third"
    assert second["year"] == "2024"
    assert second["eprint"] == "2401.00001"
    assert "title" not in third and "hand-written" in third["raw"]


def test_references_of_a_pdf_are_unknown() -> None:
    assert _facts.references(None) == []


def _result(*rules: str) -> SimpleNamespace:
    return SimpleNamespace(findings=[SimpleNamespace(rule=r, message=f"{r} message") for r in rules])


def test_check_references_skips_unstructured_and_unchecked() -> None:
    refs = _facts.references(HTML)
    answers = iter([_result("not-found"), _result("unchecked")])
    checked, findings = _facts.check_references(refs, checker=lambda _e: next(answers))
    assert checked == 1  # bib2 failed to look up, bib3 has no title
    assert [f["rule"] for f in findings] == ["not-found"]


class _Session:
    def __init__(self, head: int | Exception, get: int = 200, payload: object = None) -> None:
        self._head, self._get, self._payload = head, get, payload

    def head(self, *_a: object, **_k: object) -> SimpleNamespace:
        if isinstance(self._head, Exception):
            raise self._head
        return SimpleNamespace(status_code=self._head)

    def get(self, *_a: object, **_k: object) -> SimpleNamespace:
        return SimpleNamespace(status_code=self._get, json=lambda: self._payload)


def test_link_alive() -> None:
    assert _facts.link_alive("u", _Session(200)) is True
    assert _facts.link_alive("u", _Session(404)) is False
    assert _facts.link_alive("u", _Session(405, get=200)) is True  # HEAD refused, GET answers
    assert _facts.link_alive("u", _Session(requests.ConnectionError())) is None  # unknown, not dead


def test_reputation_unknown_paper() -> None:
    assert _facts.reputation("2610.19339", _Session(200, get=404)) is None


def test_parse_arxiv_id() -> None:
    assert parse_arxiv_id("2610.19339") == ("2610.19339", None)
    assert parse_arxiv_id("https://arxiv.org/abs/2610.19339v2") == ("2610.19339", "v2")
    assert parse_arxiv_id("arXiv:2610.19339v1") == ("2610.19339", "v1")
    with pytest.raises(FetchError):
        parse_arxiv_id("not a paper")


def test_parse_atom() -> None:
    meta = parse_atom((FIXTURES / "atom.xml").read_text())
    assert meta["version"] == "v2"
    assert meta["title"] == "Can We Believe AI Peer Review?"
    assert meta["authors"] == ["Zongrui Yang", "Gillian Dobbie"]
    assert meta["published"] == "2026-10-14"
    assert meta["comments"] == "12 pages, accepted at ICLR 2027 (oral)"
    assert meta["journal_ref"] is None


def test_parse_atom_empty() -> None:
    with pytest.raises(FetchError):
        parse_atom('<feed xmlns="http://www.w3.org/2005/Atom"></feed>')
