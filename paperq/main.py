"""paperq command line: assess papers, explain a grade, print the record schema."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from paperq import __version__, _facts, _rubric
from paperq._claims import CLASSIFIER, classify_and_check
from paperq._fetch import FetchError, Paper, cache_dir, fetch, parse_arxiv_id
from paperq._llm import LLMError
from paperq._schema import SCHEMA, new_record

EPILOG = """\
examples:
  paperq assess 2610.19339                       facts, classification, claims and graded rubric
  paperq assess 2610.19339 --facts-only          only the deterministic facts, no LLM
  paperq assess 2610.19339v1 --json              the full record
  paperq assess --from-file ids.txt --json       one arXiv id per line
  paperq explain 2610.19339v1                    the quoted findings behind a grade
  paperq schema                                  the JSON schema of a record

Records are kept in ~/.cache/paperq/records; LLM answers are cached, so assessing a paper
again costs no tokens unless --no-cache is given.

exit status: 0 assessed, 1 the paper could not be fetched, 2 usage error.
"""


def _commit() -> str:
    try:
        out = subprocess.run(
            ["git", "-C", os.path.dirname(os.path.abspath(__file__)), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return out.stdout.strip() or "unknown"


def facts_layer(paper: Paper, record: dict[str, Any], network: bool = True) -> None:
    """Fill the record's facts (and reputation) from the paper; no LLM involved."""
    titles = _facts.sections(paper.html) if paper.html else _facts.text_sections(paper.text)
    headings = _facts.sections(paper.html, paragraphs=True) if paper.html else titles
    refs = _facts.references(paper.html)
    links = _facts.artifact_links(paper.html, paper.text)
    facts: dict[str, Any] = {
        "pages": _facts.pages(paper.comments),
        "venue": _facts.venue(paper.comments, paper.journal_ref),
        "sections": titles,
        "has_evaluation_section": any(_facts.EVALUATION_HEADING.search(t) for t in titles),
        "has_limitations_section": any(_facts.LIMITATIONS_HEADING.search(t) for t in headings),
        "artifact_links": [{"url": u, "alive": _facts.link_alive(u) if network else None} for u in links],
        "statistics": _facts.statistics(paper.text),
        # a PDF's bibliography is not parsed: its count is unknown, not zero
        "references": {"count": len(refs) if paper.html else None, "structured": sum("title" in r for r in refs),
                       "checked": 0, "findings": []},
    }
    if network:
        try:
            checked, findings = _facts.check_references(refs)
            facts["references"].update(checked=checked, findings=findings)
        except Exception as exc:  # a broken lookup service must not lose the other facts
            record["errors"].append(f"reference check: {exc}")
        record["reputation"] = _facts.reputation(paper.id)
    if not paper.text_source:
        record["errors"].append("no full text: neither HTML nor PDF was found")
    record["facts"] = facts


def summary(record: dict[str, Any]) -> str:
    """Five lines a human reads; `--json` has everything else."""
    paper, facts = record["paper"], record["facts"]
    refs = facts["references"]
    links = facts["artifact_links"]
    known = [link["alive"] for link in links if link["alive"] is not None]
    alive = f"{sum(known)} alive" if known or not links else "liveness unchecked"
    stats = ", ".join(f"{k}={v}" for k, v in facts["statistics"].items() if v) or "none"
    kind = (record["classification"] or {}).get("evidence_type", "unclassified")
    rubric = record["rubric"] or {}
    graded = f" {rubric['grade']}/4 (IQR {rubric['iqr']})" if rubric.get("grade") is not None else ""
    claims = record["claims"]
    supported = sum(c["status"] == "supported" for c in claims)
    return "\n".join([
        f"{paper['id']}{paper['version'] or ''}  {paper['title']}",
        f"  verdict: {record['verdict']}{graded}   evidence: {kind}   claims: {supported}/{len(claims)} supported"
        f"   venue: {facts['venue'] or '-'}",
        f"  sections: {len(facts['sections'])}   evaluation: {'yes' if facts['has_evaluation_section'] else 'no'}"
        f"   limitations: {'yes' if facts['has_limitations_section'] else 'no'}",
        f"  artifacts: {len(links)} ({alive})   statistics: {stats}",
        f"  references: {'?' if refs['count'] is None else refs['count']}"
        f" ({refs['checked']} checked, {len(refs['findings'])} findings)",
    ])


def llm_layers(paper: Paper, record: dict[str, Any], use_cache: bool = True) -> None:
    """Classification and claims (cheap model), then the graded rubric (judges)."""
    if not paper.text:
        return
    try:
        classification, claims = classify_and_check(paper.title, paper.text, CLASSIFIER, record["usage"], use_cache)
    except LLMError as exc:
        record["errors"].append(f"classification: {exc}")
        return
    record["classification"], record["claims"] = classification, claims
    rubric, verdict = _rubric.grade(paper.title, paper.text, classification["evidence_type"], record["usage"],
                                    use_cache=use_cache)
    record["rubric"], record["verdict"] = rubric, verdict


def _record_path(paper_id: str, version: str | None) -> Path:
    return cache_dir() / "records" / f"{paper_id}{version or ''}.json"


def _refs(args: argparse.Namespace) -> list[str]:
    refs = list(args.papers)
    if args.from_file:
        lines = Path(args.from_file).read_text().splitlines()
        refs += [line.split("#")[0].strip() for line in lines if line.split("#")[0].strip()]
    return refs


def assess(args: argparse.Namespace) -> int:
    refs = _refs(args)
    if not refs:
        print("paperq: no paper given", file=sys.stderr)
        return 2
    status = 0
    for ref in refs:
        try:
            paper = fetch(ref)
        except FetchError as exc:
            print(f"paperq: {ref}: {exc}", file=sys.stderr)
            status = 1
            continue
        record = new_record(
            {"id": paper.id, "version": paper.version, "title": paper.title, "authors": paper.authors,
             "url": paper.url, "published": paper.published, "journal_ref": paper.journal_ref,
             "comments": paper.comments, "text_source": paper.text_source},
            {"name": "paperq", "version": __version__, "commit": _commit()},
        )
        facts_layer(paper, record, network=not args.offline)
        if not args.facts_only:
            llm_layers(paper, record, use_cache=not args.no_cache)
            path = _record_path(paper.id, paper.version)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(record, ensure_ascii=False))
        print(json.dumps(record, ensure_ascii=False) if args.json else summary(record))
    return status


def explain(args: argparse.Namespace) -> int:
    """The findings behind a stored grade, most severe first, each with its verbatim quote."""
    try:
        paper_id, version = parse_arxiv_id(args.paper)
    except FetchError as exc:
        print(f"paperq: {exc}", file=sys.stderr)
        return 2
    records = cache_dir() / "records"
    paths = [_record_path(paper_id, version)] if version else sorted(records.glob(f"{paper_id}v*.json"))
    paths = [p for p in paths if p.exists()]
    if not paths:
        print(f"paperq: no assessment of {args.paper}; run paperq assess first", file=sys.stderr)
        return 1
    record = json.loads(paths[-1].read_text())
    found = _rubric.findings(record["rubric"] or {})
    if args.json:
        print(json.dumps({"paper": record["paper"], "verdict": record["verdict"], "claims": record["claims"],
                          "findings": found}, ensure_ascii=False))
        return 0
    print(summary(record))
    for claim in record["claims"]:
        print(f"  claim [{claim['status']}] {claim['claim']}")
    for f in found:
        print(f"  {f['severity']:<5} {f['item']}: {f['explanation']}")
        print(f"        \"{f['quote']}\"  ({f['model']}#{f['sample']})")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="paperq", description=__doc__, epilog=EPILOG, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--version", action="version", version=f"paperq {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_assess = sub.add_parser("assess", help="assess one or more arXiv papers")
    p_assess.add_argument("papers", nargs="*", metavar="ARXIV_ID", help="arXiv id (2610.19339, 2610.19339v2) or URL")
    p_assess.add_argument("--from-file", metavar="FILE", help="also assess the ids in FILE, one per line")
    p_assess.add_argument("--no-cache", action="store_true", help="ask the models again instead of reusing answers")
    p_assess.add_argument("--facts-only", action="store_true", help="only the deterministic layer, no LLM")
    p_assess.add_argument("--offline", action="store_true",
                          help="skip link, reference and reputation lookups (fetches the paper only)")
    p_assess.add_argument("--json", action="store_true", help="one JSON record per line")
    p_assess.set_defaults(func=assess)

    p_explain = sub.add_parser("explain", help="the quoted findings behind a stored grade")
    p_explain.add_argument("paper", metavar="ARXIV_ID")
    p_explain.add_argument("--json", action="store_true")
    p_explain.set_defaults(func=explain)

    p_schema = sub.add_parser("schema", help="print the JSON schema of a record")
    p_schema.set_defaults(func=lambda _args: print(json.dumps(SCHEMA, indent=2)) or 0)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
