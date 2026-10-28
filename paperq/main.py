"""paperq command line: assess a paper, print the record schema."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from typing import Any

from paperq import __version__, _facts
from paperq._fetch import FetchError, Paper, fetch
from paperq._schema import SCHEMA, new_record

EPILOG = """\
examples:
  paperq assess 2610.19339 --facts-only          five-line summary of the deterministic facts
  paperq assess 2610.19339v1 --facts-only --json the full record
  paperq schema                                  the JSON schema of a record

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
    return "\n".join([
        f"{paper['id']}{paper['version'] or ''}  {paper['title']}",
        f"  verdict: {record['verdict']}   evidence: {kind}   venue: {facts['venue'] or '-'}",
        f"  sections: {len(facts['sections'])}   evaluation: {'yes' if facts['has_evaluation_section'] else 'no'}"
        f"   limitations: {'yes' if facts['has_limitations_section'] else 'no'}",
        f"  artifacts: {len(links)} ({alive})   statistics: {stats}",
        f"  references: {'?' if refs['count'] is None else refs['count']}"
        f" ({refs['checked']} checked, {len(refs['findings'])} findings)",
    ])


def assess(args: argparse.Namespace) -> int:
    if not args.facts_only:
        print("paperq: the LLM layers are not built yet; run with --facts-only", file=sys.stderr)
        return 2
    status = 0
    for ref in args.papers:
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
        print(json.dumps(record, ensure_ascii=False) if args.json else summary(record))
    return status


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="paperq", description=__doc__, epilog=EPILOG, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--version", action="version", version=f"paperq {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_assess = sub.add_parser("assess", help="assess one or more arXiv papers")
    p_assess.add_argument("papers", nargs="+", metavar="ARXIV_ID", help="arXiv id (2610.19339, 2610.19339v2) or URL")
    p_assess.add_argument("--facts-only", action="store_true", help="only the deterministic layer, no LLM")
    p_assess.add_argument("--offline", action="store_true",
                          help="skip link, reference and reputation lookups (fetches the paper only)")
    p_assess.add_argument("--json", action="store_true", help="one JSON record per line")
    p_assess.set_defaults(func=assess)

    p_schema = sub.add_parser("schema", help="print the JSON schema of a record")
    p_schema.set_defaults(func=lambda _args: print(json.dumps(SCHEMA, indent=2)) or 0)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
