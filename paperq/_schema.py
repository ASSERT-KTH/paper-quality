"""The assessment record: one JSON object per paper, versioned.

The record keeps every layer's output — facts, classification, claims, rubric — so a
grade can always be traced back to the evidence that produced it.
"""

from __future__ import annotations

from typing import Any

SCHEMA_VERSION = 1

EVIDENCE_TYPES = ("empirical", "system-demo", "benchmark", "position", "survey", "case-study", "replication")
CLAIM_STATUSES = ("supported", "partially", "unsupported", "not-locatable")
# graded: the rubric layer produced a grade; uncertain: samples disagree too much to grade;
# not-assessed: the LLM layers did not run (facts only)
VERDICTS = ("graded", "uncertain", "not-assessed")

_NULLABLE_STR = {"type": ["string", "null"]}
_NULLABLE_INT = {"type": ["integer", "null"]}

SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "paperq assessment record",
    "type": "object",
    "required": ["schema_version", "paper", "tool", "facts", "reputation", "classification", "claims", "rubric",
                 "verdict", "usage", "errors"],
    "properties": {
        "schema_version": {"const": SCHEMA_VERSION},
        "paper": {
            "type": "object",
            "required": ["id", "version", "title", "authors", "url"],
            "properties": {
                "id": {"type": "string"},
                "version": _NULLABLE_STR,
                "title": {"type": "string"},
                "authors": {"type": "array", "items": {"type": "string"}},
                "url": {"type": "string"},
                "published": _NULLABLE_STR,
                "journal_ref": _NULLABLE_STR,
                "comments": _NULLABLE_STR,
                "text_source": {"enum": ["html", "pdf", None]},
            },
        },
        "tool": {
            "type": "object",
            "required": ["name", "version", "commit"],
            "properties": {"name": {"const": "paperq"}, "version": {"type": "string"}, "commit": {"type": "string"}},
        },
        "facts": {
            "type": "object",
            "description": "Deterministic layer: no LLM involved.",
            "properties": {
                "pages": _NULLABLE_INT,
                "venue": {**_NULLABLE_STR, "description": "acceptance stated in arXiv comments or journal_ref"},
                "sections": {"type": "array", "items": {"type": "string"}},
                "has_evaluation_section": {"type": "boolean"},
                "has_limitations_section": {"type": "boolean"},
                "artifact_links": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["url", "alive"],
                        "properties": {"url": {"type": "string"}, "alive": {"type": ["boolean", "null"]}},
                    },
                },
                "statistics": {"type": "object", "additionalProperties": {"type": "integer"}},
                "references": {
                    "type": "object",
                    "properties": {
                        "count": {"type": ["integer", "null"],
                                  "description": "null when the bibliography was not parsed"},
                        "structured": {"type": "integer", "description": "with a parsed title"},
                        "checked": {"type": "integer"},
                        "findings": {"type": "array", "items": {"type": "object"}},
                    },
                },
            },
        },
        "reputation": {
            "type": ["object", "null"],
            "description": "Author signals, reported for context only: never an input to the grade.",
        },
        "classification": {
            "type": ["object", "null"],
            "properties": {"evidence_type": {"enum": list(EVIDENCE_TYPES)}, "rationale": {"type": "string"}},
        },
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["claim", "status"],
                "properties": {
                    "claim": {"type": "string"},
                    "location": _NULLABLE_STR,
                    "status": {"enum": list(CLAIM_STATUSES)},
                    "evidence": _NULLABLE_STR,
                },
            },
        },
        "rubric": {"type": ["object", "null"]},
        "verdict": {"enum": list(VERDICTS)},
        "usage": {
            "type": "object",
            "description": "Tokens per layer.",
            "additionalProperties": {
                "type": "object",
                "properties": {"input_tokens": {"type": "integer"}, "cached_input_tokens": {"type": "integer"},
                               "output_tokens": {"type": "integer"}},
            },
        },
        "errors": {"type": "array", "items": {"type": "string"}, "description": "sources that failed"},
    },
}


def new_record(paper: dict[str, Any], tool: dict[str, str]) -> dict[str, Any]:
    """An empty record for a paper: every layer present, none run."""
    return {
        "schema_version": SCHEMA_VERSION,
        "paper": paper,
        "tool": tool,
        "facts": {},
        "reputation": None,
        "classification": None,
        "claims": [],
        "rubric": None,
        "verdict": "not-assessed",
        "usage": {},
        "errors": [],
    }


def check(record: dict[str, Any]) -> list[str]:
    """The ways a record departs from the schema's required shape (empty when it conforms)."""
    problems = [f"missing {key}" for key in SCHEMA["required"] if key not in record]
    if record.get("schema_version") != SCHEMA_VERSION:
        problems.append(f"schema_version is {record.get('schema_version')!r}, expected {SCHEMA_VERSION}")
    if record.get("verdict") not in VERDICTS:
        problems.append(f"verdict {record.get('verdict')!r} not in {VERDICTS}")
    classification = record.get("classification")
    if classification and classification.get("evidence_type") not in EVIDENCE_TYPES:
        problems.append(f"evidence_type {classification.get('evidence_type')!r} not in {EVIDENCE_TYPES}")
    for claim in record.get("claims", []):
        if claim.get("status") not in CLAIM_STATUSES:
            problems.append(f"claim status {claim.get('status')!r} not in {CLAIM_STATUSES}")
    return problems
