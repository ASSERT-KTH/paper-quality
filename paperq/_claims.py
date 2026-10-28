"""Classification and claims layer: what kind of paper, what it claims, and whether it shows it.

Two calls to the cheap model, as in PremiseCheck: first classify the paper and extract its
central claims, then check each claim against the paper and quote the evidence. A quote
that is not in the paper is reported as such: the model's word is never taken for it.
"""

from __future__ import annotations

import json
import re
import unicodedata
from typing import Any

from paperq._llm import Answer, LLMError, ask
from paperq._schema import CLAIM_STATUSES, EVIDENCE_TYPES

CLASSIFIER = "claude-haiku-4.5"
MAX_TEXT_CHARS = 400_000  # about 100k tokens: more than any paper's body

EVIDENCE_DEFINITIONS = {
    "empirical": "measures something (experiments, a user study, a corpus analysis) and draws conclusions from it",
    "system-demo": "presents a system or tool and shows it working, without a controlled comparison",
    "benchmark": "contributes a dataset or benchmark and evaluates existing methods on it",
    "position": "argues a thesis from reasoning and literature, with no data of its own",
    "survey": "reviews and organizes prior work",
    "case-study": "examines one or a few instances in depth",
    "replication": "re-runs a prior study to confirm or refute its results",
}

CLASSIFY_PROMPT = """You are classifying a scientific paper and extracting its central claims.

evidence_type is exactly one of:
{definitions}

Judge from what the paper actually does, not from what it calls itself: a paper that says
"we evaluate" but reports no measurement is not empirical.

Extract the paper's 1 to 5 central claims: the statements the paper wants the reader to
believe at the end, each as one self-contained sentence, with the section where it is made.
Prefer claims with a measurable content ("X improves Y by Z on W") over generic ones.

Reply with: {{"evidence_type": "...", "rationale": "<one sentence>",
"claims": [{{"claim": "...", "location": "<section>"}}]}}

PAPER
Title: {title}

{text}"""

CHECK_PROMPT = """You are checking whether a scientific paper supports its own central claims.

For each claim, find the evidence the paper gives for it (a table, a figure, an experiment,
a proof, an argument) and decide:
- supported: the evidence shown establishes the claim as stated
- partially: the evidence establishes a weaker or narrower claim
- unsupported: the paper gives no evidence, or the evidence contradicts the claim
- not-locatable: the claim's evidence is announced but not in the text (e.g. "see appendix", missing)

Judge the substance only. Ignore writing quality, tone, confidence and formatting.
"quote" is one passage copied verbatim from the paper (at most 300 characters) that carries
the evidence, or the passage that shows the problem; empty when there is none. Mark any cut
inside it with "...".

Reply with: {{"claims": [{{"claim": "<as given>", "status": "...", "quote": "...",
"location": "<section>", "explanation": "<one sentence>"}}]}}

CLAIMS
{claims}

PAPER
Title: {title}

{text}"""


_PUNCTUATION = str.maketrans("\u2018\u2019\u201c\u201d\u2013\u2014\u2212\u00a0", "''\"\"--- ")


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).translate(_PUNCTUATION)
    return re.sub(r"\s+", " ", text).strip().lower()


def is_verbatim(quote: str, text: str) -> bool:
    """The quote occurs in the text, up to whitespace, case, quote marks and dashes.

    A quote elided with "..." passes when every piece of it is in the text.
    """
    pieces = [p for p in (_normalize(p) for p in re.split(r"\.\.\.|\u2026|\[\.\.\.\]", quote)) if p]
    haystack = _normalize(text)
    return bool(pieces) and all(p in haystack for p in pieces)


def paper_text(text: str) -> str:
    return text[:MAX_TEXT_CHARS]


def _add_usage(usage: dict[str, dict[str, int]], layer: str, answer: Answer) -> None:
    slot = usage.setdefault(layer, {"input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0})
    if not answer.cached:
        slot["input_tokens"] += answer.input_tokens
        slot["cached_input_tokens"] += answer.cached_input_tokens
        slot["output_tokens"] += answer.output_tokens


def classify_and_check(title: str, text: str, model: str, usage: dict[str, dict[str, int]],
                       use_cache: bool = True) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """(classification, checked claims). Raises LLMError when the model's answer is unusable."""
    body = paper_text(text)
    definitions = "\n".join(f"- {name}: {meaning}" for name, meaning in EVIDENCE_DEFINITIONS.items())
    answer = ask(model, CLASSIFY_PROMPT.format(definitions=definitions, title=title, text=body), use_cache=use_cache)
    _add_usage(usage, "classification", answer)
    kind = answer.data.get("evidence_type")
    if kind not in EVIDENCE_TYPES:
        raise LLMError(f"evidence_type {kind!r} not in {EVIDENCE_TYPES}")
    classification = {"evidence_type": kind, "rationale": str(answer.data.get("rationale", "")), "model": model}
    extracted = [c for c in answer.data.get("claims", []) if isinstance(c, dict) and c.get("claim")][:5]
    if not extracted:
        return classification, []

    claims_json = json.dumps([{"claim": c["claim"], "location": c.get("location")} for c in extracted], indent=1)
    answer = ask(model, CHECK_PROMPT.format(claims=claims_json, title=title, text=body), use_cache=use_cache)
    _add_usage(usage, "claims", answer)
    checked = []
    for item in answer.data.get("claims", []):
        if not isinstance(item, dict) or item.get("status") not in CLAIM_STATUSES:
            continue
        quote = str(item.get("quote") or "")
        checked.append({
            "claim": str(item.get("claim", "")),
            "location": item.get("location"),
            "status": item["status"],
            "evidence": quote or None,
            "quote_verbatim": is_verbatim(quote, text) if quote else None,
            "explanation": str(item.get("explanation", "")),
        })
    return classification, checked
