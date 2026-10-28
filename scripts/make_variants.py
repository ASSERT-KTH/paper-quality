"""Variants of reference-set papers for the validation: planted errors, rewrites, fake references.

Each function takes a fetched Paper and returns (variant Paper, what was changed). Planted
errors and rewrites are written by GENERATOR, a model that is not one of the judges; its
answers are cached, so a variant is the same every time it is built.
"""

from __future__ import annotations

import random
import re
from dataclasses import replace
from typing import Any

from paperq._claims import paper_text
from paperq._fetch import Paper
from paperq._llm import LLMError, ask

GENERATOR = "claude-opus-5-5"
ERRORS = 3
CHUNK = 12_000
SEED = 20261028


class VariantError(Exception):
    """The variant could not be built faithfully."""


INJECT_PROMPT = """You are building a test of automated paper review. Plant exactly {n} errors in
the scientific paper below. Each error replaces one sentence of the paper with a version that
contradicts something else the paper itself shows: a number that disagrees with a table, a
conclusion the reported results do not support, a method step inconsistent with the setup.
An expert reading carefully must be able to find it from the paper alone. Do not plant
typos, style problems or errors in the references.

"original" is the sentence copied exactly, character for character, from the paper.

Reply with: {{"errors": [{{"section": "...", "original": "...", "replacement": "...",
"contradicts": "<what in the paper it contradicts>"}}]}}

PAPER
{text}"""

REWRITE_PROMPT = """Rewrite the following passage of a scientific paper so that its wording, sentence
structure and tone change throughout, while its content is exactly preserved: keep every
claim, every number, every citation marker and every technical term, and add or remove no
information. Do not improve or weaken the argument.

Reply with: {{"text": "<the rewritten passage>"}}

PASSAGE
{text}"""


def _locate(sentence: str, text: str) -> re.Match[str] | None:
    """The sentence in the text, whatever the whitespace between its words."""
    words = sentence.split()
    if len(words) < 4:
        return None
    return re.search(r"\s+".join(re.escape(w) for w in words), text)


def _variant(paper: Paper, text: str) -> Paper:
    return replace(paper, html=None, text=text, text_source="variant")


def injected(paper: Paper) -> tuple[Paper, dict[str, Any]]:
    try:
        answer = ask(GENERATOR, INJECT_PROMPT.format(n=ERRORS, text=paper_text(paper.text)))
    except LLMError as exc:
        raise VariantError(f"generator: {exc}") from exc
    text, planted = paper.text, []
    for error in answer.data.get("errors", [])[:ERRORS]:
        original, replacement = str(error.get("original", "")), str(error.get("replacement", ""))
        found = _locate(original, text)
        if not found or not replacement or replacement == original:
            continue
        text = text[: found.start()] + replacement + text[found.end():]
        planted.append({k: str(error.get(k, "")) for k in ("section", "original", "replacement", "contradicts")})
    if len(planted) < ERRORS:
        raise VariantError(f"only {len(planted)} of {ERRORS} errors could be placed verbatim")
    return _variant(paper, text), {"kind": "injected", "generator": GENERATOR, "errors": planted}


NUMBER = re.compile(r"\d+(?:[.,]\d+)?")


def _chunks(text: str) -> list[str]:
    """Pieces of about CHUNK characters, cut after a sentence."""
    out, start = [], 0
    while start < len(text):
        end = min(len(text), start + CHUNK)
        if end < len(text):
            cut = text.rfind(". ", start + CHUNK // 2, end)
            end = cut + 2 if cut > 0 else end
        out.append(text[start:end])
        start = end
    return out


def rewritten(paper: Paper) -> tuple[Paper, dict[str, Any]]:
    """Section-by-section rewrite; a chunk whose numbers change is retried once, then kept as is."""
    pieces, kept = [], 0
    for chunk in _chunks(paper_text(paper.text)):
        for sample in range(2):
            try:
                new = str(ask(GENERATOR, REWRITE_PROMPT.format(text=chunk), sample=sample).data.get("text", ""))
            except LLMError:
                new = ""
            if new and sorted(NUMBER.findall(new)) == sorted(NUMBER.findall(chunk)):
                pieces.append(new)
                break
        else:
            pieces.append(chunk)
            kept += 1
    total = len(pieces)
    if kept > total // 4:
        raise VariantError(f"{kept} of {total} chunks kept their original wording: not a rewrite")
    return _variant(paper, " ".join(pieces)), {"kind": "rewrite", "generator": GENERATOR, "chunks": total,
                                                "kept_original": kept}


WORDS_A = ["Recursive", "Spectral", "Adversarial", "Federated", "Topological", "Causal", "Sparse", "Quantum"]
WORDS_B = ["Gradient Folding", "Lattice Distillation", "Hypergraph Compilation", "Manifold Routing",
           "Tensor Sheaves", "Kernel Annealing", "Prompt Homology", "Entropy Weaving"]
WORDS_C = ["for Self-Healing Compilers", "in Orbital Datacenters", "under Byzantine Tokenizers",
           "for Neuromorphic Peer Review", "across Holographic Benchmarks", "with Tidal Attention"]
SURNAMES = ["Valtonen", "Okonkwo-Reyes", "Brandquist", "Ishikawa-Moore", "Delacroix-Hahn", "Szymborek"]


def fake_references(key: str, n: int = 3) -> list[dict[str, str]]:
    rng = random.Random(f"{SEED}:{key}")
    refs = []
    for _ in range(n):
        title = f"{rng.choice(WORDS_A)} {rng.choice(WORDS_B)} {rng.choice(WORDS_C)}"
        authors = f"{rng.choice('ABCDJKLMRS')}. {rng.choice(SURNAMES)} and {rng.choice('ABCDJKLMRS')}. " \
            f"{rng.choice(SURNAMES)}"
        refs.append({"title": title, "author": authors, "year": str(rng.randint(2019, 2025))})
    return refs


def fabricated_refs(paper: Paper) -> tuple[Paper, dict[str, Any]]:
    """Plant references to papers that do not exist into the HTML bibliography."""
    if not paper.html or 'class="ltx_bibliography"' not in paper.html:
        raise VariantError("no HTML bibliography to plant references in")
    fakes = fake_references(f"{paper.id}{paper.version}")
    items = "".join(
        f'<li id="bib.bib9{n}" class="ltx_bibitem ltx_bib_inproceedings">'
        f'<span class="ltx_tag ltx_bib_author-year ltx_role_refnum ltx_tag_bibitem">{f["author"].split()[1]} '
        f'({f["year"]})</span> <span class="ltx_bibblock"><span class="ltx_text ltx_bib_author">{f["author"]}'
        f'</span> </span> <span class="ltx_bibblock"><span class="ltx_text ltx_bib_title">{f["title"]}</span>. '
        f"</span></li>"
        for n, f in enumerate(fakes)
    )
    html = re.sub(r'(<section[^>]*class="ltx_bibliography".*?<ul[^>]*>)', lambda m: m.group(1) + items,
                  paper.html, count=1, flags=re.S)
    if html == paper.html:
        raise VariantError("bibliography has no list")
    return replace(paper, html=html), {"kind": "refs-fabricated", "references": fakes}
