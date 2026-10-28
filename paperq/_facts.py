"""The facts layer: what can be read off a paper without an LLM.

Parsing is pure (HTML or text in, facts out) and network checks — link liveness,
reference existence, author reputation — are separate functions that never raise: a
failed lookup is reported as unknown, not as a defect of the paper.
"""

from __future__ import annotations

import html as htmllib
import re
from collections.abc import Callable
from typing import Any

import requests

from paperq._arxiv import USER_AGENT

# Section titles that announce an evaluation, and an acknowledgement of limits.
EVALUATION_HEADING = re.compile(
    r"evaluat|experiment|result|empirical|case stud|user stud|benchmark|validation|analysis|measurement", re.I
)
LIMITATIONS_HEADING = re.compile(r"limitation|threats? to validity|discussion of validity", re.I)

# Hosts of code, data and models: a link to one of them, outside the bibliography, is an artifact.
ARTIFACT_HOST = re.compile(
    r"https?://(?:www\.)?(?:[\w-]+\.github\.io|github\.com|gitlab\.com|bitbucket\.org|codeberg\.org|zenodo\.org|doi\.org/10\.5281|"
    r"huggingface\.co|figshare\.com|osf\.io|codeocean\.com|archive\.softwareheritage\.org|kaggle\.com)/[^\s\"'<>)\]]+",
    re.I,
)

# Signals of statistical reporting in the body text, counted as matches.
STATISTICS = {
    "confidence_interval": r"confidence intervals?|\b9[05]\s?%\s?CI\b",
    "p_value": r"\bp\s?[<=≤]\s?0?\.\d+|\bp-values?\b",
    "dispersion": r"±|\\pm\b|standard deviations?|standard errors?|\bstd\b|\bIQR\b",
    "seeds_or_runs": r"random seeds?|\b\d+\s+(?:seeds|runs|repetitions|trials)\b",
    "significance_test": r"t-test|wilcoxon|mann[- ]whitney|kruskal|chi-squared?|bootstrap|permutation test"
    r"|statistically significant",
    "effect_size": r"effect sizes?|cohen'?s d|cliff'?s delta|odds ratio",
    "sample_size": r"\b[nN]\s?=\s?\d+",
    "correlation_or_agreement": r"spearman|pearson|kendall|cohen'?s \\?kappa|fleiss|krippendorff|\\kappa\s?=",
}

ACCEPTANCE = re.compile(
    r"(?:accepted|to appear|published|camera[- ]ready)\s+(?:at|to|in|by|for)\s+(?:the\s+)?([^.;,()\n]{3,80})", re.I
)
PAGES = re.compile(r"(\d+)\s*pages", re.I)


def _strip_tags(fragment: str) -> str:
    # a formula becomes its LaTeX source, where `p<0.05` and `\\pm` are written
    text = re.sub(r'<math[^>]*alttext="([^"]*)"[^>]*>.*?</math>', r" \1 ", fragment, flags=re.S | re.I)
    text = re.sub(r"<(script|style|math)[^>]*>.*?</\1>", " ", text, flags=re.S | re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", htmllib.unescape(text)).strip()


def _without_bibliography(html: str) -> str:
    return re.sub(r'<section[^>]*class="ltx_bibliography".*?</section>', " ", html, flags=re.S | re.I)


def html_article(html: str) -> str:
    """The paper itself, without arXiv's page chrome and without its bibliography."""
    body = re.search(r"<article.*?</article>", html, re.S | re.I)
    return _without_bibliography(body.group(0) if body else html)


def html_body_text(html: str) -> str:
    """The paper's prose, without markup and without its bibliography."""
    return _strip_tags(html_article(html))


def sections(html: str, paragraphs: bool = False) -> list[str]:
    """Titles of the sections and subsections, in order, without their numbers.

    With `paragraphs`, also the run-in `\\paragraph{}` titles, where a "Limitations." often hides.
    """
    levels = "section|subsection|subsubsection|appendix" + ("|paragraph" if paragraphs else "")
    titles = re.findall(rf'<h[2-6][^>]*class="ltx_title ltx_title_(?:{levels})"[^>]*>(.*?)</h[2-6]>', html, re.S)
    out = []
    for title in titles:
        title = re.sub(r'<span class="ltx_tag[^"]*">.*?</span>', "", title, flags=re.S)
        out.append(_strip_tags(title))
    return out


def text_sections(text: str) -> list[str]:
    """Section headings of a pdftotext rendering: numbered lines such as `4 Evaluation`."""
    return [m.strip() for m in re.findall(r"^\s*\d+(?:\.\d+)?\.?\s+([A-Z][^\n]{2,60})$", text, re.M)]


def artifact_links(html: str | None, text: str) -> list[str]:
    """Code/data/model links the paper gives for itself, outside the bibliography, deduplicated."""
    # only the article: arXiv's HTML wraps it in links to its own GitHub issue tracker
    source = html_article(html) if html else text
    seen: dict[str, None] = {}
    for url in ARTIFACT_HOST.findall(source):
        url = htmllib.unescape(url).rstrip(".,;:")
        seen.setdefault(url, None)
    return list(seen)


def statistics(text: str) -> dict[str, int]:
    return {name: len(re.findall(pattern, text, re.I)) for name, pattern in STATISTICS.items()}


def venue(comments: str | None, journal_ref: str | None) -> str | None:
    """Where the authors say the paper was accepted, from arXiv's own metadata."""
    if journal_ref:
        return journal_ref.strip()
    if comments and (found := ACCEPTANCE.search(comments)):
        return found.group(1).strip()
    return None


def pages(comments: str | None) -> int | None:
    if comments and (found := PAGES.search(comments)):
        return int(found.group(1))
    return None


def references(html: str | None) -> list[dict[str, str]]:
    """Bibliography entries of an arXiv HTML rendering, as BibTeX-like fields.

    Entries rendered from BibTeX carry their title, authors and DOI in tagged spans; a
    hand-written `\\bibitem` has none, and is returned with its raw text only.
    """
    if not html:
        return []
    out = []
    for n, (attrs, item) in enumerate(re.findall(r'<li id="bib\.bib\d+"([^>]*)>(.*?)</li>', html, re.S), start=1):
        entry: dict[str, str] = {"ID": f"bib{n}", "raw": _strip_tags(item)[:400]}
        kind = re.search(r'class="ltx_bibitem ltx_bib_(\w+)"', attrs)
        entry["ENTRYTYPE"] = kind.group(1) if kind else "misc"
        for field, css in (("title", "ltx_bib_title"), ("author", "ltx_bib_author")):
            found = re.search(rf'<span class="ltx_text {css}">(.*?)</span>', item, re.S)
            if found:
                entry[field] = _strip_tags(found.group(1))
        if "author" in entry:
            entry["author"] = re.sub(r",?\s+and\s+|,\s+", " and ", entry["author"])
        if year := re.search(r"\((\d{4})[a-z]?\)</span>", item):
            entry["year"] = year.group(1)
        if doi := re.search(r'href="https?://(?:dx\.)?doi\.org/(10\.[^"]+)"', item):
            entry["doi"] = htmllib.unescape(doi.group(1))
        # the entry's own link, so check-hallucinations can check a web reference against its page
        if url := re.search(r'href="(https?://[^"]+)"[^>]*class="ltx_ref ltx_bib_external"', item):
            entry["url"] = htmllib.unescape(url.group(1))
        if arxiv := re.search(r"arxiv\.org/abs/(\d{4}\.\d{4,5})", item):
            entry["eprint"] = arxiv.group(1)
            entry["archiveprefix"] = "arXiv"
        out.append(entry)
    return out


def link_alive(url: str, session: Any = requests) -> bool | None:
    """True when the link answers below 400; None when the network failed, not the link."""
    try:
        response = session.head(url, headers={"User-Agent": USER_AGENT}, timeout=20, allow_redirects=True)
        if response.status_code in (403, 405, 429):  # hosts that refuse HEAD or robots
            response = session.get(url, headers={"User-Agent": USER_AGENT}, timeout=20, stream=True)
        return bool(response.status_code < 400)
    except requests.RequestException:
        return None


def check_references(entries: list[dict[str, str]], checker: Callable[[dict[str, str]], Any] | None = None
                     ) -> tuple[int, list[dict[str, str]]]:
    """Look each structured reference up with check-hallucinations.

    Returns (number checked, findings). `unchecked` findings — the lookup failed — are
    dropped: they say nothing about the paper.
    """
    if checker is None:
        from check_hallucinations.main import check_entry

        checker = check_entry
    checked, findings = 0, []
    for entry in entries:
        if "title" not in entry:
            continue
        checked += 1
        result = checker(entry)
        for finding in result.findings:
            if finding.rule == "unchecked":
                checked -= 1
                break
            findings.append({"ref": entry["raw"][:200], "rule": finding.rule, "message": finding.message})
    return checked, findings


def reputation(arxiv_id: str, session: Any = requests) -> dict[str, Any] | None:
    """Authors' h-index and works count from OpenAlex, found through the arXiv DOI.

    None when OpenAlex does not know the paper yet, which is common for new preprints.
    """
    try:
        work = session.get(
            f"https://api.openalex.org/works/doi:10.48550/arXiv.{arxiv_id}",
            headers={"User-Agent": USER_AGENT}, timeout=30,
        )
        if work.status_code != 200:
            return None
        authors = []
        for authorship in work.json().get("authorships", [])[:20]:
            author_id = (authorship.get("author") or {}).get("id", "").rsplit("/", 1)[-1]
            if not author_id:
                continue
            got = session.get(f"https://api.openalex.org/authors/{author_id}", headers={"User-Agent": USER_AGENT},
                              timeout=30)
            if got.status_code != 200:
                continue
            data = got.json()
            authors.append({
                "name": data.get("display_name"),
                "openalex": author_id,
                "h_index": (data.get("summary_stats") or {}).get("h_index"),
                "works": data.get("works_count"),
            })
        return {"source": "openalex", "authors": authors}
    except (requests.RequestException, ValueError):
        return None
