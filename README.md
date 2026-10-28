# paperq

Automated quality assessment of scientific papers: what kind of evidence a paper carries,
whether its central claims are supported, which rigor checks fail, and a grade with an
uncertainty. It doesn't write reviews: it produces one structured JSON record per paper that
code can rank and filter on, with every grade traceable to the evidence behind it.

Status: the deterministic **facts layer** is built. The LLM layers (classification and claims,
graded rubric) are not built yet, and neither is their validation.

## Install

```bash
uv venv && uv pip install -e '.[dev]'
```

## Usage

```bash
paperq assess 2610.19339 --facts-only            # five-line summary
paperq assess 2610.19339v1 --facts-only --json   # the full record, one JSON object per line
paperq assess 2610.19339 2610.32574 --facts-only --offline   # no link, reference or reputation lookups
paperq schema                                    # JSON schema of a record
```

```
2610.21046v1  PremiseCheck: Efficient Fact-Checking of Human and AI Peer Reviews
  verdict: not-assessed   evidence: unclassified   venue: -
  sections: 52   evaluation: yes   limitations: yes
  artifacts: 3 (3 alive)   statistics: confidence_interval=16, dispersion=2, significance_test=3, correlation_or_agreement=1
  references: 36 (36 checked, 0 findings)
```

Exit status: 0 assessed, 1 a paper could not be fetched, 2 usage error.

## The facts layer

No LLM is involved. Full text comes from arXiv's HTML rendering, or from the PDF through
`pdftotext` when there is no HTML. Downloads of a versioned id are cached in
`~/.cache/paperq`.

- `venue`: an acceptance taken from arXiv's `journal_ref`, or from "accepted at / to appear in"
  in the comments. It's what the authors say, not checked.
- `sections`, `has_evaluation_section`, `has_limitations_section`: found from headings. A
  run-in `\paragraph{Limitations.}` counts as a limitations section.
- `artifact_links`: the paper's own links to code, data or models, plus `*.github.io` project
  pages. Links inside the bibliography (other people's code) and arXiv's page chrome are
  excluded. `alive` is `null` when the link wasn't checked or the network failed.
- `statistics`: counts of reporting signals in the body text, with formulas kept as their LaTeX
  source: confidence intervals, p-values, dispersion, seeds or runs, significance tests, effect
  sizes, sample sizes, correlation or agreement.
- `references`: the bibliography of the HTML rendering. Entries generated from BibTeX are
  looked up with [check-hallucinations](https://pypi.org/project/check-hallucinations/); a
  lookup that failed is not a finding. A PDF bibliography isn't parsed, so its `count` is
  `null`, not 0.
- `reputation`: authors' h-index and works count from OpenAlex. It's reported for context and
  is never an input to the grade. It's `null` for preprints OpenAlex doesn't know yet, and its
  author disambiguation is unreliable for new profiles.

## Development

```bash
pytest --cov=paperq && ruff check . && mypy paperq tests
```

## License

MIT
