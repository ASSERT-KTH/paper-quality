"""LLM backend: one pinned model per call through `agent.py --model`, with an answer cache.

`agent.py` exposes no temperature, so repeated samples vary as the provider makes them vary;
the variance is measured, not suppressed. Each answer is cached under (model, prompt,
sample index), so re-running an assessment costs nothing and returns the same record.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Any

from paperq._fetch import cache_dir

AGENT = "agent.py"
TIMEOUT = 1200
NO_TOOLS = (
    "Answer from the text in this message only. Do not use any tool, do not read or write files, "
    "do not browse. Reply with one JSON object and nothing else.\n\n"
)


class LLMError(Exception):
    """The model did not answer, or answered something that is not the JSON asked for."""


@dataclass
class Answer:
    data: dict[str, Any]
    input_tokens: int
    output_tokens: int
    cached: bool  # answered from paperq's own cache: no tokens spent
    cached_input_tokens: int = 0  # input tokens the provider served from its prompt cache


def parse_json(text: str) -> dict[str, Any]:
    """The JSON object in a reply, fenced or not."""
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S)
    candidate = fenced.group(1) if fenced else text[text.find("{"): text.rfind("}") + 1]
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise LLMError(f"reply is not JSON: {text[:200]!r}") from exc
    if not isinstance(value, dict):
        raise LLMError(f"reply is not a JSON object: {text[:200]!r}")
    return value


def _run(model: str, prompt: str) -> dict[str, Any]:
    # an empty working directory: the agent has nothing to read even if it tries
    with tempfile.TemporaryDirectory() as tmp:
        out = subprocess.run(
            [AGENT, "--model", model, "--json"], input=NO_TOOLS + prompt, capture_output=True, text=True,
            timeout=TIMEOUT, cwd=tmp, check=False,
        )
    if out.returncode != 0:
        raise LLMError(f"{AGENT} --model {model} exited {out.returncode}: {out.stderr.strip()[-300:]}")
    try:
        result: dict[str, Any] = json.loads(out.stdout)
    except json.JSONDecodeError as exc:
        raise LLMError(f"{AGENT} printed no JSON: {out.stdout[:200]!r}") from exc
    return result


def ask(model: str, prompt: str, sample: int = 0, use_cache: bool = True) -> Answer:
    """One JSON answer of `model`; `sample` distinguishes repeated draws of the same prompt."""
    key = hashlib.sha256(json.dumps([model, prompt, sample]).encode()).hexdigest()
    path = cache_dir() / "llm" / f"{key}.json"
    if use_cache and path.exists():
        hit = json.loads(path.read_text())
        return Answer(hit["data"], hit["input_tokens"], hit["output_tokens"], cached=True,
                      cached_input_tokens=hit.get("cached_input_tokens", 0))
    last: Exception | None = None
    for _attempt in range(2):
        try:
            result = _run(model, prompt)
            data = parse_json(result.get("final_reply") or "")
            break
        except (LLMError, subprocess.TimeoutExpired) as exc:
            last = exc
    else:
        raise LLMError(str(last))
    usage = result.get("usage") or {}
    answer = Answer(data, int(usage.get("prompt", 0)), int(usage.get("completion", 0)), cached=False,
                    cached_input_tokens=int(usage.get("cached", 0)))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"model": model, "sample": sample, "data": data,
                                "input_tokens": answer.input_tokens, "output_tokens": answer.output_tokens,
                                "cached_input_tokens": answer.cached_input_tokens}))
    return answer
