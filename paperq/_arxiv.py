"""The one arXiv client: paced across every paperq process, patient with throttling.

arXiv asks for at most one request every 3 seconds, from all of a host's clients together.
The pace is kept in a host-wide lock file ($XDG_CACHE_HOME/arxiv.lock), so concurrent
processes (an assessment, a reference-set build, the trend-report pollers) share it
instead of each keeping its own. A throttled arXiv answers 429 or an empty
200; both are waited out (Retry-After when given), never taken for "no such paper".
"""

from __future__ import annotations

import fcntl
import time
from collections.abc import Callable

import requests

USER_AGENT = "paperq/0.1 (https://github.com/ASSERT-KTH/paper-quality)"
INTERVAL = 3.1
WAITS = (10, 30, 60, 120, 300)  # seconds between attempts when arXiv throttles
TIMEOUT = 60


class ArxivError(requests.RequestException):
    """arXiv kept refusing or failing."""


def _lock_path() -> str:
    from paperq._fetch import cache_dir

    path = cache_dir().parent / "arxiv.lock"  # beside paperq's cache: shared by every arXiv client
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)


def _paced() -> None:
    """Block until INTERVAL has passed since the last arXiv request of any process."""
    with open(_lock_path(), "a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        lock.seek(0)
        try:
            last = float(lock.read() or 0)
        except ValueError:
            last = 0.0
        wait = last + INTERVAL - time.time()
        if wait > 0:
            time.sleep(wait)
        lock.seek(0)
        lock.truncate()
        lock.write(str(time.time()))
        # the lock is released on close, after the timestamp is written


def get(url: str, ok: Callable[[requests.Response], bool], waits: tuple[float, ...] = WAITS) -> requests.Response:
    """GET an arXiv URL until `ok(response)`; 404 is returned as it is (the caller decides)."""
    problem = ""
    for wait in (0, *waits):
        if wait:
            time.sleep(wait)
        _paced()
        try:
            response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT)
        except requests.RequestException as exc:
            problem = str(exc)
            continue
        if response.status_code == 404 or ok(response):
            return response
        problem = f"HTTP {response.status_code}{', empty answer' if not response.content else ''}"
        retry_after = response.headers.get("Retry-After", "")
        if retry_after.isdigit() and int(retry_after) > 0:
            time.sleep(min(int(retry_after), 600))
    raise ArxivError(f"{url}: {problem}")


def api_ok(response: requests.Response) -> bool:
    return response.status_code == 200 and b"<feed" in response.content


def any_ok(response: requests.Response) -> bool:
    return response.status_code == 200 and bool(response.content)


def page_ok(response: requests.Response) -> bool:
    return response.status_code == 200 and b"<html" in response.content[:2000].lower()
