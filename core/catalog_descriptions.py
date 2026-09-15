"""Bounded background loading of public MOSES descriptions for visible cards."""
from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from threading import RLock
from time import monotonic
from typing import Callable

from bs4 import BeautifulSoup

from core.providers.tu_berlin import moses

CatalogKey = tuple[str, int, str]


@dataclass(frozen=True)
class CatalogDescription:
    learning_outcomes: str | None = None
    contents: str | None = None


def fetch_catalog_description(number: str, version: int, _term: str) -> CatalogDescription:
    """Read only the selected result's page, without expanding catalogs or ISIS links."""
    markup = moses.fetch_html(moses._canonical_detail_url(number, version), timeout=10)
    soup = BeautifulSoup(markup, "html.parser")
    # Reuse the scraper's section extraction so snippets match the full preview.
    return CatalogDescription(
        learning_outcomes=moses._section_text_after_heading(soup, "Lernergebnisse", "Learning outcomes"),
        contents=moses._section_text_after_heading(soup, "Lehrinhalte", "Contents"),
    )


@dataclass(frozen=True)
class DescriptionResult:
    state: str  # queued, loading, ready, error
    data: CatalogDescription | None = None


class CatalogDescriptions:
    def __init__(self, fetch: Callable[[str, int, str], CatalogDescription], *, workers: int = 3,
                 capacity: int = 256, ttl: float = 3600, clock: Callable[[], float] = monotonic):
        self._fetch = fetch
        self._workers = workers
        self._capacity = capacity
        self._ttl = ttl
        self._clock = clock
        self._lock = RLock()
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="catalog-description")
        self._jobs: dict[CatalogKey, Future] = {}
        self._cache: OrderedDict[CatalogKey, tuple[float, DescriptionResult]] = OrderedDict()

    def _remember(self, key: CatalogKey, result: DescriptionResult) -> None:
        self._cache[key] = (self._clock() + (60 if result.state == "error" else self._ttl), result)
        self._cache.move_to_end(key)
        while len(self._cache) > self._capacity:
            self._cache.popitem(last=False)

    def _collect(self) -> None:
        for key, future in list(self._jobs.items()):
            if future.done():
                try:
                    result = DescriptionResult("ready", future.result())
                except Exception:
                    result = DescriptionResult("error")
                self._remember(key, result)
                del self._jobs[key]
        for key, (expires, _) in list(self._cache.items()):
            if expires <= self._clock():
                del self._cache[key]

    def request(self, keys: list[CatalogKey]) -> dict[CatalogKey, DescriptionResult]:
        """Start at most one small batch, in displayed order; never queue a whole search."""
        with self._lock:
            self._collect()
            for key in dict.fromkeys(keys):
                if key in self._cache:
                    self._cache.move_to_end(key)
                elif key not in self._jobs and len(self._jobs) < self._workers:
                    self._jobs[key] = self._pool.submit(self._fetch, *key)
            self._collect()
            return {key: self._cache[key][1] if key in self._cache else
                DescriptionResult("loading" if key in self._jobs else "queued") for key in keys}

    def resolve(self, key: CatalogKey) -> CatalogDescription:
        """Wait for a cached or in-flight snippet; explicit requests may retry errors."""
        with self._lock:
            self._collect()
            entry = self._cache.get(key)
            if entry and entry[1].state == "ready":
                self._cache.move_to_end(key)
                return entry[1].data
            future = self._jobs.get(key)
            if future is None:
                future = self._pool.submit(self._fetch, *key)
                self._jobs[key] = future
        try:
            return future.result()
        finally:
            with self._lock:
                self._collect()

    def close(self) -> None:
        self._pool.shutdown(wait=True, cancel_futures=True)
