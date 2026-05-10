
import logging
import time
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Optional

log = logging.getLogger("citeextract.timing")

_collector: ContextVar[Optional[dict[str, float]]] = ContextVar(
    "stage_collector", default=None,
)


@contextmanager
def stage(name: str, **meta):
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        parts = [f"STAGE {name}", f"seconds={elapsed:.3f}"]
        for k, v in meta.items():
            parts.append(f"{k}={v}")
        log.info(" ".join(parts))
        bucket = _collector.get()
        if bucket is not None:
            bucket[name] = bucket.get(name, 0.0) + elapsed


@contextmanager
def collect_stages():
    """Activate a per-run accumulator. Use as a context manager around the
    top-level pipeline call. Returns a dict that gets populated as `stage()`
    contexts exit.
    """
    bucket: dict[str, float] = {}
    token = _collector.set(bucket)
    try:
        yield bucket
    finally:
        _collector.reset(token)
