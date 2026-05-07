
import logging
import time
from contextlib import contextmanager

log = logging.getLogger("citeextract.timing")


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
