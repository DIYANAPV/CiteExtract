"""Wall-clock timing for pipeline stages.

Usage:
    from src.utils.timing import stage

    with stage("L2_existence", refs=len(refs)):
        exist_map = await check_all_references(...)

Emits one log line on exit:
    STAGE L2_existence seconds=18.342 refs=42
"""

import logging
import time
from contextlib import contextmanager

log = logging.getLogger("checkcitation.timing")


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
