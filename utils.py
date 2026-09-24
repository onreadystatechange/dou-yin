from __future__ import annotations

import time
from typing import Callable, TypeVar

T = TypeVar("T")


def retry(fn: Callable[[], T], tries: int = 4, delay: float = 2.0) -> T:
    """东财接口偶发断连，指数退避重试。"""
    for attempt in range(tries):
        try:
            return fn()
        except Exception:  # noqa: BLE001
            if attempt == tries - 1:
                raise
            time.sleep(delay * (2**attempt))
    raise RuntimeError("unreachable")
