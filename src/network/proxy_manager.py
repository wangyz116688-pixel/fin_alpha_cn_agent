"""Proxy rotation and retry utilities for outbound HTTP calls."""

from __future__ import annotations

import os
import random
import time
from contextlib import contextmanager
from itertools import cycle
from typing import Callable, Iterable, List, Optional


class ProxyManager:
    """Apply rotating proxies with retry/backoff semantics."""

    def __init__(
        self,
        proxies: Optional[Iterable[str]],
        *,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 15.0,
        jitter: float = 0.5,
        logger=None,
    ) -> None:
        proxy_list = [p.strip() for p in (proxies or []) if p.strip()]
        self._proxies: List[str] = proxy_list
        self._cycle = cycle(proxy_list) if proxy_list else None
        self.max_attempts = max(1, max_attempts)
        self.base_delay = max(0.0, base_delay)
        self.max_delay = max(self.base_delay, max_delay)
        self.jitter = max(0.0, jitter)
        self.logger = logger

    @classmethod
    def from_env(cls, logger=None) -> "ProxyManager":
        force_direct = os.getenv("AKSHARE_PROXY_FORCE_DIRECT", "false").lower() in (
            "1",
            "true",
            "yes",
        )
        if force_direct:
            proxies = ["direct"]
        else:
            raw = os.getenv("AKSHARE_PROXY_LIST", "").replace(";", ",")
            proxies = [item.strip() for item in raw.split(",") if item.strip()]
            allow_direct = os.getenv("AKSHARE_PROXY_ALLOW_DIRECT", "true").lower() in (
                "1",
                "true",
                "yes",
            )
            if allow_direct and "direct" not in [p.lower() for p in proxies]:
                proxies.insert(0, "direct")

        max_attempts = int(os.getenv("AKSHARE_PROXY_MAX_ATTEMPTS", "3"))
        base_delay = float(os.getenv("AKSHARE_PROXY_BASE_DELAY", "1"))
        max_delay = float(os.getenv("AKSHARE_PROXY_MAX_DELAY", "15"))
        jitter = float(os.getenv("AKSHARE_PROXY_JITTER", "0.5"))
        return cls(
            proxies,
            max_attempts=max_attempts,
            base_delay=base_delay,
            max_delay=max_delay,
            jitter=jitter,
            logger=logger,
        )

    def _log(self, level: str, message: str) -> None:
        if self.logger:
            getattr(self.logger, level, self.logger.info)(message)

    def _next_proxy(self) -> Optional[str]:
        if not self._cycle:
            return None
        return next(self._cycle)

    @contextmanager
    def _apply_proxy(self, proxy: Optional[str]):
        previous = {
            key: os.environ.get(key)
            for key in ("HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY")
        }
        try:
            if proxy and proxy.lower() != "direct":
                os.environ["HTTP_PROXY"] = proxy
                os.environ["HTTPS_PROXY"] = proxy
                os.environ.pop("NO_PROXY", None)
            else:
                for key in ("HTTP_PROXY", "HTTPS_PROXY"):
                    os.environ.pop(key, None)
                os.environ["NO_PROXY"] = "*"
            yield proxy
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def _sleep(self, attempt: int) -> None:
        delay = min(self.base_delay * (2 ** (attempt - 1)), self.max_delay)
        if self.jitter:
            delay += random.uniform(0, self.jitter)
        time.sleep(delay)

    def run(self, func: Callable[[], any], label: str):
        last_exc: Optional[Exception] = None
        for attempt in range(1, self.max_attempts + 1):
            proxy = self._next_proxy()
            with self._apply_proxy(proxy):
                try:
                    return func()
                except Exception as exc:  # noqa: BLE001
                    last_exc = exc
                    self._log(
                        "warning",
                        f"{label} attempt {attempt}/{self.max_attempts} failed "
                        f"using proxy {proxy or 'DIRECT'}: {exc}",
                    )
            if attempt < self.max_attempts:
                self._sleep(attempt)
        self._log("error", f"{label} failed after {self.max_attempts} attempts.")
        if last_exc:
            raise last_exc
        raise RuntimeError(f"{label} failed without exception detail.")


# Shared instance for AkShare flows
proxy_manager = ProxyManager.from_env()


def _patch_akshare_no_proxy() -> None:
    """
    Patch akshare's internal request_with_retry to use trust_env=False so that
    Windows registry / system proxy settings (e.g. Clash on 127.0.0.1:7897) are
    ignored when calling domestic Chinese API endpoints that don't need a proxy.
    """
    try:
        import requests
        from requests.adapters import HTTPAdapter
        import akshare.utils.request as _ak_req

        def _patched_request_with_retry(
            url,
            params=None,
            timeout=15,
            max_retries=3,
            base_delay=1.0,
            random_delay_range=(0.5, 1.5),
        ):
            import random as _random
            import time as _time

            last_exception = None
            for attempt in range(max_retries):
                try:
                    with requests.Session() as session:
                        session.trust_env = False  # ignore system/env proxy
                        session.headers.update({
                            "User-Agent": (
                                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                                "AppleWebKit/537.36 (KHTML, like Gecko) "
                                "Chrome/120.0.0.0 Safari/537.36"
                            ),
                            "Referer": "https://quote.eastmoney.com/",
                        })
                        adapter = HTTPAdapter(pool_connections=1, pool_maxsize=1)
                        session.mount("http://", adapter)
                        session.mount("https://", adapter)
                        response = session.get(url, params=params, timeout=timeout)
                        response.raise_for_status()
                        return response
                except (requests.RequestException, ValueError) as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        delay = base_delay * (2**attempt) + _random.uniform(*random_delay_range)
                        _time.sleep(delay)
            raise last_exception

        _ak_req.request_with_retry = _patched_request_with_retry
        # Also patch the already-imported reference in func.py
        import akshare.utils.func as _ak_func
        _ak_func.request_with_retry = _patched_request_with_retry
    except Exception:
        pass


_patch_akshare_no_proxy()
