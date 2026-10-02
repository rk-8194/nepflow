"""Injectable Materials Project HTTP/client construction boundary."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from requests import Session


class TimeoutSession(Session):
    """requests session that applies a default timeout to every request."""

    def __init__(self, timeout: float) -> None:
        super().__init__()
        self._timeout = timeout

    def request(self, method: str, url: str, **kwargs: Any):  # type: ignore[override]
        kwargs.setdefault("timeout", self._timeout)
        return super().request(method, url, **kwargs)


def build_default_client(api_key: str, request_timeout: float) -> Any:
    """Construct the live MP client only at the adapter boundary."""

    try:
        from mp_api.client import MPRester
    except ImportError as exc:
        raise ImportError("Materials Project generation requires the mp-api package") from exc
    session = TimeoutSession(request_timeout)
    session.headers.update({"X-API-KEY": api_key})
    return MPRester(
        api_key,
        session=session,
        headers={"X-API-KEY": api_key},
        timeout=request_timeout,
    )


class MaterialsProjectClient:
    """Lazy, injectable client adapter for deterministic query tests."""

    def __init__(
        self,
        api_key: str,
        request_timeout: float = 60.0,
        *,
        client: Any | None = None,
        client_factory: Callable[[str, float], Any] | None = None,
    ) -> None:
        self.api_key = api_key
        self.request_timeout = request_timeout
        self._client = client
        self._client_factory = client_factory or build_default_client

    @property
    def client(self) -> Any:
        if self._client is None:
            self._client = self._client_factory(self.api_key, self.request_timeout)
        return self._client
