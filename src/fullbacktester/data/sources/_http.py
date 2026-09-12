"""Plumbing shared by sources that sign requests with the user's own credentials.

Credentials come from the caller or the environment at fetch time. They are never
written to disk, cached, logged, or put in an error message: errors name the
variable to set, never its value.
"""

from __future__ import annotations

import os
import re
import time
from collections.abc import Mapping
from typing import Any, Protocol, cast


class CredentialError(RuntimeError):
    """A credential is missing, or the provider rejected it."""


class HttpResponse(Protocol):
    status_code: int
    headers: Mapping[str, str]
    content: bytes
    text: str

    def json(self) -> Any: ...


class HttpSession(Protocol):
    def get(
        self,
        url: str,
        *,
        headers: Mapping[str, str] | None = ...,
        params: Mapping[str, Any] | None = ...,
        timeout: float = ...,
    ) -> HttpResponse: ...


def require_credential(
    explicit: str | None, env_var: str, *, provider: str, how_to_get: str
) -> str:
    """Return the credential, or explain exactly how to supply it."""
    value = explicit if explicit is not None else os.environ.get(env_var)
    if value is None or not value.strip():
        raise CredentialError(
            f"{provider} needs {env_var}, which is not set. {how_to_get} "
            "Then store it for your user account so scheduled jobs see it too. "
            f"PowerShell: [Environment]::SetEnvironmentVariable('{env_var}', '<value>', 'User')  "
            f"bash: echo 'export {env_var}=<value>' >> ~/.bashrc"
        )
    return value.strip()


def default_session(extra: str) -> HttpSession:
    try:
        import requests
    except ImportError as exc:
        raise ImportError(
            f"requests is not installed; run: pip install 'fullbacktester[{extra}]'"
        ) from exc
    return cast(HttpSession, requests.Session())


def get_with_retries(
    session: HttpSession,
    url: str,
    *,
    headers: Mapping[str, str],
    params: Mapping[str, Any] | None = None,
    timeout: float,
    max_retries: int = 4,
) -> HttpResponse:
    """GET ``url``, retrying rate limits (429) and server errors (5xx) with backoff.

    Honours a numeric ``Retry-After``. Every other status is returned for the caller
    to interpret, so each provider can give its own advice about what went wrong.
    """
    attempt = 0
    while True:
        response = session.get(url, headers=headers, params=params, timeout=timeout)
        retryable = response.status_code == 429 or response.status_code >= 500
        if not retryable or attempt >= max_retries:
            return response
        time.sleep(_backoff_seconds(response, attempt))
        attempt += 1


def error_message(response: HttpResponse) -> str:
    """The provider's own explanation, from whichever field it uses."""
    try:
        body = response.json()
    except ValueError:
        # Gateways answer auth failures with an HTML page; keep its title, not the markup.
        text = response.text.strip()
        title = re.search(r"<title>(.*?)</title>", text, re.IGNORECASE | re.DOTALL)
        plain = title.group(1) if title else re.sub(r"<[^>]+>", " ", text)
        return " ".join(plain.split())[:200] or "no response body"
    if isinstance(body, dict):
        if isinstance(body.get("message"), str):
            return str(body["message"])
        errors = body.get("errors")
        if isinstance(errors, list) and errors and isinstance(errors[0], dict):
            return str(errors[0].get("message") or errors[0])
    return str(body)[:200]


def _backoff_seconds(response: HttpResponse, attempt: int) -> float:
    retry_after = response.headers.get("Retry-After") if response.headers else None
    if retry_after is not None:
        try:
            return min(float(retry_after), 60.0)
        except ValueError:
            pass
    return float(min(2**attempt, 30))
