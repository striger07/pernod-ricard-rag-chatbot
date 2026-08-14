"""HTTP client for the FastAPI backend."""

from __future__ import annotations

import json
from typing import Any, Iterator, Optional

import httpx

from config.settings import Settings, get_settings


class BackendError(Exception):
    """Raised when the FastAPI backend cannot complete a request."""

    def __init__(self, message: str, status_code: Optional[int] = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class BackendClient:
    def __init__(self, settings: Optional[Settings] = None, timeout: float = 120.0) -> None:
        self.settings = settings or get_settings()
        self.base_url = self.settings.backend_url.rstrip("/")
        self.timeout = timeout

    def _headers(self, *, age_verified: bool) -> dict[str, str]:
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if age_verified:
            headers[self.settings.age_verification_header] = "true"
        return headers

    def health(self) -> dict[str, Any]:
        try:
            response = httpx.get(f"{self.base_url}/health", timeout=10.0)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as exc:
            raise BackendError("The assistant service is unavailable.") from exc

    def chat(
        self,
        message: str,
        *,
        session_id: Optional[str],
        age_verified: bool,
        declared_age: Optional[int],
        history: list[dict[str, str]],
    ) -> dict[str, Any]:
        payload = {
            "message": message,
            "session_id": session_id,
            "age_verified": age_verified,
            "declared_age": declared_age,
            "history": history,
        }
        try:
            response = httpx.post(
                f"{self.base_url}/chat",
                json=payload,
                headers=self._headers(age_verified=age_verified),
                timeout=self.timeout,
            )
        except httpx.HTTPError as exc:
            raise BackendError("Unable to reach the assistant service.") from exc
        if response.status_code >= 500:
            raise BackendError("The assistant service returned an error.", status_code=response.status_code)
        try:
            body = response.json()
        except json.JSONDecodeError as exc:
            raise BackendError("The assistant service returned an unreadable response.") from exc
        if "error" in body and "answer" not in body:
            error = body.get("error")
            if isinstance(error, dict):
                message_text = error.get("message", "Request failed")
            else:
                message_text = "Request failed"
            raise BackendError(str(message_text), status_code=response.status_code)
        return body

    def chat_stream(
        self,
        message: str,
        *,
        session_id: Optional[str],
        age_verified: bool,
        declared_age: Optional[int],
        history: list[dict[str, str]],
    ) -> Iterator[dict[str, Any]]:
        payload = {
            "message": message,
            "session_id": session_id,
            "age_verified": age_verified,
            "declared_age": declared_age,
            "history": history,
        }
        try:
            with httpx.stream(
                "POST",
                f"{self.base_url}/chat/stream",
                json=payload,
                headers=self._headers(age_verified=age_verified),
                timeout=self.timeout,
            ) as response:
                if response.status_code >= 500:
                    raise BackendError("Streaming is unavailable.", status_code=response.status_code)
                for line in response.iter_lines():
                    if not line:
                        continue
                    data = line[6:] if line.startswith("data: ") else line
                    try:
                        yield json.loads(data)
                    except json.JSONDecodeError:
                        continue
        except BackendError:
            raise
        except httpx.HTTPError as exc:
            raise BackendError("Streaming connection failed.") from exc

    def retrieve(
        self,
        query: str,
        *,
        session_id: Optional[str],
        age_verified: bool,
        declared_age: Optional[int],
    ) -> dict[str, Any]:
        payload = {
            "query": query,
            "session_id": session_id,
            "age_verified": age_verified,
            "declared_age": declared_age,
        }
        try:
            response = httpx.post(
                f"{self.base_url}/retrieve",
                json=payload,
                headers=self._headers(age_verified=age_verified),
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as exc:
            raise BackendError("Unable to load source previews.") from exc
