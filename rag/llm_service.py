"""Groq LLM service (OpenAI-compatible Groq API). Isolated from retrieval."""

from __future__ import annotations

from typing import Any, AsyncIterator, Optional, Sequence, TypedDict

from openai import APIConnectionError, APIStatusError, APITimeoutError, AsyncOpenAI, RateLimitError
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from config.settings import Settings, get_settings
from utils.logging import get_logger

logger = get_logger(__name__)


class ChatMessage(TypedDict):
    role: str
    content: str


class LLMServiceError(Exception):
    """Raised when the Groq API cannot produce a completion."""


class GrokLLMService:
    """Async Groq wrapper using the OpenAI SDK against https://api.groq.com/openai/v1."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        client: Optional[AsyncOpenAI] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._client = client

    @property
    def client(self) -> AsyncOpenAI:
        if self._client is None:
            if not self.settings.groq_api_key:
                raise LLMServiceError("GROQ_API_KEY is not configured")
            self._client = AsyncOpenAI(
                api_key=self.settings.groq_api_key,
                base_url=self.settings.groq_api_base,
                timeout=self.settings.groq_timeout_seconds,
            )
        return self._client

    async def generate(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None,
    ) -> str:
        payload = self._validate_messages(messages)
        completion = await self._create_completion(
            payload,
            temperature=temperature,
            max_tokens=max_tokens,
            model=model,
            stream=False,
        )
        try:
            text = completion.choices[0].message.content or ""
        except (AttributeError, IndexError) as exc:
            raise LLMServiceError("Groq returned an empty completion") from exc
        cleaned = text.strip()
        if not cleaned:
            raise LLMServiceError("Groq returned an empty completion")
        logger.info(
            "groq_generate_complete",
            model=model or self.settings.groq_model,
            chars=len(cleaned),
        )
        return cleaned

    async def generate_stream(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None,
    ) -> AsyncIterator[str]:
        payload = self._validate_messages(messages)
        stream = await self._create_completion(
            payload,
            temperature=temperature,
            max_tokens=max_tokens,
            model=model,
            stream=True,
        )
        emitted = 0
        async for event in stream:
            try:
                delta = event.choices[0].delta.content
            except (AttributeError, IndexError):
                continue
            if not delta:
                continue
            emitted += len(delta)
            yield delta
        logger.info(
            "groq_stream_complete",
            model=model or self.settings.groq_model,
            chars=emitted,
        )

    async def _create_completion(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: Optional[float],
        max_tokens: Optional[int],
        model: Optional[str],
        stream: bool,
    ) -> Any:
        kwargs: dict[str, Any] = {
            "model": model or self.settings.groq_model,
            "messages": messages,
            "temperature": self.settings.groq_temperature if temperature is None else temperature,
            "max_tokens": self.settings.groq_max_tokens if max_tokens is None else max_tokens,
            "stream": stream,
        }
        try:
            async for attempt in AsyncRetrying(
                reraise=True,
                stop=stop_after_attempt(3),
                wait=wait_exponential(multiplier=0.5, min=0.5, max=8),
                retry=retry_if_exception_type(
                    (APIConnectionError, APITimeoutError, RateLimitError)
                ),
            ):
                with attempt:
                    return await self.client.chat.completions.create(**kwargs)
        except LLMServiceError:
            raise
        except APIStatusError as exc:
            logger.error("groq_api_status_error", status=getattr(exc, "status_code", None))
            raise LLMServiceError("Groq API returned an error status") from exc
        except Exception as exc:
            logger.error("groq_api_failed", error_type=type(exc).__name__)
            raise LLMServiceError("Groq API request failed") from exc

    def _validate_messages(self, messages: Sequence[ChatMessage]) -> list[dict[str, str]]:
        if not messages:
            raise LLMServiceError("messages must not be empty")
        payload: list[dict[str, str]] = []
        allowed = {"system", "user", "assistant"}
        for message in messages:
            role = str(message.get("role", "")).strip()
            content = str(message.get("content", "")).strip()
            if role not in allowed:
                raise LLMServiceError(f"Unsupported chat role: {role!r}")
            if not content:
                raise LLMServiceError("Chat message content must not be empty")
            payload.append({"role": role, "content": content})
        return payload
