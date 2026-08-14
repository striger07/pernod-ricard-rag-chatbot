from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from config.settings import Settings
from rag.llm_service import GrokLLMService, LLMServiceError


def _settings() -> Settings:
    return Settings(grok_api_key="test-key", grok_api_base="https://api.x.ai/v1", grok_model="grok-3", _env_file=None)


@pytest.mark.asyncio
async def test_generate_returns_message():
    client = MagicMock()
    client.chat.completions.create = AsyncMock(
        return_value=SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="Absolut is a Swedish vodka."))]
        )
    )
    service = GrokLLMService(settings=_settings(), client=client)
    text = await service.generate([{"role": "user", "content": "What is Absolut?"}])
    assert text == "Absolut is a Swedish vodka."
    kwargs = client.chat.completions.create.await_args.kwargs
    assert kwargs["model"] == "grok-3"
    assert kwargs["stream"] is False
    assert kwargs["messages"][0]["role"] == "user"


@pytest.mark.asyncio
async def test_generate_stream_yields_deltas():
    async def _stream():
        yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="Abso"))])
        yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="lut"))])
        yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=None))])

    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=_stream())
    service = GrokLLMService(settings=_settings(), client=client)
    parts = [part async for part in service.generate_stream([{"role": "user", "content": "Name the brand"}])]
    assert "".join(parts) == "Absolut"


@pytest.mark.asyncio
async def test_generate_rejects_empty_messages():
    service = GrokLLMService(settings=_settings(), client=MagicMock())
    with pytest.raises(LLMServiceError):
        await service.generate([])


def test_missing_api_key_raises():
    service = GrokLLMService(settings=Settings(grok_api_key="", _env_file=None))
    with pytest.raises(LLMServiceError):
        _ = service.client
