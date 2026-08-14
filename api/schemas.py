"""Pydantic API contracts."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class HistoryTurn(BaseModel):
    role: str
    content: str = Field(..., min_length=1, max_length=8000)

    @field_validator("role")
    @classmethod
    def validate_role(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in {"user", "assistant"}:
            raise ValueError("role must be 'user' or 'assistant'")
        return normalized


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    session_id: Optional[str] = Field(default=None, max_length=128)
    age_verified: bool = False
    declared_age: Optional[int] = Field(default=None, ge=0, le=120)
    history: List[HistoryTurn] = Field(default_factory=list)


class Citation(BaseModel):
    title: str
    url: str
    score: float


class ChatResponse(BaseModel):
    answer: str
    blocked: bool
    policy: str
    confidence: Optional[float] = None
    citations: List[Citation] = Field(default_factory=list)
    session_id: str
    redirect_url: Optional[str] = None


class RetrieveRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4000)
    session_id: Optional[str] = Field(default=None, max_length=128)
    age_verified: bool = False
    declared_age: Optional[int] = Field(default=None, ge=0, le=120)
    top_k: Optional[int] = Field(default=None, ge=1, le=50)


class RetrieveResponse(BaseModel):
    query: str
    blocked: bool = False
    policy: str = "allow"
    confidence: float
    chunks: List[Dict[str, Any]]
    session_id: str
    answer: Optional[str] = None


class IngestRequest(BaseModel):
    urls: Optional[List[str]] = None
    recreate: bool = False
    crawl_only: bool = False


class IngestResponse(BaseModel):
    documents: int
    chunks: int
    upserted: int


class ErrorBody(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    error: ErrorBody
