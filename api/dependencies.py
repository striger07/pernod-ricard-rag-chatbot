"""FastAPI dependencies."""

from __future__ import annotations

from fastapi import Depends, Header, HTTPException, Request, status

from config.settings import Settings, get_settings
from rag.engine import RagEngine


def get_engine(request: Request) -> RagEngine:
    engine = getattr(request.app.state, "engine", None)
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine is not initialised")
    return engine


def age_header(
    request: Request,
    settings: Settings = Depends(get_settings),
    x_age_verified: str | None = Header(default=None, alias="X-Age-Verified"),
) -> bool:
    header_name = settings.age_verification_header
    raw = request.headers.get(header_name) or x_age_verified
    if not raw:
        return False
    return raw.strip().lower() in {"1", "true", "yes", "verified"}


def require_admin(
    settings: Settings = Depends(get_settings),
    x_admin_token: str | None = Header(default=None, alias="X-Admin-Token"),
    authorization: str | None = Header(default=None),
) -> None:
    expected = (settings.admin_api_token or "").strip()
    if not expected:
        if settings.is_production:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Ingestion is not configured",
            )
        return
    provided = (x_admin_token or "").strip()
    if not provided and authorization:
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() == "bearer":
            provided = token.strip()
    if provided != expected:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid admin token")
