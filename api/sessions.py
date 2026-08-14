"""In-memory conversation sessions with server-side age-gate state."""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock
from typing import Optional
from uuid import uuid4


@dataclass
class SessionState:
    session_id: str
    age_verified: bool = False
    history: list[dict[str, str]] = field(default_factory=list)

    def append(self, role: str, content: str, *, max_turns: int = 20) -> None:
        self.history.append({"role": role, "content": content})
        overflow = len(self.history) - max_turns * 2
        if overflow > 0:
            self.history = self.history[overflow:]


class SessionStore:
    """Process-local session store. Replace with Redis for multi-instance deployments."""

    def __init__(self) -> None:
        self._sessions: dict[str, SessionState] = {}
        self._lock = Lock()

    def get_or_create(self, session_id: Optional[str]) -> SessionState:
        with self._lock:
            if session_id and session_id in self._sessions:
                return self._sessions[session_id]
            new_id = session_id or str(uuid4())
            state = SessionState(session_id=new_id)
            self._sessions[new_id] = state
            return state
