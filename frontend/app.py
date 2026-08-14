"""Pernod Ricard RAG chatbot — Streamlit UI."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from config.settings import get_settings
from frontend.client import BackendClient, BackendError

SETTINGS = get_settings()
CLIENT = BackendClient(SETTINGS)

SUGGESTED_QUESTIONS = (
    ("Product knowledge", "What is Absolut vodka and where is it produced?"),
    ("History", "What is the heritage of Jameson Irish whiskey?"),
    ("Cocktail recipe", "How do I serve a Beefeater gin and tonic?"),
)

THEME_CSS = """
<style>
    .stApp { background: #0b0d12; color: #f4efe6; }
    header[data-testid="stHeader"] { background: #0b0d12; }
    [data-testid="stSidebar"] { background: #11141c; border-right: 1px solid #2a2418; }
    .pr-hero { font-family: Georgia, "Times New Roman", serif; letter-spacing: 0.08em;
               text-transform: uppercase; color: #c4a35a; font-size: 0.82rem; margin-bottom: 0.2rem; }
    .pr-title { font-family: Georgia, "Times New Roman", serif; color: #f4efe6;
                font-size: 1.8rem; margin: 0 0 0.4rem 0; }
    .pr-sub { color: #b7b1a6; font-size: 0.95rem; margin-bottom: 1.2rem; }
    .confidence-badge { display: inline-block; padding: 0.2rem 0.7rem; border-radius: 999px;
                        border: 1px solid #c4a35a; color: #c4a35a; font-size: 0.8rem; }
    .citation-chip a { color: #c4a35a; text-decoration: none; }
    .citation-chip a:hover { text-decoration: underline; }
    .user-bubble, .assistant-bubble { padding: 0.9rem 1rem; border-radius: 12px; margin: 0.35rem 0; }
    .user-bubble { background: #1c2333; border: 1px solid #2c3548; }
    .assistant-bubble { background: #141821; border: 1px solid #2a2418; }
    .error-box { background: #2a1515; border: 1px solid #8a3b3b; padding: 0.8rem 1rem; border-radius: 10px; }
    @media (max-width: 768px) {
        .pr-title { font-size: 1.35rem; }
        [data-testid="stSidebar"] { min-width: 100%; }
    }
</style>
"""


def _init_state() -> None:
    defaults: dict[str, Any] = {
        "age_verified": False,
        "declared_age": None,
        "session_id": None,
        "messages": [],
        "last_user_message": None,
        "last_error": None,
        "last_confidence": None,
        "last_citations": [],
        "preview_source": None,
        "pending_prompt": None,
        "drawer_open": True,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _history_payload() -> list[dict[str, str]]:
    payload: list[dict[str, str]] = []
    for item in st.session_state.messages:
        if item.get("role") in {"user", "assistant"} and item.get("content"):
            payload.append({"role": item["role"], "content": item["content"]})
    return payload[-12:]


def _render_age_gate() -> None:
    st.markdown('<div class="pr-hero">Pernod Ricard</div>', unsafe_allow_html=True)
    st.markdown('<div class="pr-title">Age verification</div>', unsafe_allow_html=True)
    st.markdown(
        f"This assistant is intended for adults of legal drinking age "
        f"({SETTINGS.minimum_legal_drinking_age}+)."
    )
    with st.form("age_gate_form"):
        declared_age = st.number_input(
            "Your age",
            min_value=0,
            max_value=120,
            value=SETTINGS.minimum_legal_drinking_age,
            step=1,
        )
        confirmed = st.checkbox("I confirm that I am of legal drinking age.")
        submitted = st.form_submit_button("Enter")
    if not submitted:
        st.stop()
    if not confirmed or int(declared_age) < SETTINGS.minimum_legal_drinking_age:
        st.error("Access is restricted to adults of legal drinking age.")
        st.stop()
    st.session_state.age_verified = True
    st.session_state.declared_age = int(declared_age)
    st.rerun()


def _confidence_label(value: Optional[float]) -> str:
    if value is None:
        return "Confidence unavailable"
    pct = int(round(value * 100))
    if value >= 0.75:
        band = "High"
    elif value >= 0.5:
        band = "Moderate"
    else:
        band = "Low"
    return f"{band} retrieval confidence · {pct}%"


def _render_citations(citations: list[dict[str, Any]], *, key_prefix: str) -> None:
    if not citations:
        st.caption("No citations for this reply.")
        return
    for index, citation in enumerate(citations):
        title = citation.get("title") or "Source"
        url = citation.get("url") or ""
        score = citation.get("score")
        score_text = f"{float(score):.2f}" if score is not None else "—"
        cols = st.columns([4, 1])
        with cols[0]:
            if url:
                st.markdown(
                    f'<div class="citation-chip">[{index + 1}] '
                    f'<a href="{url}" target="_blank" rel="noopener">{title}</a> · score {score_text}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(f"[{index + 1}] {title} · score {score_text}")
        with cols[1]:
            button_key = f"preview-{key_prefix}-{index}-{url}"
            if st.button("Preview", key=button_key):
                st.session_state.preview_source = citation


def _load_preview_content(citation: dict[str, Any]) -> str:
    url = citation.get("url") or ""
    try:
        retrieved = CLIENT.retrieve(
            st.session_state.last_user_message or "",
            session_id=st.session_state.session_id,
            age_verified=True,
            declared_age=st.session_state.declared_age,
        )
    except BackendError:
        return "Source preview is unavailable."
    for chunk in retrieved.get("chunks") or []:
        if chunk.get("url") == url:
            return str(chunk.get("content") or "No excerpt available.")
    return "No matching excerpt was returned for this source."


def _render_preview_modal() -> None:
    citation = st.session_state.preview_source
    if not citation:
        return
    title = citation.get("title") or "Source"
    url = citation.get("url") or ""
    score = citation.get("score")
    with st.expander(f"Source preview · {title}", expanded=True):
        if url:
            st.markdown(f"[{url}]({url})")
        if score is not None:
            st.caption(f"Retrieval score: {float(score):.3f}")
        st.write(_load_preview_content(citation))
        if st.button("Close preview", key="close-source-preview"):
            st.session_state.preview_source = None
            st.rerun()


def _send_prompt(prompt: str) -> None:
    st.session_state.last_user_message = prompt
    st.session_state.last_error = None
    st.session_state.messages.append({"role": "user", "content": prompt})
    assistant_slot: dict[str, Any] = {
        "role": "assistant",
        "content": "",
        "citations": [],
        "confidence": None,
        "blocked": False,
        "policy": "allow",
    }
    with st.chat_message("user"):
        st.markdown(f'<div class="user-bubble">{prompt}</div>', unsafe_allow_html=True)
    with st.chat_message("assistant"):
        stream_box = st.empty()
        status = st.status("Retrieving sources and composing a reply...", expanded=True)
        collected = ""
        try:
            for event in CLIENT.chat_stream(
                prompt,
                session_id=st.session_state.session_id,
                age_verified=True,
                declared_age=st.session_state.declared_age,
                history=_history_payload(),
            ):
                event_type = event.get("type")
                if event_type == "blocked":
                    assistant_slot["content"] = event.get("answer") or "This request cannot be completed."
                    assistant_slot["blocked"] = True
                    assistant_slot["policy"] = event.get("policy") or "blocked"
                    st.session_state.session_id = event.get("session_id") or st.session_state.session_id
                    stream_box.markdown(assistant_slot["content"])
                    break
                if event_type == "meta":
                    assistant_slot["citations"] = event.get("citations") or []
                    assistant_slot["confidence"] = event.get("confidence")
                    st.session_state.last_citations = assistant_slot["citations"]
                    st.session_state.last_confidence = assistant_slot["confidence"]
                    st.session_state.session_id = event.get("session_id") or st.session_state.session_id
                    status.update(label="Generating answer...", state="running")
                elif event_type == "token":
                    collected += str(event.get("text") or "")
                    assistant_slot["content"] = collected
                    stream_box.markdown(collected)
                elif event_type == "done":
                    assistant_slot["content"] = event.get("answer") or collected
                    assistant_slot["policy"] = event.get("policy") or assistant_slot["policy"]
                    st.session_state.session_id = event.get("session_id") or st.session_state.session_id
                    stream_box.markdown(assistant_slot["content"])
                elif event_type == "error":
                    raise BackendError(str(event.get("message") or "Streaming failed."))
            if not assistant_slot["content"]:
                fallback = CLIENT.chat(
                    prompt,
                    session_id=st.session_state.session_id,
                    age_verified=True,
                    declared_age=st.session_state.declared_age,
                    history=_history_payload(),
                )
                assistant_slot["content"] = fallback.get("answer") or ""
                assistant_slot["citations"] = fallback.get("citations") or []
                assistant_slot["confidence"] = fallback.get("confidence")
                assistant_slot["blocked"] = bool(fallback.get("blocked"))
                assistant_slot["policy"] = fallback.get("policy") or "allow"
                st.session_state.session_id = fallback.get("session_id") or st.session_state.session_id
                st.session_state.last_citations = assistant_slot["citations"]
                st.session_state.last_confidence = assistant_slot["confidence"]
                stream_box.markdown(assistant_slot["content"])
            status.update(label="Reply ready", state="complete")
        except BackendError as exc:
            st.session_state.last_error = str(exc)
            assistant_slot["content"] = "The assistant could not complete that request. Use Retry to try again."
            assistant_slot["blocked"] = True
            assistant_slot["policy"] = "error"
            stream_box.markdown(assistant_slot["content"])
            status.update(label="Request failed", state="error")
        except Exception:
            st.session_state.last_error = "An unexpected interface error occurred."
            assistant_slot["content"] = "Something went wrong while rendering the reply. Use Retry to try again."
            assistant_slot["policy"] = "error"
            stream_box.markdown(assistant_slot["content"])
            status.update(label="Request failed", state="error")
    st.session_state.messages.append(assistant_slot)


def main() -> None:
    st.set_page_config(
        page_title="Pernod Ricard Assistant",
        page_icon="◆",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(THEME_CSS, unsafe_allow_html=True)
    _init_state()

    if not st.session_state.age_verified:
        _render_age_gate()

    with st.sidebar:
        st.markdown('<div class="pr-hero">Collection</div>', unsafe_allow_html=True)
        st.markdown("**Citation drawer**")
        st.caption("Inspect every source used in the latest answer.")
        if st.session_state.last_confidence is not None:
            st.markdown(
                f'<span class="confidence-badge">{_confidence_label(st.session_state.last_confidence)}</span>',
                unsafe_allow_html=True,
            )
        _render_citations(st.session_state.last_citations, key_prefix="drawer")
        st.divider()
        if st.button("New conversation"):
            st.session_state.messages = []
            st.session_state.session_id = None
            st.session_state.last_citations = []
            st.session_state.last_confidence = None
            st.session_state.last_error = None
            st.rerun()
        st.caption("Backend: " + SETTINGS.backend_url)

    st.markdown('<div class="pr-hero">Pernod Ricard</div>', unsafe_allow_html=True)
    st.markdown('<div class="pr-title">Brand assistant</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="pr-sub">Premium guidance on group brands, heritage, and serves. '
        "Always drink responsibly.</div>",
        unsafe_allow_html=True,
    )

    if st.session_state.last_error:
        st.markdown(f'<div class="error-box">{st.session_state.last_error}</div>', unsafe_allow_html=True)
        if st.button("Retry last question") and st.session_state.last_user_message:
            if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
                st.session_state.messages.pop()
            if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
                st.session_state.messages.pop()
            _send_prompt(st.session_state.last_user_message)
            st.rerun()

    st.markdown("**Suggested questions**")
    suggestion_cols = st.columns(3)
    for column, (label, question) in zip(suggestion_cols, SUGGESTED_QUESTIONS, strict=True):
        with column:
            if st.button(label, use_container_width=True):
                st.session_state.pending_prompt = question

    for message_index, message in enumerate(st.session_state.messages):
        role = message.get("role")
        css = "user-bubble" if role == "user" else "assistant-bubble"
        with st.chat_message("user" if role == "user" else "assistant"):
            st.markdown(f'<div class="{css}">{message.get("content", "")}</div>', unsafe_allow_html=True)
            if role == "assistant":
                confidence = message.get("confidence")
                if confidence is not None:
                    st.caption(_confidence_label(confidence))
                citations = message.get("citations") or []
                if citations:
                    with st.expander("Sources"):
                        _render_citations(citations, key_prefix=f"history-{message_index}")

    _render_preview_modal()

    pending = st.session_state.pending_prompt
    prompt = st.chat_input("Ask about a Pernod Ricard brand...")
    if pending:
        st.session_state.pending_prompt = None
        _send_prompt(pending)
        st.rerun()
    elif prompt:
        _send_prompt(prompt)
        st.rerun()


if __name__ == "__main__":
    main()
