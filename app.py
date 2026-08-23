# -*- coding: utf-8 -*-
"""Streamlit entry point for the Symptom Triage Agent."""

import os
import re

import streamlit as st
from dotenv import load_dotenv

from guardrails import emergency_guardrail_history, emergency_response
from intake_state import ClinicalIntake, track_asked_question, update_intake
from triage_agent import (
    SYSTEM_PROMPT,
    create_client,
    enforce_single_follow_up,
    extract_triage_level,
    retry_triage_response,
    stream_triage_response,
)

TRIAGE_LEVELS = {
    "emergency": {"label": "Emergency", "icon": "&#128680;", "class_name": "triage-emergency"},
    "urgent": {"label": "Urgent", "icon": "&#9888;&#65039;", "class_name": "triage-urgent"},
    "routine": {"label": "Routine", "icon": "&#128197;", "class_name": "triage-routine"},
    "self-care": {"label": "Self-Care", "icon": "&#10003;", "class_name": "triage-selfcare"},
}
DEMO_PRESETS = (
    (
        "Preset 1: Emergency",
        "Sudden chest pain radiating to left arm and shortness of breath",
        "Tests the immediate emergency guardrail.",
    ),
    (
        "Preset 2: Urgent",
        "High fever of 39.5\u00b0C for 2 days, persistent vomiting",
        "Tests care within 24 hours.",
    ),
    (
        "Preset 3: Routine",
        "Mild knee ache when walking for the past 3 weeks",
        "Tests a standard GP appointment.",
    ),
    (
        "Preset 4: Self-Care",
        "Mild runny nose and sneezing since this morning, no fever",
        "Tests home care advice.",
    ),
)

WELCOME_MESSAGE = (
    "Hello! 👋 I can help you work out how urgently you may need care. "
    "To get started, tell me about your **main symptom** and, if you can, include:\n\n"
    "- when it began\n"
    "- how severe it feels (mild, moderate, severe, or 1–10)\n"
    "- whether it is improving, staying the same, or getting worse\n"
    "- whether you have any warning signs, such as trouble breathing, chest pain or "
    "pressure, fainting, new one-sided weakness or trouble speaking, severe bleeding, "
    "or swelling of the face or throat\n\n"
    "You can answer in your own words—whatever you know is helpful."
)

st.set_page_config(page_title="Symptom Triage Agent", page_icon="🩺", layout="wide", initial_sidebar_state="expanded")


def resolve_api_key() -> str | None:
    load_dotenv()
    if key := os.getenv("GEMINI_API_KEY"):
        return key
    try:
        return st.secrets.get("GEMINI_API_KEY")
    except Exception:
        return None


def render_style() -> None:
    st.markdown("""<style>
    .header-container { background: linear-gradient(135deg,#667eea,#764ba2); padding:2rem; border-radius:10px; margin-bottom:2rem; }
    .header-title,.header-subtitle { color:white; text-align:center; } .header-title { margin:0; }
    .emergency-response { background:#ffebee; border:2px solid #f44336; border-radius:8px; padding:1.5rem; color:#000; }
    .triage-card-title { display:flex; align-items:center; gap:.6rem; padding:.8rem 1rem; border-radius:.55rem; margin-bottom:.8rem; font-size:1.05rem; font-weight:700; }
    .triage-badge { color:white; padding:.25rem .75rem; border-radius:999px; font-weight:700; font-size:.9rem; letter-spacing:.01em; }
    .triage-card-title.triage-emergency { background:#ffebee; border-left:5px solid #d32f2f; color:#8b0000; }
    .triage-card-title.triage-urgent { background:#fff3e0; border-left:5px solid #ef6c00; color:#9a4d00; }
    .triage-card-title.triage-routine { background:#e3f2fd; border-left:5px solid #1976d2; color:#0d47a1; }
    .triage-card-title.triage-selfcare { background:#e8f5e9; border-left:5px solid #388e3c; color:#1b5e20; }
    .triage-badge.triage-emergency { background:#d32f2f; } .triage-badge.triage-urgent { background:#ef6c00; }
    .triage-badge.triage-routine { background:#1976d2; } .triage-badge.triage-selfcare { background:#388e3c; }
    .emergency-response .triage-emergency { background:#d32f2f; color:white; padding:.25rem .75rem; border-radius:999px; font-weight:700; }
    </style>""", unsafe_allow_html=True)


def render_assistant_response(response: str) -> None:
    """Render triage conclusions as a color-coded card; preserve normal chat otherwise."""
    triage = extract_triage_level(response)
    if not triage:
        st.markdown(response, unsafe_allow_html=response.startswith("<div"))
        return

    level, body = triage
    config = TRIAGE_LEVELS[level]
    with st.container(border=True):
        st.markdown(
            f"<div class='triage-card-title {config['class_name']}'>"
            f"<span>{config['icon']}</span><span>Triage assessment</span>"
            f"<span class='triage-badge {config['class_name']}'>{config['label']}</span></div>",
            unsafe_allow_html=True,
        )
        if body:
            st.markdown(body)


def initialise_messages() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "assistant", "content": WELCOME_MESSAGE},
        ]
    if "clinical_intake" not in st.session_state:
        st.session_state.clinical_intake = ClinicalIntake().as_dict()


def intake_from_session() -> ClinicalIntake:
    return ClinicalIntake.from_dict(st.session_state.clinical_intake)


def safe_api_failure_response(intake: ClinicalIntake) -> str:
    """Return conservative, local guidance when Gemini is unavailable."""
    severity_score = re.search(r"\b(10|[1-9])/10\b", intake.severity or "")
    is_severe = (intake.severity or "").casefold() == "severe" or (
        severity_score is not None and int(severity_score.group(1)) >= 7
    )
    is_worsening = (intake.progression or "").casefold() == "worsening"

    if is_severe or is_worsening:
        return (
            "Triage Level: Urgent\n\n"
            "I couldn't complete the automated assessment. Because your symptoms are "
            "severe or worsening, please seek prompt advice from a qualified healthcare "
            "professional or urgent-care service now. If symptoms become life-threatening "
            "or you develop trouble breathing, chest pain, fainting, new weakness or speech "
            "trouble, severe bleeding, or face/throat swelling, call your local emergency "
            "number immediately."
        )

    if question := intake.next_question():
        return (
            "I couldn't complete the automated assessment right now. If your symptoms are "
            "severe or getting worse, seek urgent professional medical help; if you develop "
            "an emergency warning sign, call your local emergency number immediately.\n\n"
            f"To continue the intake safely: {question}"
        )

    return (
        "I couldn't complete the automated assessment, so I can't safely assign a triage "
        "level. Please contact a qualified healthcare professional for advice. Seek urgent "
        "help if symptoms are severe or worsening, and call your local emergency number "
        "immediately for any emergency warning signs."
    )


def show_and_store_assistant_response(response: str) -> None:
    st.session_state.messages.append({"role": "assistant", "content": response})
    intake = track_asked_question(intake_from_session(), response)
    st.session_state.clinical_intake = intake.as_dict()
    render_assistant_response(response)


def render_intake_status(intake: ClinicalIntake) -> None:
    labels = {"onset_duration": "Onset & duration", "severity": "Severity", "progression": "Progression", "red_flags": "Red flags screened"}
    with st.sidebar:
        st.markdown("### Clinical intake")
        for field, label in labels.items():
            value = getattr(intake, field)
            st.caption(f"{'✅' if value is not None else '◻️'} {label}: {value or 'pending'}")


def render_sidebar() -> str | None:
    selected_preset = None
    with st.sidebar:
        if st.button("🔄 Start New Assessment", use_container_width=True):
            st.session_state.clear()
            st.rerun()
        st.markdown("### Demo presets")
        for label, symptom_text, description in DEMO_PRESETS:
            if st.button(label, key=f"demo_preset_{label}", use_container_width=True):
                selected_preset = symptom_text
            st.caption(description)
        st.markdown("### Important disclaimer")
        st.error("This is not a medical diagnosis tool. Consult a qualified healthcare professional for medical advice.")
        st.markdown("### Emergency numbers")
        st.markdown("Call your local emergency number (such as **911**, **112**, or **999**) for an emergency.")
        st.markdown("### Triage levels")
        st.markdown("🔴 Emergency — immediate care\n\n🟠 Urgent — care within 24 hours\n\n🔵 Routine — standard appointment\n\n🟢 Self-care — home care and monitoring")
    return selected_preset


def main() -> None:
    render_style()
    st.markdown("<div class='header-container'><h1 class='header-title'>🩺 Symptom Triage Agent</h1><p class='header-subtitle'>Professional Symptom Assessment & Triage System</p></div>", unsafe_allow_html=True)
    initialise_messages()
    selected_preset = render_sidebar()
    render_intake_status(intake_from_session())

    for message in st.session_state.messages:
        if message["role"] != "system":
            with st.chat_message(message["role"]):
                if message["role"] == "assistant":
                    render_assistant_response(message["content"])
                else:
                    st.markdown(message["content"])

    prompt = selected_preset or st.chat_input("Type your response...")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        intake = update_intake(intake_from_session(), prompt)
        st.session_state.clinical_intake = intake.as_dict()
        result = emergency_guardrail_history(st.session_state.messages)
        with st.chat_message("user"):
            st.markdown(prompt)

        if result:
            response = emergency_response(result)
            st.session_state.messages.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                render_assistant_response(response)
            st.stop()

        api_key = resolve_api_key()
        if not api_key:
            with st.chat_message("assistant"):
                show_and_store_assistant_response(safe_api_failure_response(intake))
            st.stop()
        with st.chat_message("assistant"):
            placeholder = st.empty()
            full_response = ""
            try:
                client = create_client(api_key)
                placeholder.markdown("Assessing your symptoms...")
                for chunk in stream_triage_response(client, st.session_state.messages, intake):
                    if chunk.text:
                        full_response += chunk.text
                if not extract_triage_level(full_response):
                    placeholder.markdown("Validating the response...")
                    full_response = retry_triage_response(
                        client, st.session_state.messages, intake, full_response
                    )
                full_response = enforce_single_follow_up(full_response, intake)
                placeholder.empty()
                render_assistant_response(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                intake = track_asked_question(intake, full_response)
                st.session_state.clinical_intake = intake.as_dict()
            except Exception:
                placeholder.empty()
                show_and_store_assistant_response(safe_api_failure_response(intake))


if __name__ == "__main__":
    main()
