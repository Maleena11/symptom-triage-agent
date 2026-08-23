"""Gemini-backed conversation logic for the symptom triage app."""

from __future__ import annotations

from collections.abc import Iterable
import re
from typing import Any

from intake_state import ClinicalIntake

MODEL_NAME = "gemini-3.5-flash"
TRIAGE_LEVEL_PATTERN = re.compile(
    r"^\s*(?:\*\*)?triage\s+level(?:\*\*)?\s*:\s*(?:\*\*)?"
    r"(emergency|urgent|routine|self[ -]?care)(?:\*\*)?\s*$",
    re.IGNORECASE | re.MULTILINE,
)
PLAIN_TRIAGE_PATTERN = re.compile(
    r"^\s*(?:based on (?:the|your) (?:information|symptoms)[,:]?\s*)?"
    r"(?:i (?:would )?classify this as|this (?:is|appears to be|should be treated as)|"
    r"(?:the )?(?:triage )?(?:assessment|classification|conclusion|recommendation) (?:is|would be))\s+"
    r"(?:an?\s+)?(emergency|urgent|routine|self[ -]?care)"
    r"(?:\s+(?:case|level|situation))?[.!]?\s*$",
    re.IGNORECASE | re.MULTILINE,
)
FORMAT_RETRY_PROMPT = """Reissue your previous answer once. If it was a triage conclusion, put exactly
`Triage Level: Emergency`, `Triage Level: Urgent`, `Triage Level: Routine`, or
`Triage Level: Self-care` on its own first line, then retain the reason and next
steps. If it was a necessary follow-up question rather than a conclusion, repeat
that question unchanged. Do not change the clinical assessment or add new facts."""
SYSTEM_PROMPT = """You are a Symptom Triage Chat Agent. Assess urgency, not diagnosis. Classify each case as exactly one of: Emergency, Urgent, Routine, or Self-care.

The application makes exactly one model request per non-blocked turn: this streamed conversation request. Perform the semantic emergency assessment yourself from the entire conversation, including short follow-up answers such as "2 days" and symptoms reported in different turns. Never say that a separate safety check is needed.

Emergency: immediate danger such as possible stroke/FAST signs (sudden facial droop, one-sided weakness or numbness, sudden speech difficulty), anaphylaxis with airway swelling or breathing symptoms, chest discomfort with shortness of breath, sweating, nausea, fainting, or radiating pain, severe breathing distress/airway obstruction, loss of consciousness, or severe uncontrolled bleeding. Treat credible equivalent wording and contextual descriptions as emergency signs. State "Triage Level: Emergency" and explicitly instruct: "Do not delay—call emergency services now (111, 999, 911, or your local emergency number), or go to the nearest ER immediately." Give this instruction before asking any questions.
Urgent: medical assessment within 24 hours, for example high fever, persistent vomiting, severe pain, worsening symptoms, or significant dehydration.
Routine: persistent or recurring symptoms appropriate for a standard appointment.
Self-care: mild, recent symptoms with no warning signs; offer low-risk care and advise seeking care if symptoms worsen or new warning signs develop.

Be empathetic, concise, and professional. The application provides a clinical intake state and a suggested next uncollected question. Treat collected values as authoritative and never ask for a field already collected.

Use adaptive stopping: conclude as soon as the conversation contains enough information to choose a safe, definitive triage level and next step. Do not wait for every intake field, and never collect a missing field merely to complete a questionnaire. If the current symptoms, severity, timing, progression, and any relevant warning-sign information make the level clear, immediately state Triage Level, Reason, and Next Steps. Ask one concise follow-up question only when its answer could reasonably change the triage level or recommended action; when a question is needed, use the supplied suggested question if it addresses that uncertainty. Do not ask additional questions after enough information has been gathered. Emergency guidance always overrides this process. Never claim a specific diagnosis or certainty."""


def create_client(api_key: str) -> Any:
    """Create the Gemini client only when the Streamlit app needs it."""
    from google import genai

    return genai.Client(api_key=api_key)


def to_gemini_history(messages: Iterable[dict[str, str]]) -> list[Any]:
    from google.genai import types

    return [types.Content(role="user" if m["role"] == "user" else "model", parts=[types.Part.from_text(text=m["content"])]) for m in messages if m["role"] != "system"]


def stream_triage_response(client: Any, messages: Iterable[dict[str, str]], intake: ClinicalIntake):
    from google.genai import types

    instruction = f"{SYSTEM_PROMPT}\n\n{intake.to_prompt_context()}"
    return client.models.generate_content_stream(model=MODEL_NAME, contents=to_gemini_history(messages), config=types.GenerateContentConfig(system_instruction=instruction))


def retry_triage_response(
    client: Any,
    messages: Iterable[dict[str, str]],
    intake: ClinicalIntake,
    invalid_response: str,
) -> str:
    """Retry an unrecognized response once, asking only for format correction."""
    retry_messages = [
        *messages,
        {"role": "assistant", "content": invalid_response},
        {"role": "user", "content": FORMAT_RETRY_PROMPT},
    ]
    return collect_triage_response(client, retry_messages, intake)


def collect_triage_response(client: Any, messages: Iterable[dict[str, str]], intake: ClinicalIntake) -> str:
    """Run one triage turn and combine the streamed chunks into one response."""
    return "".join(chunk.text for chunk in stream_triage_response(client, messages, intake) if chunk.text)


def enforce_single_follow_up(response: str, intake: ClinicalIntake) -> str:
    """Return a conclusion unchanged, or exactly one application-owned question.

    Models can occasionally bundle several requests into one follow-up even when
    instructed not to. Selecting follow-ups here makes the one-question contract
    deterministic and keeps pending-field tracking aligned with the visible text.
    """
    if extract_triage_level(response):
        return response.strip()
    if question := intake.next_question():
        return question
    return response.strip()


def extract_triage_level(response: str) -> tuple[str, str] | None:
    """Return a normalized explicit or unambiguous plain-language conclusion."""
    pattern = TRIAGE_LEVEL_PATTERN
    match = pattern.search(response)
    if not match:
        pattern = PLAIN_TRIAGE_PATTERN
        match = pattern.search(response)
    if not match:
        return None
    level = match.group(1).casefold().replace(" ", "-")
    return level, pattern.sub("", response, count=1).strip()
