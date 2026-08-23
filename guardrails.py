"""Low-latency deterministic emergency checks for symptom triage.

These checks intentionally cover only high-confidence, immediately dangerous
presentations. Everything else is classified by the single streamed model call.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

EMERGENCY = "Emergency"
URGENT = "Urgent"
ROUTINE = "Routine"
SELF_CARE = "Self-care"
DEFAULT_HISTORY_WINDOW = 8


@dataclass(frozen=True)
class TriageResult:
    level: str
    reason: str


def _matches(message: str, *patterns: str) -> bool:
    return any(re.search(pattern, message) for pattern in patterns)


def emergency_guardrail(text: str) -> TriageResult | None:
    """Return an emergency result for a high-confidence explicit red flag.

    This function is local, regex-based, and performs no network I/O. Semantic
    interpretation remains in the prompt used by the one streaming model call.
    """
    message = " ".join(text.casefold().replace("’", "'").split())

    # Respiratory failure / airway obstruction.
    if _matches(
        message,
        r"\b(?:i )?(?:can(?:not|'t)|cannot|can't|unable to) breathe\b",
        r"\b(?:gasping|gasping for air|turning blue|blue (?:lips|face))\b",
        r"\b(?:choking|airway blocked)\b.*\b(?:can't|cannot|unable to) (?:breathe|speak)\b",
        r"\b(?:severe|extreme) (?:shortness of breath|difficulty|trouble) breathing\b",
    ):
        return TriageResult(EMERGENCY, "Severe breathing difficulty or an obstructed airway is an emergency warning sign.")

    # Stroke: FAST signs, particularly sudden facial asymmetry, one-sided arm
    # weakness/numbness, or speech difficulty.
    if _matches(
        message,
        r"\b(?:sudden(?:ly)? )?(?:face|facial) (?:droop|drooping|weakness|numbness|asymmetry)\b",
        r"\b(?:face|facial) suddenly (?:started )?(?:to )?(?:droop|drooping)\b",
        r"\b(?:sudden(?:ly)? )?(?:one[ -]sided|left[ -]sided|right[ -]sided) (?:weakness|numbness|paralysis)\b",
        r"\b(?:sudden(?:ly)? )?(?:slurred speech|speech (?:difficulty|trouble)|trouble speaking|unable to speak)\b",
        r"\b(?:fast stroke|stroke symptoms?|having a stroke)\b",
    ):
        return TriageResult(EMERGENCY, "Possible stroke (FAST) symptoms require immediate emergency evaluation.")

    # Anaphylaxis: an allergic reaction affecting breathing or the airway.
    allergic_context = _matches(message, r"\b(?:allergic reaction|allergy|after (?:eating|a bite|.* sting)|hives)\b")
    airway_swelling = _matches(message, r"\b(?:throat|tongue|lips?|face) (?:is |are )?(?:swollen|swelling)\b", r"\bthroat (?:feels )?(?:tight|closing)\b")
    breathing_distress = _matches(message, r"\b(?:short(?:ness)? of breath|trouble breathing|difficulty breathing|wheezing)\b")
    if (allergic_context and (airway_swelling or breathing_distress)) or (airway_swelling and breathing_distress):
        return TriageResult(EMERGENCY, "A severe allergic reaction with airway or breathing symptoms is an emergency.")

    # Cardiac presentations. Chest discomfort with a serious associated symptom
    # is a high-confidence emergency pattern.
    chest_discomfort = _matches(message, r"\b(?:chest (?:pain|pressure|tightness|discomfort)|pressure (?:in|on) (?:my )?chest)\b")
    cardiac_associated = _matches(
        message,
        r"\b(?:short(?:ness)? of breath|difficulty breathing|trouble breathing)\b",
        r"\b(?:cold sweat|sweating|nausea|vomiting|dizzy|dizziness|faint(?:ing|ed)?)\b",
        r"\b(?:pain|pressure) (?:radiat(?:es|ing)|going|spreading) (?:to )?(?:my )?(?:arm|jaw|back|shoulder)\b",
    )
    if chest_discomfort and cardiac_associated:
        return TriageResult(EMERGENCY, "Chest discomfort with associated warning symptoms may be a cardiac emergency.")

    # Severe haemorrhage, including explicit uncontrolled external bleeding or
    # major internal bleeding indicators.
    if _matches(
        message,
        r"\b(?:severe|heavy|uncontrolled|profuse) bleeding\b",
        r"\bbleeding (?:won't|will not|doesn't|does not) stop\b",
        r"\b(?:spurting|gushing) blood\b",
        r"\bblood (?:is |was )?(?:spurting|gushing)\b",
        r"\b(?:vomiting|throwing up|coughing up) blood\b",
        r"\b(?:soaking|soaked) (?:a |\d+ )?(?:pad|tampon|bandage) (?:an? )?hour\b",
    ):
        return TriageResult(EMERGENCY, "Severe or uncontrolled bleeding requires immediate emergency care.")

    if _matches(message, r"\b(?:loss of consciousness|lost consciousness|unconscious|passed out)\b"):
        return TriageResult(EMERGENCY, "Loss of consciousness is an emergency warning sign.")
    return None


def emergency_guardrail_history(
    messages: Iterable[Mapping[str, str]], *, max_user_messages: int = DEFAULT_HISTORY_WINDOW
) -> TriageResult | None:
    """Check recent user turns together for emergency red-flag combinations.

    Assistant and system messages are deliberately excluded: scanning generated
    emergency advice could otherwise cause a false emergency on later turns.
    """
    if max_user_messages < 1:
        raise ValueError("max_user_messages must be at least 1")

    user_turns = [
        message.get("content", "")
        for message in messages
        if message.get("role") == "user" and message.get("content")
    ]
    return emergency_guardrail("\n".join(user_turns[-max_user_messages:]))


def emergency_response(result: TriageResult) -> str:
    """Build the non-delay clinical emergency instruction shown to the user."""
    return (
        "<div class='emergency-response'>"
        f"<h3>🚨 <span class='triage-emergency'>Triage Level: {result.level}</span></h3>"
        f"<p><strong>Reason:</strong> {result.reason}</p>"
        "<p><strong>Action Required — do not delay:</strong> Call emergency services "
        "now (111, 999, 911, or your local emergency number), or go to the nearest "
        "ER (emergency room) immediately.</p></div>"
    )
