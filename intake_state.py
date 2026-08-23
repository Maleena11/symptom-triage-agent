"""Explicit, session-scoped clinical intake tracking."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re


INTAKE_FIELDS = ("onset_duration", "severity", "progression", "red_flags")
QUESTIONS = {
    "onset_duration": "When did this start, and how long has it been going on?",
    "severity": "How severe is it right now: mild, moderate, severe, or a number from 1 to 10?",
    "progression": "Is it improving, staying about the same, or getting worse?",
    "red_flags": "Before I assess urgency, are you having warning signs such as trouble breathing, chest pain or pressure, fainting, new one-sided weakness or speech trouble, severe bleeding, or swelling of the face or throat?",
}


@dataclass
class ClinicalIntake:
    onset_duration: str | None = None
    severity: str | None = None
    progression: str | None = None
    red_flags: str | None = None
    pending_field: str | None = None

    def next_field(self) -> str | None:
        return next((field for field in INTAKE_FIELDS if getattr(self, field) is None), None)

    def next_question(self) -> str | None:
        field = self.next_field()
        return QUESTIONS[field] if field else None

    def is_complete(self) -> bool:
        return self.next_field() is None

    def to_prompt_context(self) -> str:
        def display(value: str | None) -> str:
            return value if value is not None else "Not yet collected"

        lines = ["Clinical intake state (application-maintained; do not infer missing values):"]
        lines.extend(f"- {field.replace('_', ' ').title()}: {display(getattr(self, field))}" for field in INTAKE_FIELDS)
        if question := self.next_question():
            lines.append(f"- Suggested next uncollected question (ask only if needed for triage): {question}")
        else:
            lines.append("- All tracked intake fields are collected; give the appropriate triage response.")
        return "\n".join(lines)

    def as_dict(self) -> dict[str, str | None]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, str | None]) -> "ClinicalIntake":
        return cls(**{field: value.get(field) for field in (*INTAKE_FIELDS, "pending_field")})


ONSET_PATTERN = re.compile(r"\b(?:since|for|started|began|onset)\b.{0,60}|\b(?:today|yesterday|last night|this morning|\d+\s*(?:minutes?|hours?|days?|weeks?|months?)\s*ago)\b", re.IGNORECASE)
SEVERITY_SCORE_PATTERN = re.compile(r"\b(10|[1-9])\s*(?:/|out of)\s*10\b", re.IGNORECASE)
SEVERITY_WORD_PATTERN = re.compile(r"\b(mild|moderate|severe)\b", re.IGNORECASE)


def update_intake(intake: ClinicalIntake, user_text: str) -> ClinicalIntake:
    """Record facts from one user turn without guessing missing clinical facts."""
    text = " ".join(user_text.strip().split())
    lower = text.casefold()
    previous_pending = intake.pending_field
    if match := ONSET_PATTERN.search(text):
        intake.onset_duration = match.group(0)
    if match := SEVERITY_SCORE_PATTERN.search(text):
        intake.severity = f"{match.group(1)}/10"
    elif match := SEVERITY_WORD_PATTERN.search(text):
        intake.severity = match.group(1).casefold()
    if re.search(r"\b(?:worsening|worse|getting worse|progressing)\b", lower):
        intake.progression = "worsening"
    elif re.search(r"\b(?:improving|better|getting better|settling)\b", lower):
        intake.progression = "improving"
    elif re.search(r"\b(?:constant|unchanged|the same|staying the same|not changing)\b", lower):
        intake.progression = "constant"
    if previous_pending == "red_flags" or "warning sign" in lower or "red flag" in lower:
        if re.search(r"\b(?:yes|yeah|yep|i do|i am|having|some)\b", lower):
            intake.red_flags = "positive"
        elif re.search(r"\b(?:no|nope|none|not at all|i don't|i do not|without)\b", lower):
            intake.red_flags = "negative"
    intake.pending_field = intake.next_field()
    return intake
