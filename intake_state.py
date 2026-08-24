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


ONSET_PATTERN = re.compile(
    r"\b(?:"
    r"(?:started|began)\s+(?:today|yesterday|last night|this morning|\d+\s*(?:minutes?|hours?|days?|weeks?|months?)\s+ago)"
    r"|since\s+(?:today|yesterday|last night|this morning|\d+\s*(?:minutes?|hours?|days?|weeks?|months?)\s+ago)"
    r"|for\s+(?:(?:the\s+)?past\s+)?\d+\s*(?:minutes?|hours?|days?|weeks?|months?)"
    r"|onset\s+(?:was\s+)?(?:today|yesterday|last night|this morning|\d+\s*(?:minutes?|hours?|days?|weeks?|months?)\s+ago)"
    r"|today|yesterday|last night|this morning|\d+\s*(?:minutes?|hours?|days?|weeks?|months?)\s+ago"
    r")\b",
    re.IGNORECASE,
)
SEVERITY_SCORE_PATTERN = re.compile(r"\b(10|[1-9])\s*(?:/|out of)\s*10\b", re.IGNORECASE)
SEVERITY_WORD_PATTERN = re.compile(r"\b(mild|moderate|severe)\b", re.IGNORECASE)

QUESTION_PATTERNS = {
    "onset_duration": re.compile(
        r"\b(?:when\b[^?]*(?:start|begin)|how long\b[^?]*(?:going on|had|have|last))",
        re.IGNORECASE,
    ),
    "severity": re.compile(
        r"\b(?:how severe|rate\b[^?]*(?:1\s*(?:to|[-–])\s*10|out of 10)|"
        r"(?:mild|moderate|severe)\b[^?]*(?:which|would|is it))",
        re.IGNORECASE,
    ),
    "progression": re.compile(
        r"\b(?:improving\b[^?]*(?:same|worse)|getting (?:better|worse)|"
        r"staying (?:about )?the same)",
        re.IGNORECASE,
    ),
    "red_flags": re.compile(
        r"\b(?:warning signs?|red flags?|trouble breathing|chest pain|fainting|"
        r"one-sided weakness|severe bleeding|face or throat)",
        re.IGNORECASE,
    ),
}


def track_asked_question(intake: ClinicalIntake, assistant_text: str) -> ClinicalIntake:
    """Track a supported intake field only when the assistant actually asks it."""
    questions = re.findall(r"[^?]*\?", " ".join(assistant_text.split()))
    intake.pending_field = None
    for question in reversed(questions):
        for field, pattern in QUESTION_PATTERNS.items():
            if getattr(intake, field) is None and pattern.search(question):
                intake.pending_field = field
                return intake
    return intake


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

    # Short answers often have meaning only in the context of the question just
    # asked (for example, "two days" or "pretty bad"). Preserve that answer in
    # its intended slot when explicit extraction above did not fill it.
    if text and previous_pending in INTAKE_FIELDS and getattr(intake, previous_pending) is None:
        if previous_pending != "red_flags":
            setattr(intake, previous_pending, text)
    if previous_pending == "red_flags" or "warning sign" in lower or "red flag" in lower:
        # Chat replies commonly stretch short words (for example, "nooo").
        # Treat those variants as a completed negative screen so the intake
        # does not ask the same question forever.
        negative = re.search(
            r"\b(?:no+|nope+|none|nah+|not at all|i don't|i do not|without)\b",
            lower,
        )
        explicit_red_flag = re.search(
            r"\b(?:trouble breathing|shortness of breath|chest (?:pain|pressure)|faint(?:ing|ed)?|"
            r"passed out|one[ -]sided weakness|speech trouble|severe bleeding|"
            r"(?:face|throat|tongue|lips?) swelling)\b",
            lower,
        )
        if negative:
            intake.red_flags = "negative"
        elif explicit_red_flag or re.search(r"\b(?:yes|yeah|yep|i do|i am|having|some)\b", lower):
            # Preserve the actual warning sign where possible so the model can
            # reason about it; a bare affirmative is still useful context.
            intake.red_flags = text if explicit_red_flag else "positive"
    # The reply resolves the pending question. A new pending field is recorded
    # only after inspecting the assistant's next response.
    intake.pending_field = None
    return intake
