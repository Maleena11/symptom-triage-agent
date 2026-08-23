"""Unit tests for the local emergency guardrail."""

import unittest

from guardrails import EMERGENCY, TriageResult, emergency_guardrail, emergency_guardrail_history, emergency_response
from intake_state import ClinicalIntake, QUESTIONS, update_intake
from triage_agent import SYSTEM_PROMPT
from app import DEMO_PRESETS, extract_triage_level


class EmergencyGuardrailTests(unittest.TestCase):
    def test_chest_pain_and_breathing_difficulty_is_emergency(self):
        self.assertEqual(emergency_guardrail("I have chest pain and shortness of breath.").level, EMERGENCY)

    def test_loss_of_consciousness_is_emergency(self):
        self.assertEqual(emergency_guardrail("My partner just passed out.").level, EMERGENCY)

    def test_uncontrolled_bleeding_is_emergency(self):
        self.assertEqual(emergency_guardrail("The bleeding won't stop after a cut.").level, EMERGENCY)

    def test_fast_stroke_symptom_is_emergency(self):
        self.assertEqual(emergency_guardrail("My face suddenly started drooping.").level, EMERGENCY)

    def test_anaphylaxis_with_airway_symptoms_is_emergency(self):
        self.assertEqual(emergency_guardrail("After a bee sting, my throat is swelling.").level, EMERGENCY)

    def test_cardiac_warning_pattern_is_emergency(self):
        self.assertEqual(emergency_guardrail("Chest pressure and cold sweating.").level, EMERGENCY)

    def test_respiratory_failure_pattern_is_emergency(self):
        self.assertEqual(emergency_guardrail("I can't breathe.").level, EMERGENCY)

    def test_spurting_blood_is_emergency(self):
        self.assertEqual(emergency_guardrail("Blood is spurting from my arm.").level, EMERGENCY)

    def test_non_red_flag_is_deferred_to_triage_agent(self):
        self.assertIsNone(emergency_guardrail("I have had a mild headache since this morning."))

    def test_split_turn_chest_pressure_and_breathing_difficulty_is_emergency(self):
        messages = [
            {"role": "user", "content": "I have chest pressure."},
            {"role": "assistant", "content": "When did it start?"},
            {"role": "user", "content": "I'm also short of breath."},
        ]
        self.assertEqual(emergency_guardrail_history(messages).level, EMERGENCY)

    def test_history_scan_ignores_assistant_text(self):
        messages = [
            {"role": "user", "content": "I have a mild headache."},
            {"role": "assistant", "content": "Chest pain and shortness of breath can be an emergency."},
        ]
        self.assertIsNone(emergency_guardrail_history(messages))

    def test_history_scan_respects_rolling_window(self):
        messages = [
            {"role": "user", "content": "I have chest pain."},
            {"role": "user", "content": "Actually my toe hurts."},
            {"role": "user", "content": "I am short of breath."},
        ]
        self.assertIsNone(emergency_guardrail_history(messages, max_user_messages=2))

    def test_emergency_response_requires_immediate_call_without_delay(self):
        response = emergency_response(TriageResult(EMERGENCY, "Test emergency reason."))

        self.assertIn("do not delay", response.casefold())
        self.assertIn("call emergency services now", response.casefold())
        self.assertIn("111", response)
        self.assertIn("999", response)
        self.assertIn("911", response)
        self.assertIn("er (emergency room) immediately", response.casefold())


class ClinicalIntakeTests(unittest.TestCase):
    def test_tracker_extracts_all_explicit_slots(self):
        intake = ClinicalIntake()
        update_intake(intake, "It started 2 days ago, is moderate, and is getting worse.")
        self.assertEqual(intake.onset_duration, "started 2 days ago, is moderate, and is getting worse.")
        self.assertEqual(intake.severity, "moderate")
        self.assertEqual(intake.progression, "worsening")
        self.assertEqual(intake.next_field(), "red_flags")

    def test_red_flag_screen_requires_the_screening_turn(self):
        intake = ClinicalIntake(onset_duration="today", severity="mild", progression="constant")
        intake.pending_field = intake.next_field()
        update_intake(intake, "No, none of those warning signs.")
        self.assertEqual(intake.red_flags, "negative")
        self.assertTrue(intake.is_complete())

    def test_next_question_is_first_uncollected_slot(self):
        intake = ClinicalIntake()
        self.assertEqual(intake.next_question(), QUESTIONS["onset_duration"])
        update_intake(intake, "Since yesterday")
        self.assertEqual(intake.next_question(), QUESTIONS["severity"])

    def test_prompt_context_marks_missing_question_as_optional(self):
        context = ClinicalIntake().to_prompt_context()
        self.assertIn("Suggested next uncollected question", context)
        self.assertIn("ask only if needed for triage", context)


class AdaptiveStoppingPromptTests(unittest.TestCase):
    def test_prompt_allows_a_conclusion_before_every_intake_field_is_collected(self):
        self.assertIn("Use adaptive stopping", SYSTEM_PROMPT)
        self.assertIn("Do not wait for every intake field", SYSTEM_PROMPT)
        self.assertIn("never collect a missing field merely to complete a questionnaire", SYSTEM_PROMPT)

    def test_prompt_limits_follow_up_questions_to_decision_relevant_uncertainty(self):
        self.assertIn("Ask one concise follow-up question only", SYSTEM_PROMPT)
        self.assertIn("could reasonably change the triage level or recommended action", SYSTEM_PROMPT)


class TriagePresentationTests(unittest.TestCase):
    def test_demo_presets_cover_each_triage_level_with_expected_symptoms(self):
        self.assertEqual(len(DEMO_PRESETS), 4)
        self.assertEqual(
            [label for label, _, _ in DEMO_PRESETS],
            ["Preset 1: Emergency", "Preset 2: Urgent", "Preset 3: Routine", "Preset 4: Self-Care"],
        )
        self.assertEqual(
            [symptoms for _, symptoms, _ in DEMO_PRESETS],
            [
                "Sudden chest pain radiating to left arm and shortness of breath",
                "High fever of 39.5\u00b0C for 2 days, persistent vomiting",
                "Mild knee ache when walking for the past 3 weeks",
                "Mild runny nose and sneezing since this morning, no fever",
            ],
        )

    def test_extracts_each_triage_level_and_removes_the_plain_text_label(self):
        for level, expected in (
            ("Emergency", "emergency"),
            ("Urgent", "urgent"),
            ("Routine", "routine"),
            ("Self-care", "self-care"),
        ):
            self.assertEqual(
                extract_triage_level(f"Triage Level: {level}\n\n**Next steps:** Example."),
                (expected, "**Next steps:** Example."),
            )

    def test_accepts_a_bold_triage_level_label(self):
        self.assertEqual(extract_triage_level("**Triage Level: Urgent**\n\nSee a clinician."), ("urgent", "See a clinician."))

    def test_non_conclusion_remains_regular_chat(self):
        self.assertIsNone(extract_triage_level("When did these symptoms begin?"))


class ClassificationAccuracyTests(unittest.TestCase):
    """Regression cases for non-emergency triage conclusions.

    These use representative model responses rather than calling Gemini, keeping
    the unit suite deterministic and independent of credentials or network access.
    """

    CASES = (
        (
            "Urgent",
            "I have been vomiting all day, cannot keep fluids down, and feel dizzy.",
            "Triage Level: Urgent\n\nYou should be assessed within 24 hours because dehydration is possible.",
            "urgent",
        ),
        (
            "Routine",
            "I have had recurring mild heartburn after meals for three weeks, with no warning signs.",
            "Triage Level: Routine\n\nBook a standard appointment to discuss persistent symptoms.",
            "routine",
        ),
        (
            "Self-care",
            "I have had a mild runny nose and sore throat for one day, without fever or breathing problems.",
            "Triage Level: Self-care\n\nRest, drink fluids, and seek care if symptoms worsen.",
            "self-care",
        ),
    )

    def test_representative_non_emergency_cases_have_expected_triage_level(self):
        for case_name, user_symptoms, model_response, expected_level in self.CASES:
            with self.subTest(case=case_name, symptoms=user_symptoms):
                result = extract_triage_level(model_response)
                self.assertIsNotNone(result)
                actual_level, _ = result
                self.assertEqual(actual_level, expected_level)


if __name__ == "__main__":
    unittest.main()
