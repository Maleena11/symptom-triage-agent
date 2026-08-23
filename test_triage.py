"""Unit tests for the local emergency guardrail."""

import unittest

from guardrails import EMERGENCY, TriageResult, emergency_guardrail, emergency_guardrail_history, emergency_response
from intake_state import ClinicalIntake, QUESTIONS, track_asked_question, update_intake
from triage_agent import SYSTEM_PROMPT, extract_triage_level
from app import DEMO_PRESETS, WELCOME_MESSAGE, safe_api_failure_response


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

    def test_high_priority_wording_variants_are_emergencies(self):
        variants = (
            "I have crushing chest pain.",
            "I'm breathless.",
            "My partner just collapsed.",
            "My stool is black and tarry.",
            "I am coughing blood.",
            "I suddenly lost my vision.",
            "I have a severe headache that came on suddenly.",
            "I want to kill myself.",
            "I'm thinking about self-harm.",
        )

        for message in variants:
            with self.subTest(message=message):
                result = emergency_guardrail(message)
                self.assertIsNotNone(result)
                self.assertEqual(result.level, EMERGENCY)

    def test_obviously_negated_self_harm_statement_is_not_emergency(self):
        self.assertIsNone(emergency_guardrail("I am not suicidal."))

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
    def test_welcome_message_requests_core_intake_details(self):
        welcome = WELCOME_MESSAGE.casefold()

        for expected in (
            "main symptom",
            "when it began",
            "how severe",
            "getting worse",
            "warning signs",
            "trouble breathing",
            "chest pain",
        ):
            with self.subTest(expected=expected):
                self.assertIn(expected, welcome)

        self.assertIn("in your own words", welcome)

    def test_tracker_extracts_all_explicit_slots(self):
        intake = ClinicalIntake()
        update_intake(intake, "It started 2 days ago, is moderate, and is getting worse.")
        self.assertEqual(intake.onset_duration, "started 2 days ago")
        self.assertEqual(intake.severity, "moderate")
        self.assertEqual(intake.progression, "worsening")
        self.assertEqual(intake.next_field(), "red_flags")

    def test_onset_extraction_stops_before_following_clauses(self):
        examples = (
            ("Since yesterday, the pain has been severe.", "Since yesterday"),
            ("It began last night and is improving.", "began last night"),
            ("I have had it for the past 3 weeks, and it is constant.", "for the past 3 weeks"),
        )

        for message, expected in examples:
            with self.subTest(message=message):
                intake = ClinicalIntake()
                update_intake(intake, message)
                self.assertEqual(intake.onset_duration, expected)

    def test_red_flag_screen_requires_the_screening_turn(self):
        intake = ClinicalIntake(onset_duration="today", severity="mild", progression="constant")
        track_asked_question(intake, QUESTIONS["red_flags"])
        update_intake(intake, "No, none of those warning signs.")
        self.assertEqual(intake.red_flags, "negative")
        self.assertTrue(intake.is_complete())

    def test_next_question_is_first_uncollected_slot(self):
        intake = ClinicalIntake()
        self.assertEqual(intake.next_question(), QUESTIONS["onset_duration"])
        update_intake(intake, "Since yesterday")
        self.assertEqual(intake.next_question(), QUESTIONS["severity"])

    def test_short_onset_answer_is_stored_for_the_question_actually_asked(self):
        intake = ClinicalIntake()
        track_asked_question(intake, QUESTIONS["onset_duration"])

        update_intake(intake, "two days")

        self.assertEqual(intake.onset_duration, "two days")
        self.assertIsNone(intake.pending_field)

    def test_short_severity_answer_is_stored_for_the_question_actually_asked(self):
        intake = ClinicalIntake(onset_duration="two days")
        track_asked_question(intake, "How severe does it feel right now?")

        update_intake(intake, "pretty bad")

        self.assertEqual(intake.severity, "pretty bad")

    def test_progression_answer_is_stored_for_the_question_actually_asked(self):
        intake = ClinicalIntake(onset_duration="two days", severity="7/10")
        track_asked_question(intake, QUESTIONS["progression"])

        update_intake(intake, "getting worse")

        self.assertEqual(intake.progression, "worsening")

    def test_unasked_short_reply_is_not_assigned_to_the_next_missing_field(self):
        intake = ClinicalIntake()

        update_intake(intake, "two days")

        self.assertIsNone(intake.onset_duration)
        self.assertIsNone(intake.pending_field)

    def test_prompt_context_marks_missing_question_as_optional(self):
        context = ClinicalIntake().to_prompt_context()
        self.assertIn("Suggested next uncollected question", context)
        self.assertIn("ask only if needed for triage", context)


class ApiFailureFallbackTests(unittest.TestCase):
    def test_severe_symptoms_receive_urgent_professional_guidance(self):
        response = safe_api_failure_response(ClinicalIntake(severity="severe"))

        self.assertIn("Triage Level: Urgent", response)
        self.assertIn("qualified healthcare professional", response)

    def test_worsening_symptoms_receive_urgent_professional_guidance(self):
        response = safe_api_failure_response(ClinicalIntake(progression="worsening"))

        self.assertIn("Triage Level: Urgent", response)

    def test_non_severe_symptoms_get_next_missing_intake_question(self):
        intake = ClinicalIntake(onset_duration="today", severity="mild")
        response = safe_api_failure_response(intake)

        self.assertIn(QUESTIONS["progression"], response)
        self.assertIn("severe or getting worse", response)

    def test_complete_intake_does_not_repeat_a_question(self):
        intake = ClinicalIntake("today", "mild", "constant", "negative")
        response = safe_api_failure_response(intake)

        self.assertIn("can't safely assign a triage level", response)
        self.assertNotIn("To continue the intake safely", response)


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

    def test_accepts_unambiguous_plain_language_conclusions(self):
        examples = (
            ("This should be treated as an emergency.\n\nCall now.", "emergency"),
            ("I would classify this as urgent.\n\nSeek care today.", "urgent"),
            ("The triage assessment is routine.\n\nBook an appointment.", "routine"),
            ("Based on your symptoms, this appears to be self care.\n\nRest.", "self-care"),
        )
        for response, expected in examples:
            with self.subTest(response=response):
                self.assertEqual(extract_triage_level(response)[0], expected)

    def test_does_not_treat_conditional_care_advice_as_a_conclusion(self):
        self.assertIsNone(extract_triage_level("Seek urgent care if your symptoms get worse."))

    def test_non_conclusion_remains_regular_chat(self):
        self.assertIsNone(extract_triage_level("When did these symptoms begin?"))


if __name__ == "__main__":
    unittest.main()
