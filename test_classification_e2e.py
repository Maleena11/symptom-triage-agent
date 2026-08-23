"""Conversation-to-triage regression tests.

The deterministic tests mock Gemini at its network boundary, not at the response
parser. The fake reads the conversation sent by the agent and derives its answer
from those symptoms. Optional live tests exercise the real configured model.
"""

import os
from types import SimpleNamespace
import unittest

from intake_state import ClinicalIntake
from triage_agent import MODEL_NAME, collect_triage_response, create_client, extract_triage_level, retry_triage_response


CASES = (
    ("urgent", [{"role": "user", "content": "I have been vomiting all day and cannot keep fluids down. I feel dizzy."}]),
    ("routine", [{"role": "user", "content": "I have recurring mild heartburn after meals for three weeks, with no warning signs."}]),
    ("self-care", [{"role": "user", "content": "I have a mild runny nose and sore throat since yesterday, without fever or breathing problems."}]),
    (
        "urgent",
        [
            {"role": "user", "content": "I have been vomiting."},
            {"role": "assistant", "content": "Can you keep fluids down?"},
            {"role": "user", "content": "No, and now I feel dizzy."},
        ],
    ),
)


class ConversationAwareModels:
    """Small contract fake that classifies the contents it is actually passed."""

    def __init__(self):
        self.calls = []

    def generate_content_stream(self, *, model, contents, config):
        turns = [part.text for content in contents for part in content.parts]
        self.calls.append({"model": model, "turns": turns, "instruction": config.system_instruction})
        user_text = " ".join(turns[::2] if len(turns) > 1 else turns).casefold()
        if "vomit" in user_text and ("cannot keep fluids" in user_text or "feel dizzy" in user_text):
            level = "Urgent"
        elif "three weeks" in user_text or "recurring" in user_text:
            level = "Routine"
        elif "mild runny nose" in user_text and "without fever" in user_text:
            level = "Self-care"
        else:
            raise AssertionError(f"The agent did not send a recognized representative conversation: {turns!r}")
        return iter((SimpleNamespace(text="Triage Level: "), SimpleNamespace(text=level), SimpleNamespace(text="\n\nTest advice.")))


class ClassificationPipelineTests(unittest.TestCase):
    def test_invalid_response_is_retried_once_with_formatting_instruction(self):
        class RetryModels:
            def __init__(self):
                self.calls = 0

            def generate_content_stream(self, *, model, contents, config):
                self.calls += 1
                self.assertion_contents = [part.text for content in contents for part in content.parts]
                return iter((SimpleNamespace(text="Triage Level: Urgent\n\nSeek care today."),))

        models = RetryModels()
        client = SimpleNamespace(models=models)
        intake = ClinicalIntake()
        response = retry_triage_response(client, [{"role": "user", "content": "Symptoms"}], intake, "Please seek care today.")

        self.assertEqual(models.calls, 1)
        self.assertEqual(extract_triage_level(response)[0], "urgent")
        self.assertIn("Please seek care today.", models.assertion_contents)
        self.assertIn("Reissue your previous answer once", models.assertion_contents[-1])

    def test_representative_conversations_drive_expected_triage_decisions(self):
        models = ConversationAwareModels()
        client = SimpleNamespace(models=models)
        intake = ClinicalIntake(onset_duration="provided", severity="provided", progression="provided", red_flags="negative")

        for expected, messages in CASES:
            with self.subTest(expected=expected, conversation=messages):
                response = collect_triage_response(client, messages, intake)
                result = extract_triage_level(response)
                self.assertIsNotNone(result)
                self.assertEqual(result[0], expected)

        self.assertEqual(len(models.calls), len(CASES))
        self.assertTrue(all(call["model"] == MODEL_NAME for call in models.calls))
        self.assertTrue(all("Red Flags: negative" in call["instruction"] for call in models.calls))
        self.assertEqual(models.calls[-1]["turns"], [turn["content"] for turn in CASES[-1][1]])


@unittest.skipUnless(os.getenv("RUN_LIVE_TRIAGE_TESTS") == "1" and os.getenv("GEMINI_API_KEY"), "set RUN_LIVE_TRIAGE_TESTS=1 and GEMINI_API_KEY")
class LiveClassificationAccuracyTests(unittest.TestCase):
    """Real-model evaluations; opt in because they use network access and quota."""

    @classmethod
    def setUpClass(cls):
        cls.client = create_client(os.environ["GEMINI_API_KEY"])
        cls.intake = ClinicalIntake(onset_duration="provided", severity="provided", progression="provided", red_flags="negative")

    def test_real_model_classifies_representative_conversations(self):
        for expected, messages in CASES:
            with self.subTest(expected=expected, conversation=messages):
                response = collect_triage_response(self.client, messages, self.intake)
                result = extract_triage_level(response)
                self.assertIsNotNone(result, response)
                self.assertEqual(result[0], expected, response)


if __name__ == "__main__":
    unittest.main()
