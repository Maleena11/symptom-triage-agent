# Symptom Triage Agent

A Streamlit chat application that provides urgency guidance for reported symptoms. It is not a diagnosis tool or a substitute for professional medical advice.

## Structure

- `app.py` — Streamlit UI, chat state, visual triage badges, and API-key setup.
- `guardrails.py` — deterministic emergency red-flag checks across recent user turns.
- `triage_agent.py` — Gemini client, clinical triage prompt, and response streaming.
- `test_triage.py` — automated tests for deterministic emergency handling.
- `requirements.txt` — locked runtime dependencies.

## Setup

1. Create and activate a virtual environment.
2. Install dependencies: `pip install -r requirements.txt`
3. Copy `.env.example` to `.env`, then set `GEMINI_API_KEY`.
4. Start the app: `streamlit run app.py`

Run local safety tests with `python -m unittest test_triage.py`.

## Safety design

Each new user message is added to the conversation and the most recent eight user turns are checked together before it reaches the conversational agent:

1. A deterministic local guardrail catches explicit high-confidence red flags: FAST stroke symptoms, anaphylaxis with airway/breathing symptoms, cardiac warning patterns, respiratory failure/airway obstruction, loss of consciousness, and severe haemorrhage. It also works when relevant symptoms are reported in separate turns.
2. Assistant and system messages are excluded from that scan so previously generated advice cannot create a false positive.

The local layer is immediate and remains usable if the model request fails. For non-deterministic cases, the streamed triage request is the only Gemini call on a turn and receives the full conversation context. The system instructions place semantic emergency classification inside that same request, eliminating the former synchronous structured guardrail call and its extra per-turn latency. The prompt requires one of four levels: Emergency, Urgent, Routine, or Self-care.

## Clinical intake state

The app keeps an explicit, session-scoped intake record for onset and duration, severity, progression, and a red-flag screen. It asks only the first missing item, passes the current state and exact next question to Gemini on every turn, and waits for this intake before giving non-emergency triage. The sidebar shows the same state so the assessment is transparent.
