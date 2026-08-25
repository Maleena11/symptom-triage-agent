# Symptom Triage Agent

A simple **Streamlit chat application** that helps users understand how urgently they should seek medical care based on the symptoms they describe.

The app provides one of four guidance levels:

* **Emergency** — Get emergency medical help immediately.
* **Urgent** — Seek medical care as soon as possible.
* **Routine** — Make a normal appointment with a healthcare professional.
* **Self-care** — The symptoms may be suitable for basic self-care and monitoring.

> **Important:** This application provides triage guidance only. It does **not diagnose medical conditions** and does not replace professional medical advice. If someone may be in immediate danger, contact local emergency services immediately.

## How It Works

The application uses a combination of **local safety rules**, a **guided symptom intake**, and **Google Gemini AI**.

When a user describes their symptoms:

1. The app stores the conversation and updates the available symptom information.
2. Local safety rules first check the user's recent messages for obvious emergency warning signs.
3. If an emergency warning sign is found, the app immediately recommends emergency care without calling Gemini.
4. If no emergency rule is triggered, the app checks whether the required intake information has been collected.
5. If important information is missing, the app asks **one follow-up question at a time** to collect it.
6. The user may therefore be asked several follow-up questions across multiple turns until the intake is complete.
7. Once the required intake information has been collected, Gemini reviews the conversation and symptom information.
8. The app returns an **Emergency, Urgent, Routine, or Self-care** result with a recommended next step.
9. If Gemini or the Gemini API is unavailable, the app can use a **deterministic fallback** to provide triage guidance after the required intake information has been collected.

This design ensures that obvious emergency warning signs are handled immediately, while non-emergency cases go through a structured intake before AI-based or fallback triage is performed.

## Emergency Safety Checks

Before sending anything to Gemini, the app checks the user's recent messages for clear emergency warning signs.

Examples include:

* Suicidal or self-harm thoughts
* Severe difficulty breathing or inability to breathe
* Blue lips or face
* Signs of stroke, such as facial drooping, one-sided weakness, or sudden speech problems
* Serious allergic reactions with swelling or breathing difficulty
* Crushing or squeezing chest pain
* Chest pain with shortness of breath, sweating, fainting, nausea, or pain spreading to another area
* Severe or uncontrolled bleeding
* Coughing or vomiting blood
* Loss of consciousness or collapse
* Sudden loss of vision
* A sudden extremely severe or "worst ever" headache

If one of these emergency rules matches, the app immediately recommends contacting emergency services or going to the nearest emergency department.

These checks are an additional safety layer and **do not cover every possible medical emergency**.

## Guided Intake

For cases that do not trigger an emergency safety rule, the app collects the symptom information needed for triage before calling Gemini.

If required information is missing, the app asks **one question at a time**. Depending on the information already provided, this process may continue across several conversation turns.

Once the required intake information is complete, the app proceeds with the triage assessment.

This approach helps ensure that the triage step has enough structured context rather than sending an incomplete symptom description directly to Gemini.

## Why Streamlit?

**Streamlit** is used to create the user interface.

It provides:

* A simple chat interface
* Session state for multi-turn conversations
* Sidebar controls and demo examples
* Straightforward display of completed triage responses
* Easy local setup using Python

The interface waits for the complete triage response before displaying it rather than streaming partial Gemini output.

Streamlit allows the project to provide an interactive multi-turn experience without building a separate frontend application.

## Why Gemini?

**Gemini** is used to understand symptom descriptions written in natural language after the required intake information has been collected.

It can consider information from multiple messages and help determine the appropriate urgency level.

However, Gemini is **not the only safety or triage mechanism**. Clear emergency situations are checked locally before an API request is made, and a deterministic fallback can provide guidance when Gemini is unavailable.

## Project Structure

```text
app.py
    Streamlit user interface and application setup

guardrails.py
    Local emergency safety checks

intake_state.py
    Tracks symptom information and follow-up questions

triage_agent.py
    Gemini API communication, triage response handling,
    and fallback triage behavior

test_triage.py
    Unit tests for the main application logic

test_classification_e2e.py
    End-to-end and optional live Gemini tests

requirements.txt
    Python dependencies

.env.example
    Example environment configuration for the Gemini API key

.gitignore
    Specifies files and directories that should not be committed
```

## Setup

Python **3.10 or later is required** because the project source uses Python 3.10 syntax.

### 1. Create a virtual environment

```powershell
python -m venv .venv
```

### 2. Activate it

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 4. Create the environment file

```powershell
Copy-Item .env.example .env
```

### 5. Add your Gemini API key

Open `.env` and add:

```text
GEMINI_API_KEY=your_gemini_api_key_here
```

The Gemini API key is used for AI-based triage after the guided intake is complete. If Gemini or the API key is unavailable, the application can use its deterministic fallback to provide guidance after collecting the required intake information.

## Run the Application

Start the Streamlit app with:

```powershell
python -m streamlit run app.py
```

Then open the Streamlit URL shown in the terminal.

## Demo Examples

The sidebar contains example symptoms for testing the four possible triage levels.

> **Note:** The **Emergency** preset is designed to trigger an immediate emergency result because it matches a local emergency safety rule. The **Urgent**, **Routine**, and **Self-care** presets begin the guided intake process and may require additional information before a final triage result is shown.

### Emergency

```text
Chest pain spreading to the arm with shortness of breath.
```

This example should trigger the local emergency safety checks and produce an immediate emergency recommendation without waiting for the guided intake or calling Gemini.

### Urgent

```text
High fever with persistent vomiting.
```

This example begins the guided intake. The app may ask follow-up questions before producing the final triage result.

### Routine

```text
Mild knee pain that has lasted for three weeks.
```

This example begins the guided intake. The app may ask follow-up questions before producing the final triage result.

### Self-care

```text
Mild runny nose and sneezing without fever.
```

This example begins the guided intake. The app may ask follow-up questions before producing the final triage result.

## Testing

Run all local tests:

```powershell
python -m unittest
```

Run only the main unit tests:

```powershell
python -m unittest test_triage.py
```

To optionally run tests using the live Gemini API:

```powershell
$env:RUN_LIVE_TRIAGE_TESTS = "1"
python -m unittest test_classification_e2e.py
```

> Live tests require a valid Gemini API key and may use API quota.

## Fallback Behavior

Gemini is not required for every possible application outcome.

Emergency warning signs are handled locally before Gemini is called. For non-emergency cases, the application first collects the required intake information.

After the intake is complete, the application normally uses Gemini for the triage assessment. However, if Gemini cannot be used — for example, because the API key is missing or the API is unavailable — the application can use a **deterministic fallback** to produce triage guidance.

This means the application can still provide guidance in some situations without a working Gemini connection, although the fallback is more limited than the AI-based assessment.

## Limitations

* The application provides **urgency guidance, not a medical diagnosis**.
* It should not replace advice from a qualified healthcare professional.
* The local emergency rules cannot detect every possible emergency.
* Gemini may misunderstand symptoms or produce incorrect results.
* The deterministic fallback is limited and should not be treated as a substitute for professional medical assessment.
* A Gemini API key and internet access are required for **Gemini-based assessments**, but the application can still use its deterministic fallback when Gemini is unavailable after the required intake information has been collected.
* The application currently checks only the latest eight user messages for local emergency rules.
* The quality of the triage result depends on the information provided by the user.
* This project has **not been clinically validated**.
* It should **not be used for real clinical decision-making** without professional clinical review, safety validation, privacy and security assessment, and any required regulatory approval.

## Disclaimer

This project is intended as a **prototype and educational demonstration**.

It is not a medical device, diagnostic system, or emergency service. The triage levels are intended only to demonstrate how symptom information can be routed into different levels of urgency.

Always seek advice from a qualified healthcare professional when concerned about symptoms. For a possible medical emergency, contact local emergency services immediately.
