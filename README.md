# Symptom Triage Agent

A simple **Streamlit chat application** that helps users understand how urgently they should seek medical care based on the symptoms they describe.

The app provides one of four guidance levels:

* 🚨 **Emergency** — Get emergency medical help immediately.
* ⚠️ **Urgent** — Seek medical care as soon as possible.
* 🩺 **Routine** — Make a normal appointment with a healthcare professional.
* 🏠 **Self-care** — The symptoms may be suitable for basic self-care and monitoring.

> **Important:** This application provides triage guidance only. It does **not diagnose medical conditions** and does not replace professional medical advice. If someone may be in immediate danger, contact local emergency services immediately.

## How It Works

The application uses a combination of **local safety rules** and **Google Gemini AI**.

When a user describes their symptoms:

1. The app stores the conversation and updates the symptom information.
2. Local safety rules first check for obvious emergency warning signs.
3. If an emergency warning sign is found, the app immediately recommends emergency care without calling Gemini.
4. If no emergency rule is triggered, Gemini reviews the conversation and symptom information.
5. The app returns an **Emergency, Urgent, Routine, or Self-care** result with a recommended next step.
6. If more information is needed, the app may ask one important follow-up question.

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

## Why Streamlit?

**Streamlit** is used to create the user interface.

It provides:

* A simple chat interface
* Session state for multi-turn conversations
* Sidebar controls and demo examples
* Streaming responses
* Easy local setup using Python

This allows the project to work without building a separate frontend application.

## Why Gemini?

**Gemini** is used to understand symptom descriptions written in natural language.

It can consider information from multiple messages and help determine the appropriate urgency level.

However, Gemini is **not the only safety mechanism**. Clear emergency situations are checked locally before an API request is made.

## Project Structure

```text
app.py
    Streamlit user interface and application setup

guardrails.py
    Local emergency safety checks

intake_state.py
    Tracks symptom information and follow-up questions

triage_agent.py
    Gemini API communication and triage response handling

test_triage.py
    Unit tests for the main application logic

test_classification_e2e.py
    End-to-end and optional live Gemini tests

requirements.txt
    Python dependencies
```

## Setup

Python **3.10 or later** is recommended.

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

## Run the Application

Start the Streamlit app with:

```powershell
python -m streamlit run app.py
```

Then open the Streamlit URL shown in the terminal.

## Demo Examples

The sidebar contains example symptoms for testing the four possible results.

**Emergency**

```text
Chest pain spreading to the arm with shortness of breath.
```

**Urgent**

```text
High fever with persistent vomiting.
```

**Routine**

```text
Mild knee pain that has lasted for three weeks.
```

**Self-care**

```text
Mild runny nose and sneezing without fever.
```

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

## Limitations

* The application provides **urgency guidance, not a medical diagnosis**.
* It should not replace advice from a qualified healthcare professional.
* The local emergency rules cannot detect every possible emergency.
* Gemini may misunderstand symptoms or produce incorrect results.
* Most non-emergency assessments require internet access and a valid Gemini API key.
* The application currently checks only the latest eight user messages for local emergency rules.
* This project has **not been clinically validated**.
* It should **not be used for real clinical decision-making** without professional clinical review, safety validation, privacy and security assessment, and any required regulatory approval.

## Disclaimer

This project is intended as a **prototype and educational demonstration**.

It is not a medical device, diagnostic system, or emergency service. Always seek advice from a qualified healthcare professional when concerned about symptoms. For a possible medical emergency, contact local emergency services immediately.
