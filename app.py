import os
import json
from pydantic import BaseModel, Field

import streamlit as st
from google import genai
from google.genai import types
from dotenv import load_dotenv


# ---------------------------------------------------------
# Load environment variables
# ---------------------------------------------------------

load_dotenv()


# ---------------------------------------------------------
# Fetch Gemini API key
# ---------------------------------------------------------

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    try:
        api_key = st.secrets.get("GEMINI_API_KEY")
    except Exception:
        api_key = None

if not api_key:
    st.error(
        "Please set your GEMINI_API_KEY in a .env file "
        "or Streamlit Secrets."
    )
    st.stop()


# ---------------------------------------------------------
# Initialize Gemini client
# ---------------------------------------------------------

client = genai.Client(api_key=api_key)


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="Symptom Triage Agent",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 Symptom Triage Chat Agent")

st.caption(
    "This prototype assesses symptom urgency. "
    "It does not diagnose medical conditions."
)


# ---------------------------------------------------------
# System Prompt
# ---------------------------------------------------------

SYSTEM_PROMPT = """
You are a Symptom Triage Chat Agent.

Your goal is to assess the urgency of the user's symptoms.
You must classify urgency into one of these four levels:

1. Emergency
Immediate danger requiring immediate medical attention.
Examples include:
- Chest pain with difficulty breathing
- Severe difficulty breathing
- Sudden severe weakness
- Loss of consciousness
- Severe or uncontrolled bleeding
- Other symptoms that indicate an immediate threat to life

For Emergency cases:
- Clearly state "Triage Level: Emergency"
- Explain briefly why the situation is concerning
- Tell the user to call local emergency services or go to the nearest emergency department immediately
- Do not continue a long questionnaire before giving the emergency advice

2. Urgent
Requires medical assessment within 24 hours.
Examples may include:
- High fever
- Persistent vomiting
- Severe pain
- Worsening symptoms
- Significant dehydration
- Other concerning symptoms that do not appear immediately life-threatening

For Urgent cases:
- Clearly state "Triage Level: Urgent"
- Explain the reason briefly
- Recommend seeking medical care within 24 hours

3. Routine
Persistent or recurring symptoms that should be assessed through a standard medical appointment.

For Routine cases:
- Clearly state "Triage Level: Routine"
- Explain the reason briefly
- Recommend arranging a standard medical appointment

4. Self-care
Mild, recent symptoms without emergency or urgent warning signs.

For Self-care cases:
- Clearly state "Triage Level: Self-care"
- Give general, low-risk self-care guidance
- Tell the user to seek medical attention if symptoms worsen or new warning signs appear

IMPORTANT GUIDELINES:

1. Be empathetic, clear, concise, and professional.

2. Ask follow-up questions one at a time.

3. Gather relevant information such as:
   - Main symptom
   - Severity
   - When it started
   - Whether it is getting worse
   - Associated symptoms
   - Relevant context when necessary

4. Do NOT diagnose the user with a specific medical condition.

5. You are assessing urgency, not providing a medical diagnosis.

6. If the user mentions an emergency warning sign at ANY point in the conversation,
   immediately classify the situation as Emergency and provide emergency advice.

7. Do not wait until the end of the conversation to respond to a newly reported emergency symptom.

8. Do not repeatedly ask questions when enough information is already available.

9. Once you have enough information, provide a final triage decision using:
   - Triage Level
   - Reason
   - Next Steps

10. If information is unclear, ask one short clarification question rather than making
    an unsupported assumption.

11. Never claim certainty about a diagnosis.

12. The user may answer questions in natural language rather than using exact words
    such as "Mild", "Moderate", "Severe", "Yes", or "No".
"""


# ---------------------------------------------------------
# Emergency Guardrail
# ---------------------------------------------------------

def emergency_guardrail(text):
    """
    Hardcoded safety check that runs BEFORE sending
    the user's message to the LLM.
    """

    text = text.lower()

    # Rule 1: Chest pain + breathing difficulty
    if (
        ("chest pain" in text or "chest pressure" in text)
        and (
            "difficulty breathing" in text
            or "shortness of breath" in text
            or "short of breath" in text
            or "can't breathe" in text
            or "cannot breathe" in text
            or "trouble breathing" in text
            or "struggling to breathe" in text
            or "struggle to breathe" in text
            or "gasping for air" in text
        )
    ):
        return (
            "Emergency",
            "Chest pain together with difficulty breathing is a serious emergency warning sign."
        )

    # Rule 2: Loss of consciousness
    if (
        "loss of consciousness" in text
        or "passed out" in text
        or "lost consciousness" in text
        or "unconscious" in text
    ):
        return (
            "Emergency",
            "Loss of consciousness is an emergency warning sign."
        )

    # Rule 3: Severe or uncontrolled bleeding
    if (
        "severe bleeding" in text
        or "heavy bleeding" in text
        or "bleeding won't stop" in text
        or "bleeding will not stop" in text
        or "uncontrolled bleeding" in text
    ):
        return (
            "Emergency",
            "Severe or uncontrolled bleeding requires immediate emergency care."
        )

    # Rule 4: Severe breathing difficulty
    if (
        "severe difficulty breathing" in text
        or "i can't breathe" in text
        or "i cannot breathe" in text
        or "unable to breathe" in text
    ):
        return (
            "Emergency",
            "Severe difficulty breathing is an emergency warning sign."
        )

    return None


# ---------------------------------------------------------
# Semantic Guardrail Schema & Function
# ---------------------------------------------------------

class EmergencyAssessment(BaseModel):
    is_emergency: bool = Field(
        description="True if the text indicates an immediate, life-threatening medical emergency "
                    "(e.g., stroke, anaphylaxis, shock, cardiac arrest, severe respiratory distress, choking, etc.)."
    )
    reason: str = Field(
        description="A brief explanation of why this is or is not an immediate medical emergency."
    )


def semantic_guardrail(text):
    """
    Semantic safety check using the LLM to identify context-based red flags
    (like symptoms indicating stroke, anaphylaxis, or shock) that keywords might miss.
    """
    try:
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=EmergencyAssessment,
            temperature=0.0,
            system_instruction=(
                "You are a medical safety guardrail. Analyze the user's symptom description "
                "to determine if they are experiencing an immediate, life-threatening medical emergency. "
                "Specifically look for context-based red flags indicating conditions such as stroke, "
                "anaphylaxis (severe allergic reaction), shock, cardiac arrest, severe respiratory distress, "
                "or other acute life-threatening situations."
            )
        )
        response = client.models.generate_content(
            model="gemini-3.5-flash",
            contents=text,
            config=config
        )
        
        # Parse the structured response
        data = json.loads(response.text)
        if data.get("is_emergency"):
            return "Emergency", data.get("reason", "Immediate medical attention is required.")
            
    except Exception as e:
        # In case of API failure, log to console but don't disrupt the user experience
        print(f"Error in semantic guardrail: {e}")
        
    return None


# ---------------------------------------------------------
# Initialize Session State
# ---------------------------------------------------------

if "messages" not in st.session_state:

    st.session_state.messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "assistant",
            "content": (
                "Hello! 👋 I can help assess the urgency of your symptoms.\n\n"
                "Please describe what you are experiencing today."
            )
        }
    ]


# ---------------------------------------------------------
# New Assessment button
# ---------------------------------------------------------

if st.button("🔄 New Assessment"):

    st.session_state.clear()

    st.rerun()


# ---------------------------------------------------------
# Display Previous Messages
# ---------------------------------------------------------

for message in st.session_state.messages:

    # Do not display the system prompt
    if message["role"] == "system":
        continue

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ---------------------------------------------------------
# User Input
# ---------------------------------------------------------

if prompt := st.chat_input("Type your response..."):

    # -----------------------------------------------------
    # STEP 1: Run emergency guardrail BEFORE LLM
    # -----------------------------------------------------
    # Run deterministic guardrail first
    guardrail_result = emergency_guardrail(prompt)

    # Run semantic guardrail if deterministic does not trigger
    if not guardrail_result:
        guardrail_result = semantic_guardrail(prompt)

    if guardrail_result:

        level, reason = guardrail_result

        response = (
            f"### 🚨 Triage Level: {level}\n\n"
            f"**Reason:** {reason}\n\n"
            "Please call your local emergency services or go to "
            "the nearest emergency department immediately."
        )

        # Save user message
        st.session_state.messages.append(
            {
                "role": "user",
                "content": prompt
            }
        )

        # Save emergency response
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": response
            }
        )

        # Display emergency response
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            st.markdown(response)

        st.stop()


    # -----------------------------------------------------
    # STEP 2: Display and save user message
    # -----------------------------------------------------

    with st.chat_message("user"):
        st.markdown(prompt)

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )


    # -----------------------------------------------------
    # STEP 3: Call Gemini with streaming
    # -----------------------------------------------------

    with st.chat_message("assistant"):

        message_placeholder = st.empty()

        full_response = ""

        try:
            # Format history for Gemini API
            gemini_messages = []
            for msg in st.session_state.messages:
                if msg["role"] == "system":
                    continue
                # Map roles: 'user' -> 'user', 'assistant' -> 'model'
                role = "user" if msg["role"] == "user" else "model"
                gemini_messages.append(
                    types.Content(
                        role=role,
                        parts=[types.Part.from_text(text=msg["content"])]
                    )
                )

            # Request streaming content from Gemini
            response_stream = client.models.generate_content_stream(
                model="gemini-3.5-flash",
                contents=gemini_messages,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT
                )
            )

            for chunk in response_stream:
                if chunk.text:
                    full_response += chunk.text
                    message_placeholder.markdown(full_response + "▌")

            # Display final response
            message_placeholder.markdown(full_response)

            # Save assistant response
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": full_response
                }
            )

        except Exception as e:

            error_message = (
                "Sorry, I could not process your request right now.\n\n"
                "Please check that your Gemini API key is configured correctly "
                "and try again."
            )

            message_placeholder.error(f"{error_message}\n\n**Error details:** `{str(e)}`")

            # Do not expose technical/API details to the user