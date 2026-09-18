# Streamlit chat UI (assignment sections 6 and 7)
# fusion.py is unchanged. uploads go to temp files and are deleted.

import os
import sys
import tempfile
import string
import streamlit as st

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from config import PROJECT_DIR, DATA_DIR
from fusion import AirportAssistant, answer, MIN_SCORE

if not os.path.exists(DATA_DIR):
    st.error("I cannot find the data folder on Google Drive:")
    st.code(DATA_DIR)
    st.write("Mount Drive, check PROJECT_DIR in the notebook, then run the setup cell again.")
    st.stop()

def spoken_reply(reply):
    # turn the KB dict into a highly formatted, clean Markdown answer
    if not reply["matched"]:
        text = "I do not get your question.\n\n"
        text += "Please ask a person at the information desk."
        return text


    name = reply["name"]
    text = f"### {name}\n"
    
    if reply.get("description"):
        text += f"*{reply['description']}*\n\n"
    if reply.get("direction_text"):
        text += f"**How to get there:** {reply['direction_text']}\n\n"
    if reply.get("opening_hours"):
        text += f"**Opening hours:** {reply['opening_hours']}\n\n"
    if reply.get("accessibility"):
        text += f"**Access:** {reply['accessibility']}\n\n"
    if reply.get("assistance_contact"):
        text += f"**Need a person?** {reply['assistance_contact']}\n\n"
    return text


def save_upload(file_obj, fallback_name):
    if file_obj is None:
        return None
    name = getattr(file_obj, "name", fallback_name)
    suffix = os.path.splitext(name)[1]
    if suffix == "":
        suffix = os.path.splitext(fallback_name)[1]
        if suffix == "":
            suffix = ".bin"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(file_obj.getbuffer())
    tmp.close()
    return tmp.name


def run_turn(text, image_file, audio_file, recorded):
    tmp_paths = []
    image_path = None
    audio_path = None
    try:
        if image_file is not None:
            image_path = save_upload(image_file, "sign.png")
            tmp_paths.append(image_path)
        
        # Audio from live recording only
        audio_src = recorded
        audio_name = "record.wav"
        if audio_src is not None:
            audio_path = save_upload(audio_src, audio_name)
            tmp_paths.append(audio_path)

        q = text if text is not None and str(text).strip() != "" else None

        # --- INTERCEPTORS (Greetings & Gratitude) ---
        if q is not None and image_path is None and audio_path is None:
            clean_q = q.lower().translate(str.maketrans('', '', string.punctuation)).strip()
            
            # Catch greetings
            if clean_q in ["hi", "hello", "hey", "greetings", "hiya", "good morning", "good afternoon", "good evening"]:
                return {
                    "matched": True,
                    "name": "Hello! 👋",
                    "description": "I am your smart multimodal assistant. I can help you find gates, baggage claim, or services around the airport. What can I help you find today?",
                    "direction_text": None,
                    "opening_hours": None,
                    "accessibility": None,
                    "assistance_contact": None,
                    "score": 1.0, 
                    "mode": "Text",
                    "transcript": None
                }
            
            # Catch thank yous
            if clean_q in ["thanks", "thank you", "thx", "thanks a lot", "thank you very much", "appreciate it", "perfect thanks", "ok thanks"]:
                return {
                    "matched": True,
                    "name": "You're very welcome! ✈️",
                    "description": "I'm glad I could help you navigate Kathmandu International Airport. Have a wonderful flight and a safe journey!",
                    "direction_text": None,
                    "opening_hours": None,
                    "accessibility": None,
                    "assistance_contact": None,
                    "score": 1.0, 
                    "mode": "Text",
                    "transcript": None
                }

        # If it is NOT a greeting or thank you, run the normal AI backend search
        reply = answer(assistant, text=q, image_path=image_path, audio_path=audio_path)
        return reply
    finally:
        for path in tmp_paths:
            if path is not None and os.path.exists(path):
                os.remove(path)


# --- DASHBOARD UI DESIGN ---
st.set_page_config(page_title="Airport Assistant", layout="wide", initial_sidebar_state="collapsed")

# FORCING the giant text using INLINE CSS so Streamlit cannot ignore it
st.markdown(
    """
    <div style="text-align: center; padding-bottom: 20px;">
        <h1 style="font-size: 4.5rem; font-weight: 700; color: #1E3A8A; line-height: 1.0; margin: 0; padding: 0;">
            Kathmandu International Airport
        </h1>
        <p style="font-size: 1.5rem; color: #64748B; margin-top: 10px;">
            Smart Multimodal Passenger Assistant
        </p>
    </div>
    """, 
    unsafe_allow_html=True
)
st.divider()

@st.cache_resource
def load_assistant():
    print("Loading models for Streamlit...")
    return AirportAssistant(DATA_DIR)

assistant = load_assistant()

# Initialize session states for chat and live analytics
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "text": "Hello. I am a Smart Airport Assistant. I can help you navigate the airport security, terminal, gates and also help you with signs. \n\nType a question, or use the **Attach / Record** button below to upload a sign photo or record your voice.",
            "image": None,
        }
    ]
if "latest_score" not in st.session_state:
    st.session_state.latest_score = 0.0
if "latest_transcript" not in st.session_state:
    st.session_state.latest_transcript = None
if "latest_mode" not in st.session_state:
    st.session_state.latest_mode = "None"

# Create the dual-pane layout (70% chat, 30% analytics)
chat_col, analytics_col = st.columns([7, 3], gap="large")

# --- RIGHT PANE: LIVE ANALYTICS & INFO ---
with analytics_col:
    st.markdown("### System Analytics")
    
    st.metric(label="Retrieval Confidence Score", 
              value=f"{st.session_state.latest_score:.3f}", 
              delta=f"Threshold: {MIN_SCORE}", 
              delta_color="off")
    
    st.markdown(f"**Input Modality Detected:** `{st.session_state.latest_mode}`")
    
    if st.session_state.latest_transcript:
        st.info(f"**Whisper Transcript:**\n\n\"{st.session_state.latest_transcript}\"")
        
    st.divider()
    st.warning(
        "**Usage Guidelines:**\n\n"
        "• This is not a fully functional chatbot model. Do **not** upload a boarding pass, passport, or a photo with a face for privacy reasons.\n\n"
        "• Temporary files are deleted immediately.\n"
    )
   
# --- LEFT PANE: CHAT INTERFACE ---
with chat_col:
    # Render chat history
    chat_container = st.container(height=300, border=False)
    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                if msg.get("image") is not None:
                    st.image(msg["image"], width=220, caption="Uploaded Image")
                st.markdown(msg["text"])

    # Input controls
    input_col1, input_col2 = st.columns([2, 8], vertical_alignment="center")
    
    with input_col1:
        if hasattr(st, "popover"):
            attach_box = st.popover("📎 Attach / Record", use_container_width=True)
        else:
            attach_box = st.expander("📎 Attach")
        with attach_box:
            image_file = st.file_uploader("Upload Sign Photo", type=["png", "jpg", "jpeg"], key="chat_image")
            recorded = None
            if hasattr(st, "audio_input"):
                recorded = st.audio_input("Record Voice Query")
            send_attach = st.button("Submit Media", use_container_width=True)

    with input_col2:
        with st.form("composer", border=False, clear_on_submit=True):
            box, go = st.columns([8, 2], vertical_alignment="center")
            with box:
                prompt = st.text_input(
                    "Message",
                    label_visibility="collapsed",
                    placeholder="Ask a question...",
                )
            with go:
                submitted = st.form_submit_button("Send", use_container_width=True)

    # Process interactions
    should_run = False
    user_text = None
    if submitted and prompt is not None and str(prompt).strip() != "":
        should_run = True
        user_text = prompt.strip()
    elif send_attach:
        if image_file is not None or recorded is not None:
            should_run = True
            user_text = None
            st.toast("Media successfully attached!", icon="✅")
        else:
            st.toast("Please attach a photo or record a voice clip first.", icon="⚠️")

    if should_run:
        user_label = user_text if user_text is not None else "[Media Uploaded]"
        user_image = image_file.getvalue() if image_file is not None else None
        
        st.session_state.messages.append({
            "role": "user",
            "text": user_label,
            "image": user_image,
        })

        # Run the backend inference
        reply = run_turn(user_text, image_file, None, recorded)
        body = spoken_reply(reply)
        
        # Update the live analytics state variables
        st.session_state.latest_score = reply.get('score', 0.0)
        st.session_state.latest_transcript = reply.get('transcript', None)
        st.session_state.latest_mode = reply.get('mode', 'Text')
            
        st.session_state.messages.append({
            "role": "assistant",
            "text": body,
            "image": None,
        })
        st.rerun()
