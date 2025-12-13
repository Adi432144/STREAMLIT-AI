import streamlit as st
import time
import requests
from datetime import datetime
import random
import os
import re
import joblib
import nltk
import google.generativeai as genai

# 🔊 SPEECH TO TEXT IMPORTS
from streamlit_mic_recorder import mic_recorder
import speech_recognition as sr
import tempfile

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ---------------- NLTK SETUP ----------------
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# ---------------- GEMINI CONFIG ----------------
try:
    genai.configure(api_key="YOUR_GEMINI_API_KEY")
except:
    st.error("Gemini API Key missing")

# ---------------- MOOD ASSETS ----------------
color_palette = {
    "Happy": ["#FFD700", "#FF5722"],
    "Sad": ["#202020", "#000000"],
    "Neutral": ["#EAEAEA", "#CCCCCC"]
}

mood_emojis = {
    "Happy": "https://media.giphy.com/media/11sBLVxNs7v6WA/giphy.gif",
    "Sad": "https://media.giphy.com/media/fhLgA6nJec3Cw/giphy.gif",
    "Neutral": "https://media.giphy.com/media/iyCUpd3MOYLf8COyPQ/giphy.gif"
}

utility_emojis = {
    "clock": "https://media.giphy.com/media/2zdVnsL3mbrs4xg4fr/giphy.gif",
    "globe": "https://media.giphy.com/media/mf8UbIDew7e8g/giphy.gif"
}

# ---------------- PAGE CONFIG ----------------
st.set_page_config(layout="wide", page_title="AI Mood Adaptive Story")

# ---------------- SESSION STATE ----------------
if "username" not in st.session_state:
    st.session_state.username = ""
if "mood" not in st.session_state:
    st.session_state.mood = "Neutral"
if "story" not in st.session_state:
    st.session_state.story = "Your story will appear here."
if "mood_history" not in st.session_state:
    st.session_state.mood_history = []

# ---------------- THEME FUNCTION ----------------
def apply_mood_theme(mood):
    bg = mood_emojis.get(mood, mood_emojis["Neutral"])
    st.markdown(f"""
    <style>
    [data-testid="stAppViewContainer"] {{
        background: url('{bg}') no-repeat center center fixed;
        background-size: cover;
    }}
    .story-box {{
        background: rgba(0,0,0,0.6);
        color: white;
        padding: 20px;
        border-radius: 10px;
        font-size: 18px;
    }}
    </style>
    """, unsafe_allow_html=True)

apply_mood_theme(st.session_state.mood)

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.header("👤 Profile")
    st.write(st.session_state.username or "Guest")
    st.subheader("📝 Mood History")

    if not st.session_state.mood_history:
        st.info("No moods yet")
    else:
        for m in reversed(st.session_state.mood_history):
            st.write("•", m)

# ---------------- LOCATION + TIME ----------------
def get_location():
    try:
        res = requests.get("http://ip-api.com/json").json()
        return f"{res['city']}, {res['country']}"
    except:
        return "Unknown"

location = get_location()

col1, col2 = st.columns(2)
with col1:
    st.markdown(f"🕒 **{datetime.now().strftime('%d %b %I:%M %p')}**")
with col2:
    st.markdown(f"🌍 **{location}**")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return joblib.load("sentiment_model1.pkl")

pipeline = load_model()

# ---------------- TEXT PREPROCESS ----------------
def preprocess_text(text):
    sw = set(stopwords.words("english"))
    text = re.sub(r"http\S+|@\w+", "", text.lower())
    tokens = [w for w in nltk.word_tokenize(text) if w.isalpha() and w not in sw]
    lem = WordNetLemmatizer()
    return " ".join([lem.lemmatize(w) for w in tokens])

# 🔊 ---------------- SPEECH TO TEXT FUNCTION ----------------
def speech_to_text(audio_bytes):
    r = sr.Recognizer()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        f.write(audio_bytes)
        path = f.name

    with sr.AudioFile(path) as source:
        audio = r.record(source)

    try:
        return r.recognize_google(audio)
    except:
        return ""

# ---------------- STORY GENERATION ----------------
def generate_story(sentiment, limit):
    prompt = f"""
    Write a suspenseful short story under {limit} words.
    The tone should reflect a {sentiment} emotional state.
    Include mystery, emotional tension, and a twist ending.
    """
    model = genai.GenerativeModel("gemini-2.5-flash")
    return model.generate_content(prompt).text

# ---------------- MAIN UI ----------------
st.title("🧠 AI Mood Adaptive Story")

if not st.session_state.username:
    st.session_state.username = st.text_input("Enter your name")
    st.stop()

st.subheader(f"Hello {st.session_state.username} 👋")

st.markdown("### 🎙️ Speak or Type Your Mood")

colA, colB = st.columns([2, 1])

with colA:
    mood_text = st.text_area("Type your mood", height=100)

with colB:
    audio = mic_recorder(
        start_prompt="🎤 Start",
        stop_prompt="🛑 Stop",
        just_once=True
    )

    if audio and audio["bytes"]:
        st.info("Listening...")
        spoken_text = speech_to_text(audio["bytes"])
        if spoken_text:
            mood_text = spoken_text
            st.success("Voice captured!")
            st.write("🗣️", spoken_text)

word_limit = st.slider("Story length", 50, 400, 150)

if st.button("✨ Generate Story"):
    if mood_text:
        clean = preprocess_text(mood_text)
        pred = pipeline.predict([clean])[0]
        mood = "Happy" if pred == 4 else "Sad"

        st.session_state.mood = mood
        apply_mood_theme(mood)

        story = generate_story(mood, word_limit)
        st.session_state.story = story
        st.session_state.mood_history.append(mood_text)
        st.rerun()

# ---------------- OUTPUT ----------------
st.markdown("## 📖 Your Story")
st.markdown(f"<div class='story-box'>{st.session_state.story}</div>", unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("🌟 *Stories that feel what you feel*")
