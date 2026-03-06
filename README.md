# 🤖 Streamlit AI: Multi-Model Chat Interface
[![Live Demo](https://img.shields.io/badge/Render-Live_Demo-00d1b2?style=for-the-badge&logo=render&logoColor=white)](https://streamlit-ai-by5h.onrender.com)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776ab?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io/)

An production-grade AI Chatbot interface built with **Streamlit** and powered by the **Google Gemini 1.5 Flash** model. This application demonstrates how to bridge high-performance LLMs with a reactive web UI for real-time natural language processing.

---

## 🌟 Key Features
* **Gemini 1.5 Flash Integration:** Leverages Google's high-speed, low-latency model for near-instant responses.
* **Stateful Chat History:** Maintains conversation context throughout the session using Streamlit's `session_state`.
* **Markdown Rendering:** Full support for code snippets, tables, and formatted text in AI responses.
* **Responsive Deployment:** Optimized for both desktop and mobile viewing via Render.

## 🛠️ Technical Architecture
This project follows a clean **Controller-View** pattern within the Streamlit framework:
* **Frontend:** Streamlit's reactive components handle user input and message rendering.
* **LLM Orchestration:** `google-generativeai` SDK manages the secure connection to Google AI Studio.
* **Environment Management:** Secure API key handling via Streamlit Secrets (for production) and `.env` (for local development).

## 🚀 Quick Start (Local Development)

### 1. Installation
```bash
git clone [https://github.com/Adi432144/STREAMLIT-AI.git](https://github.com/Adi432144/STREAMLIT-AI.git)
cd STREAMLIT-AI
pip install -r requirements.txt
