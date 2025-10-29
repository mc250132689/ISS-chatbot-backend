import os
import json
import re
import requests
from datetime import datetime
from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ==============================
# Configuration
# ==============================
HF_TOKEN = os.getenv("HF_TOKEN") or "your_huggingface_token_here"
HF_MODEL = "google/flan-t5-base"  # ✅ lightweight, multilingual-compatible
HF_API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL}"

LOG_FILE = "chat_logs.json"  # ✅ Store logs in a simple JSON file

# ==============================
# App setup
# ==============================
app = FastAPI(title="🕌 Islamic Spiritual Sickness Chatbot (Malay-English)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================
# Schema
# ==============================
class ChatRequest(BaseModel):
    message: str

# ==============================
# Utility Functions
# ==============================
def detect_language(text: str) -> str:
    """Basic Malay/English detection."""
    malay_keywords = [
        "saya", "anda", "apa", "kenapa", "bila", "tidur", "mimpi", "ular",
        "doa", "sakit", "ruqyah", "hati", "jin", "gangguan", "solat"
    ]
    malay_count = sum(1 for w in malay_keywords if re.search(rf"\b{w}\b", text.lower()))
    return "ms" if malay_count >= 2 else "en"

def log_to_json(user_message: str, ai_reply: str, lang: str):
    """Save chat logs into a local JSON file."""
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "language": lang,
        "user_message": user_message,
        "ai_reply": ai_reply,
    }

    # Load existing logs or create new
    try:
        if os.path.exists(LOG_FILE):
            with open(LOG_FILE, "r", encoding="utf-8") as f:
                logs = json.load(f)
        else:
            logs = []
    except json.JSONDecodeError:
        logs = []

    logs.append(entry)

    # Keep log size reasonable
    if len(logs) > 1000:
        logs = logs[-500:]

    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)

# ==============================
# Hugging Face Query
# ==============================
def query_huggingface(prompt: str, lang: str = "en"):
    try:
        headers = {
            "Authorization": f"Bearer {HF_TOKEN}",
            "Content-Type": "application/json",
        }

        if lang == "ms":
            system_prompt = (
                "Anda ialah pembantu Islam yang memahami bahasa Melayu. "
                "Bantu pengguna memahami mimpi, gangguan spiritual, ruqyah, dan penyembuhan Islam. "
                "Gunakan nada sopan, ringkas, dan berdasarkan Al-Quran serta sunnah."
            )
        else:
            system_prompt = (
                "You are an Islamic assistant who understands English. "
                "Help users interpret dreams, spiritual issues, and Islamic healing. "
                "Be polite, concise, and grounded in the Qur'an and Sunnah."
            )

        payload = {
            "inputs": f"{system_prompt}\n\nUser: {prompt}\nAssistant:",
            "parameters": {"max_new_tokens": 250, "temperature": 0.7},
        }

        response = requests.post(HF_API_URL, headers=headers, json=payload, timeout=60)
        response.raise_for_status()

        data = response.json()

        if isinstance(data, list) and "generated_text" in data[0]:
            return data[0]["generated_text"]
        elif isinstance(data, dict) and "generated_text" in data:
            return data["generated_text"]
        else:
            return str(data)

    except requests.exceptions.RequestException as e:
        return f"⚠️ Ralat pelayan Hugging Face: {e}"

# ==============================
# Routes
# ==============================
@app.get("/")
def home():
    return {"message": "🕌 Islamic Spiritual Sickness Chatbot Backend is running."}

@app.post("/chat")
async def chat(request: ChatRequest):
    user_message = request.message.strip()
    if not user_message:
        return {"reply": "Sila masukkan soalan anda."}

    lang = detect_language(user_message)
    ai_reply = query_huggingface(user_message, lang)

    # ✅ Log chat interaction
    log_to_json(user_message, ai_reply, lang)

    return {"reply": ai_reply or "Maaf, saya tidak dapat memahami pertanyaan anda."}

@app.post("/token")
async def token(message: str = Form(...)):
    lang = detect_language(message)
    ai_reply = query_huggingface(message, lang)
    log_to_json(message, ai_reply, lang)
    return {"reply": ai_reply}

@app.get("/logs")
def get_logs():
    """Retrieve chat logs."""
    if not os.path.exists(LOG_FILE):
        return {"logs": []}
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        logs = json.load(f)
    return {"logs": logs}
