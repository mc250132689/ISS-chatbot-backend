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
HF_MODEL = "mistralai/Mistral-7B-Instruct"   # ✅ bilingual + active on new router
HF_API_URL = f"https://router.huggingface.co/hf-inference/models/{HF_MODEL}"

LOG_FILE = "chat_logs.json"

# ==============================
# App setup
# ==============================
app = FastAPI(title="🕌 Islamic Spiritual Sickness Chatbot (HF Router API)")

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
# Helpers
# ==============================
def detect_language(text: str) -> str:
    malay_words = [
        "mimpi","saya","anda","apa","kenapa","bila","ruqyah","sakit","jin","gangguan","solat","doa","hati"
    ]
    return "ms" if any(w in text.lower() for w in malay_words) else "en"

def log_to_json(user_message: str, ai_reply: str, lang: str):
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "language": lang,
        "user_message": user_message,
        "ai_reply": ai_reply,
    }
    try:
        logs = json.load(open(LOG_FILE, "r", encoding="utf-8")) if os.path.exists(LOG_FILE) else []
    except json.JSONDecodeError:
        logs = []
    logs.append(entry)
    if len(logs) > 1000:
        logs = logs[-500:]
    json.dump(logs, open(LOG_FILE, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

# ==============================
# Query Hugging Face Router
# ==============================
def query_huggingface(prompt: str, lang: str = "en"):
    try:
        headers = {
            "Authorization": f"Bearer {HF_TOKEN}",
            "Content-Type": "application/json",
        }

        if lang == "ms":
            system_prompt = (
                "Anda ialah pembantu Islam yang memahami Bahasa Melayu. "
                "Bantu pengguna memahami mimpi, gangguan spiritual, ruqyah, dan penyembuhan Islam "
                "berdasarkan Al-Quran dan Sunnah."
            )
        else:
            system_prompt = (
                "You are an Islamic assistant who helps interpret dreams and identify spiritual sickness "
                "based on the Qur'an and Sunnah. Reply politely and clearly."
            )

        payload = {
            "inputs": f"{system_prompt}\n\nUser: {prompt}\nAssistant:",
            "parameters": {"max_new_tokens": 200, "temperature": 0.7},
        }

        response = requests.post(HF_API_URL, headers=headers, json=payload, timeout=60)
        response.raise_for_status()

        data = response.json()
        if isinstance(data, list) and "generated_text" in data[0]:
            return data[0]["generated_text"]
        if isinstance(data, dict) and "generated_text" in data:
            return data["generated_text"]
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
    log_to_json(user_message, ai_reply, lang)
    return {"reply": ai_reply or "Maaf, saya tidak dapat memahami pertanyaan anda."}

@app.get("/logs")
def get_logs():
    if not os.path.exists(LOG_FILE):
        return {"logs": []}
    return {"logs": json.load(open(LOG_FILE, "r", encoding="utf-8"))}
