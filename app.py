import os
import json
from datetime import datetime
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from huggingface_hub import InferenceClient

# ==============================
# CONFIG
# ==============================
HF_TOKEN = os.getenv("HF_TOKEN") or "your_huggingface_token_here"
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"

# Initialize Hugging Face client
client = InferenceClient(
    model=MODEL_NAME,
    token=HF_TOKEN,
    provider="featherless-ai",  # ✅ must use new provider
)

LOG_FILE = "chat_logs.json"

# ==============================
# APP SETUP
# ==============================
app = FastAPI(title="🕌 Islamic Spiritual Sickness Chatbot (2025-HF)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================
# SCHEMA
# ==============================
class ChatRequest(BaseModel):
    message: str

# ==============================
# HELPERS
# ==============================
def detect_language(text: str) -> str:
    malay_keywords = [
        "mimpi","ruqyah","jin","sihir","doa","solat","sakit","gangguan","syaitan","hati","tidur"
    ]
    return "ms" if any(w in text.lower() for w in malay_keywords) else "en"

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
# MAIN CHAT FUNCTION
# ==============================
def ask_model(prompt: str, lang: str) -> str:
    try:
        system_prompt = (
            "Anda ialah pembantu Islam yang memahami Bahasa Melayu. "
            "Bantu pengguna memahami mimpi, gangguan spiritual, ruqyah, dan penyembuhan Islam "
            "berdasarkan Al-Quran dan Sunnah."
            if lang == "ms" else
            "You are an Islamic assistant that interprets dreams and spiritual sickness "
            "based on Qur'an and Sunnah. Reply politely and concisely."
        )

        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            temperature=0.7,
        )
        return completion.choices[0].message["content"].strip()
    except Exception as e:
        return f"⚠️ Ralat pelayan Hugging Face: {e}"

# ==============================
# ROUTES
# ==============================
@app.get("/")
def home():
    return {"message": "🕌 Islamic Spiritual Sickness Chatbot Backend (HF 2025) is running."}

@app.post("/chat")
async def chat(request: ChatRequest):
    user_message = request.message.strip()
    if not user_message:
        return {"reply": "Sila masukkan soalan anda."}

    lang = detect_language(user_message)
    ai_reply = ask_model(user_message, lang)
    log_to_json(user_message, ai_reply, lang)

    return {"reply": ai_reply or "Maaf, saya tidak dapat memahami pertanyaan anda."}

@app.get("/logs")
def get_logs():
    if not os.path.exists(LOG_FILE):
        return {"logs": []}
    return {"logs": json.load(open(LOG_FILE, "r", encoding="utf-8"))}
