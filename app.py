import os
import requests
from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ==============================
# Configuration
# ==============================
HF_TOKEN = os.getenv("HF_TOKEN") or "your_huggingface_token_here"
HF_MODEL = "google/flan-t5-base"  # ✅ supported & lightweight
HF_API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL}"

# ==============================
# App setup
# ==============================
app = FastAPI(title="🕌 Islamic Spiritual Sickness Chatbot (Lightweight)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (frontend)
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
# Hugging Face Query Function
# ==============================
def query_huggingface(prompt: str):
    try:
        headers = {
            "Authorization": f"Bearer {HF_TOKEN}",
            "Content-Type": "application/json",
        }

        # System context
        system_prompt = (
            "You are a bilingual (Malay + English) Islamic assistant. "
            "Respond based on ruqyah, dreams, and guidance from Islamic spiritual healing. "
            "Keep answers short and respectful. If unclear, ask politely for clarification."
        )

        payload = {
            "inputs": f"{system_prompt}\n\nUser: {prompt}\nAssistant:",
            "parameters": {"max_new_tokens": 250, "temperature": 0.7},
        }

        response = requests.post(HF_API_URL, headers=headers, json=payload, timeout=60)
        response.raise_for_status()

        data = response.json()

        if isinstance(data, list) and len(data) > 0 and "generated_text" in data[0]:
            return data[0]["generated_text"]
        elif isinstance(data, dict) and "generated_text" in data:
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

    ai_reply = query_huggingface(user_message)
    return {"reply": ai_reply or "Maaf, saya tidak dapat memahami pertanyaan anda."}

@app.post("/token")
async def token(message: str = Form(...)):
    ai_reply = query_huggingface(message)
    return {"reply": ai_reply}
