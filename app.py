import os
import requests
from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ==============================
# Configuration
# ==============================
HF_TOKEN = os.getenv("HF_TOKEN") or "your_huggingface_token_here"
HF_MODEL = "bert-base-multilingual-cased"  # You can use any model name here
HF_API_URL = f"https://router.huggingface.co/hf-inference/models/{HF_MODEL}"

# ==============================
# App setup
# ==============================
app = FastAPI(title="Islamic Spiritual Sickness Chatbot (Lightweight)")

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
        payload = {"inputs": prompt}

        response = requests.post(HF_API_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()

        data = response.json()

        # Handle text generation or classification outputs gracefully
        if isinstance(data, list):
            if "generated_text" in data[0]:
                return data[0]["generated_text"]
            elif "label" in data[0]:
                return data[0]["label"]
        elif isinstance(data, dict) and "generated_text" in data:
            return data["generated_text"]

        return str(data)

    except requests.exceptions.RequestException as e:
        return f"Ralat pelayan Hugging Face: {e}"

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

# ==============================
# Form test route (optional)
# ==============================
@app.post("/token")
async def token(message: str = Form(...)):
    ai_reply = query_huggingface(message)
    return {"reply": ai_reply}
