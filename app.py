import os
import requests
from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ==============================
# Configuration
# ==============================
# Use environment variable or fallback token
HF_TOKEN = os.getenv("HF_TOKEN") or "your_huggingface_token_here"

# Use a lightweight multilingual model for Render free tier
HF_MODEL = os.getenv("HF_MODEL") or "distilbert-base-multilingual-cased"

# ✅ New Hugging Face Inference Providers endpoint (Jan 2025 update)
HF_API_URL = f"https://router.huggingface.co/hf-inference/{HF_MODEL}"

# ==============================
# App setup
# ==============================
app = FastAPI(title="🕌 Islamic Spiritual Sickness Chatbot (Lightweight)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all origins (for your frontend)
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
    """Send a prompt to Hugging Face Inference Providers API."""
    try:
        headers = {
            "Authorization": f"Bearer {HF_TOKEN}",
            "Content-Type": "application/json",
        }
        payload = {"inputs": prompt}

        response = requests.post(HF_API_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()

        data = response.json()

        # Handle both text-generation and classification outputs
        if isinstance(data, list):
            # Text generation
            if "generated_text" in data[0]:
                return data[0]["generated_text"]
            # Classification
            elif "label" in data[0]:
                return data[0]["label"]

        elif isinstance(data, dict):
            # Some models return directly a dict
            if "generated_text" in data:
                return data["generated_text"]
            elif "label" in data:
                return data["label"]

        return str(data)

    except requests.exceptions.RequestException as e:
        return f"Ralat pelayan Hugging Face: {e}"
    except ValueError:
        return "Ralat pelayan: Tidak dapat memproses respons daripada Hugging Face."

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
# Simple HTML form test route
# ==============================
@app.post("/token")
async def token(message: str = Form(...)):
    ai_reply = query_huggingface(message)
    return {"reply": ai_reply}
