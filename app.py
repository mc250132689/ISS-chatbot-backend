from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import requests
import os

app = FastAPI()

# ✅ Allow frontend connection (adjust to your domain if needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["https://your-frontend-domain.com"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Hugging Face Inference Providers API
HF_API_URL = "https://router.huggingface.co/hf-inference/models/mistralai/Mistral-7B-Instruct"
HF_API_KEY = os.getenv("HF_API_KEY")

HEADERS = {
    "Authorization": f"Bearer {HF_API_KEY}",
    "Content-Type": "application/json"
}

@app.get("/")
def root():
    return {"status": "ok", "message": "🕌 Islamic Spiritual Sickness Chatbot API (HF Providers Ready)"}

@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    user_message = data.get("message", "").strip()

    if not user_message:
        return {"response": "Sila masukkan soalan atau pertanyaan anda."}

    prompt = f"""
    You are an Islamic counselor that helps identify and explain possible spiritual sicknesses (penyakit rohani)
    according to Islamic principles and ruqyah syar'iyyah.
    Respond calmly and respectfully, using the same language as the user (Malay or English).
    
    Question: {user_message}
    Answer:
    """

    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": 200, "temperature": 0.7}
    }

    try:
        response = requests.post(HF_API_URL, headers=HEADERS, json=payload, timeout=60)
        result = response.json()

        # Handle different HF response formats
        reply = None
        if isinstance(result, list) and len(result) > 0 and "generated_text" in result[0]:
            reply = result[0]["generated_text"]
        elif isinstance(result, dict) and "generated_text" in result:
            reply = result["generated_text"]

        if not reply:
            reply = "Saya tidak pasti, sila cuba semula atau gunakan bahasa yang lebih jelas."

        # Clean output
        reply = reply.split("Answer:")[-1].strip()
        return {"response": reply}

    except Exception as e:
        return {"response": f"Ralat pelayan: {str(e)}"}
