from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import requests
import os

app = FastAPI()

# ✅ Allow frontend to access this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or restrict to your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Hugging Face Inference API setup
HF_API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct"
HF_API_KEY = os.getenv("HF_API_KEY")  # set in Render environment variables

HEADERS = {"Authorization": f"Bearer {HF_API_KEY}"}


@app.get("/")
def root():
    return {"status": "ok", "message": "🕌 Islamic Spiritual Sickness Chatbot API running"}


@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    user_message = data.get("message", "")

    if not user_message:
        return {"response": "Sila masukkan soalan."}

    prompt = f"""
    You are an Islamic counselor who identifies and explains possible spiritual sickness (penyakit rohani)
    according to Islamic guidance and ruqyah syar'iyyah. 
    Respond in the same language as the user (Malay or English).
    Question: {user_message}
    """

    payload = {"inputs": prompt, "parameters": {"max_new_tokens": 250}}
    try:
        response = requests.post(HF_API_URL, headers=HEADERS, json=payload, timeout=60)
        result = response.json()

        # HF returns a list sometimes
        if isinstance(result, list) and len(result) > 0 and "generated_text" in result[0]:
            reply = result[0]["generated_text"].split("Question:")[-1].strip()
        elif isinstance(result, dict) and "generated_text" in result:
            reply = result["generated_text"]
        else:
            reply = "Saya tidak pasti, sila cuba lagi."

        return {"response": reply}

    except Exception as e:
        return {"response": f"Ralat pelayan: {str(e)}"}
