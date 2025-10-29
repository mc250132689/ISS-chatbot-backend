from fastapi import FastAPI, Request, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import requests
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# Allow all origins (for your web frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

HF_API_KEY = os.getenv("HF_API_KEY")
MODEL = "mistralai/Mistral-7B-Instruct"

@app.get("/")
def root():
    return {"message": "Spiritual Sickness Chatbot API is running."}

@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    user_input = data.get("message", "")

    if not HF_API_KEY:
        return JSONResponse({"error": "Missing Hugging Face API key"}, status_code=500)

    prompt = f"""
You are an Islamic spiritual advisor (rawatan Islam expert). 
Only answer questions related to Islamic healing, spiritual sickness, or ruqyah.
You may respond in Malay or English depending on the question.

User: {user_input}
Assistant:
"""

    try:
        response = requests.post(
            f"https://api-inference.huggingface.co/models/{MODEL}",
            headers={"Authorization": f"Bearer {HF_API_KEY}"},
            json={"inputs": prompt},
            timeout=30,
        )
        result = response.json()

        if isinstance(result, list) and "generated_text" in result[0]:
            reply = result[0]["generated_text"].split("Assistant:")[-1].strip()
        elif "error" in result:
            reply = f"⚠️ Model error: {result['error']}"
        else:
            reply = "I'm sorry, I couldn’t generate a response right now."

        return {"response": reply}

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/token")
async def get_token(username: str = Form(...), password: str = Form(...)):
    if username == "admin" and password == os.getenv("ADMIN_PASSWORD", "admin123"):
        return {"token": "fake-jwt-token"}
    return JSONResponse({"error": "Invalid credentials"}, status_code=401)
