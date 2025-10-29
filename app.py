from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import requests
import os

# --- Basic FastAPI setup ---
app = FastAPI(title="Islamic Spiritual Sickness Chatbot")

# --- Allow all CORS origins (for frontend) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # you can restrict to your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Hugging Face Config ---
HF_API_KEY = os.getenv("HUGGINGFACE_API_KEY", "")
MODEL_NAME = "microsoft/DialoGPT-small"  # you can replace with your own model

if not HF_API_KEY:
    print("⚠️ WARNING: HUGGINGFACE_API_KEY is missing in environment variables.")

headers = {
    "Authorization": f"Bearer {HF_API_KEY}",
    "Content-Type": "application/json"
}

HF_URL = f"https://router.huggingface.co/hf-inference/models/{MODEL_NAME}"

# --- Root route ---
@app.get("/")
async def root():
    return {"message": "🕌 Islamic Spiritual Sickness Chatbot API is running."}

# --- Chat endpoint ---
@app.post("/chat")
async def chat(request: Request):
    try:
        data = await request.json()
        user_input = data.get("message", "").strip()

        if not user_input:
            return JSONResponse(content={"reply": "Sila masukkan soalan anda."}, status_code=400)

        payload = {"inputs": user_input}

        response = requests.post(HF_URL, headers=headers, json=payload, timeout=30)

        if response.status_code != 200:
            print("HF API ERROR:", response.status_code, response.text)
            return JSONResponse(
                content={"reply": f"Ralat pelayan Hugging Face: {response.status_code}"},
                status_code=500
            )

        result = response.json()
        # Some HF models return a list; handle both cases
        if isinstance(result, list) and len(result) > 0:
            bot_reply = result[0].get("generated_text", "Maaf, tiada jawapan dijumpai.")
        elif isinstance(result, dict) and "generated_text" in result:
            bot_reply = result["generated_text"]
        else:
            bot_reply = "Maaf, tiada jawapan dari model."

        return {"reply": bot_reply}

    except Exception as e:
        print("❌ Server Error:", e)
        return JSONResponse(content={"reply": f"Ralat pelayan: {str(e)}"}, status_code=500)


# --- For Render deployment ---
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
