import os
import json
import requests
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from datetime import datetime, timedelta
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from passlib.context import CryptContext
import jwt
from dotenv import load_dotenv

# Load env
load_dotenv()
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "mistralai/Mistral-7B-Instruct")
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./data.db")
SECRET_KEY = os.getenv("SECRET_KEY", "change_this_secret")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 60))

# FastAPI setup
app = FastAPI()
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)

# SQLite DB
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(128), unique=True, nullable=False)
    hashed_password = Column(String(256), nullable=False)

Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Auth helpers
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm="HS256")

def verify_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        return payload.get("sub")
    except Exception:
        return None

def must_auth(token: str = Depends(oauth2_scheme)):
    sub = verify_token(token)
    if not sub:
        raise HTTPException(status_code=403, detail="Unauthorized")
    return sub

# Load knowledge base
with open("./data/knowledge_base.json", "r", encoding="utf-8") as f:
    KNOWLEDGE = json.load(f)

# Schemas
class ChatIn(BaseModel):
    query: str

# Auth endpoints
@app.post("/token")
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == form_data.username).first()
    if not user or not pwd_context.verify(form_data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_access_token({"sub": user.username})
    return {"access_token": token, "token_type": "bearer"}

@app.post("/register")
def register(username: str, password: str, db: Session = Depends(get_db)):
    if db.query(User).filter(User.username == username).first():
        raise HTTPException(status_code=400, detail="User exists")
    hashed = pwd_context.hash(password)
    u = User(username=username, hashed_password=hashed)
    db.add(u)
    db.commit()
    return {"message": "ok"}

# Chat endpoint (Hugging Face Inference API)
@app.post("/chat")
def chat(inp: ChatIn):
    context_text = "\n".join([item["content"] for item in KNOWLEDGE])
    system = (
        "You are a bilingual (Malay + English) Islamic spiritual-health assistant. "
        "Use the context for ruqyah and guidance. "
        "If outside scope, reply briefly that you only assist on spiritual sickness and Rawatan Islam."
    )
    prompt = f"{system}\n\nContext:\n{context_text}\n\nUser: {inp.query}\nAssistant:"
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}", "Content-Type": "application/json"}
    payload = {"inputs": prompt, "parameters": {"max_new_tokens": 300}}
    r = requests.post(f"https://api-inference.huggingface.co/models/{MODEL_NAME}", json=payload, headers=headers, timeout=60)
    if r.status_code != 200:
        raise HTTPException(status_code=500, detail=f"HF API error: {r.status_code} {r.text}")
    out = r.json()
    if isinstance(out, list) and len(out) > 0 and "generated_text" in out[0]:
        return {"response": out[0]["generated_text"]}
    return {"response": str(out)}
