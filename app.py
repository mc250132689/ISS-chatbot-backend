import os
import json
import requests
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from datetime import datetime, timedelta
from sqlalchemy import create_engine, Column, Integer, String, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from passlib.context import CryptContext
import jwt
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from dotenv import load_dotenv

# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv()
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "mistralai/Mistral-7B-Instruct")
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./data.db")
SECRET_KEY = os.getenv("SECRET_KEY", "change_this_secret")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 60))

# -----------------------------
# App & CORS
# -----------------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Database setup (SQLite)
# -----------------------------
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

# -----------------------------
# Auth helpers
# -----------------------------
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

# -----------------------------
# Knowledge base (RAG)
# -----------------------------
EMBED_MODEL = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
INDEX_PATH = "./data/knowledge_index.faiss"
MAPPING_PATH = "./data/knowledge_mapping.json"
os.makedirs("./data", exist_ok=True)

faiss_index = None
mapping = []

# Load static knowledge base if DB empty
kb_file = "./data/knowledge_base.json"
if os.path.exists(kb_file):
    with open(kb_file, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    texts = [d["content"] for d in mapping]
    emb = EMBED_MODEL.encode(texts, convert_to_numpy=True).astype("float32")
    faiss_index = faiss.IndexFlatL2(emb.shape[1])
    faiss_index.add(emb)
    faiss.write_index(faiss_index, INDEX_PATH)
    with open(MAPPING_PATH, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)
else:
    faiss_index = None

def retrieve_context(query, top_k=3):
    if faiss_index is None or faiss_index.ntotal == 0:
        return ""
    qvec = EMBED_MODEL.encode([query], convert_to_numpy=True).astype("float32")
    D, I = faiss_index.search(qvec, top_k)
    res = []
    for i in I[0]:
        if i < len(mapping):
            res.append(mapping[i]["content"])
    return "\n".join(res)

# -----------------------------
# Default admin
# -----------------------------
db = SessionLocal()
if db.query(User).count() == 0:
    hashed = pwd_context.hash("admin123")
    user = User(username="admin", hashed_password=hashed)
    db.add(user); db.commit()
    print("Created default admin: admin / admin123")
db.close()

# -----------------------------
# Schemas
# -----------------------------
class ChatIn(BaseModel):
    query: str

# -----------------------------
# Auth endpoints
# -----------------------------
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
    db.add(u); db.commit()
    return {"message": "ok"}

# -----------------------------
# Chat endpoint (HF Inference API)
# -----------------------------
@app.post("/chat")
def chat(inp: ChatIn):
    query = inp.query
    context = retrieve_context(query)
    system = (
        "You are a bilingual (Malay + English) Islamic spiritual-health assistant. "
        "Use the context for ruqyah and guidance. "
        "If outside scope, reply briefly that you only assist on spiritual sickness and Rawatan Islam."
    )
    prompt = f"{system}\n\nContext:\n{context}\n\nUser: {query}\nAssistant:"
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}", "Content-Type": "application/json"}
    payload = {"inputs": prompt, "parameters": {"max_new_tokens": 300}}
    r = requests.post(f"https://api-inference.huggingface.co/models/{MODEL_NAME}", json=payload, headers=headers, timeout=60)
    if r.status_code != 200:
        raise HTTPException(status_code=500, detail=f"HF API error: {r.status_code} {r.text}")
    out = r.json()
    if isinstance(out, list) and len(out) > 0 and "generated_text" in out[0]:
        return {"response": out[0]["generated_text"]}
    return {"response": str(out)}
