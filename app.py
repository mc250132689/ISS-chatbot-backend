import os, json, requests
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

# Load env
from dotenv import load_dotenv
load_dotenv()

HUGGINGFACE_API_KEY = os.getenv('HUGGINGFACE_API_KEY')
MODEL_NAME = os.getenv('MODEL_NAME', 'mistralai/Mistral-7B-Instruct')
DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///./data.db')
SECRET_KEY = os.getenv('SECRET_KEY', 'change_this_secret')
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv('ACCESS_TOKEN_EXPIRE_MINUTES', '60'))

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

# DB setup
engine = create_engine(DATABASE_URL, connect_args={'check_same_thread': False} if 'sqlite' in DATABASE_URL else {})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()
pwd_context = CryptContext(schemes=['bcrypt'], deprecated='auto')

class KBEntry(Base):
    __tablename__ = 'knowledge_base'
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(255), nullable=False)
    content = Column(Text, nullable=False)

class User(Base):
    __tablename__ = 'users'
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
oauth2_scheme = OAuth2PasswordBearer(tokenUrl='token')
def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({'exp': expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm='HS256')

def verify_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=['HS256'])
        return payload.get('sub')
    except Exception:
        return None

# RAG setup
EMBED_MODEL = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
INDEX_PATH = os.path.join(os.path.dirname(__file__), 'data', 'knowledge_index.faiss')
MAPPING_PATH = os.path.join(os.path.dirname(__file__), 'data', 'knowledge_mapping.json')
faiss_index = None
mapping = []

def build_index_from_db(db: Session):
    global faiss_index, mapping
    rows = db.query(KBEntry).all()
    mapping = [{'title': r.title, 'content': r.content} for r in rows]
    if not mapping:
        faiss_index = None
        return
    texts = [m['content'] for m in mapping]
    emb = EMBED_MODEL.encode(texts, convert_to_numpy=True).astype('float32')
    idx = faiss.IndexFlatL2(emb.shape[1])
    idx.add(emb)
    faiss.write_index(idx, INDEX_PATH)
    with open(MAPPING_PATH, 'w', encoding='utf-8') as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)
    faiss_index = idx

# Initialize index from DB or fallback to static KB file
def init_index():
    global faiss_index, mapping
    db = SessionLocal()
    rows = db.query(KBEntry).count()
    if rows > 0:
        build_index_from_db(db)
    else:
        # load static KB file if DB empty
        kbfile = os.path.join(os.path.dirname(__file__), 'data', 'knowledge_base.json')
        if os.path.exists(kbfile):
            with open(kbfile, 'r', encoding='utf-8') as f:
                docs = json.load(f)
            mapping = [{'title': d.get('title',''), 'content': d.get('content','')} for d in docs]
            texts = [d['content'] for d in docs]
            emb = EMBED_MODEL.encode(texts, convert_to_numpy=True).astype('float32')
            idx = faiss.IndexFlatL2(emb.shape[1])
            idx.add(emb)
            faiss.write_index(idx, INDEX_PATH)
            with open(MAPPING_PATH, 'w', encoding='utf-8') as f:
                json.dump(mapping, f, ensure_ascii=False, indent=2)
            faiss_index = idx
    db.close()

init_index()

def retrieve_context(query, top_k=3):
    if faiss_index is None or faiss_index.ntotal == 0:
        return ''
    qvec = EMBED_MODEL.encode([query], convert_to_numpy=True).astype('float32')
    D, I = faiss_index.search(qvec, top_k)
    res = []
    for i in I[0]:
        if i < len(mapping):
            res.append(mapping[i]['content'])
    return '\n'.join(res)

# Create default admin if none exists
def ensure_default_admin():
    db = SessionLocal()
    if db.query(User).count() == 0:
        hashed = pwd_context.hash('admin123')
        user = User(username='admin', hashed_password=hashed)
        db.add(user); db.commit()
        print('Created default admin: admin / admin123')
    db.close()

ensure_default_admin()

# Schemas
class KBIn(BaseModel):
    title: str
    content: str

class ChatIn(BaseModel):
    query: str

# Auth endpoints
@app.post('/token')
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == form_data.username).first()
    if not user or not pwd_context.verify(form_data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail='Invalid credentials')
    token = create_access_token({'sub': user.username})
    return {'access_token': token, 'token_type': 'bearer'}

@app.post('/register')
def register(username: str, password: str, db: Session = Depends(get_db)):
    if db.query(User).filter(User.username == username).first():
        raise HTTPException(status_code=400, detail='User exists')
    hashed = pwd_context.hash(password)
    u = User(username=username, hashed_password=hashed)
    db.add(u); db.commit(); return {'message':'ok'}

# KB CRUD (protected)
def must_auth(token: str = Depends(oauth2_scheme)):
    sub = verify_token(token)
    if not sub:
        raise HTTPException(status_code=403, detail='Unauthorized')
    return sub

@app.get('/kb')
def list_kb(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    must_auth(token)
    rows = db.query(KBEntry).all()
    return [{'id':r.id,'title':r.title,'content':r.content} for r in rows]

@app.post('/kb')
def add_kb(item: KBIn, token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    must_auth(token)
    row = KBEntry(title=item.title, content=item.content)
    db.add(row); db.commit(); db.refresh(row)
    build_index_from_db(db)
    return {'id':row.id,'title':row.title,'content':row.content}

@app.put('/kb/{entry_id}')
def update_kb(entry_id: int, item: KBIn, token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    must_auth(token)
    row = db.query(KBEntry).filter(KBEntry.id==entry_id).first()
    if not row:
        raise HTTPException(status_code=404, detail='Not found')
    row.title = item.title; row.content = item.content
    db.commit(); db.refresh(row)
    build_index_from_db(db)
    return {'id':row.id,'title':row.title,'content':row.content}

@app.delete('/kb/{entry_id}')
def delete_kb(entry_id: int, token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    must_auth(token)
    row = db.query(KBEntry).filter(KBEntry.id==entry_id).first()
    if not row:
        raise HTTPException(status_code=404, detail='Not found')
    db.delete(row); db.commit()
    build_index_from_db(db)
    return {'message':'deleted'}

# Chat endpoint using HF Inference API
@app.post('/chat')
def chat(inp: ChatIn):
    query = inp.query
    context = retrieve_context(query)
    system = 'You are a bilingual (Malay + English) Islamic spiritual-health assistant. Use the context for ruqyah and guidance. '             'If outside scope, reply briefly that you only assist on spiritual sickness and Rawatan Islam.'
    prompt = f"{system}\n\nContext:\n{context}\n\nUser: {query}\nAssistant:"
    headers = {'Authorization': f'Bearer {HUGGINGFACE_API_KEY}', 'Content-Type': 'application/json'}
    payload = {'inputs': prompt, 'parameters': {'max_new_tokens': 300}}
    r = requests.post(f'https://api-inference.huggingface.co/models/{MODEL_NAME}', json=payload, headers=headers, timeout=60)
    if r.status_code != 200:
        raise HTTPException(status_code=500, detail=f'HF API error: {r.status_code} {r.text}')
    out = r.json()
    if isinstance(out, list) and len(out)>0 and 'generated_text' in out[0]:
        return {'response': out[0]['generated_text']}
    # fallback
    return {'response': str(out)}
