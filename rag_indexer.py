#!/usr/bin/env python3
import json, os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

BASE = os.path.dirname(__file__)
KB_PATH = os.path.join(BASE, "data", "knowledge_base.json")
INDEX_PATH = os.path.join(BASE, "data", "knowledge_index.faiss")
MAPPING_PATH = os.path.join(BASE, "data", "knowledge_mapping.json")

def build():
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    with open(KB_PATH, 'r', encoding='utf-8') as f:
        docs = json.load(f)
    texts = [d['content'] for d in docs]
    if len(texts) == 0:
        print("No docs to index.")
        return
    embeddings = embedder.encode(texts, convert_to_numpy=True).astype('float32')
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, INDEX_PATH)
    with open(MAPPING_PATH, 'w', encoding='utf-8') as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)
    print(f"Indexed {len(texts)} documents.")

if __name__ == '__main__':
    build()
