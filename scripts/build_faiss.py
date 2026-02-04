import json
from sentence_transformers import SentenceTransformer # to generate embeddings
import faiss
import os
import pickle

# ---- CONFIG ----
CHUNKS_FILE = "../chunks.json"
FAISS_INDEX_FILE = "../embeddings/faiss_index.bin"
META_FILE = "../embeddings/chunk_metadata.pkl"
EMBEDDING_MODEL = "all-mpnet-base-v2"  # encode chunks into embeddings
BATCH_SIZE = 16

# ---- LOAD FUNCTIONS ----
# Load chunks
with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
    chunks = json.load(f)

texts = [chunk["text"] for chunk in chunks]

# Add metadata with chunk index
metadata = [{"pdf": chunk["pdf"], "chunk_id": i} for i, chunk in enumerate(chunks)]

# Load embedding model
print("Loading embedding model...")
model = SentenceTransformer(EMBEDDING_MODEL, device="cuda")

# ---- ENCODE AND BUILD INDEX ----
# Encode chunks
print(f"Encoding {len(texts)} chunks...")
embeddings = model.encode(
    texts,
    batch_size=BATCH_SIZE,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True  # important for cosine similarity
)

# Create FAISS index
d = embeddings.shape[1]  # embedding dimension
index = faiss.IndexFlatIP(d)  # use inner product with normalized embeddings = cosine similarity
index.add(embeddings)

# ---- SAVE INDEX AND METADATA ----
os.makedirs("../embeddings", exist_ok=True)
faiss.write_index(index, FAISS_INDEX_FILE)

with open(META_FILE, "wb") as f:
    pickle.dump(metadata, f)

print(f"FAISS index saved to {FAISS_INDEX_FILE}")
print(f"Metadata saved to {META_FILE}")
