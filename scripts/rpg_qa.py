import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
import torch

# ---- CONFIG ----
FAISS_INDEX_FILE = "../embeddings/faiss_index.bin"
META_FILE = "../embeddings/chunk_metadata.pkl"
TOP_K = 5
SIMILARITY_THRESHOLD = 0.35  # optional threshold for chunk relevance
EMBEDDING_MODEL = "all-mpnet-base-v2"
LLM_MODEL = "mistralai/Mistral-7B-v0.1"

# ---- LOAD FUNCTIONS ----
print("Loading FAISS index...")
index = faiss.read_index(FAISS_INDEX_FILE)

with open("../chunks.json", "r", encoding="utf-8") as f:
    chunks_data = json.load(f)  # contains 'pdf' and 'text'

print("Loading embedding model...")
embed_model = SentenceTransformer(EMBEDDING_MODEL, device="cuda")

print("Loading LLM model...")
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
tokenizer.pad_token = tokenizer.eos_token  # set pad token for generation
llm_model = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL,
    device_map="auto",
    torch_dtype=torch.float16 
)

# ---- HELPER FUNCTIONS ----
def clean_chunk(chunk_text):
    text = chunk_text.replace("\n", " ").strip()
    text = re.sub(r"\s{2,}", " ", text)
    # Remove page numbers like "Page 3/20"
    text = re.sub(r"Page \d+/\d+", "", text)
    return text

# Retrieve relevant chunks, use cosine similarity
def retrieve_chunks(question, top_k=TOP_K, similarity_threshold=SIMILARITY_THRESHOLD):

    # Encode question with normalized embeddings
    q_emb = embed_model.encode([question], convert_to_numpy=True, normalize_embeddings=True)

    # Query the FAISS index
    distances, indices = index.search(q_emb, top_k)

    retrieved = []
    for idx, score in zip(indices[0], distances[0]):
        if score < similarity_threshold: # ignore low similarity chunks
            continue
        text = clean_chunk(chunks_data[idx]['text'])
        retrieved.append(text)
    
    return retrieved

# Format context for prompt
def format_context(chunks, max_chars=3000):
    context = "\n\n".join([f"[Chunk {i+1}] {c}" for i, c in enumerate(chunks)])
    return context[:max_chars]


# ---- MAIN QUESTION-ANSWERING FUNCTION ----

def answer_question(question):
    chunks = retrieve_chunks(question)
    if not chunks:
        return "I don't know based on the provided documents."
    
    context = format_context(chunks)

    prompt = f"""
            You are an expert tabletop RPG assistant.

            Answer the question using ONLY the information in the context.
            Answer concisely and clearly.
            If the context does not explicitly contain the answer, say:
            "I don't know based on the provided documents."

            Context:
            {context}

            Question:
            {question}

            Answer (one short paragraph or list):
            """

    # Tokenize and generate answer
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=7000).to("cuda")
    outputs = llm_model.generate(
        **inputs,
        max_new_tokens=300,
        temperature=0.3,
        do_sample=False,  # deterministic, avoids repetition and prompt leakage
        repetition_penalty=1.1,
        pad_token_id=tokenizer.eos_token_id
    )

    # Extract generated answer only
    answer_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)
    return answer.strip()

# ---- INTERACTIVE LOOP ----

print("\nRPG QA Bot is ready! Type your question or 'exit' to quit.\n")
while True:
    question = input("Question: ")
    if question.lower() in ["exit", "quit"]:
        break
    answer = answer_question(question)
    print(f"\nAnswer: {answer}\n")
