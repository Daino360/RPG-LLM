import pdfplumber
import os
import json

# ---- CONFIG ----
PDF_FOLDER = "pdfs"
OUTPUT_FILE = "chunks.json"
CHUNK_SIZE = 400   # words per chunk
OVERLAP = 50       # overlapping words

# --- LOADER FUNCTIONS ----
def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

# Split text into chunks with overlap
def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i + chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap
    return chunks

def main():
    all_chunks = []
    for filename in os.listdir(PDF_FOLDER):
        if filename.endswith(".pdf"):
            path = os.path.join(PDF_FOLDER, filename)
            print(f"Processing {filename}...")
            text = extract_text_from_pdf(path)
            chunks = chunk_text(text)
            print(f" -> Created {len(chunks)} chunks")
            for chunk in chunks:
                all_chunks.append({"pdf": filename, "text": chunk})

    # Save chunks to JSON
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(all_chunks)} chunks to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
