# RPG-LLM

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.10.0-red?logo=pytorch&logoColor=white)
![Transformers](https://img.shields.io/badge/Transformers-HuggingFace-purple)
![CUDA](https://img.shields.io/badge/CUDA-13.1-orange?logo=nvidia&logoColor=white)
![Machine Learning](https://img.shields.io/badge/Machine_Learning-ML-brightgreen)
![NLP](https://img.shields.io/badge/NLP-Natural_Language_Processing-blue)
![FAISS](https://img.shields.io/badge/FAISS-vector_search-purple)



RPG-LLM is a Python project that allows you to **query a PDF rulebook of any role-playing game** using a Language Model (LLM).  
It extracts the text, creates embeddings with **FAISS**, and answers questions intelligently about the content.

>**DISCLAIMER**: This is the first version of RPG-LLM, tested only with a free quickstart of [Dungeons and Dragons](https://media.wizards.com/2020/dnd/downloads/dnd_starter_rulebook.pdf)

---

## Project Structure

```

RPG-LLM/
├── embeddings/         # Stores generated FAISS index and metadata
├── pdfs/               # Place your PDFs here
├── scripts/
│   ├── pdf_loader.py   # Extracts text from PDFs
│   ├── build_faiss.py  # Builds FAISS embeddings index
│   └── rpg_qa.py       # Queries the LLM using FAISS
├── rpg.yaml            # Conda environment file
├── .gitignore
└── README.md

```

---

## Installation

1. **Clone the repository:**

```bash
git clone https://github.com/Daino360/RPG-LLM.git
cd RPG-LLM
```

2. **Create the Conda environment from the YAML file:**

```bash
conda env create -f rpg.yaml
conda activate rpg
```

---

## Usage

1. **Add PDFs:**
   Place any RPG rulebook PDFs you want to query inside the `pdfs/` folder.

2. **Extract text from PDFs:**

```bash
python scripts/pdf_loader.py
```

This will convert PDFs into text chunks.

3. **Build the FAISS embeddings index:**

```bash
python scripts/build_faiss.py
```

4. **Ask questions about the RPG rulebook:**

```bash
python scripts/rpg_qa.py
```

## Author
Stefano Dainelli — [LinkedIn](https://www.linkedin.com/in/stefanodainelli)
