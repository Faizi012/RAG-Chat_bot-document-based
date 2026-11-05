#RAG-Based Document Question-Answering Chatbot
A Streamlit application that lets users upload PDF, DOCX, TXT and Markdown documents, then ask natural-language questions about their contents. It uses text extraction, chunking, Hugging Face embeddings, Chroma similarity search and a local TinyLlama model served by Ollama.
Features
- Upload one or multiple supported documents.
- Extract text from PDF, DOCX, TXT and Markdown.
- Split extracted content into overlapping chunks.
- Create embeddings with BAAI/bge-small-en-v1.5.
- Store and retrieve chunks with Chroma.
- Generate context-based answers using TinyLlama through Ollama.
- Display source filenames and page metadata for retrieved chunks.
Technology stack
Python, Streamlit, pypdf, python-docx, LangChain text splitters, Hugging Face sentence-transformers, Chroma and Ollama.
Requirements
- Python 3.10+ (use a version compatible with the installed packages).
- Ollama installed.
- TinyLlama downloaded in Ollama.
- Internet access for initial package/model downloads.
Setup (Windows PowerShell)
cd path\to\RAG_chatbot
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
ollama pull tinyllama
python -m streamlit run app.py
If PowerShell blocks activation for the current terminal:
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
Open the local URL printed by Streamlit (usually http://localhost:8501).
How to use
1. Start Ollama and ensure tinyllama is installed (ollama list).
2. Run the Streamlit application.
3. Upload PDF, DOCX, TXT or Markdown files.
4. Click Process Documents.
5. Enter a question and click Get Answer.
6. Review the answer and the displayed source information.
Project structure
RAG_chatbot/
├── app.py
├── README.md
├── requirements.txt
├── RAG_Document_Chatbot_Project_Documentation.docx
├── .gitignore
└── documents/     # Optional test documents; do not commit private files
Important implementation notes
- Chunk size: 1000 characters.
- Chunk overlap: 200 characters.
- Retrieval: top 4 similar chunks.
- Embedding model: BAAI/bge-small-en-v1.5.
- Vector store: Chroma.
- LLM: tinyllama through Ollama.
- The described implementation keeps the Chroma collection in memory/session state; it is not persistent across app restarts.
- Scanned PDFs require OCR. This implementation extracts embedded text and does not include OCR.
- The model is instructed to answer from retrieved context, but generated answers should always be checked against the source document.
Troubleshooting
- Missing Python module: activate the correct virtual environment and run python -m pip install -r requirements.txt.
- Ollama/model unavailable: start Ollama and run ollama pull tinyllama.
- No text extracted: the file may be scanned/image-only, empty or unsupported.
- Embedding input error: make sure the embedding/vector-store API receives a list of plain text strings when using Chroma.from_texts.
Documentation
See RAG_Document_Chatbot_Project_Documentation.docx for the abstract, introduction, problem statement, objectives, architecture, workflow, implementation, setup, testing, limitations and future enhancements.
