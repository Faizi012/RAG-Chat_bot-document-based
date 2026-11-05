
import streamlit as st
from io import BytesIO
from pypdf import PdfReader
from docx import Document as DocxDocument

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama

import uuid


# --------------------------------
# 1. Streamlit page
# --------------------------------

st.set_page_config(
    page_title="RAG Document Chatbot",
    page_icon="📚"
)

st.title("📚 RAG-Based Document Chatbot")
st.write("Upload documents and ask questions about their content.")


# --------------------------------
# 2. Extract text from documents
# --------------------------------

def extract_text(uploaded_file):

    file_name = uploaded_file.name
    file_extension = file_name.split(".")[-1].lower()

    extracted_pages = []

    # PDF
    if file_extension == "pdf":

        pdf = PdfReader(BytesIO(uploaded_file.getvalue()))

        for page_number, page in enumerate(pdf.pages, start=1):

            text = page.extract_text() or ""

            if text.strip():
                extracted_pages.append({
                    "text": text,
                    "source": file_name,
                    "page": page_number
                })

    # DOCX
    elif file_extension == "docx":

        document = DocxDocument(
            BytesIO(uploaded_file.getvalue())
        )

        text = "\n".join(
            paragraph.text
            for paragraph in document.paragraphs
            if paragraph.text.strip()
        )

        if text.strip():
            extracted_pages.append({
                "text": text,
                "source": file_name,
                "page": "N/A"
            })

    # TXT and Markdown
    elif file_extension in ["txt", "md"]:

        text = uploaded_file.getvalue().decode(
            "utf-8",
            errors="replace"
        )

        if text.strip():
            extracted_pages.append({
                "text": text,
                "source": file_name,
                "page": "N/A"
            })

    else:
        raise ValueError("Unsupported file format.")

    return extracted_pages


# --------------------------------
# 3. Upload documents
# --------------------------------

uploaded_files = st.file_uploader(
    "Upload your documents",
    type=["pdf", "docx", "txt", "md"],
    accept_multiple_files=True
)


# --------------------------------
# 4. Process documents
# --------------------------------

if st.button("Process Documents"):

    if not uploaded_files:

        st.warning("Please upload at least one document.")

    else:

        try:

            all_pages = []

            for uploaded_file in uploaded_files:

                pages = extract_text(uploaded_file)
                all_pages.extend(pages)

            if not all_pages:
                st.error(
                    "No readable text was found in the uploaded files. "
                    "Scanned PDFs may require OCR."
                )
                st.stop()

            # Split extracted text into chunks

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )

            text_chunks = []
            metadatas = []

            for page in all_pages:

                chunks = splitter.split_text(page["text"])

                for chunk in chunks:

                    if chunk.strip():

                        # Only plain strings go into the embedding process
                        text_chunks.append(chunk)

                        metadatas.append({
                            "source": page["source"],
                            "page": str(page["page"])
                        })

            if not text_chunks:
                st.error("No text chunks were generated.")
                st.stop()

            # --------------------------------
            # 5. Create embedding model
            # --------------------------------

            embedding_model = HuggingFaceEmbeddings(
                model_name="BAAI/bge-small-en-v1.5",
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True}
            )

            # --------------------------------
            # 6. Create Chroma vector database
            # --------------------------------

            vectorstore = Chroma.from_texts(
                texts=text_chunks,
                embedding=embedding_model,
                metadatas=metadatas,
                collection_name=f"rag_{uuid.uuid4().hex[:12]}"
            )

            # Store vector database in session
            st.session_state.vectorstore = vectorstore
            st.session_state.processed = True

            st.success("Documents processed successfully!")

            st.write("Number of documents:", len(uploaded_files))
            st.write("Number of text chunks:", len(text_chunks))

        except Exception as e:

            st.error(f"Processing failed: {str(e)}")


# --------------------------------
# 7. Ask questions
# --------------------------------

if st.session_state.get("processed", False):

    st.divider()
    st.subheader("Ask questions about your documents")

    question = st.text_input(
        "Enter your question:"
    )

    if st.button("Get Answer"):

        if question.strip():

            try:

                vectorstore = st.session_state.vectorstore

                # Retrieve relevant chunks
                relevant_docs = vectorstore.similarity_search(
                    question,
                    k=4
                )

                if not relevant_docs:
                    st.warning("No relevant information was found.")

                else:

                    # Combine retrieved text
                    context = "\n\n".join(
                        doc.page_content
                        for doc in relevant_docs
                    )

                    # --------------------------------
                    # 8. LLM
                    # --------------------------------

                    llm = ChatOllama(
                        model="tinyllama",
                        temperature=0.2
                    )

                    prompt = f"""
                    You are a document question-answering assistant.

                    Answer the user's question using only the
                    information provided in the context below.

                    If the answer is not available in the context,
                    clearly say that the uploaded documents do not
                    contain enough information.

                    Context:
                    {context}

                    Question:
                    {question}

                    Answer:
                    """

                    response = llm.invoke(prompt)

                    st.subheader("Answer")
                    st.write(response.content)

                    # --------------------------------
                    # 9. Show sources
                    # --------------------------------

                    st.subheader("Sources")

                    for doc in relevant_docs:

                        st.write(
                            "File:",
                            doc.metadata.get("source", "Unknown")
                        )

                        st.write(
                            "Page:",
                            doc.metadata.get("page", "Unknown")
                        )

                        st.caption(
                            doc.page_content[:300]
                        )

            except Exception as e:

                st.error(f"Question answering failed: {str(e)}")

        else:

            st.warning("Please enter a question.")
