from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import fitz  # PyMuPDF
import os
import uuid
import faiss
import numpy as np
from sqlalchemy import Column, Integer, String, create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
import logging
from sentence_transformers import SentenceTransformer
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database Setup
Base = declarative_base()
DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

class Document(Base):
    __tablename__ = "documents"
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, unique=True, index=True)
    filepath = Column(String)
    upload_date = Column(String)

Base.metadata.create_all(bind=engine)

# FastAPI Setup
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust origins for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploaded_pdfs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Load Sentence Transformer Model
model = SentenceTransformer('all-MiniLM-L6-v2')

@app.get("/")
def read_root():
    return {"message": "Welcome to the PDF Q&A Application!"}

def extract_text_from_pdf(filepath):
    """Extracts text from a PDF file."""
    try:
        pdf_document = fitz.open(filepath)
        text = ""
        for page in pdf_document:
            text += page.get_text()
        return text
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")

def generate_embeddings(texts):
    """Generates text embeddings using Sentence-Transformers."""
    embeddings = model.encode(texts)
    return embeddings

def split_into_sentences(text):
    """Splits text into sentences using regex."""
    sentence_endings = r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s"
    sentences = re.split(sentence_endings, text)
    return [sentence.strip() for sentence in sentences if sentence.strip()]

def filter_answer(answer):
    """Filter out personal information (e.g., phone number, email)."""
    answer = re.sub(r"\+?\d{1,4}?[\s-]?\(?\d{1,3}?\)?[\s-]?\d{1,4}[\s-]?\d{1,4}[\s-]?\d{1,9}", "[PHONE_NUMBER]", answer)
    answer = re.sub(r"\S+@\S+\.\S+", "[EMAIL_ADDRESS]", answer)
    return answer

@app.post("/upload/")
async def upload_pdf(file: UploadFile):
    """Uploads a PDF file."""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Invalid file type. Only PDFs are allowed.")

    unique_filename = f"{uuid.uuid4()}_{file.filename}"
    file_path = os.path.join(UPLOAD_DIR, unique_filename)

    try:
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        logger.info(f"File uploaded: {file_path}")

        extracted_text = extract_text_from_pdf(file_path)
        logger.info("Text extracted from PDF")

        # Split the extracted text into sentences for better chunking
        sentences = split_into_sentences(extracted_text)

        # Generate embeddings for document sentences
        embeddings = generate_embeddings(sentences)
        logger.info("Embeddings generated")

        # Create FAISS index
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings.astype("float32"))
        faiss.write_index(index, f"{file_path}.index")
        logger.info("FAISS index created")

        session = SessionLocal()
        new_document = Document(filename=unique_filename, filepath=file_path, upload_date=str(uuid.uuid1()))
        session.add(new_document)
        session.commit()
        logger.info("Document saved to database")

        return JSONResponse(content={"message": "File uploaded successfully.", "filename": unique_filename})
    except Exception as e:
        logger.error(f"Error uploading file: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/question/")
async def ask_question(filename: str = Form(...), question: str = Form(...)):
    """Processes a question based on uploaded PDF content."""
    index_path = os.path.join(UPLOAD_DIR, f"{filename}.index")
    if not os.path.exists(index_path):
        raise HTTPException(status_code=404, detail="Index for the document not found.")

    try:
        # Read FAISS index
        index = faiss.read_index(index_path)

        # Generate embedding for the question
        question_embedding = generate_embeddings([question])[0].reshape(1, -1).astype("float32")

        # Perform search on the FAISS index
        D, I = index.search(question_embedding, k=1)
        logger.info(f"Search result - Distance: {D[0][0]}, Index: {I[0][0]}")

        # Retrieve the most relevant chunk (sentence) from the document
        session = SessionLocal()
        document = session.query(Document).filter(Document.filename == filename).first()
        file_path = document.filepath
        extracted_text = extract_text_from_pdf(file_path)

        # Split the extracted text into sentences
        sentences = split_into_sentences(extracted_text)
        relevant_sentence = sentences[I[0][0]]

        # Optionally, filter out personal info like phone number and email
        filtered_answer = filter_answer(relevant_sentence)

        # Return the filtered answer
        return JSONResponse(content={"answer": filtered_answer, "confidence": float(D[0][0])})

    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
