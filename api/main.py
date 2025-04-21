import os
from fastapi import FastAPI, UploadFile, File,Depends
from pydantic import BaseModel
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from api.database import SessionLocal, engine
from api.auth_utils import get_current_user_id
from sqlalchemy.orm import Session
from fastapi.middleware.cors import CORSMiddleware
import api.firebase_config
from api.models import Base, User, ChatHistory

from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi import Depends, HTTPException
from firebase_admin import auth
security = HTTPBearer()
Base.metadata.create_all(bind=engine)
# Allow requests from your React frontend
origins = [
    "http://localhost:3000"
]

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
# Load API keys from .env
load_dotenv()
api_key = os.getenv('API_KEY')

# Paths
FAISS_PATH = "FAISSSTORE"
TEMP_FOLDER = "temp"

# Ensure directories exist
os.makedirs(FAISS_PATH, exist_ok=True)
os.makedirs(TEMP_FOLDER, exist_ok=True)

app = FastAPI(title="PDF Upload & Q&A API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
def get_current_user_id(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        decoded_token = auth.verify_id_token(credentials.credentials)
        return {
            "uid": decoded_token["uid"],
            "name": decoded_token.get("name", ""),
            "email": decoded_token.get("email", "")  # ✅ Extract email
        }
    except Exception as e:
        print("Firebase token verification failed:", e)
        raise HTTPException(status_code=401, detail="Invalid Firebase token")
def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    text = ""
    reader = PdfReader(pdf_path)
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def process_pdf_and_store(pdf_path):
    """Extract text, split into chunks, and store in FAISS."""
    text = extract_text_from_pdf(pdf_path)
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
    text_chunks = text_splitter.split_text(text)
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    if os.path.exists(f"{FAISS_PATH}/index.faiss") and os.path.exists(f"{FAISS_PATH}/index.pkl"):
        faiss_store = FAISS.load_local(FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    else:
        faiss_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    
    faiss_store.save_local(FAISS_PATH)
    return "PDF Processed & Stored in FAISS!"

@app.post("/upload/")
async def upload_pdf(file: UploadFile = File(...)):
    """API endpoint to upload PDF and process it."""
    file_path = os.path.join(TEMP_FOLDER, file.filename)
    
    with open(file_path, "wb") as f:
        f.write(file.file.read())
    
    message = process_pdf_and_store(file_path)
    os.remove(file_path)  # Clean up after processing
    
    return {"message": message}

class QueryModel(BaseModel):
    question: str

def get_text_conversation_chain():
    """Define the prompt template and load the language model."""
    prompt_template = """
      You are a helpful AI assistant. Use ONLY the context below to answer the question. 
      If the context does not contain the answer, respond with: "I could not find the answer in the provided documents.


        Context: {context}
        Question: {question}

        Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# @app.post("/ask/")
# async def ask_question(query: QueryModel, db: Session = Depends(get_db), user_id: str = Depends(get_current_user_id)):
#     """API endpoint to answer user questions based on stored embeddings."""
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

#     if os.path.exists(f"{FAISS_PATH}/index.faiss") and os.path.exists(f"{FAISS_PATH}/index.pkl"):
#         faiss_store = FAISS.load_local(FAISS_PATH, embeddings, allow_dangerous_deserialization=True)

#         docs = faiss_store.similarity_search(query.question, k=5)

#         # context = "\n\n".join(doc.page_content for doc in docs)
        
#         chain = get_text_conversation_chain()
#         response = chain({"input_documents": docs, "question": query.question}, return_only_outputs=True)
        
#         return {"answer": response["output_text"]}
    
#     return {"error": "FAISS index not found. Please upload and process PDFs first!"}


@app.post("/ask/")
async def ask_question(query: QueryModel, db: Session = Depends(get_db), user_data: dict = Depends(get_current_user_id)):
    """API endpoint to answer user questions based on shared admin-uploaded FAISS index."""
    user_id = user_data["uid"]
    user_email = user_data["email"]
    user_name = user_data["name"]

    user = db.query(User).filter((User.id == user_id) | (User.email == user_email)).first()

    if not user:
        new_user = User(
            id=user_id,
            name=user_name,
            email=user_email
        )
        db.add(new_user)
        db.commit()

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Shared FAISS index path
    if os.path.exists(f"{FAISS_PATH}/index.faiss") and os.path.exists(f"{FAISS_PATH}/index.pkl"):
        faiss_store = FAISS.load_local(FAISS_PATH, embeddings, allow_dangerous_deserialization=True)

        docs = faiss_store.similarity_search(query.question, k=5)

        chain = get_text_conversation_chain()
        response = chain({"input_documents": docs, "question": query.question}, return_only_outputs=True)

        # Optional: Store user Q&A in MySQL
        chat = ChatHistory(user_id=user.id, question=query.question, answer=response["output_text"])
        db.add(chat)
        db.commit()

        return {"answer": response["output_text"]}

    return {"error": "FAISS index not found. Please ask admin to upload PDFs first."}

@app.get("/history/")
def get_history(user_id: str = Depends(get_current_user_id), db: Session = Depends(get_db)):
    chats = db.query(ChatHistory).filter(ChatHistory.user_id == user_id).all()
    return [{"question": c.question, "answer": c.answer, "timestamp": c.timestamp} for c in chats]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
