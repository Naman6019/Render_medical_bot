from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
import torch
import os

app = FastAPI()

# Paths
DB_FAISS_PATH = os.path.join(os.getcwd(), 'vectorstore', 'db_faiss')

# Initialize QA bot
def qa_bot():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': device})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    llm = CTransformers(
        model="TheBloke/Llama-2-7B-Chat-GGML",
        model_type="llama",
        max_new_tokens=512,
        temperature=0.7,
        device=device
    )
    qa_prompt = PromptTemplate(template="""Use the following pieces of information to answer the user's question.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

    Context: {context}
    Question: {question}

    Only return the helpful answer below and nothing else.
    Helpful answer:
    """, input_variables=['context', 'question'])
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=db.as_retriever(search_kwargs={'k': 2}),
        return_source_documents=True,
        chain_type_kwargs={'prompt': qa_prompt}
    )
    return qa

qa = qa_bot()

class Query(BaseModel):
    query: str

@app.post("/ask")
async def ask(query: Query):
    conversation_context = query.query
    try:
        result = qa({"query": conversation_context})
        answer = result["result"]
        sources = result["source_documents"]
        response = {"answer": answer, "sources": sources}
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"message": "Welcome to the Medical Chatbot API"}
