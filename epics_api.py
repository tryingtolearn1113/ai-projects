from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
import requests
from bs4 import BeautifulSoup
import os
from dotenv import load_dotenv

load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")

app = FastAPI(title="EPICS QA API")

# Load RAG chain at startup
chain = None

from collections import defaultdict
sessions = defaultdict(list)

class Question(BaseModel):
    text: str
    session_id: str = "default"

def fetch_and_build_chain():
    urls = [
        "https://docs.epics-controls.org/en/latest/guides/EPICS_Intro.html",
        "https://pyepics.github.io/pyepics/overview.html",
    ]
    all_docs = []
    for url in urls:
        try:
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            for tag in soup(['script', 'style']):
                tag.decompose()
            text = soup.get_text(separator='\n')
            lines = [l.strip() for l in text.splitlines() if l.strip()]
            clean_text = '\n'.join(lines)
            if clean_text:
                all_docs.append(Document(page_content=clean_text, metadata={"source": url}))
        except Exception as e:
            print(f"Failed: {url}: {e}")

    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(all_docs)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vectorstore = Chroma.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

    prompt = ChatPromptTemplate.from_template("""
You are an EPICS control system expert.
Answer based on the documentation below.
If you don't know, say "I don't know."

Documentation: {context}
Question: {question}
""")

    chain = (
        {
            "context": RunnableLambda(lambda x: x["question"]) | retriever,
            "question": RunnableLambda(lambda x: x["question"])
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

@app.on_event("startup")
async def startup_event():
    global chain
    print("Loading EPICS documentation...")
    chain = fetch_and_build_chain()
    print("API ready.")

@app.get("/")
def root():
    return {"message": "EPICS QA API is running"}

@app.post("/ask")
def ask(question: Question):
    if chain is None:
        return {"error": "API not ready yet, please wait"}
    
    if not question.text.strip():
        return {"error": "Question cannot be empty"}
    
    try:
        history = sessions[question.session_id]
        history_text = "\n".join(history[-6:])
        
        answer = chain.invoke({
            "question": question.text,
            "history": history_text
        })
        
        history.append(f"User: {question.text}")
        history.append(f"AI: {answer}")
        
        return {
            "question": question.text,
            "answer": answer,
            "session_id": question.session_id
        }
    except Exception as e:
        return {"error": f"Something went wrong: {str(e)}"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)