import os
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_text_splitters import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")

def fetch_webpage(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url,
                                headers=headers,
                                timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        for tag in soup(['script', 'style']):
            tag.decompose()
            
        text = soup.get_text(separator='\n')
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        return '\n'.join(lines)
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return ""
    
def build_rag(urls):
    all_docs = []
    for url in urls:
        text = fetch_webpage(url)
        if text:
            doc = Document(
                page_content=text,
                metadata={"source": url}
            )
            all_docs.append(doc)
            print(f"Got {len(text)} characters")
    if not all_docs:
        print("No documents is loaded")
        return None
    
    print("\n Splitting into chunks")
    splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(all_docs)
    print(f"Created{len(chunks)} chunks")
    
    print("\n Embeddings...")
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001"
    )
    print("\nBuilding vector database...")
    vectorstore = Chroma.from_documents(chunks, embeddings)
    print("Ready \n")
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash",
        temperature=0
    )

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )
    
    prompt = ChatPromptTemplate.from_template("""
You are an EPICS control system expert.

Documentation:{context}
History:{history}
Question: {question}
""")
    chain = (
        {
            "context": (lambda x: x["question"]) | retriever,
            "question": RunnablePassthrough(),
            "history": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

def main():
    urls = [
         # EPICS official docs
    "https://docs.epics-controls.org/en/latest/getting-started/installation.html",
    "https://docs.epics-controls.org/en/latest/guides/EPICS_Intro.html",
    
    # PyEPICS - Python library you used at PAL
    "https://pyepics.github.io/pyepics/overview.html",
    "https://pyepics.github.io/pyepics/pv.html",
    
    # PyDM - what you converted to
    "https://slaclab.github.io/pydm/",
    "https://slaclab.github.io/pydm/tutorials/index.html",
    ]
    chain = build_rag(urls)
    if not chain:
        return
    chat_history = []
    
    while True:
        question = input("\nYou: ")
        
        if question.lower() == "quit":
            break
            
        if question.strip() == "":
            continue
        
        response = chain.invoke({
            "question": question,
            "history": "\n".join(chat_history)
        })
        
        print(f"\nAI: {response}")
        
        chat_history.append(f"User: {question}")
        chat_history.append(f"AI: {response}")
        
        if len(chat_history) > 6:
            chat_history = chat_history[-6:]
    
    
    
    
if __name__ == "__main__":
    main()
    