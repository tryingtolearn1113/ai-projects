"""
Web RAG - reads websites and answers questions
Perfect for EDM to PyDM conversion knowledge!
"""
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
    """Fetch and clean text from a webpage"""
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, 
                              headers=headers,
                              timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove scripts and styles
        for tag in soup(['script', 'style']):
            tag.decompose()
            
        text = soup.get_text(separator='\n')
        # Clean up whitespace
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        return '\n'.join(lines)
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return ""

def build_web_rag(urls):
    """Build RAG from multiple webpages"""
    
    all_docs = []
    for url in urls:
        print(f"Reading: {url}")
        text = fetch_webpage(url)
        if text:
            doc = Document(
                page_content=text,
                metadata={"source": url}
            )
            all_docs.append(doc)
            print(f"✅ Got {len(text)} characters")

    if not all_docs:
        print("No documents loaded!")
        return None

    print("\nSplitting into chunks...")
    splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(all_docs)
    print(f"Created {len(chunks)} chunks!")

    print("Creating embeddings...")
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001"
    )

    print("Building vector database...")
    vectorstore = Chroma.from_documents(chunks, embeddings)
    print("Ready! ✅\n")

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash",
        temperature=0
    )

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )

    prompt = ChatPromptTemplate.from_template("""
You are an EPICS control system expert.
Answer based on the documentation below.
If you don't know, say "I don't know".

Documentation:
{context}

Question: {question}
""")

    chain = (
        {"context": retriever, 
         "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain

def main():
    print("=== EDM to PyDM Web RAG ===\n")

    # EDM and PyDM documentation URLs
    urls = [
        "https://slaclab.github.io/edm2pydm/",
        "https://slaclab.github.io/edm2pydm/example-walk-through/convert_existing_edm.html",
        "https://slaclab.github.io/edm2pydm/how-to/visibilityPV.html",
        "https://slaclab.github.io/edm2pydm/how-to/colorPV.html",
    ]

    chain = build_web_rag(urls)
    if not chain:
        return

    print("Ask questions about EDM to PyDM conversion!")
    print("Type 'quit' to exit\n")
    print("=" * 30)

    while True:
        question = input("\nYou: ")

        if question.lower() == "quit":
            break

        if question.strip() == "":
            continue

        print("Thinking...")
        response = chain.invoke(question)
        print(f"\nAI: {response}")

if __name__ == "__main__":
    main()