import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from langchain_text_splitters import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
import requests
from bs4 import BeautifulSoup

load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0
)
    
@tool
def web_tool(query: str) -> str:
    """Search the web for current or recent information."""
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        url = f"https://www.google.com/search?q={query}"
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        results = soup.find_all('div', class_='BNeawe')
        if results:
            return ' '.join([r.get_text() for r in results[:5]])
        return "No results found."
    except Exception as e:
        return f"Search failed: {e}"

def build_doc_tool(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    @tool
    def search_docs(query: str) -> str:
        """Search EPICS documentation for technical information about EPICS, PyEPICS, or PyDM."""
        docs = retriever.invoke(query)
        if docs:
            return "\n\n".join([d.page_content for d in docs])
        return "No relevant documents found."
    
    return search_docs

def fetch_webpage(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        for tag in soup(['script', 'style']):
            tag.decompose()
        text = soup.get_text(separator='\n')
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        return '\n'.join(lines)
    except Exception as e:
        return ""

def build_vectorstore(urls):
    print("Loading EPICS documentation...")
    all_docs = []
    for url in urls:
        text = fetch_webpage(url)
        if text:
            all_docs.append(Document(page_content=text, metadata={"source": url}))
    
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(all_docs)
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vectorstore = Chroma.from_documents(chunks, embeddings)
    print(f"Ready — {len(chunks)} chunks loaded.\n")
    return vectorstore

def main():
    urls = [
        "https://docs.epics-controls.org/en/latest/guides/EPICS_Intro.html",
        "https://pyepics.github.io/pyepics/overview.html",
        "https://pyepics.github.io/pyepics/pv.html",
    ]
    
    vectorstore = build_vectorstore(urls)
    doc_tool = build_doc_tool(vectorstore)
    tools = [web_tool, doc_tool]
        
    agent = create_react_agent(
    llm,
    tools=[web_tool, doc_tool],
    prompt="You are an EPICS control system expert. Use your tools to answer questions."
)
    
    
    print("=== EPICS Agent ===")
    print("Ask anything about EPICS. Type 'quit' to exit.\n")
    
    while True:
        question = input("\nYou: ")
        
        if question.lower() == "quit":
            break
            
        if question.strip() == "":
            continue
        
        print("\nThinking...\n")
        response = agent.invoke({"messages": [{"role": "user", "content": question}]})
        content = response['messages'][-1].content
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get('type') == 'text':
                    print(f"\nAI: {block['text']}")
        else:
            print(f"\nAI: {content}")

if __name__ == "__main__":
    main()