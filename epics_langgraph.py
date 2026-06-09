import os
from dotenv import load_dotenv
from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.tools import tool
import requests
from bs4 import BeautifulSoup

load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")

llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0)

class State(TypedDict):
    question: str          # the user's question
    documents: List[str]   # retrieved doc chunks
    web_results: str       # web search results
    answer: str            # final answer
    use_web: bool          # did docs fail?
    
def build_retriever(urls):
    print("Loading documentation...")
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
    print(f"Ready — {len(chunks)} chunks\n")
    return vectorstore.as_retriever(search_kwargs={"k": 3})

def make_retrieve_node(retriever):
    def retrieve(state: State) -> State:
        """Search ChromaDB for relevant documents."""
        docs = retriever.invoke(state["question"])
        documents = [d.page_content for d in docs]
        return {"documents": documents}
    return retrieve

def grade(state: State) -> State:
    """Check if retrieved documents are useful."""
    documents = state["documents"]
    
    # Simple check — if no documents or all very short, use web
    if not documents:
        return {"use_web": True}
    
    total_content = " ".join(documents)
    if len(total_content) < 100:
        return {"use_web": True}
    
    return {"use_web": False}

def route_after_grade(state: State) -> str:
    """Decide next node based on grade result."""
    if state["use_web"]:
        return "web_search"
    return "generate"

def web_search(state: State) -> State:
    """Search the web when docs don't have the answer."""
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        url = f"https://www.google.com/search?q={state['question']}"
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        results = soup.find_all('div', class_='BNeawe')
        if results:
            web_results = ' '.join([r.get_text() for r in results[:5]])
        else:
            web_results = "No results found."
    except Exception as e:
        web_results = f"Search failed: {e}"
    
    return {"web_results": web_results}

def generate(state: State) -> State:
    """Generate final answer from docs or web results."""
    question = state["question"]
    
    if state.get("use_web") and state.get("web_results"):
        context = state["web_results"]
        source = "web search"
    else:
        context = "\n\n".join(state.get("documents", []))
        source = "documentation"
    
    prompt = f"""Answer this question using the {source} below.
If you don't know, say "I don't know."

Context: {context}

Question: {question}"""
    
    response = llm.invoke(prompt)
    return {"answer": response.content}

def build_graph(retriever):
    graph = StateGraph(State)
    
    # Add nodes
    graph.add_node("retrieve", make_retrieve_node(retriever))
    graph.add_node("grade", grade)
    graph.add_node("web_search", web_search)
    graph.add_node("generate", generate)
    
    # Add edges
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "grade")
    graph.add_conditional_edges(
        "grade",
        route_after_grade,
        {
            "web_search": "web_search",
            "generate": "generate"
        }
    )
    graph.add_edge("web_search", "generate")
    graph.add_edge("generate", END)
    
    return graph.compile()

def main():
    urls = [
        "https://docs.epics-controls.org/en/latest/guides/EPICS_Intro.html",
        "https://pyepics.github.io/pyepics/overview.html",
        "https://pyepics.github.io/pyepics/pv.html",
    ]
    
    retriever = build_retriever(urls)
    graph = build_graph(retriever)
    
    print("=== EPICS LangGraph QA ===")
    print("Type 'quit' to exit.\n")
    
    while True:
        question = input("\nYou: ")
        
        if question.lower() == "quit":
            break
        
        if question.strip() == "":
            continue
        
        print("\nThinking...\n")
        result = graph.invoke({
            "question": question,
            "documents": [],
            "web_results": "",
            "answer": "",
            "use_web": False
        })
        answer = result['answer']
        if isinstance(answer, list):
            for block in answer:
                if isinstance(block, dict) and block.get('type') == 'text':
                    print(f"\nAI: {block['text']}")
        else:
            print(f"\nAI: {answer}")
        
        if result['use_web']:
            print("(answered from web search)")
        else:
            print("(answered from documentation)")

if __name__ == "__main__":
    main()