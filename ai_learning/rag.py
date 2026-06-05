"""
Simple RAG (Retrieval Augmented Generation)
Feed it documents → ask questions → AI answers
from YOUR documents!
"""
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Load API key
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")

def build_rag(document_path):
    """Build RAG from a text document"""

    print(f"Loading document: {document_path}")
    loader = TextLoader(document_path,
                        encoding='utf-8')
    documents = loader.load()

    print("Splitting into chunks...")
    splitter = CharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=30
    )
    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks!")

    print("Creating embeddings...")
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001"
    )

    print("Building vector database...")
    vectorstore = Chroma.from_documents(
        chunks,
        embeddings
    )
    print("Vector database ready! ✅\n")

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash",
        temperature=0
    )

    retriever = vectorstore.as_retriever(
            search_kwargs={"k": 5}
        )

    prompt = ChatPromptTemplate.from_template("""
        Answer the question based on the context below.
        If you don't know, say "I don't know".

        Context: {context}
        Question: {question}
        """)

    chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

    return chain

def main():
    print("=== RAG System ===\n")

    # Create test document if not exists
    doc_path = "my_document.txt"
    if not os.path.exists(doc_path):
        print("Creating test document...")
        with open(doc_path, 'w') as f:
            f.write("""
EPICS stands for Experimental Physics and Industrial Control System.
It is used in particle accelerators worldwide.
PAL stands for Pohang Accelerator Laboratory.
PAL is located in Pohang, South Korea.
PLS-II is the main storage ring at PAL.
The beam current at PLS-II is 300mA maximum.
LINAC stands for Linear Accelerator.
PyEPICS is a Python library for EPICS.
SOFB stands for Slow Orbit Feedback.
FOFB stands for Fast Orbit Feedback.
BPM stands for Beam Position Monitor.
EDM stands for Extensible Display Manager.
""")
        print("Test document created!\n")

    # Build RAG
    chain = build_rag(doc_path)

    # Interactive Q&A
    print("Ready! Ask questions about your document.")
    print("Type 'quit' to exit\n")
    print("=" * 30)

    while True:
        question = input("\nYou: ")

        if question.lower() == "quit":
            print("Goodbye!")
            break

        if question.strip() == "":
            continue

        print("Thinking...")
        response = chain.invoke(question)
        print(f"\nAI: {response}")

if __name__ == "__main__":
    main()