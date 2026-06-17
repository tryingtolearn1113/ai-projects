
# EPICS & PyDM AUTONOMOUS RAG AGENT (Production Version)
# Architecture: Gemma-4-12B (GGUF) + ChromaDB + Gradio Web UI
# ==============================================================================
import os
import chromadb
import gradio as gr
from llama_cpp import Llama

# 1. Environment Path Fix for Cloud GPUs (Kaggle/Colab)
os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")

print("=== Starting Enterprise EPICS AI Web Server ===")

# 2. Connect to the Persistent Vector Database
print("Connecting to ChromaDB Knowledge Base...")
# We use a try-except block to handle different environments gracefully
db_path = "/kaggle/working/chroma_db" if os.path.exists("/kaggle") else "./chroma_db"
chroma_client = chromadb.PersistentClient(path=db_path)
epics_docs_db = chroma_client.get_or_create_collection(name="epics_manuals_ultimate")
print(f"Database loaded! Found {epics_docs_db.count()} official documentation chunks. 📚")

# 3. Locate and Load the AI Model
target_file = "Gemma-4-12B-OBLITERATED-Q4_K_M.gguf"
absolute_model_path = None

# Scan common directories (Current Folder, Kaggle Input, Kaggle Working)
search_paths = ['.', '/kaggle/input', '/kaggle/working', '/content']
for search_path in search_paths:
    if absolute_model_path: break
    if os.path.exists(search_path):
        for root, dirs, files in os.walk(search_path):
            if target_file in files:
                absolute_model_path = os.path.join(root, target_file)
                break

if absolute_model_path is None:
    print("❌ ERROR: Model file missing! Please ensure the .gguf file is downloaded.")
else:
    print(f"Loading Model from: {absolute_model_path}")
    llm = Llama(
        model_path=absolute_model_path, 
        n_gpu_layers=-1,  # 100% GPU offloading
        n_ctx=12288,      # 12K Context Window (Safe for 15GB VRAM)
        n_threads=4,      # Multi-threading for speed
        verbose=False
    )
    print("Model Loaded Successfully! ✅")

    # ==========================================
    # CORE FUNCTION 1: The Auto-Converter
    # ==========================================
    def convert_file(file_obj):
        if file_obj is None:
            yield "Please upload a file first!"
            return
            
        try:
            # Handle Gradio file object formats safely
            if isinstance(file_obj, list): file_obj = file_obj[0]
            filepath = getattr(file_obj, "name", None) or getattr(file_obj, "path", None) or file_obj
            
            print(f"📂 [Processing File]: {filepath}")
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                file_content = f.read()
                
            # Truncate extremely large files to prevent GPU Out-of-Memory crashes
            if len(file_content) > 10000:
                file_content = file_content[:10000] + "\n\n# ... [FILE TRUNCATED TO PREVENT MEMORY OVERFLOW] ..."

            # RAG Search for conversion rules
            results = epics_docs_db.query(query_texts=["How to convert EDM screens and visibility to PyDM PyQt6"], n_results=3)
            rag_context = "\n---\n".join(results['documents'][0]) if results['documents'][0] else ""

            system_prompt = (
                "You are a professional, senior EPICS and PyDM Software Engineer. "
                "Convert the provided legacy EDL/EDM file into clean, modern PyQt6 Python code. "
                "Output ONLY the complete, working Python code. No explanations."
            )

            prompt = f"[OFFICIAL PyDM CONVERSION RULES]:\n{rag_context}\n\n[LEGACY FILE TO CONVERT]:\n{file_content}"

            stream = llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2048, 
                temperature=0.1, # Strict, non-creative coding
                stream=True
            )
            
            partial_code = ""
            for chunk in stream:
                if "content" in chunk["choices"][0]["delta"]:
                    partial_code += chunk["choices"][0]["delta"]["content"]
                    yield partial_code
                    
        except Exception as e:
            yield f"System Error during conversion: {str(e)}"

    # ==========================================
    # CORE FUNCTION 2: The RAG Chatbot
    # ==========================================
    def respond(message, history):
        system_prompt = (
            "You are a Senior EPICS and PyDM Software Engineer. "
            "Answer all questions directly, professionally, and with high technical accuracy. "
            "Always rely on the official documentation excerpts provided to you."
        )

        # RAG Search
        results = epics_docs_db.query(query_texts=[message], n_results=2)
        past_memories = ""
        if results['documents'][0]:
            past_memories = f"[OFFICIAL DOCUMENTATION EXCERPTS:\n{results['documents'][0][0]}\n]\n\n"

        # Build Sliding Window History (Keep last 3 turns)
        formatted_messages = [{"role": "system", "content": system_prompt}]
        for user_msg, ai_msg in history[-3:]:
            formatted_messages.append({"role": "user", "content": user_msg})
            formatted_messages.append({"role": "assistant", "content": ai_msg})
            
        formatted_messages.append({"role": "user", "content": past_memories + message})
        
        stream = llm.create_chat_completion(
            messages=formatted_messages, 
            max_tokens=1024, 
            temperature=0.1,
            stream=True
        )
        
        partial_message = ""
        for chunk in stream:
            if "content" in chunk["choices"][0]["delta"]:
                partial_message += chunk["choices"][0]["delta"]["content"]
                yield partial_message

    # ==========================================
    # WEB UI (Gradio Frontend)
    # ==========================================
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("# ⚛️ EPICS Auto-Converter & RAG Assistant")
        gr.Markdown("Powered by `Gemma-4-12B (GGUF)`, `ChromaDB`, and an Automated Web Crawler. Deployed for On-Premise capability.")
        
        with gr.Tabs():
            # Tab 1: Converter
            with gr.TabItem("📄 EDL to PyQt6 Converter"):
                with gr.Row():
                    with gr.Column():
                        file_input = gr.File(label="Upload Legacy Screen (.edl, .edm, .txt)")
                        convert_button = gr.Button("Convert File to Python 🚀", variant="primary")
                    with gr.Column():
                        output_code = gr.Textbox(label="Generated PyQt6 Code", lines=20, show_copy_button=True)
                convert_button.click(convert_file, inputs=file_input, outputs=output_code)
                
            # Tab 2: Chatbot
            with gr.TabItem("🤖 Technical RAG Assistant"):
                gr.ChatInterface(respond)

    print("\n🚀 Launching Web Server...")
    # share=True creates a public URL for Kaggle/Colab!
    demo.launch(share=True)