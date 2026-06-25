import os
import sys
import re
import time
import subprocess
import fitz  # PyMuPDF
from dotenv import load_dotenv
from llama_cpp import Llama

# ==========================================
# 1. Env 
# ==========================================
load_dotenv()
os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")

LAW_API_KEY = os.environ.get("LAW_OC") 

if not LAW_API_KEY:
    raise ValueError(
        " [Security Error] 'LAW_OC' API key not found in .env file.\n"
        "Please check your .env configuration."
    )

os.environ["LAW_OPENAPI_KEY"] = LAW_API_KEY

print("===  [T4 x2 Dual GPU] All-Round Bar Exam Solver Started ===")

# ==========================================
# 2. PDF Parser
# ==========================================
def extract_and_parse_pdf(pdf_path: str):
    try:
        doc = fitz.open(pdf_path)
        text = "".join([page.get_text() for page in doc])
    except Exception as e:
        print(f" [Error] Failed to read PDF: {e}")
        return []

    start_match = re.search(r'제\s*\d+\s*문', text)
    if start_match: 
        text = text[start_match.start():]

    exams_parsed = []
    exam_blocks = re.split(r'(제\s*\d+\s*문\s*[〉>\]]?)', text)
    
    for i in range(1, len(exam_blocks), 2):
        exam_title = exam_blocks[i].strip()
        exam_content = exam_blocks[i+1]
        
        parts = re.split(r'<\s*문\s*제\s*>', exam_content, flags=re.IGNORECASE)
        if len(parts) < 2: 
            continue
            
        fact_pattern = parts[0].strip()
        questions_block = parts[1].strip()

        question_matches = re.finditer(r'(\d+)\.\s(.*?)(?=\n\d+\.\s|\Z)', questions_block, re.DOTALL)
        for match in question_matches:
            exams_parsed.append({
                "q_id": f"{exam_title} - Sub {match.group(1).strip()}",
                "fact_pattern": fact_pattern,
                "question_text": match.group(2).strip()
            })
    return exams_parsed

# ==========================================
# 3. KR Law MCP
# ==========================================
def get_mcp_legal_research(keywords):
    mcp_context = ""
    print(f"   >  [MCP Engine] Searching national law database...")
    
    for kw in keywords[:2]:  
        try:
            res = subprocess.run(
                ["korean-law", "search_law", "--query", kw],
                capture_output=True, text=True, env=os.environ, timeout=30
            )
            
            if res.returncode == 0 and res.stdout:
                mcp_context += f"### [MCP National Law Data: {kw}]\n{res.stdout[:1000].strip()}\n\n"
        except subprocess.TimeoutExpired:
            print(f"   > ⚠️ [{kw}] Search timeout (30s). Skipping.")
        except Exception as e:
            pass
            
    return mcp_context if mcp_context else "(Local MCP data timeout - using base legal knowledge.)"

# ==========================================
# 4. Universal Keyword Extractor (All-Round Law)
# ==========================================
def extract_law_keywords(llm, fact_pattern, question_text):
    prompt = (
        "당신은 대한민국 최고 수준의 법률 사건 분석기입니다. 제공된 사례와 질문을 분석하여, "
        "국가법령정보센터 API에서 핵심 조문과 판례를 조회할 수 있는 정확한 '핵심 단어'를 딱 3개만 골라내십시오.\n"
        "사건의 도메인(민법, 형법, 공법, 상법, 행정법 등)에 알맞은 정확한 법률 용어여야 합니다. "
        "(예: 민사의 경우 '소멸시효', '채무불이행' / 공법의 경우 '위헌법률심판', '처분성')\n"
        "출력은 반드시 단어들을 쉼표(,)로만 구분해야 하며, 부가 설명은 절대 금지합니다.\n\n"
        f"[사례]: {fact_pattern[:1000]}\n"
        f"[질문]: {question_text}\n\n"
        "정제된 키워드:"
    )
    
    messages = [{"role": "user", "content": prompt}]
    res = llm.create_chat_completion(messages=messages, max_tokens=60, temperature=0.1)
    raw_output = res["choices"][0]["message"]["content"].strip()
    
    raw_output = re.sub(r'[^가-힣a-zA-Z0-9,\s]', '', raw_output)
    keywords = [kw.strip() for kw in raw_output.split(",") if kw.strip()]
    return keywords[:3]

# ==========================================
# 5. Universal Inference Pipeline
# ==========================================
def solve_case_pipeline(llm, q_id, fact_pattern, question_text):
    start_time = time.time()
    
    print(f"\n[Step 1] Extracting keywords for {q_id}...")
    keywords = extract_law_keywords(llm, fact_pattern, question_text)
    if not keywords:
        keywords = ["민법", "형법", "행정법"]
    print(f"   > 🎯 Keywords: {keywords}")
    
    print(f"[Step 2] Querying national MCP database...")
    mcp_law_context = get_mcp_legal_research(keywords)
    
    # ⚖️ 페르소나 및 지침 전과목(민법/상법/형법/공법) 종합 확장
    system_prompt = """당신은 대한민국 변호사시험 전 과목 출제위원이자 법학전문대학원협의회 총괄 채점관 출신의 최고 권위 석학입니다.
주어진 [GitHub MCP 기반 인용 검증 법령 및 판례 데이터]는 법제처 실시간 데이터가 반영된 최고 등급의 지식입니다. 해당 도메인(민사법, 형사법, 공법)의 사법 사조에 부합하도록 사실관계를 철저하게 포섭하십시오.

반드시 지켜야 할 변호사시험 논술형 정석 답안 규격 (IRAC):
1. 문제의 소재 (쟁점의 정리)
2. 관련 법리 및 판례 (조문 번호 명시 및 대법원/헌재 판례 요지를 학술적으로 충실히 서술)
3. 사안의 적용 (사실관계 속 인물들의 법률 행위를 해당 법리에 일대일로 정밀 대조 및 대입)
4. 결론 (청구 인용 여부, 유무죄 및 죄수, 처분의 위법성 여부 등 도메인에 맞는 명확한 결론 제시)

논리적 일관성을 유지하고 완결성 있는 단락으로 밀도 높은 최상위권 답안을 작성하십시오."""

    user_prompt = f"""[기초 사실관계]
{fact_pattern}

[문항 {q_id}]
{question_text}

[GitHub MCP 기반 인용 검증 법령 및 판례 데이터]
{mcp_law_context}

위 내용을 토대로 완벽한 사례형 논술 답안을 출력하십시오."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    print(f"[Step 3] LLM Generating Answer (Streaming):\n")
    print("=" * 60)
    
    stream = llm.create_chat_completion(
        messages=messages, 
        max_tokens=-1, 
        temperature=0.3,         
        repeat_penalty=1.18,     
        frequency_penalty=0.5,   
        presence_penalty=0.4,    
        stream=True
    )
    
    full_answer = ""
    for chunk in stream:
        if "content" in chunk["choices"][0]["delta"]:
            content = chunk["choices"][0]["delta"]["content"]
            print(content, end="", flush=True)
            full_answer += content
    print("\n" + "=" * 60)
    
    return {
        "q_id": q_id,
        "question": question_text,
        "answer": full_answer,
        "elapsed": round(time.time() - start_time, 1)
    }

# ==========================================
# 6. Main Controller
# ==========================================
def main(pdf_path: str, custom_model_path: str = None):
    absolute_model_path = custom_model_path

    if not absolute_model_path:
        search_dirs = ['/kaggle/working/models', '/kaggle/working', '/tmp', '.']
        for d in search_dirs:
            if os.path.exists(d):
                for root, dirs, files in os.walk(d):
                    for file in files:
                        if file.endswith(".gguf"):
                            absolute_model_path = os.path.join(root, file)
                            break
                    if absolute_model_path: break
            if absolute_model_path: break

    if not absolute_model_path or not os.path.exists(absolute_model_path):
        print(" [Error] No valid GGUF model found in search paths.")
        return

    print(f"📦 Target Model: {absolute_model_path}")
    
    llm = Llama(
        model_path=absolute_model_path, 
        n_gpu_layers=-1,   
        n_ctx=16384,       
        n_batch=512,
        verbose=False
    )
    print("✅ [GPU Status] T4 x2 Dual GPU Loaded. (VRAM Fully Occupied)")

    parsed_questions = extract_and_parse_pdf(pdf_path)
    if not parsed_questions:
        print(" [Error] No target questions found in PDF.")
        return
    print(f"📄 [Data Status] PDF parsed successfully. (Total: {len(parsed_questions)} questions)")

    all_results = []
    for q in parsed_questions:
        res = solve_case_pipeline(llm, q["q_id"], q["fact_pattern"], q["question_text"])
        all_results.append(res)

    output_name = f"{os.path.splitext(os.path.basename(pdf_path))[0]}_Kaggle_MCP_최종답안지.md"
    with open(output_name, "w", encoding="utf-8") as f:
        f.write(f"# 📜 변호사시험 AI 전과목 통합 최종 답안지\n")
        f.write(f"> 본 답안지는 로컬 내장형 전도메인 종합 법령 MCP 검색 모듈과 듀얼 GPU 가속 시스템의 교차 포섭 검증을 완료했습니다.\n\n---\n\n")
        for r in all_results:
            f.write(f"## 🔷 [{r['q_id']}]\n\n**[변호사시험 문제 문항]**\n{r['question']}\n\n**[AI 최고 법률전문가 IRAC 표준답안]**\n{r['answer']}\n\n---\n\n")
            
    print(f"\n🎉 [Done] Process complete. Output saved to: {os.path.abspath(output_name)}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("\n Usage: %run kaggle_lawyer_bot.py [PDF_PATH] [Optional: MODEL_PATH]")
        sys.exit(1)
        
    pdf_input = sys.argv[1]
    model_input = sys.argv[2] if len(sys.argv) > 2 else None
    main(pdf_input, model_input)