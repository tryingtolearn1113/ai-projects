%%writefile /kaggle/working/bar_exam_solver_RAG.py

import os
import sys
import re
import requests
import time
import fitz
from dotenv import load_dotenv
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from llama_cpp import Llama

load_dotenv(dotenv_path="/kaggle/working/.env")

# ==========================================
# 1. PDF Parser
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
    exam_blocks = re.split(r'(제\s*\d+\s*문(?:의\s*\d+)?\s*[〉>\]]?)', text)

    seen_titles = {}  # 중복 블록 제거용: 정규화된 제목 → 인덱스

    for i in range(1, len(exam_blocks), 2):
        exam_title = exam_blocks[i].strip()
        exam_content = exam_blocks[i + 1]

        parts = re.split(r'<\s*문\s*제\s*>', exam_content, flags=re.IGNORECASE)
        if len(parts) < 2:
            continue

        # 제목 정규화 (〉, >, ] 등 제거해서 비교)
        normalized = re.sub(r'[〉>\]\s]', '', exam_title)

        fact_pattern = parts[0].strip()
        questions_block = parts[1].strip()

        # 소문항 파싱: 숫자. 패턴 우선
        question_matches = list(re.finditer(
            r'\n\s*(\d+)\.\s(.*?)(?=\n\s*\d+\.\s|\Z)',
            "\n" + questions_block, re.DOTALL
        ))

        if not question_matches:
            continue

        questions = [
            {
                "q_id": f"{normalized}〉 - 문 {m.group(1).strip()}",
                "fact_pattern": fact_pattern,
                "question_text": m.group(2).strip()
            }
            for m in question_matches
        ]

        # 같은 제목의 블록이 이미 있으면 소문항이 더 많은 걸로 교체
        if normalized in seen_titles:
            existing_count = sum(1 for q in exams_parsed if normalized in q['q_id'])
            if len(questions) > existing_count:
                exams_parsed = [q for q in exams_parsed if normalized not in q['q_id']]
                exams_parsed.extend(questions)
                seen_titles[normalized] = len(exams_parsed) - len(questions)
        else:
            seen_titles[normalized] = len(exams_parsed)
            exams_parsed.extend(questions)

    print(f"✅ 파싱 완료: {len(exams_parsed)}개 소문항")
    return exams_parsed


# ==========================================
# 2. 하이브리드 검색기
#    법전(korean_law_core) + 판례(korean_precedent_core) 동시 검색.
#    판례 DB가 없어도 법전만으로 정상 작동.
# ==========================================
class LegalRetriever:
    def __init__(self):
        src_path  = '/kaggle/input/datasets/shleepracticing/korean-law-chromadb-bge-m3'
        dest_path = '/kaggle/working/legal_db_persisted'

        # Kaggle input은 읽기 전용 → working으로 복사
        if not os.path.exists(dest_path):
            if not os.path.exists(src_path):
                print(" ⚠️ [Error] DB 원본 경로를 찾을 수 없습니다.")
                sys.exit(1)
            import shutil
            print(f" 📋 DB 복사 중: {src_path} → {dest_path}")
            shutil.copytree(src_path, dest_path)
            print(" ✅ DB 복사 완료")
        else:
            print(f" ✅ DB 이미 존재: {dest_path}")

        client = PersistentClient(path=dest_path)

        # 법전 컬렉션 (필수)
        self.law_collection = client.get_collection(name="korean_law_bge_m3")

        # 판례 컬렉션 (선택 — 없으면 법전만 사용)
        try:
            self.prec_collection = client.get_collection(name="korean_precedent_core")
            prec_count = self.prec_collection.count()
            print(f" ⚡ [System] 판례 컬렉션 로드: {prec_count:,}개 청크")
        except Exception:
            self.prec_collection = None
            print(" ℹ️  [System] 판례 컬렉션 없음 — 법전만 사용")

        self.embedding_model = SentenceTransformer(
            "BAAI/bge-m3", device="cpu"
        )

        # BM25는 법전만 (판례는 양이 많아 메모리 문제 발생 가능)
        all_data = self.law_collection.get()
        self.documents = all_data["documents"]
        self.ids       = all_data["ids"]

        tokenized_corpus = [self._bigram(doc) for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized_corpus)
        print(f" ⚡ [System] 법전 {len(self.documents)}개 조문 BM25 인덱스 생성 완료.")

    # BM25는 법전만 (판례는 양이 많아 메모리 문제 발생 가능)gemini added
        all_data = self.law_collection.get()
        self.documents = all_data["documents"]
        self.ids       = all_data["ids"]

        tokenized_corpus = [self._bigram(doc) for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized_corpus)
        print(f" ⚡ [System] 법전 {len(self.documents)}개 조문 BM25 인덱스 생성 완료.")

        # 💡 [여기부터 추가] 검증을 위한 조문 프리픽스(법명_제O조) 고속 캐싱
        self.valid_law_prefixes = set()
        for db_id in self.ids:
            # db_id 예시: "형법_제164조_①_123" 또는 "형법_제164조_full_123"
            parts = db_id.split('_')
            if len(parts) >= 2:
                prefix = f"{parts[0]}_{parts[1]}" # "형법_제164조"
                self.valid_law_prefixes.add(prefix)
        print(f" ⚡ [System] 조문 자동검증용 캐시 생성 완료: {len(self.valid_law_prefixes):,}개 고유 조문")

    def _bigram(self, text: str):
        words = text.split()
        tokens = []
        for word in words:
            w = re.sub(r'[^\w]', '', word)
            if not w:
                continue
            if len(w) <= 2:
                tokens.append(w)
            else:
                for i in range(len(w) - 1):
                    tokens.append(w[i:i + 2])
        return tokens

    def _search_collection(self, collection, query_vector: list, top_k: int) -> list:
        """단일 컬렉션 dense 검색. (doc_id, text, similarity) 리스트 반환."""
        results = collection.query(
            query_embeddings=[query_vector], n_results=top_k
        )
        output = []
        if results["ids"] and results["ids"][0]:
            for doc_id, doc, dist in zip(
                results["ids"][0],
                results["documents"][0],
                results["distances"][0]
            ):
                output.append((doc_id, doc, 1.0 - dist))
        return output

    def retrieve(self, query: str, top_k: int = 12) -> str:
        query_vector = self.embedding_model.encode([query]).tolist()[0]

        # ── 법전 Dense 검색 ──
        law_dense = self._search_collection(self.law_collection, query_vector, top_k * 2)

        law_dense_ids = {r[0] for r in law_dense}
        law_dense_sim = {r[0]: r[2] for r in law_dense}

        # ── 법전 BM25 검색 ──
        tokenized_query = self._bigram(query)
        sparse_scores   = self.bm25.get_scores(tokenized_query)
        top_sparse      = sorted(
            range(len(sparse_scores)),
            key=lambda i: sparse_scores[i], reverse=True
        )[:top_k * 2]

        BM25_THRESHOLD = 15.0
        bm25_whitelist = set()
        if top_sparse and sparse_scores[top_sparse[0]] >= BM25_THRESHOLD:
            bm25_whitelist.add(self.ids[top_sparse[0]])

        # ── 법전 RRF 합산 ──
        rrf = {}
        for rank, (doc_id, _, _) in enumerate(law_dense):
            rrf[doc_id] = rrf.get(doc_id, 0) + 1.0 / (60 + rank)
        for rank, idx in enumerate(top_sparse):
            doc_id = self.ids[idx]
            if sparse_scores[idx] > 0:
                rrf[doc_id] = rrf.get(doc_id, 0) + 1.0 / (60 + rank)

        # dense 미검출 + BM25 약한 매칭 제외
        allowed = law_dense_ids | bm25_whitelist
        filtered_law = {
            doc_id: score for doc_id, score in rrf.items()
            if doc_id in allowed
        }
        top_law = sorted(filtered_law.items(), key=lambda x: x[1], reverse=True)[:top_k]

        # ── 판례 Dense 검색 (있을 때만) ──
        top_prec = []
        if self.prec_collection is not None:
            prec_k    = max(4, top_k // 3)   # 판례는 조문의 1/3 비중
            prec_hits = self._search_collection(self.prec_collection, query_vector, prec_k)
            # 유사도 0.50 이상만 포함 (너무 먼 판례는 노이즈)
            top_prec  = [(doc_id, doc) for doc_id, doc, sim in prec_hits if sim >= 0.50]

        best_sim = max(law_dense_sim.values()) if law_dense_sim else 0.0
        print(
            f"   > 검색 완료 (법전 유사도: {best_sim:.4f}, "
            f"법전 조문: {len(top_law)}개, 판례: {len(top_prec)}개)"
        )

        # ── 컨텍스트 조합: 법전 먼저, 판례 뒤에 ──
        context = "=== [관련 법령 조문] ===\n"
        for doc_id, _ in top_law:
            idx = self.ids.index(doc_id)
            context += self.documents[idx] + "\n\n"

        if top_prec:
            context += "\n=== [관련 판례] ===\n"
            for _, doc in top_prec:
                context += doc + "\n\n"

        return context.strip()

    def verify_citations(self, answer_text: str) -> list:
        """
        답안에 쓰인 '제N조' 패턴을 추출해서 DB에 실제로 있는지 확인.
        """
        pattern = r'([가-힣]+법)\s*제(\d+)조'
        citations = re.findall(pattern, answer_text)
        warnings = []

        for law_name, article_no in sorted(set(citations)):
            candidate_id = f"{law_name}_제{article_no}조"
            
            # 💡 [변경됨] DB 조회 없이 만들어둔 Set에 있는지 바로 확인
            if candidate_id not in self.valid_law_prefixes:
                warnings.append(
                    f"⚠️ '{law_name} 제{article_no}조' — DB에서 확인되지 않음 "
                    f"(조문 번호 재확인 필요, 존재하지 않는 조문일 가능성)"
                )
        return warnings


# ==========================================
# 3. 소문항 사실관계 추출
# ==========================================
def get_relevant_fact_segment(fact_pattern: str, question_text: str) -> str:
    match = re.search(r'\((\d+)\)\s*에서', question_text)
    if match:
        num = match.group(1)
        start = re.search(rf'\({num}\)', fact_pattern)
        if start:
            segment = fact_pattern[start.start():]
            nxt = re.search(rf'\({int(num) + 1}\)', segment)
            return segment[:nxt.start()].strip() if nxt else segment.strip()
    return fact_pattern.strip()


# ==========================================
# 4. 법률 키워드 추출
# ==========================================
def generate_search_keywords(llm, fact_pattern: str, question_text: str) -> str:
    print("   🧠 [Agent] 법률 키워드 추출 중...")
    prompt = (
        "당신은 대한민국 변호사시험 출제위원입니다.\n"
        "아래 사안을 읽고, 이 문제를 풀기 위해 법전에서 찾아야 할 "
        "핵심 죄명이나 법률 개념어를 정확히 5개 이내로 추출하십시오.\n\n"
        "🚨 [절대 준수 규칙]\n"
        "1. 반드시 순수 한국어(표준어)로만 작성할 것 (한자·중국어·영어 절대 금지)\n"
        "2. 쉼표(,)로만 구분하여 단어만 출력할 것\n"
        "3. 서론, 부연 설명, 번호 매기기 등은 절대 쓰지 말 것\n"
        "4. 사안에 등장하는 모든 행위자와 피해자의 관계, 행위의 수단과 목적,\n"
        "   행위 완성 여부를 모두 반영하여 키워드를 도출할 것\n"
        "   (수단이 되는 행위와 목적이 되는 행위가 분리되어 있다면 둘 다 포함)\n"
        "5. 가중·감경 사유가 있는지 사실관계에서 확인하고,\n"
        "   있다면 그 가중·감경에 해당하는 법률 개념어도 포함할 것\n\n"
        f"[사안]: {fact_pattern[:500]}\n"
        f"[질문]: {question_text}"
    )
    res = llm.create_chat_completion(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=80, temperature=0.1, stream=False
    )
    raw = res["choices"][0]["message"]["content"]
    clean = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
    clean = re.sub(r'[^가-힣,\s]', '', clean).strip()
    print(f"   🔑 [쟁점 키워드]: {clean}")
    return clean


# ==========================================
# 5. 시스템 프롬프트
#    원칙: 특정 과목·사건의 정답을 미리 알려주지 않음.
#    "어떻게 사고해야 하는지(절차)"만 남기고,
#    "무엇이 정답인지(내용)"는 RAG 검색과 모델 판단에 맡김.
# ==========================================
SYSTEM_PROMPT = """당신은 대한민국 최고의 변호사시험 채점 위원입니다.

━━━ [출력 형식 — 절대 준수] ━━━
반드시 아래 4개 목차 구조로만 답변하십시오.
## 1. 문제의 소재
## 2. 관련 법리 및 판례
## 3. 사안의 적용
## 4. 결론

• 출력의 첫 줄은 반드시 ## 1. 문제의 소재 로 시작하십시오.
• 영어 사고과정, 드래프트, 자가검증, 사실관계 단순 나열 금지.
• 각 행위자(甲, 乙, 丙 등)의 죄책을 반드시 개별적으로 서술하십시오.

━━━ [답안 작성 전 필수 사고 절차] ━━━

① 관계·신분 확인
   사실관계에 등장하는 모든 인물의 관계(가족·직업·신분 등)를 나열하십시오.
   한국 법률에는 신분관계나 특수 지위에 따라 형이 가중되거나 감경되는
   조문이 존재합니다. 검색 결과에서 그 관계에 대응하는 가중·감경 조문이
   있는지 반드시 확인하고, 있으면 일반 조문 대신 그 조문을 적용하십시오.

② 행위의 수단과 목적 분리
   사실관계에 등장하는 모든 행위를 시간 순으로 나열하십시오.
   어떤 행위가 수단이고 어떤 행위가 목적인지 구분하십시오.
   수단에 해당하는 범죄와 목적에 해당하는 범죄는 각각 별도로 성립 여부를
   검토하십시오. 둘 중 하나만 골라 적용하지 말고 둘 다 검토한 뒤
   ⑤에서 죄수 관계로 정리하십시오.

③ 각 행위자의 가담 형태
   모든 행위자 각각에 대해 다음 중 어디에 해당하는지 판단하십시오.
   직접 실행(정범) / 타인에게 실행을 사주(교사) / 함께 실행(공동정범) /
   도와주기만 함(종범). 교사와 공동정범은 혼동하지 마십시오.

④ 각 범죄의 완성 단계
   각 범죄에 대해 구성요건이 충족되었는지 검색 결과의 조문 내용으로
   확인하여 기수 여부를 판단하십시오.
   기수에 이르지 못했다면, 행위자가 자의로 중단했는지(중지미수)
   외부 요인으로 중단됐는지(장애미수)를 구별하십시오.

⑤ 죄수 관계
   하나의 행위로 여러 죄가 성립하면 상상적 경합,
   별개의 행위로 여러 죄가 성립하면 실체적 경합으로 처리하십시오.
   법조경합(특별관계·흡수관계) 여부도 검토하십시오.

⑥ 절차법 특수 쟁점 (해당되는 경우에만)
   항소·상고·환송이 등장하면 다음을 반드시 검토하십시오.
   - 누가 항소·상고했는지 확인하고, 그에 따라 상급심이 심판할 수 있는
     범위가 제한되는지, 그리고 원심보다 무거운 결론(형이나 유·무죄)으로
     바꾸는 것이 피고인에게 불리한 변경에 해당하지 않는지 검토하십시오.
     결론을 내리기 전에 "이 변경이 누구에게 유리한가/불리한가"를
     먼저 판단하고, 그 판단이 절차법의 일반 원칙과 충돌하지 않는지
     검색 결과의 조문으로 재확인하십시오.
   - 상급심이 하급심의 사실인정을 뒤집으려면 어떤 근거가 필요한지
     (새로운 증거의 유무, 심리 절차, 논리·경험칙 위반 여부 등)를
     검색 결과에서 확인하고, 그 근거가 사실관계에 실제로 있는지
     엄격히 대조하십시오. 근거 없이 결론만 바꾸는 것이 허용되는지
     스스로 재검토한 뒤 서술하십시오.
   - 고소·고발·친고 여부가 절차 적법성에 영향을 미치는지 확인하십시오.
   - 체포·구속의 적법성 요건(죄명·긴급성·영장 등)을 검색 결과로 확인하십시오.
   - 이 쟁점들은 결론이 갈리기 쉬우므로, 결론을 쓰기 전 반대 결론도
     한 번 검토해보고 어느 쪽이 검색된 조문의 문언에 더 부합하는지
     비교한 뒤 최종 결론을 선택하십시오.

━━━ [조문 선택 원칙] ━━━
1. 검색 결과를 참고하되, 사실관계에 가장 부합하는 조문을 선택하십시오.
2. 유사 조문이 여럿이면 더 구체적이고 좁은 조문을 우선 적용하십시오.
3. 조문 번호가 불확실하면 번호를 지어내지 말고 "관련 조문 확인 필요"라고 명시하십시오.
4. 검색 결과에 없어도 사실관계상 명백히 필요한 조문이면 적용하십시오.
5. 검색 결과에 없는 조문 번호를 임의로 지어내는 것은 절대 금지합니다.
   확신이 없는 조문 번호는 쓰지 말고 "정확한 조문 확인 필요"라고 쓰십시오."""


# ==========================================
# 6. 답안 완전성 검증
# ==========================================
def check_completeness(answer: str) -> list:
    required = [
        "## 1. 문제의 소재",
        "## 2. 관련 법리 및 판례",
        "## 3. 사안의 적용",
        "## 4. 결론"
    ]
    return [s for s in required if s not in answer]


# ==========================================
# 7. 답안 생성
# ==========================================
def solve_case_pipeline(llm, q_id: str, target_fact: str,
                         question_text: str, law_context: str) -> str:
    print(f"   ✍️ [Agent] 답안 작성 중...")

    user_prompt = (
        f"{law_context}\n\n"
        f"[기초 사실관계]\n{target_fact}\n\n"
        f"[문항]\n{question_text}\n\n"
        "위 문항에 대해 체크리스트 순서대로 사실관계를 분석한 뒤, "
        "IRAC 구조로 답안을 작성하십시오."
    )

    res = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        max_tokens=3500,
        temperature=0.1,
        stream=False
    )

    raw = res["choices"][0]["message"]["content"]
    clean = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()

    # ## 1. 문제의 소재 이전 내용 제거
    matches = list(re.finditer(r'##\s*1\.\s*문제의\s*소재', clean))
    if matches:
        clean = clean[matches[-1].start():]

    # 완전성 검증
    missing = check_completeness(clean)
    if missing:
        clean += (
            f"\n\n> ⚠️ [생성 불완전] 누락 섹션: {', '.join(missing)}"
        )
        print(f"   ⚠️ 누락 섹션: {missing}")

    print("\n" + "=" * 60)
    print(clean)
    print("=" * 60)
    return clean.strip()


# ==========================================
# 8. Main
# ==========================================
def main(pdf_path: str, model_path: str):
    print(f"📦 모델 로드: {model_path}")
    llm = Llama(
        model_path=model_path,
        n_gpu_layers=-1,
        n_ctx=16384,
        n_batch=512,
        n_ubatch=256,
        n_threads=4,
        tensor_split=[1, 1],  # 듀얼 GPU
        flash_attn=True,
        verbose=False
    )
    print("✅ Kaggle Dual GPU & Llama 준비 완료.")

    parsed_questions = extract_and_parse_pdf(pdf_path)
    if not parsed_questions:
        print(" [Error] PDF에서 문제를 찾을 수 없습니다.")
        return

    retriever = LegalRetriever()

    output_name = (
        f"/kaggle/working/"
        f"{os.path.splitext(os.path.basename(pdf_path))[0]}_AI최종답안지.md"
    )
    # 헤더만 먼저 쓰고 이후 append — 중간 크래시 시 이전 결과 보존
    with open(output_name, "w", encoding="utf-8") as f:
        f.write("# 📜 변호사시험 AI 전과목 통합 최종 답안지\n\n---\n\n")

    for q in parsed_questions:
        print(f"\n[System] 풀이 중: {q['q_id']}")
        start_time = time.time()

        target_fact = get_relevant_fact_segment(q['fact_pattern'], q['question_text'])

        # 1단계: 법률 키워드 추출
        keywords = generate_search_keywords(llm, target_fact, q['question_text'])

        # 2단계: 키워드 + 질문 + 사실관계 조합으로 검색
        search_query = f"{q['question_text']} {keywords} {target_fact[:200]}"
        law_context = retriever.retrieve(search_query, top_k=12)

        # 3단계: 답안 생성
        answer = solve_case_pipeline(
            llm, q["q_id"], target_fact, q["question_text"], law_context
        )

        # 4단계: 답안에 쓰인 조문 번호 DB 교차검증
        # (없는 조문을 지어내는 것을 방지하는 최종 안전장치)
        warnings = retriever.verify_citations(answer)
        if warnings:
            answer += "\n\n### [자동 검증 결과]\n" + "\n".join(warnings)
            print(f"   검증 경고 {len(warnings)}건 발생 — 답안지에 첨부됨")

        elapsed = time.time() - start_time
        time_str = f"{elapsed:.2f}sec ({elapsed / 60:.1f}m)"
        print(f"⏱️ [{q['q_id']}] 완료: {time_str}")

        # append 저장 — 문제마다 즉시 기록
        with open(output_name, "a", encoding="utf-8") as f:
            f.write(
                f"## 🔷 [{q['q_id']}]\n\n"
                f"**[소요 시간]** {time_str}\n\n"
                f"**[문항]**\n{q['question_text']}\n\n"
                f"**[답안]**\n{answer}\n\n---\n\n"
            )

    print(f"\n🎉 완료: {output_name}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("\n Usage: python bar_exam_solver_RAG.py [PDF_PATH] [MODEL_PATH]")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
