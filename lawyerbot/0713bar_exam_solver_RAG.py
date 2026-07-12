#%%writefile /kaggle/working/bar_exam_solver_RAG.py
import os
import sys
import re
import time
import fitz
import torch
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

    seen_titles = {}

    for i in range(1, len(exam_blocks), 2):
        exam_title = exam_blocks[i].strip()
        exam_content = exam_blocks[i + 1]

        parts = re.split(r'<\s*문\s*제\s*>', exam_content, flags=re.IGNORECASE)
        if len(parts) < 2:
            continue

        normalized = re.sub(r'[〉>\]\s]', '', exam_title)

        fact_pattern = parts[0].strip()
        questions_block = parts[1].strip()

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
# 2. 하이브리드 검색기 (조/항 단위 재분할 DB + 정확매칭 강제포함)
# ==========================================
class LegalRetriever:
    def __init__(self):
        law_path = '/kaggle/working/legal_db_persisted_law_split'
        prec_path = '/kaggle/working/legal_db_persisted_precedent'

        if not os.path.exists(law_path):
            print(f" ⚠️ [Error] 법전 DB 경로({law_path})를 찾을 수 없습니다.")
            sys.exit(1)

        law_client = PersistentClient(path=law_path)
        law_cols = law_client.list_collections()
        if not law_cols:
            print(" ⚠️ [Error] 법전 DB 안에 컬렉션이 비어있습니다.")
            sys.exit(1)
        self.law_collection = law_client.get_collection(name=law_cols[0].name)
        print(f" ⚡ [System] 법전 컬렉션 연결 완료: {law_cols[0].name} ({self.law_collection.count():,} 청크)")

        self.prec_collection = None
        if os.path.exists(prec_path):
            try:
                prec_client = PersistentClient(path=prec_path)
                prec_cols = prec_client.list_collections()
                if prec_cols:
                    self.prec_collection = prec_client.get_collection(name=prec_cols[0].name)
                    print(f" ⚡ [System] 판례 컬렉션 연결 완료: {prec_cols[0].name} ({self.prec_collection.count():,} 청크)")
                else:
                    print(" ℹ️  [System] 판례 DB 내부 컬렉션 감지 불가 - 법전만 가동")
            except Exception as e:
                print(f" ℹ️  [System] 판례 연결 실패 (법전 단독 가동): {e}")

        retriever_device = "cpu"
        print(f" 📥 임베딩 모델 로드 중 (BAAI/bge-m3, 장치: {retriever_device})")
        self.embedding_model = SentenceTransformer("BAAI/bge-m3", device=retriever_device)
        self.embedding_model.max_seq_length = 1024

        all_data = self.law_collection.get(include=["documents", "metadatas"])
        self.documents = all_data["documents"]
        self.ids       = all_data["ids"]
        self.metadatas = all_data["metadatas"]

        tokenized_corpus = [self._bigram(doc) for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized_corpus)
        print(f" ⚡ [System] 법전 {len(self.documents)}개 조/항 BM25 인덱스 빌드 완료.")

        # 형법·형소법 "총칙성" 조문 — 구체적 사실관계와 임베딩 유사도가 낮아
        # 검색에서 누락되기 쉽지만, 거의 모든 형사법 문제에 공통으로 필요한 조문.
        # 정답(적용 여부·죄명)은 여전히 모델이 판단하며, 여기서는 해당 조문들의
        # '존재와 정확한 번호'만 항상 참고 가능하도록 보장한다.
        self._GENERAL_KEYWORDS = [
            "미수범", "중지범", "불능범", "예비", "음모",
            "공동정범", "교사범", "종범", "간접정범",
            "상상적 경합", "경합범", "누범", "자수",
            "정당방위", "긴급피난", "심신장애", "심신상실",
            "친고죄", "고소", "고소기간", "고소의 취소",
            "상소", "항소", "상고", "파기환송", "불이익변경금지",
            "공소시효", "공소제기",
        ]
        self.general_provisions_context = self._build_general_provisions()
        print(f" ⚡ [System] 총칙성 상시포함 조문 {self.general_provisions_context.count('[형법]') + self.general_provisions_context.count('[형사소송법]')}개 확보.")

    def _build_general_provisions(self) -> str:
        selected = []
        seen_headers = set()
        for doc, meta in zip(self.documents, self.metadatas):
            if meta.get("law_name") not in ("형법", "형사소송법"):
                continue
            header = doc.split("\n", 1)[0]
            if any(kw in header for kw in self._GENERAL_KEYWORDS):
                if header not in seen_headers:
                    seen_headers.add(header)
                    selected.append(doc)
        return "\n\n".join(selected)

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

    def _exact_match_force_include(self, keywords: str) -> list:
        """
        키워드 추출 단계에서 나온 '죄명'이 조문 제목(예: [형법] 제323조 (권리행사방해))과
        겹치면, 임베딩 유사도 순위와 무관하게 해당 청크를 강제로 포함시킨다.
        완전 일치뿐 아니라 '사주살인'처럼 여러 개념이 합성된 키워드도, 그 안에 '살인' 같은
        핵심 개념이 bigram 단위로 겹치면 잡아내도록 완화했다 (번호 환각의 주 원인 방지).
        """
        forced_ids = set()
        terms = [t.strip() for t in re.split(r'[,\s]+', keywords) if len(t.strip()) >= 2]
        term_bigrams = [set(self._bigram(t)) for t in terms]

        for doc_id, doc in zip(self.ids, self.documents):
            header_line = doc.split("\n", 1)[0]
            title_match = re.search(r'\(([^)]+)\)', header_line)
            title = title_match.group(1) if title_match else ""
            if not title:
                continue
            title_bigrams = set(self._bigram(title))
            if not title_bigrams:
                continue

            for term, tb in zip(terms, term_bigrams):
                if not tb:
                    continue
                if term in title or (len(title) >= 2 and title in term):
                    forced_ids.add(doc_id)
                    break
                # 키워드 bigram의 상당 부분이 조문 제목과 겹치면 (예: '사주살인'의 '살인'이
                # '살인, 존속살해' 제목과 겹침) 강제 포함
                overlap = len(tb & title_bigrams) / len(tb)
                if overlap >= 0.4:
                    forced_ids.add(doc_id)
                    break
        return list(forced_ids)

    def retrieve(self, query: str, keywords: str = "", top_k: int = 12) -> str:
        query_vector = self.embedding_model.encode([query]).tolist()[0]

        law_dense = self._search_collection(self.law_collection, query_vector, top_k * 2)
        law_dense_ids = {r[0] for r in law_dense}
        law_dense_sim = {r[0]: r[2] for r in law_dense}

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

        # 죄명 정확매칭 강제포함
        forced_ids = set(self._exact_match_force_include(keywords)) if keywords else set()

        rrf = {}
        for rank, (doc_id, _, _) in enumerate(law_dense):
            rrf[doc_id] = rrf.get(doc_id, 0) + 1.0 / (60 + rank)
        for rank, idx in enumerate(top_sparse):
            doc_id = self.ids[idx]
            if sparse_scores[idx] > 0:
                rrf[doc_id] = rrf.get(doc_id, 0) + 1.0 / (60 + rank)
        for doc_id in forced_ids:
            rrf[doc_id] = rrf.get(doc_id, 0) + 1.0  # 최상위권 강제 부스팅

        allowed = law_dense_ids | bm25_whitelist | forced_ids
        filtered_law = {
            doc_id: score for doc_id, score in rrf.items()
            if doc_id in allowed
        }
        top_law = sorted(filtered_law.items(), key=lambda x: x[1], reverse=True)[:top_k]

        top_prec = []
        if self.prec_collection is not None:
            prec_k    = max(4, top_k // 3)
            prec_hits = self._search_collection(self.prec_collection, query_vector, prec_k)
            top_prec  = [(doc_id, doc) for doc_id, doc, sim in prec_hits if sim >= 0.45]

        best_sim = max(law_dense_sim.values()) if law_dense_sim else 0.0
        print(
            f"    > 검색 완료 (법전 유사도: {best_sim:.4f}, "
            f"법전 조문: {len(top_law)}개, 강제포함: {len(forced_ids)}개, 판례: {len(top_prec)}개)"
        )

        context = ""
        if self.general_provisions_context:
            context += "=== [형법·형사소송법 총칙 조문 - 항상 제공] ===\n"
            context += self.general_provisions_context + "\n\n"

        context += "=== [관련 법령 조문] ===\n"
        for doc_id, _ in top_law:
            try:
                idx = self.ids.index(doc_id)
                context += self.documents[idx] + "\n\n"
            except ValueError:
                continue

        if top_prec:
            context += "\n=== [관련 판례] ===\n"
            for _, doc in top_prec:
                context += doc + "\n\n"

        return context.strip()


# ==========================================
# 3. 소문항 사실관계 추출 (도입부 보존)
# ==========================================
def get_relevant_fact_segment(fact_pattern: str, question_text: str) -> str:
    first_marker = re.search(r'\(1\)', fact_pattern)
    intro = fact_pattern[:first_marker.start()].strip() if first_marker else ""

    match = re.search(r'\((\d+)\)\s*에서', question_text)
    if match:
        num = match.group(1)
        start = re.search(rf'\({num}\)', fact_pattern)
        if start:
            segment = fact_pattern[start.start():]
            nxt = re.search(rf'\({int(num) + 1}\)', segment)
            specific = segment[:nxt.start()].strip() if nxt else segment.strip()
            if intro and intro not in specific:
                return f"{intro}\n{specific}".strip()
            return specific
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
        "5. 가중·감경 사유가 사실관계에 명시적으로 있는 경우에만 그에 해당하는\n"
        "   법률 개념어를 포함하고, 명시되지 않은 사유를 추측해서 넣지 말 것\n"
        "6. 정식 죄명(예: 권리행사방해죄, 현주건조물방화죄)이 있다면 반드시\n"
        "   정식 명칭 그대로 포함할 것 (줄여쓰거나 다른 말로 바꾸지 말 것)\n"
        "7. '사주살인', '방화미수'처럼 여러 개념을 하나로 합성한 신조어를 만들지 말고,\n"
        "   '살인', '교사', '현주건조물방화', '미수'처럼 각 개념을 별도의 단어로\n"
        "   분리하여 출력할 것\n\n"
        f"[사안]: {fact_pattern[:500]}\n"
        f"[질문]: {question_text}"
    )
    res = llm.create_chat_completion(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=80, temperature=0.1, repeat_penalty=1.1, stream=False
    )
    raw = res["choices"][0]["message"]["content"]
    clean = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
    clean = re.sub(r'[^가-힣,\s]', '', clean).strip()
    print(f"   🔑 [쟁점 키워드]: {clean}")
    return clean


# ==========================================
# 5. 시스템 프롬프트
# ==========================================
SYSTEM_PROMPT = """당신은 대한민국 최고의 변호사시험 채점 위원입니다.

━━━ [언어 규칙 — 절대 준수] ━━━
🚨 반드시 순수 한국어(표준어)로만 작성하십시오. 한자, 간체자, 번체자,
   중국어, 일본어를 단 한 글자도 사용하지 마십시오. (예: "교사"라고 쓰고
   "敎唆"나 "教唆"라고 쓰지 마십시오. "방조"라고 쓰고 "帮助"라고 쓰지 마십시오.)

━━━ [출력 형식 — 절대 준수] ━━━
반드시 아래 4개 목차 구조로만 답변하십시오.
## 1. 문제의 소재
## 2. 관련 법리 및 판례
## 3. 사안의 적용
## 4. 결론

• 출력의 첫 줄은 반드시 ## 1. 문제의 소재 로 시작하십시오.
• 영어 사고과정, 드래프트, 자가검증, 사실관계 단순 나열 금지.
• 사안에 등장하는 각 당사자(행위자, 이해관계인 등)의 법적 지위나 책임을
  반드시 개별적으로 서술하십시오.
• 🚨 "*수정*", "*보정*", "*반론*" 같은 자기수정 표시나, 결론을 냈다가
  다시 뒤집는 반복적 서술을 최종 답안에 남기지 마십시오. 검토 중 다른
  견해가 있다면 "일부 견해는 ~로 보나, 통설·판례는 ~로 본다"처럼
  최대 한 번만 언급한 뒤, 하나의 확정된 결론으로 단정하여 서술하십시오.

━━━ [최우선 원칙 — 모든 과목 공통] ━━━
🚨 사실관계에 명시적으로 언급되지 않은 사실, 신분, 관계, 사유를
   추측하거나 임의로 가정하지 마십시오.
🚨 검색 결과(법령·판례)에 존재하는 내용이라도, 사안과 직접 관련이
   없다면 언급하지 마십시오. 컨텍스트에 있는 정보를 전부 다 써야 한다는
   압박을 갖지 마십시오. 관련 없는 조문·판례는 과감히 버리십시오.
🚨 어떤 쟁점이 사실관계상 해당사항이 없다면, 억지로 논점을 만들지 말고
   "해당 사안에는 이에 해당하는 사유 없음"이라고 짧게 명시한 뒤 넘어가십시오.

━━━ [답안 작성 전 필수 사고 절차] ━━━

① 당사자 관계·지위 확인
   사실관계에 등장하는 모든 인물의 관계(가족·직업·계약관계·신분 등)를
   사실관계에 적힌 그대로 나열하십시오. 그 관계에 대응하는 법률효과
   (가중·감경, 권리·의무 발생 등)가 검색 결과에 있는지 확인하고,
   있으면 적용하십시오. 사실관계에 없는 관계를 만들어내지 마십시오.

② 행위·사실관계의 시간 순 정리
   사실관계에 등장하는 사건·행위를 시간 순으로 나열하십시오.
   각 행위/사실이 어떤 법률효과의 요건에 해당하는지 개별적으로 검토하고,
   여러 쟁점이 있다면 하나만 고르지 말고 모두 검토한 뒤 ⑤에서 관계를 정리하십시오.

③ 각 당사자의 관여 형태
   형사법이라면: 직접 실행(정범) / 교사 / 공동정범 / 종범 중 어디인지 판단하십시오.
   민사·행정법이라면: 각 당사자가 권리자/의무자/처분청/상대방 등 어떤
   지위에 있는지 판단하십시오. 과목과 무관한 항목은 건너뛰십시오.

④ 법률효과의 성립·완성 여부
   각 쟁점에 대해 구성요건·요건사실이 검색 결과의 조문 내용상 충족되었는지
   확인하십시오. 형사법이라면 기수/미수(중지·장애) 여부까지 판단하십시오.

⑤ 경합·관계 정리
   형사법이라면 죄수 관계(상상적 경합/실체적 경합/법조경합)를 정리하십시오.
   민사·행정법이라면 청구권 경합, 처분의 효력 범위 등 해당하는 경우에만 정리하십시오.

⑥ 절차법 특수 쟁점 (해당되는 경우에만)
   항소·상고·환송, 소송요건, 고소·고발, 체포·구속의 적법성 등이 사실관계에
   등장하는 경우에만 검색 결과로 확인하여 검토하십시오. 등장하지 않으면 생략하십시오.

━━━ [조문 선택 원칙 — 절대 준수] ━━━
1. 🚨 조문 번호는 반드시 [관련 법령 조문]에 실제로 제시된 조문 번호만
   사용하십시오. 죄명이나 개념은 알아도 정확한 조 번호가 검색 결과에
   없다면, 절대로 기억이나 추측으로 번호를 지어내지 말고
   "OO죄(조문 번호 확인 필요)"라고만 쓰십시오. 이는 예외 없는 규칙입니다.
2. 🚨 [관련 판례]가 검색 결과에 없다면 판례를 인용하지 마십시오.
   대법원 판례 사건번호는 검색 결과에 실제로 제시된 것만 인용
   가능하며, 기억나는 사건번호를 임의로 적어서는 절대 안 됩니다.
   판례가 없으면 "본 사안에 관한 구체적 판례는 검색되지 않음.
   학설상 일반론에 따라 검토함"이라고 명시하십시오.
3. 유사 조문이 여럿이면 더 구체적이고 좁은 조문을 우선 적용하십시오."""


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


def verify_cited_articles(answer: str, law_context: str) -> list:
    """
    답안에 등장한 '제N조' 번호 중, 실제로 검색된 law_context에 없는 번호를 찾아낸다.
    모델이 검색 결과에 없는 조문 번호를 기억으로 지어냈는지(환각) 기계적으로 검증하는 안전장치.
    """
    cited = set(re.findall(r'제\s*(\d+)\s*조', answer))
    retrieved = set(re.findall(r'제\s*(\d+)\s*조', law_context))
    suspicious = sorted((cited - retrieved), key=int)
    return suspicious


def check_foreign_chars(answer: str) -> list:
    """
    한자/중국어/일본어 문자가 답안에 섞여 있는지 확인 (CJK 통합한자 범위).
    단, 甲乙丙丁 등 변호사시험에서 당사자를 지칭하는 정식 관행 표기는
    오류가 아니므로 제외한다.
    """
    PARTY_NAME_HANJA = set("甲乙丙丁戊己庚辛壬癸")
    foreign = [c for c in re.findall(r'[\u4e00-\u9fff]', answer) if c not in PARTY_NAME_HANJA]
    return sorted(set(foreign))


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

    start = time.time()
    res = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        max_tokens=3500,
        temperature=0.1,
        repeat_penalty=1.1,
        stream=False
    )
    elapsed = time.time() - start

    usage = res.get("usage", {})
    completion_tokens = usage.get("completion_tokens", 0)
    if completion_tokens and elapsed > 0:
        print(f"   ⏱️ 생성 속도: {completion_tokens / elapsed:.2f} tok/s ({completion_tokens} tokens, {elapsed:.1f}s)")

    raw = res["choices"][0]["message"]["content"]
    clean = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()

    matches = list(re.finditer(r'##\s*1\.\s*문제의\s*소재', clean))
    if matches:
        clean = clean[matches[-1].start():]

    missing = check_completeness(clean)
    if missing:
        clean += f"\n\n> ⚠️ [생성 불완전] 누락 섹션: {', '.join(missing)}"
        print(f"   ⚠️ 누락 섹션: {missing}")

    suspicious_articles = verify_cited_articles(clean, law_context)
    if suspicious_articles:
        clean += (
            f"\n\n> ⚠️ [검증 필요] 검색 결과에 없는 조문 번호가 인용됨: "
            f"제{', 제'.join(suspicious_articles)}조 (환각 가능성, 수동 확인 요망)"
        )
        print(f"   ⚠️ 검증 필요 조문: 제{', 제'.join(suspicious_articles)}조")

    foreign_chars = check_foreign_chars(clean)
    if foreign_chars:
        clean += f"\n\n> ⚠️ [검증 필요] 한자/중국어 혼입 발견: {' '.join(foreign_chars)}"
        print(f"   ⚠️ 한자 혼입 발견: {' '.join(foreign_chars)}")

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
        n_ubatch=512,
        n_threads=4,
        tensor_split=[1, 1],
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
    with open(output_name, "w", encoding="utf-8") as f:
        f.write("# 📜 변호사시험 AI 전과목 통합 최종 답안지\n\n---\n\n")

    for q in parsed_questions:
        print(f"\n[System] 풀이 중: {q['q_id']}")
        start_time = time.time()

        target_fact = get_relevant_fact_segment(q['fact_pattern'], q['question_text'])

        keywords = generate_search_keywords(llm, target_fact, q['question_text'])

        search_query = f"{q['question_text']} {keywords} {target_fact[:200]}"
        law_context = retriever.retrieve(search_query, keywords=keywords, top_k=12)

        answer = solve_case_pipeline(
            llm, q["q_id"], target_fact, q["question_text"], law_context
        )

        elapsed = time.time() - start_time
        time_str = f"{elapsed:.2f}sec ({elapsed / 60:.1f}m)"
        print(f"⏱️ [{q['q_id']}] 완료: {time_str}")

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
