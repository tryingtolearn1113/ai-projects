from llama_cpp import Llama

# n_gpu_layers=0 으로 설정하세요! (CPU만 사용한다는 뜻)
llm = Llama(
    model_path="./Gemma-4-12B-OBLITERATED-Q4_K_M.gguf", 
    n_ctx=512,           # 메모리 절약
    n_threads=2,         
    n_gpu_layers=0,      
    use_mlock=False,     # RAM 강제 할당 방지
    embedding=False,     # 임베딩 기능 끄기 (메모리 절약)
    verbose=True         # 무슨 에러인지 상세히 출력
)

# ... 나머지 코드는 동일합니다 ...