"""실행 설정. pydantic-settings로 환경변수에서 읽고, 없으면 기본값을 쓴다.

값은 프로세스 환경변수에서 읽는다(env_file 미사용). 진입점의
`load_dotenv()`가 `.env`를 `os.environ`으로 올린 뒤 `get_settings()`를 호출하면
그 값이 반영된다 — import 시점이 아니라 호출 시점에 읽어야 dotenv 순서와 무관하게
동작한다.

필드 → 환경변수(대문자, 대소문자 무시) 매핑:
- openrouter_api_key   ← OPENROUTER_API_KEY  (필수)
- openrouter_base_url  ← OPENROUTER_BASE_URL
- rlm_root_model       ← RLM_ROOT_MODEL      (코드 생성용 루트 모델)
- rlm_sub_model        ← RLM_SUB_MODEL       (위임용 sub 모델)
- openai_api_key       ← OPENAI_API_KEY      (RAG 임베딩용)
- rag_embed_model      ← RAG_EMBED_MODEL     (임베딩 모델)
- rag_chunk_size       ← RAG_CHUNK_SIZE      (passage 청킹 길이)
- rag_chunk_overlap    ← RAG_CHUNK_OVERLAP   (청크 겹침)
- rag_top_k            ← RAG_TOP_K           (생성기에 넣을 최종 passage 수)
- rag_top_n            ← RAG_TOP_N           (리랭킹 전 후보 수)
- rag_use_hyde         ← RAG_USE_HYDE        (HyDE 쿼리 확장 사용)
- rag_use_rerank       ← RAG_USE_RERANK      (LLM 리랭킹 사용)
- rag_cache_dir        ← RAG_CACHE_DIR       (인덱스 디스크 캐시 위치)
"""
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")

    openrouter_api_key: Optional[str] = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    rlm_root_model: str = "openai/gpt-5.6-sol"
    rlm_sub_model: str = "openai/gpt-5.6-luna"

    # RAG 비교 벤치용 — 임베딩·청킹·검색 파라미터
    openai_api_key: Optional[str] = None          # RAG 임베딩용
    rag_embed_model: str = "text-embedding-3-large"
    rag_chunk_size: int = 800
    rag_chunk_overlap: int = 150
    rag_top_k: int = 6                             # 생성기에 넣을 최종 passage 수
    rag_top_n: int = 24                            # 리랭킹 전 후보 수
    rag_use_hyde: bool = True
    rag_use_rerank: bool = True
    rag_cache_dir: str = ".rag_cache"


def get_settings() -> Settings:
    """호출 시점의 환경변수로 Settings를 새로 만든다."""
    return Settings()
