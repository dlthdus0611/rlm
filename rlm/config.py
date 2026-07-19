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
"""
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")

    openrouter_api_key: Optional[str] = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    rlm_root_model: str = "openai/gpt-5.5"
    rlm_sub_model: str = "openai/gpt-5.5-mini"


def get_settings() -> Settings:
    """호출 시점의 환경변수로 Settings를 새로 만든다."""
    return Settings()
