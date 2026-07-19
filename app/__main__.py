"""`python -m app` — streamlit 앱을 코드에서 직접 띄운다(디버거용 진입점).

`streamlit run app/streamlit_app.py`와 동작은 같지만, streamlit CLI가 아니라
이 파이썬 모듈을 진입점으로 실행한다. 그래서 IDE 디버거(VS Code 등)나
`python -m debugpy --listen 5678 -m app`로 그대로 붙일 수 있고,
streamlit이 앱 스크립트를 in-process로 실행하므로 `streamlit_app.py`·`rlm/`
안의 중단점이 잡힌다.
"""
import sys
from pathlib import Path

from streamlit.web import cli as stcli


def main() -> int:
    script = Path(__file__).resolve().parent / "streamlit_app.py"
    # streamlit CLI가 기대하는 argv를 흉내낸다: `streamlit run <script>`.
    sys.argv = ["streamlit", "run", str(script)]
    return stcli.main()


if __name__ == "__main__":
    raise SystemExit(main())
