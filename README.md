# mini-rlm

논문 "Recursive Language Models"(arXiv:2512.24601, repo: alexzhang13/rlm)의 핵심
메커니즘을 LangGraph로 재현한 학습용 토이.

## ⚠️ 보안 경고
이 프로젝트는 **모델이 생성한 Python 코드를 in-process `exec()`로 실제 실행**합니다.
신뢰할 수 있는 task와 본인 머신에서만 쓰세요. 프로덕션 사용 금지.

## 셋업
```bash
python3.10 -m venv .venv && source .venv/bin/activate   # Python 3.10+
pip install -r requirements.txt
export OPENROUTER_API_KEY=...        # 필수
# (선택) LangSmith 트레이싱
# export LANGSMITH_API_KEY=...; export LANGSMITH_TRACING=true; export LANGSMITH_PROJECT=mini-rlm
```

## 테스트 (네트워크/키 불필요)
```bash
pytest -v
```

## 데모 (실제 API 필요)
```bash
python demo_tickets.py
```
