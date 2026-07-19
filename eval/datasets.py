"""평가용 데이터셋 위치 — 단일 진실 소스.

테스트셋 경로와 context 파일 위치는 eval 도메인 사실이다. CLI(runner)와 UI(평가 페이지)가
이 한곳을 공유해, 데이터 파일 이름이 바뀌어도 두 진입점이 어긋나지 않게 한다.
"""

CONTEXT_PATH = "data/samsung_2023.txt"

# 짧은 이름 → 테스트셋 JSON 경로 목록.
TESTSETS = {
    "single": ["data/qa_testset.json"],
    "cross": ["data/qa_crosssection.json"],
    "both": ["data/qa_testset.json", "data/qa_crosssection.json"],
}
