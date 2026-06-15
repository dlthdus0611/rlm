import json

from rlm.eval import QAItem, load_testset


def test_load_testset_single(tmp_path):
    p = tmp_path / "qa.json"
    p.write_text(json.dumps([
        {"id": "Q001", "difficulty": "low", "question": "삼성 언제 생겼어?",
         "answer": "1969년 1월 13일", "page": 5, "section": "01_회사개요_연혁",
         "question_textbook": "삼성전자 설립일은?"}
    ], ensure_ascii=False), encoding="utf-8")

    items = load_testset(str(p))

    assert len(items) == 1
    it = items[0]
    assert isinstance(it, QAItem)
    assert it.id == "Q001"
    assert it.difficulty == "low"
    assert it.answer == "1969년 1월 13일"
    assert it.section == "01_회사개요_연혁"


def test_load_testset_cross_uses_sections(tmp_path):
    p = tmp_path / "cross.json"
    p.write_text(json.dumps([
        {"id": "C001", "difficulty": "high", "question": "q", "answer": "a",
         "page": 6, "sections": ["01_회사개요_연혁", "14_계열회사"]}
    ], ensure_ascii=False), encoding="utf-8")

    items = load_testset(str(p))

    assert items[0].sections == ["01_회사개요_연혁", "14_계열회사"]
    assert items[0].section == ""
