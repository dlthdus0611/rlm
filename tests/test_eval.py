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


from rlm.eval import select_items


def _items():
    return [
        QAItem(id=f"Q{i}", difficulty=("low" if i % 2 == 0 else "high"),
               question="q", answer="a")
        for i in range(10)
    ]


def test_select_items_filters_by_difficulty():
    out = select_items(_items(), difficulties=["high"])
    assert len(out) == 5
    assert all(it.difficulty == "high" for it in out)


def test_select_items_samples_n_deterministically():
    a = select_items(_items(), n=3, seed=42)
    b = select_items(_items(), n=3, seed=42)
    assert len(a) == 3
    assert [it.id for it in a] == [it.id for it in b]  # seed 고정 → 동일


def test_select_items_n_larger_than_pool_returns_all():
    out = select_items(_items(), n=999)
    assert len(out) == 10
