"""티켓 집계 데모: '환불을 요청한 티켓 수'와 '그중 사유가 배송 지연인 수'를 RLM으로 전수 집계.

현실의 고객 문의처럼 표현이 제각각이고 함정이 섞인 '지저분한' 데이터를 쓴다.
- 환불 요청인데 '환불'이라는 단어가 없는 경우("결제 취소하고 돈 돌려주세요")
- '환불' 단어는 있지만 요청이 아닌 경우(정책 문의, 교환 요청, "환불 안 해도 됨")
- 배송 지연을 말하지만 환불은 아닌 경우(단순 배송 추적)
- 지연 환불인데 '배송 지연' 리터럴 없이 "2주째 안 와서"로 표현

따라서 단순 키워드 카운트(`context.count("환불")`)로는 틀리고, sub-LLM 의미 분류로만
맞게 된다 — RLM이 빛나는 지점. 실제 OpenRouter 호출이 필요하다(OPENROUTER_API_KEY).
"""
import random

from dotenv import load_dotenv

from rlm import run

load_dotenv()  # 루트의 .env 에서 OPENROUTER_API_KEY 등을 환경변수로 로드

CAUSES = ["배송 지연", "품질 불량", "단순 변심"]
SUBJECTS = ["주문 문의", "계정 문제", "주소 변경", "배송 추적", "제품 사용법"]

GREETINGS = [
    "안녕하세요", "안녕하세요!", "수고 많으십니다", "담당자님 안녕하세요",
    "안녕하세요, 고객센터죠?", "안뇽하세요", "여보세요, 거기 고객센터 맞죠?",
]
SIGNOFFS = [
    "빠른 처리 부탁드립니다.", "회신 기다릴게요.", "감사합니다!",
    "꼭 좀 부탁드려요ㅠㅠ", "수고하세요.", "확인 부탁드려요.", "",
]

# 사유를 리터럴 'CAUSES' 단어 없이 다양하게 표현 — 의미 분류가 필요하게.
CAUSE_PHRASES = {
    "배송 지연": [
        "배송이 너무 늦어서", "주문한 지 2주가 지났는데 아직 안 와서",
        "일주일째 계속 '배송중'이라", "약속한 날짜를 한참 넘겨서",
        "택배가 깜깜무소식이라", "발송이 자꾸 미뤄져서",
    ],
    "품질 불량": [
        "받아보니 제품이 불량이라", "물건이 깨져서 도착했길래",
        "설명이랑 실물이 너무 달라서", "품질이 기대 이하라서",
    ],
    "단순 변심": [
        "그냥 마음이 바뀌어서", "필요가 없어져서",
        "다른 제품을 사려고요. ", "단순 변심이긴 한데",
    ],
}
# 절반쯤은 '환불'이라는 단어가 없음 — 키워드 카운트를 비껴가게.
REFUND_REQUESTS = [
    "환불 부탁드립니다.", "환불해 주세요.", "환불 요청합니다.",
    "결제 취소하고 돈 돌려주세요.", "주문 취소하고 환급받고 싶어요.",
    "돈 돌려받을 수 있을까요?", "그냥 취소할게요. 결제한 금액은 돌려주시겠죠?",
]
# 함정 ①: '환불' 단어는 있지만 이 주문에 대한 환불 '요청'은 아님.
REFUND_DISTRACTORS = [
    "환불 정책이 궁금해서요. 단순 변심도 환불 되나요? 지금 신청하려는 건 아니에요.",
    "지난번 주문은 환불 잘 받았어요. 이번 {ref}은 교환만 하고 싶습니다.",
    "환불 말고 사이즈 교환만 가능할까요? {ref} 건이에요.",
    "환불은 안 하셔도 돼요. 배송지가 맞는지만 확인 부탁드려요. {ref}",
]
# 함정 ②: 배송 지연을 말하지만 환불 요청은 아님(추적/일정 문의).
DELAY_DISTRACTORS = [
    "{ref} 배송이 너무 늦는데 지금 어디쯤인지 알 수 있을까요?",
    "주문한 지 열흘인데 아직도 안 왔어요. 언제 도착하나요? {ref}",
    "배송이 계속 지연되네요... 취소는 아니고 일정만 알려주세요. {ref}",
]
NON_REFUND_PLAIN = [
    "{ref} 관련 {subj} 문의예요. 확인 부탁드려요.",
    "{ref} {subj} 때문에 연락드립니다.",
    "{subj}은 어떻게 하나요? {ref} 건입니다.",
]


def _order_ref(rng, i):
    forms = [
        f"주문 #{1000 + i}", f"주문번호 {20240000 + i}", f"ORD-{1000 + i}",
        f"주문 {1000 + i}번", f"({1000 + i}번 주문)",
    ]
    return rng.choice(forms)


def _messy(rng, text):
    """가벼운 노이즈: 이중 공백, 말끝 추임새."""
    if rng.random() < 0.25:
        text = text.replace(" ", "  ", 1)
    if rng.random() < 0.3:
        text += rng.choice([" ㅠㅠ", " ㅎㅎ", "..", "!!"])
    return text


def make_tickets(n: int = 60, seed: int = 42):
    # 정답 라벨용 RNG와 표현/노이즈용 RNG를 분리 — 표현을 바꿔도 카운트는 불변.
    label_rng = random.Random(seed)
    style_rng = random.Random(seed + 1)
    tickets = []
    refund_count = 0
    delay_refund_count = 0
    for i in range(n):
        mentions_refund = label_rng.random() < 0.4
        cause = label_rng.choice(CAUSES) if mentions_refund else None

        ref = _order_ref(style_rng, i)
        greet = style_rng.choice(GREETINGS)
        sign = style_rng.choice(SIGNOFFS)

        if mentions_refund:
            refund_count += 1
            if cause == "배송 지연":
                delay_refund_count += 1
            cause_phrase = style_rng.choice(CAUSE_PHRASES[cause])
            request = style_rng.choice(REFUND_REQUESTS)
            body = f"{greet}, {ref} 건인데요 {cause_phrase} {request} {sign}"
        else:
            kind = style_rng.choices(
                ["plain", "refund_distractor", "delay_distractor"],
                weights=[5, 3, 2],
            )[0]
            if kind == "plain":
                tmpl = style_rng.choice(NON_REFUND_PLAIN)
                body = f"{greet}, " + tmpl.format(ref=ref, subj=style_rng.choice(SUBJECTS))
            elif kind == "refund_distractor":
                body = f"{greet}, " + style_rng.choice(REFUND_DISTRACTORS).format(ref=ref)
            else:
                body = f"{greet}, " + style_rng.choice(DELAY_DISTRACTORS).format(ref=ref)
            body = f"{body} {sign}"

        tickets.append(f"=== 티켓 #{i} ===\n{_messy(style_rng, body).strip()}")
    context = "\n\n".join(tickets)
    return context, refund_count, delay_refund_count


def main():
    context, refund_count, delay_refund_count = make_tickets()
    question = (
        "context에는 여러 개의 고객 지원 티켓이 '=== 티켓 #N ===' 로 구분되어 있습니다. "
        "이 주문에 대해 실제로 '환불을 요청'한 티켓 수와, 그 환불 요청들 중 사유가 "
        "'배송 지연'(늦은 배송)인 티켓 수를 구하세요. "
        "환불 정책 문의·교환 요청처럼 환불을 요청하지 않은 단순 언급은 세지 마세요. "
        "표현이 제각각이니('돈 돌려주세요', '결제 취소' 등도 환불 요청) 의미로 판단하세요. "
        "최종 답은 정확히 '환불=<수>, 배송지연=<수>' 형식으로만 제출하세요."
    )

    naive = context.count("환불")  # 키워드만 세는 순진한 기준선(틀릴 것).
    print(f"[데이터] 티켓 {context.count('=== 티켓')}개, {len(context)}자")
    print(f"[정답]   환불={refund_count}, 배송지연={delay_refund_count}")
    print(f"[참고]   '환불' 단어가 박힌 티켓(순진한 카운트): {naive}개")
    print("[실행]   RLM 호출 중...\n")

    answer = run(question, context, max_iterations=12)

    print(f"\n[모델 답] {answer}")
    expected = f"환불={refund_count}, 배송지연={delay_refund_count}"
    # 공백을 제거하고 두 카운트 부분 문자열이 모두 있는지로 채점(형식 변형 허용).
    normalized = (answer or "").replace(" ", "")
    ok = (
        f"환불={refund_count}" in normalized
        and f"배송지연={delay_refund_count}" in normalized
    )
    print(f"[정답]    {expected}")
    print(f"[일치]    {'✅' if ok else '❌'}")


if __name__ == "__main__":
    main()
