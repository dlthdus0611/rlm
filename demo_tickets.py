"""티켓 집계 데모: '환불 언급 수'와 '그중 배송지연 원인 수'를 RLM으로 전수 집계.

문서가 거대하지 않아도, 전수/체계 집계는 한 번의 forward pass보다 RLM이 정확하다는
두 번째 축을 보여주는 예제. 실제 OpenRouter 호출이 필요하다(OPENROUTER_API_KEY).
"""
import random

from dotenv import load_dotenv

from rlm import run

load_dotenv()  # 루트의 .env 에서 OPENROUTER_API_KEY 등을 환경변수로 로드

CAUSES = ["배송 지연", "품질 불량", "단순 변심"]
SUBJECTS = ["주문 문의", "계정 문제", "주소 변경", "배송 추적", "제품 사용법"]


def make_tickets(n: int = 60, seed: int = 42):
    rng = random.Random(seed)
    tickets = []
    refund_count = 0
    delay_refund_count = 0
    for i in range(n):
        mentions_refund = rng.random() < 0.4
        if mentions_refund:
            refund_count += 1
            cause = rng.choice(CAUSES)
            if cause == "배송 지연":
                delay_refund_count += 1
            body = (
                f"안녕하세요, 주문 #{1000 + i} 관련 문의입니다. "
                f"{cause} 때문에 환불을 요청하고 싶습니다. 처리 부탁드립니다."
            )
        else:
            subject = rng.choice(SUBJECTS)
            body = (
                f"안녕하세요, 주문 #{1000 + i} 관련 {subject}입니다. "
                f"확인 부탁드립니다."
            )
        tickets.append(f"=== 티켓 #{i} ===\n{body}")
    context = "\n\n".join(tickets)
    return context, refund_count, delay_refund_count


def main():
    context, refund_count, delay_refund_count = make_tickets()
    question = (
        "context에는 여러 개의 고객 지원 티켓이 '=== 티켓 #N ===' 로 구분되어 있습니다. "
        "환불(refund)을 언급한 티켓 수와, 그중 원인이 '배송 지연'인 티켓 수를 구하세요. "
        "최종 답은 정확히 '환불=<수>, 배송지연=<수>' 형식으로만 제출하세요."
    )

    print(f"[데이터] 티켓 {context.count('=== 티켓')}개, {len(context)}자")
    print(f"[정답]   환불={refund_count}, 배송지연={delay_refund_count}")
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
