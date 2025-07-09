from __future__ import annotations
import argparse
import sys
import traceback
from multiagent_system import MultiAgentSystem


DEMO_PROMPT = (
    """☀️ 5 dấu hiệu có thể là ung thư dạ dày giai đoạn đầu"""
)

def pretty_print_result(res: dict):
    print("\n" + "=" * 50)
    print("🏆 TÓM TẮT CHAIN OF THOUGHT")
    print("=" * 50)

    orch_think = res["orchestrator_plan"].get("thinking", "")
    if orch_think:
        print("📊 ORCHESTRATOR THINKING:\n", orch_think)

    print("\n📊 GENERATOR & EVALUATOR THINKING:")
    for log in res["thinking_log"]:
        print(f"\n--- LẦN LẶP {log['iteration']} ---")
        if log["generator_thinking"]:
            print("Generator:\n", log["generator_thinking"])
        if log["evaluator_thinking"]:
            print("Evaluator:\n", log["evaluator_thinking"])
        print(f"Score: {log['score']:.2f}")

    print("\n" + "=" * 50)
    print("🏆 KẾT QUẢ CUỐI CÙNG")
    print("=" * 50)
    print(f"📊 Điểm số: {res['score']:.2f}")
    print(f"🔄 Số vòng lặp: {res['iterations']}")
    print(f"✅ Trạng thái: {'ĐẠT CHUẨN' if res['score'] >= 0.75 else 'CHƯA ĐẠT CHUẨN'}")

    print("\n📄 TEMPLATE CUỐI CÙNG:")
    print("-" * 50)
    print(res["content"])

    if res.get("docx_path"):
        print(f"\n📄 File Word: {res['docx_path']}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="?", help="Yêu cầu người dùng")
    parser.add_argument("--iter", type=int, default=3, help="Số vòng lặp tối đa")
    parser.add_argument(
        "--threshold", type=float, default=0.8, help="Ngưỡng điểm pass (0–1)"
    )
    parser.add_argument("--demo", action="store_true", help="Chạy prompt demo B2S")
    args = parser.parse_args()

    if args.demo:
        question = DEMO_PROMPT
    elif args.question:
        question = args.question
    else:
        parser.error("Bạn phải cung cấp 'question' hoặc --demo")

    print("🚀 KHỞI ĐỘNG HỆ THỐNG MULTI-AGENT RAG")
    print("=" * 80)

    try:
        system = MultiAgentSystem()
        result = system.run(
            user_request=question,
            max_iterations=args.iter,
            pass_threshold=args.threshold,
        )
        pretty_print_result(result)

    except Exception as exc:  # pylint: disable=broad-except
        print(f"❌ Lỗi: {exc}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
    
    
"""
1. python -m pip install -r requirements.txt
2. python -m utils.pull_ollama_model
3. python -m tools.build_index <data_path> (optional)
4. python -m main "<prompt>"
"""
