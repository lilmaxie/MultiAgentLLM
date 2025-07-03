# agents/evaluator.py
from __future__ import annotations
import json
import re
from typing import Any, Dict, Tuple
from utils import call_llm


AFFINA_BASE_TEMPLATE = """
## Chương Trình
| I. THÔNG TIN CHUNG
| ·       Thời gian diễn ra chương trình: […]
| ·       Phạm vi: […]
| ·       Đối tượng áp dụng: […]

| II. MỤC TIÊU
| 1. […]
| 2. […]
| 3. […]

| III. NỘI DUNG ƯU ĐÃI
| […]
| STT | Dịch vụ/Sản phẩm VAS | Ưu đãi
| 1 | […] |  Ưu đãi: […]
| NaN | NaN |  Giá trị tối đa: […]
| NaN | NaN |  […]
| Tổng | NaN | […]

| IV. HÌNH THỨC TRIỂN KHAI
| […]
| V. ĐIỀU KIỆN & QUY ĐỊNH
| […]

| Tháng | Doanh thu VAS ước tính | Chi phí ưu đãi | % khuyến mãi trung bình |
| [Tháng 1] | […] | […] | […] |
| [Tháng 2] | […] | […] | […] |
| [Tháng 3] | […] | […] | […] |
| Tổng | […] | […] | […] |

## Dữ Liệu
| Dữ liệu 3 tháng liền kề |
| Tháng | NaN | DMP KH tiềm năng | TBYT KH tiềm năng | Checkup KH tiềm năng | Califonia KH tiềm năng | Tổng |
| Tháng [A] | […] | NaN | NaN | NaN | NaN | NaN |
| Tháng [B] | […] | NaN | NaN | NaN | NaN | NaN |
| Tháng [C] | […] | NaN | NaN | NaN | NaN | NaN |

| Estimate Dữ liệu 3 tháng ([Tháng 1]-[Tháng 3]) |
| …

| Ticket size trung bình |
| …

| Doanh thu VAS ước tính |
| …

| Chi phí khuyến mãi |
| …

## Theo dõi
| Khách Hàng Thực Tế |
| …
| Doanh thu VAS |
| …
| Chi phí khuyến mãi |
| …

TEMPLATE_END
""".strip()

# ─────────────────────────── EvaluatorAgent ────────────────────────────── #
class EvaluatorAgent:
    """
    Đánh giá độ đúng FORMAT template AFFINA.
    Trả về dict:
        {
            "thinking": str,
            "score": float 0–1,
            "feedback": str,
            "full_response": str
        }
    """

    def __init__(self, llm, base_template: str = AFFINA_BASE_TEMPLATE):
        self.llm = llm
        self.base_template = base_template

        self.SYS = """Bạn là **EvaluatorAgent** – chấm điểm độ đúng **FORMAT** của template khuyến mãi AFFINA.

LUẬT GATE FORMAT (bắt buộc)
- Nếu template sai bất kỳ điều sau ➜ cho score = 0.0, lý do trong feedback, NGỪNG chấm tiếp:
  • Thiếu / thừa heading ## Chương Trình, ## Dữ Liệu, ## Theo dõi
  • Sai cấu trúc bảng (số cột, vị trí “NaN”, ký tự · bị đổi)
  • Thêm markdown khác (###, **, ``` …)
  • Không giữ đúng thứ tự dòng trong “Chương Trình”

NẾU QUA GATE ➜ chấm theo trọng số:
1. **Chính xác FORMAT** (50 %) – giống hệt template gốc, placeholder đúng.
2. **Hoàn chỉnh NỘI DUNG** (30 %) – đủ bảng & section.
3. **Hợp lý dữ liệu** (15 %) – số, tỉ lệ, tổng logic.
4. **Ứng dụng** (5 %) – dùng ngay được.

Tổng = Σ( tiêu chí × trọng số ). Điểm thập phân 0 – 1. Đạt khi ≥ 0.8.

ĐỊNH DẠNG ĐẦU RA DUY NHẤT
🧠 CHAIN OF THOUGHT - EVALUATOR:
<thinking>
(1) Chạy “Format Gate”: kiểm tra 4 lỗi nghiêm trọng (heading, |, NaN, markdown lạ, thứ tự dòng).
   • Nếu FAIL → đặt score = 0.0, feedback = lỗi & STOP.
(2) Tính điểm từng tiêu chí:
   • Format (50 %) – mô tả thiếu/sai cụ thể.
   • Hoàn chỉnh (30 %) – thiếu bảng, thiếu mục.
   • Hợp lý dữ liệu (15 %) – placeholder còn nhiều? số & % bất logic?
   • Ứng dụng (5 %) – dễ dùng ngay?
(3) Tính tổng, so sánh ngưỡng 0.8.
(4) Viết feedback đầy đủ, súc tích.
</thinking>

VÍ DỤ FORMAT RESULT
<result>{"score": <score>, "feedback": "Format chuẩn, đủ bảng; cần điều chỉnh tổng dòng Ticket size."}</result>
EVAL_END

QUY TẮC XUẤT
• **Chỉ** 2 khối `<thinking>` & `<result>` + dòng `EVAL_END`.
• **KHÔNG** bọc JSON trong ```json, không thêm ký hiệu khác.
"""

    def evaluate(self, candidate: Any) -> Dict[str, Any]:
        """
        candidate: có thể là str (template) hoặc dict {'content': str}
        """
        tpl = candidate["content"] if isinstance(candidate, dict) else str(candidate)

        prompt = f"""{self.SYS}

TEMPLATE GỐC, TEMPLATE CHUẨN:
{self.base_template}

TEMPLATE CẦN ĐÁNH GIÁ:
{tpl}

ĐÁNH GIÁ THEO CHUẨN TEMPLATE AFFINA:"""

        raw = call_llm(self.llm, prompt).strip()
        return self._extract(raw)

    @staticmethod
    def _extract(raw: str) -> Dict[str, Any]:
        thinking, score, feedback = "", 0.0, "N/A"

        # 1) thinking
        m = re.search(r"<thinking>(.*?)</thinking>", raw, re.S)
        if m:
            thinking = m.group(1).strip()

        # 2) score & feedback
        score, feedback = EvaluatorAgent._parse_result_json(raw)

        return {"thinking": thinking, "score": score, "feedback": feedback, "full_response": raw}

    @staticmethod
    def _parse_result_json(raw: str) -> Tuple[float, str]:
        """
        Parse JSON ở mọi dạng khả thi:
          <result>{ ... }</result>
          { "score": 0.9, "feedback": "…" }
        """
        patterns = [
            r"<result>\s*(\{[^{}]*\"score\"[^{}]*\"feedback\"[^{}]*\})\s*</result>",
            r"\{[^{}]*\"score\"[^{}]*\"feedback\"[^{}]*\}",
        ]
        for pat in patterns:
            m = re.search(pat, raw, re.S | re.I)
            if not m:
                continue
            js = m.group(1) if m.groups() else m.group(0)
            try:
                data = json.loads(js)
                score = max(0.0, min(1.0, float(data["score"])))
                return score, str(data["feedback"])
            except Exception:  # pylint: disable=broad-except
                continue

        # Fallback nếu parse thất bại
        print("⚠️  EvaluatorAgent: không parse được JSON – raw:\n", raw)
        return 0.0, "Evaluator trả về format sai – check prompt/regex."
