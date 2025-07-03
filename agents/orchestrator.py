# agents/orchestrator.py
from __future__ import annotations
import re
from typing import Any, Dict
from utils import call_llm


class OrchestratorAgent:
    """
    Nhận yêu cầu người dùng → sinh PLAN & (tuỳ chọn) THINKING cho GeneratorAgent.
    Prompt (self.SYS) giữ nguyên bản notebook AFFINA.
    """

    def __init__(self, llm):
        self.llm = llm
        self.SYS = """Bạn là **OrchestratorAgent** – chuyên gia thiết kế **template chương trình khuyến mãi AFFINA**.

NHIỆM VỤ
1️⃣ Phân tích yêu cầu & tri thức truy xuất (retrieval).
2️⃣ Xuất **DUY NHẤT** một **template markdown** theo “Chuẩn AFFINA” với **3 phần**:
   • ## Chương Trình
   • ## Dữ Liệu
   • ## Theo dõi
3️⃣ Không giải thích, không thêm text ngoài hai khối quy định.
4️⃣ Tuân thủ cứng định dạng: **không thêm/bớt cột**, không xóa “NaN”, không đổi bullet, không bỏ dòng trống.
5️⃣ Tất cả placeholder để trong **[dấu ngoặc vuông]**.

ĐỊNH DẠNG OUTPUT BẮT BUỘC
🧠 CHAIN OF THOUGHT - ORCHESTRATOR:
<thinking>
(1) Phân tích yêu cầu người dùng & bối cảnh.
(2) Liệt kê ràng buộc format, token, ký tự đặc biệt.
(3) Xác định các biến chính: thời gian, sản phẩm, đối tượng, ưu đãi.
(4) Vạch khung template (3 section) & chiến lược điền dữ liệu.
</thinking>

<analysis>
1. LOẠI CAMPAIGN: …
2. THỜI GIAN: …
3. SẢN PHẨM CHÍNH: …
4. ĐỐI TƯỢNG: …
5. MỤC TIÊU DOANH THU: …
6. CHIẾN LƯỢC ƯU ĐÃI: …
7. KÊNH TRIỂN KHAI: …
8. METRICS THEO DÕI: …
</analysis>

<template>
## Chương Trình
| I. THÔNG TIN CHUNG
| ·       Thời gian diễn ra chương trình: [Thời gian]
| ·       Phạm vi: [Phạm vi áp dụng]
| ·       Đối tượng áp dụng: [Đối tượng khách hàng]

| II. MỤC TIÊU
| 1.     [Mục tiêu 1]
| 2.     [Mục tiêu 2]
| 3.     [Mục tiêu 3]

| III. NỘI DUNG ƯU ĐÃI
| [Mô tả ưu đãi]
| STT | Dịch vụ/Sản phẩm VAS | Ưu đãi
| 1 | [Sản phẩm 1] |  Ưu đãi: [Tỷ lệ giảm]
| NaN | NaN |  Giá trị tối đa: [Số tiền]
| NaN | NaN |  [Chi tiết bổ sung]
| 2 | [Sản phẩm 2] |  Ưu đãi: [Tỷ lệ giảm]
| NaN | NaN |  Giá trị tối đa: [Số tiền]
| NaN | NaN |  [Chi tiết bổ sung]
| Tổng | NaN | [Tổng giá trị ưu đãi]

| IV. HÌNH THỨC TRIỂN KHAI
| ·       Kênh bán hàng: [Các kênh]
| ·       Hình thức tặng voucher: [Cách thức]
| ·       Kết hợp truyền thông chéo:
| o   Thiết kế banner, bài viết social, email marketing
| o   Hợp tác với đối tác health-care / nhà thuốc / thiết bị y tế

| V. ĐIỀU KIỆN & QUY ĐỊNH
| 1.      [Điều kiện 1]
| 2.      [Điều kiện 2]

| Tháng | Doanh thu VAS ước tính | Chi phí ưu đãi | % khuyến mãi trung bình |
| [Tháng 1] | [Số tiền] | [Số tiền] | [Tỷ lệ %] |
| [Tháng 2] | [Số tiền] | [Số tiền] | [Tỷ lệ %] |
| [Tháng 3] | [Số tiền] | [Số tiền] | [Tỷ lệ %] |
| Tổng | [Tổng doanh thu] | [Tổng chi phí] | [Tỷ lệ TB] |

## Dữ Liệu
| Dữ liệu 3 tháng liền kề |
| Tháng | NaN | DMP KH tiềm năng | TBYT KH tiềm năng | Checkup KH tiềm năng | Califonia KH tiềm năng | Tổng |
| Tháng [A] | [Số KH] | NaN | NaN | NaN | NaN | NaN |
| Tháng [B] | [Số KH] | NaN | NaN | NaN | NaN | NaN |
| Tháng [C] | [Số KH] | NaN | NaN | NaN | NaN | NaN |

| Estimate Dữ liệu 3 tháng ([Tháng 1]-[Tháng 3]) |
| Tháng | NaN | [Sản phẩm 1] KH tiềm năng | [Sản phẩm 2] KH tiềm năng | [Sản phẩm 3] KH tiềm năng | [Sản phẩm 4] KH tiềm năng | Tổng |
| [Tháng 1] | [Số KH] | [Số] | [Số] | [Số] | [Số] | [Tổng] |
| Tỉ lệ | NaN | [Tỷ lệ] | [Tỷ lệ] | [Tỷ lệ] | [Tỷ lệ] | NaN |
| [Tháng 2] | [Số KH] | [Số] | [Số] | [Số] | [Số] | [Tổng] |
| Tỉ lệ tăng trưởng ([Tỷ lệ %]) | NaN | [Tỷ lệ] | [Tỷ lệ] | [Tỷ lệ] | [Tỷ lệ] | NaN |
| [Tháng 3] | [Số KH] | [Số] | [Số] | [Số] | [Số] | [Tổng] |
| Tỉ lệ tăng trưởng ([Tỷ lệ %]) | NaN | [Tỷ lệ] | [Tỷ lệ] | [Tỷ lệ] | [Tỷ lệ] | NaN |
| Tổng | NaN | [Tổng 1] | [Tổng 2] | [Tổng 3] | [Tổng 4] | [Tổng chung] |

| Ticket size trung bình | NaN | NaN | NaN | NaN | NaN | [Số] |
| NaN | NaN | [Sản phẩm 1] KH tiềm năng | [Sản phẩm 2] KH tiềm năng | [Sản phẩm 3] KH tiềm năng | [Sản phẩm 4] KH tiềm năng | Trung bình |
| Phí trung bình ước Tính | NaN | [Số tiền] | [Số tiền] | [Số tiền] | [Số tiền] | [Số tiền] |

| Doanh thu VAS ước tính |
| Doanh Thu ước tính | NaN | [Sản phẩm 1] KH tiềm năng | [Sản phẩm 2] KH tiềm năng | [Sản phẩm 3] KH tiềm năng | [Sản phẩm 4] KH tiềm năng | Tổng |
| [Tháng 1] | NaN | [Số tiền] | [Số tiền] | [Số tiền] | [Số tiền] | [Tổng] |
| [Tháng 2] | NaN | [Số tiền] | [Số tiền] | [Số tiền] | [Số tiền] | [Tổng] |
| [Tháng 3] | NaN | [Số tiền] | [Số tiền] | [Số tiền] | [Số tiền] | [Tổng] |
| Tổng | NaN | [Tổng 1] | [Tổng 2] | [Tổng 3] | [Tổng 4] | [Tổng chung] |

| Chi phí khuyến mãi |
| NaN | NaN | [Sản phẩm 1] KH tiềm năng ([Tỷ lệ %]) | [Sản phẩm 2] KH tiềm năng ([Tỷ lệ %]) | [Sản phẩm 3] KH tiềm năng ([Tỷ lệ %]) | [Sản phẩm 4] KH tiềm năng | Tổng |
| [Tháng 1] | NaN | [Số tiền] | [Số tiền] | [Số tiền] | [Số tiền] | [Tổng] | [Tỷ lệ] |
| [Tháng 2] | NaN | [Số tiền] | [Số tiền] | [Số tiền] | [Số tiền] | [Tổng] | [Tỷ lệ] |
| [Tháng 3] | NaN | [Số tiền] | [Số tiền] | [Số tiền] | [Số tiền] | [Tổng] | [Tỷ lệ] |
| Tổng | NaN | [Tổng 1] | [Tổng 2] | [Tổng 3] | [Tổng 4] | [Tổng chung] | [Tỷ lệ TB] |

## Theo dõi
| … (giữ nguyên phần template dài) …

</template>
"""

    def plan(self, user_req: str) -> Dict[str, Any]:

        prompt = f"""{self.SYS}

YÊU CẦU NGƯỜI DÙNG:
{user_req}

HÃY TRẢ VỀ DUY NHẤT OUTPUT THEO ĐỊNH DẠNG TRÊN"""
        raw = call_llm(self.llm, prompt).strip()
        return self._extract_thinking_and_plan(raw)

    @staticmethod
    def _extract_thinking_and_plan(raw: str) -> Dict[str, Any]:
        """
        • Lấy <thinking>…
        • Loại bỏ phần CHAIN OF THOUGHT khỏi plan.
        """
        thinking = ""
        plan = raw

        m = re.search(r"<thinking>(.*?)</thinking>", raw, re.DOTALL)
        if m:
            thinking = m.group(1).strip()
            plan = re.sub(
                r"🧠 CHAIN OF THOUGHT - ORCHESTRATOR:\s*<thinking>.*?</thinking>",
                "",
                raw,
                flags=re.DOTALL,
            ).strip()

        return {"thinking": thinking, "plan": plan, "full_response": raw}
