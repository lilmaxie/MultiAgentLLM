from __future__ import annotations
import re
from typing import Any, Dict, List
from utils import call_llm


class GeneratorAgent:
    def __init__(self, llm):
        self.llm = llm
        self.SYS = """Bạn là GeneratorAgent – chuyên gia tạo **template chương trình khuyến mãi AFFINA**.

QUY TẮC BẤT DI BẤT DỊCH
1. Chỉ được phép in đúng **3 heading cấp 2**, xuất hiện **một lần** và đúng thứ tự:
   • ## Chương Trình
   • ## Dữ Liệu
   • ## Theo dõi
   → Ngay sau mỗi heading, dòng kế tiếp BẮT BUỘC bắt đầu bằng ký tự “|”.

2. TUYỆT ĐỐI không tạo bất kỳ markdown nào khác: ✗ `###`, ✗ `####`, ✗ `*`/`-` list, ✗ chữ đậm `**`, ✗ code-block ``` … ```.

3. Toàn bộ phần thân phải là bảng với dấu **|**; giữ nguyên “NaN”, bullet **·** và ký hiệu **ü**; đúng số cột, đúng thứ tự dòng.

4. Nếu không có số liệu, GIỮ nguyên placeholder trong **[dấu ngoặc vuông]**
   (vd. `[Số tiền]`, `[Tỷ lệ]`, `[Sản phẩm 1]` …) – không tự chế giá trị.

5. Nếu có FEEDBACK, hãy chỉnh vào template nhưng **không** ghi chú giải.

6. Sau khi in template, kết thúc bằng dòng `TEMPLATE_END`.

ĐỊNH DẠNG OUTPUT
🧠 CHAIN OF THOUGHT - GENERATOR:
<thinking>
(1) Nhập PLAN (8 mục) → xác định: THỜI GIAN, ƯU ĐÃI, SẢN PHẨM, %GIẢM.
(2) Áp dụng FEEDBACK (nếu có) → sửa đúng ô; KHÔNG thêm lời giải thích.
(3) Kiểm tra CONTEXT: có đủ 3 section gốc? Nếu thiếu → GIỮ placeholder, KHÔNG tự bịa số.
(4) Dựng TEMPLATE: copy khung cố định, điền giá trị / placeholder.
(5) Kiểm tra format: 3 heading, các dấu |, NaN, ·; xoá markdown lạ (###, **, -, ```).
</thinking>

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
| Khách Hàng Thực Tế |
| Tháng | NaN | [Sản phẩm 1] KH tiềm năng | [Sản phẩm 2] KH tiềm năng | [Sản phẩm 3] KH tiềm năng | [Sản phẩm 4] KH tiềm năng | Tổng |
| [Tháng 1] | [Số] | NaN | NaN | NaN | NaN | [Số] |
| Tỉ lệ | NaN | [Tỷ lệ] | [Tỷ lệ] | [Tỷ lệ] | [Tỷ lệ] | NaN |
| [Tháng 2] | [Số] | NaN | NaN | NaN | NaN | [Số] |
| Tỉ lệ tăng trưởng ([Tỷ lệ %]) | NaN | [Tỷ lệ] | [Tỷ lệ] | [Tỷ lệ] | [Tỷ lệ] | [Tỷ lệ] |
| [Tháng 3] | [Số] | NaN | NaN | NaN | NaN | [Số] |
| Tỉ lệ tăng trưởng ([Tỷ lệ %]) | NaN | [Tỷ lệ] | [Tỷ lệ] | [Tỷ lệ] | [Tỷ lệ] | [Tỷ lệ] |
| Tổng | NaN | [Tổng 1] | [Tổng 2] | [Tổng 3] | [Tổng 4] | [Tổng] |

| Doanh thu VAS |
| Doanh Thu ước tính | NaN | [Sản phẩm 1] KH tiềm năng | [Sản phẩm 2] KH tiềm năng | [Sản phẩm 3] KH tiềm năng | [Sản phẩm 4] KH tiềm năng | Tổng |
| [Tháng 1] | NaN | [Số tiền] | [Số tiền] | [Số tiền] | [Số tiền] | [Tổng] |
| [Tháng 2] | NaN | [Số tiền] | [Số tiền] | [Số tiền] | [Số tiền] | [Tổng] |
| [Tháng 3] | NaN | [Số tiền] | [Số tiền] | [Số tiền] | [Số tiền] | [Tổng] |
| Tổng | NaN | [Tổng 1] | [Tổng 2] | [Tổng 3] | [Tổng 4] | [Tổng chung] |

| Chi phí khuyến mãi |
| NaN | NaN | [Sản phẩm 1] KH tiềm năng ([Tỷ lệ %]) | [Sản phẩm 2] KH tiềm năng ([Tỷ lệ %]) | [Sản phẩm 3] KH tiềm năng ([Tỷ lệ %]) | [Sản phẩm 4] KH tiềm năng | Tổng |
| [Tháng 1] | NaN | [Số tiền] | [Số tiền] | [Số tiền] | [Số tiền] | [Tổng] |
| [Tháng 2] | NaN | [Số tiền] | [Số tiền] | [Số tiền] | [Số tiền] | [Tổng] |
| [Tháng 3] | NaN | [Số tiền] | [Số tiền] | [Số tiền] | [Số tiền] | [Tổng] |
| Tổng | NaN | [Tổng 1] | [Tổng 2] | [Tổng 3] | [Tổng 4] | [Tổng chung] |
</template>
TEMPLATE_END

HƯỚNG DẪN NỘI BỘ:
• Thay [Tháng 1-3] bằng tháng cụ thể; điền dữ liệu hợp lý.
• Giữ nguyên cấu trúc & ký tự đặc biệt.
• Không thêm giải thích sau `TEMPLATE_END`.
• Nếu thêm bất cứ ký tự nào sau TEMPLATE_END → score = 0
"""

    def generate(
        self,
        user_req: str,
        plan_data: Any,
        context_segments: List[str] | None = None,
        feedback: str = "",
    ) -> Dict[str, Any]:
        # --- Build context block (nếu có) ---
        ctx_block = ""
        if context_segments:
            clipped = [
                seg[:800] + " …" if len(seg) > 800 else seg for seg in context_segments[:5]
            ]
            ctx_block = "\n\n### NGỮ CẢNH THAM KHẢO (KHÔNG SAO CHÉP)\n" + "\n\n".join(f"- {c}" for c in clipped)

        plan_text = plan_data.get("plan", "") if isinstance(plan_data, dict) else str(plan_data)
        fb_block = f"\n\n### FEEDBACK VÒNG TRƯỚC\n{feedback}" if feedback else ""

        prompt = f"""{self.SYS}

### YÊU CẦU NGƯỜI DÙNG
{user_req}

### CHIẾN LƯỢC TỪ ORCHESTRATOR
{plan_text}{ctx_block}{fb_block}

### HƯỚNG DẪN CUỐI
• BẮT BUỘC có đủ 3 heading đúng thứ tự và phải là dạng BẢNG GIỐNG NHƯ TEMPLATE:
  1. ## Chương Trình (5 phần: THÔNG TIN CHUNG, MỤC TIÊU, NỘI DUNG ƯU ĐÃI, HÌNH THỨC TRIỂN KHAI, ĐIỀU KIỆN & QUY ĐỊNH)
  2. ## Dữ Liệu (5 phần bảng dữ liệu)
  3. ## Theo dõi (3 phần bảng theo dõi)
• Sau dòng cuối cùng ghi đúng `TEMPLATE_END`.
"""
        raw = call_llm(self.llm, prompt).strip()
        return self._extract_thinking_and_content(raw)

    @staticmethod
    def _extract_thinking_and_content(raw: str) -> Dict[str, Any]:
        thinking = ""
        content = raw

        # 1) Tách thinking
        for pat in [
            r"🧠 CHAIN OF THOUGHT - GENERATOR:\s*<thinking>.*?</thinking>",
            r"<thinking>.*?</thinking>",
            r"<think>.*?</think>",
        ]:
            m = re.search(pat, content, re.DOTALL | re.IGNORECASE)
            if m:
                thinking = m.group(0)
                content = re.sub(pat, "", content, flags=re.DOTALL | re.IGNORECASE).strip()
                break

        # 2) Loại bỏ văn bản thừa
        for pat in [
            r"TEMPLATE:\s*",
            r"Template như sau:\s*",
            r"Dựa trên.*?template:\s*",
            r"Theo yêu cầu.*?:\s*",
            r"\*\(Based on.*?\)\*\s*",
            r"-{3,}\s*",
            r"^\s*\*+.*?\*+\s*$",
        ]:
            content = re.sub(pat, "", content, flags=re.MULTILINE | re.IGNORECASE).strip()

        # 3) Bảo đảm bắt đầu đúng heading
        if not content.startswith("## Chương Trình"):
            m = re.search(r"## Chương Trình", content, re.IGNORECASE)
            if m:
                content = content[m.start():].strip()

        return {"thinking": thinking, "content": content, "full_response": raw}

    @staticmethod
    def _clean_response(response: str) -> str:
        """Clean extra artifacts (không dùng trong pipeline chính, nhưng giữ phòng)."""
        if not response:
            return ""
        for pat in [
            r"<think>.*?</think>",
            r"<thinking>.*?</thinking>",
            r"\*\(Based on.*?\)\*",
            r"TEMPLATE.*?:",
            r"Dựa trên.*?:",
            r"-{3,}",
            r"^\s*###.*$",
            r"^\s*\*\*.*?\*\*.*$",
        ]:
            response = re.sub(pat, "", response, flags=re.MULTILINE | re.DOTALL | re.IGNORECASE)
        response = re.sub(r"\n\s*\n\s*\n", "\n\n", response).strip()
        return response