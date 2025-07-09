# import re
# from typing import List

# __all__ = ["normalize_title", "parse_markdown_table"]

# def normalize_title(title: str) -> str:
#     return re.sub(r"[^\w\s]+", "-", title.lower()).strip("-")

# def parse_markdown_table(table_lines: List[str]) -> str:
#     if not table_lines:
#         return ""
#     formatted = []
#     for line in table_lines:
#         line = line.strip()
#         if not line or not line.startswith("|"):
#             continue
#         cells = [c.strip() for c in line.strip("|").split("|")]
#         if all(
#             c in {"", "-", "---", "----"} or (c.startswith("-") and c.endswith("-"))
#             for c in cells
#         ):
#             continue
#         processed = [c if c not in {"NaN", ""} else "" for c in cells]
#         formatted.append(" | ".join(processed))
#     return "\n".join(formatted)
