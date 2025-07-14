# from __future__ import annotations
# import os
# import re
# from typing import List, Dict
# from llama_index.core.schema import TextNode

# # Re-use helpers đã khai báo sẵn trong utils.text_utils
# from .text_utils import normalize_title, parse_markdown_table


# class AFFINAMarkdownChunker:
#     """
#     Chia cấu trúc tài liệu AFFINA thành 3 tầng:
#         • Root  (preview)
#         • Parent-section (Chương Trình – Dữ Liệu – Theo dõi)
#         • *Không tạo* child_section/table để đơn giản hoá
#     """

#     def __init__(self, markdown_content: str):
#         self.content = markdown_content
#         self.chunk_counter = 0

#     def generate_chunk_id(self, prefix: str = "chunk") -> str:
#         """Tạo ID duy nhất cho mỗi chunk."""
#         self.chunk_counter += 1
#         return f"{prefix}_{self.chunk_counter}"

#     @staticmethod
#     def extract_section(content: str, start_marker: str, end_marker: str | None = None) -> str:
#         """Lấy phần nội dung giữa hai marker '## ...' (không bao gồm end_marker)."""
#         start_idx = content.find(start_marker)
#         if start_idx == -1:
#             return ""

#         if end_marker:
#             end_idx = content.find(end_marker, start_idx + len(start_marker))
#             if end_idx == -1:
#                 return content[start_idx:]
#             return content[start_idx:end_idx]

#         return content[start_idx:]

#     def hierarchical_chunking_markdown(self) -> List[Dict]:
#         """
#         Trả về list[dict] chunk:
#             • Root
#             • 3 parent section: Chương Trình, Dữ Liệu, Theo dõi
#         """
#         chunks: List[Dict] = []

#         # 1) Root preview
#         preview = self.content[:1000] + "..." if len(self.content) > 1000 else self.content
#         chunks.append({
#             "title": "AFFINA Promotion Program Document",
#             "content": preview,
#             "type": "root",
#             "level": "high",
#             "section": "root",
#         })

#         # 2) Ba section chính
#         sections = [
#             ("Chương Trình", "## Chương Trình", "## Dữ Liệu"),
#             ("Dữ Liệu",      "## Dữ Liệu",      "## Theo dõi"),
#             ("Theo dõi",     "## Theo dõi",     None),
#         ]

#         for name, start, end in sections:
#             sec_content = self.extract_section(self.content, start, end)
#             if not sec_content:
#                 continue

#             chunks.append({
#                 "title": name,
#                 "content": sec_content,
#                 "type": "parent_section",
#                 "level": "high",
#                 "section": name.lower().replace(" ", "_"),
#             })

#         return chunks

#     def create_nodes_from_chunks(
#         self,
#         chunks: List[Dict],
#         source_file: str = "affina_doc",
#     ) -> List[TextNode]:
#         """Convert chunks → list TextNode (LlamaIndex)."""
#         nodes: List[TextNode] = []

#         for idx, chunk in enumerate(chunks):
#             chunk_id = f"{source_file}_{chunk['type']}_{normalize_title(chunk['title'])}"

#             metadata = {
#                 "source_file": source_file,
#                 "chunk_type": chunk["type"],
#                 "title": chunk["title"],
#                 "level": chunk.get("level", "medium"),
#                 "chunk_id": chunk_id,
#                 "chunk_index": idx,
#                 "section": chunk.get("section", "unknown"),
#             }

#             # Giữ compat với child_section/table (nếu muốn mở rộng sau)
#             if "roman_section" in chunk:
#                 metadata["roman_section"] = chunk["roman_section"]
#             if "table_title" in chunk:
#                 metadata["table_title"] = chunk["table_title"]

#             nodes.append(TextNode(text=chunk["content"], metadata=metadata))

#         return nodes

#     def process_document(self, source_file: str = "affina_doc") -> List[TextNode]:
#         chunks = self.hierarchical_chunking_markdown()
#         return self.create_nodes_from_chunks(chunks, source_file)

#     def print_chunks_structure(self, chunks: List[Dict]):
#         """In cấu trúc phân mảnh (debug)."""
#         print("Document Chunks Structure")
#         print("=" * 40)
#         for chunk in chunks:
#             indent = "  " if chunk["type"].startswith("child_") else ""
#             print(f"{indent}- {chunk['title']} ({chunk['type']})")
#             print(f"{indent}  length: {len(chunk['content'])} chars\n")
            

# def process_affina_markdown(
#     markdown_content: str,
#     source_file: str = "affina_doc",
# ) -> List[TextNode]:
#     """Xử lý chuỗi markdown AFFINA, trả về list TextNode."""
#     return AFFINAMarkdownChunker(markdown_content).process_document(source_file)

# def process_markdown_file(md_file_path: str) -> List[TextNode]:
#     """Đọc file .md rồi gọi `process_affina_markdown`."""
#     try:
#         with open(md_file_path, "r", encoding="utf-8") as f:
#             md_content = f.read()
#         src = os.path.splitext(os.path.basename(md_file_path))[0]
#         return process_affina_markdown(md_content, source_file=src)
#     except Exception as exc:  # pylint: disable=broad-except
#         print(f"Lỗi xử lý {md_file_path}: {exc}")
#         return []

# def save_nodes_to_json(nodes: List[TextNode], filename: str):
#     """Lưu list TextNode → JSON (content + metadata)."""
#     import json

#     data = [{"text": n.text, "metadata": n.metadata} for n in nodes]
#     with open(filename, "w", encoding="utf-8") as f:
#         json.dump(data, f, ensure_ascii=False, indent=2)

# def load_nodes_from_json(filename: str) -> List[TextNode]:
#     """Đọc JSON đã lưu thành list TextNode."""
#     import json

#     with open(filename, "r", encoding="utf-8") as f:
#         data = json.load(f)

#     return [TextNode(text=d["text"], metadata=d["metadata"]) for d in data]


# __all__ = [
#     "AFFINAMarkdownChunker",
#     "process_affina_markdown",
#     "process_markdown_file",
#     "save_nodes_to_json",
#     "load_nodes_from_json",
# ]
