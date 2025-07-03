import pandas as pd

def excel_to_markdown(file_path, sheet_name=0):
    """
    Chuyển đổi file Excel sang chuỗi Markdown table.
    :param file_path: Đường dẫn file Excel (.xlsx, .xls)
    :param sheet_name: Tên sheet hoặc chỉ số sheet (mặc định: sheet đầu tiên)
    :return: Chuỗi markdown
    """
    # Đọc sheet
    df = pd.read_excel(file_path, sheet_name=sheet_name)

    # Chuyển đổi sang markdown
    markdown = df.to_markdown(index=False, tablefmt="github")

    return markdown

# Ví dụ sử dụng
if __name__ == "__main__":
    md = excel_to_markdown("AFFINA - CONTENT PLAN T7_2025.xlsx")
    print(md)
