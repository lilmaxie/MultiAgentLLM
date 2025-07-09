from pathlib import Path
import yaml

CONFIG_PATH = Path(__file__).with_name("config.yaml")

if not CONFIG_PATH.exists():
    raise FileNotFoundError(
        f"⚠️  Không tìm thấy {CONFIG_PATH}. "
        "Hãy tạo file config.yaml (hoặc copy config.yaml.sample) rồi chỉnh giá trị."
    )

with CONFIG_PATH.open("r", encoding="utf-8") as f:
    CONFIG: dict = yaml.safe_load(f)
