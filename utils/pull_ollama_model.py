# utils/pull_ollama_model.py
import sys, shutil, subprocess
from pathlib import Path
from config import CONFIG

def _exists(model: str) -> bool:
    root = Path.home() / ".ollama" / "models"
    return root.exists() and any(model in p.name for p in root.glob(f"**/{model.replace(':','_')}*"))

def pull(model: str):
    if _exists(model):
        print(f"✅  {model} đã có.")
        return
    print(f"⏬  Pull {model} …")
    subprocess.run(["ollama", "pull", model], check=True)
    print(f"✅  {model} OK")

def main(models):
    if shutil.which("ollama") is None:
        sys.exit("❌ 'ollama' chưa cài.")
    for m in models:
        pull(m.strip())

if __name__ == "__main__":
    cli = sys.argv[1:]
    models = cli or CONFIG.get("OLLAMA_MODELS", ["qwen3:0.6b"])
    main(models)
