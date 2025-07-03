import subprocess, time, shutil, atexit, requests, os
from config import CONFIG

HOST = "http://localhost:11434"
DEFAULT_MODEL = CONFIG["OLLAMA_MODELS"][0]

def _running() -> bool:
    try:
        return requests.get(f"{HOST}/api/tags", timeout=2).status_code == 200
    except requests.exceptions.RequestException:
        return False

def _start_daemon():
    if shutil.which("ollama") is None:
        raise RuntimeError("❌ Lệnh 'ollama' chưa cài!")
    proc = subprocess.Popen(
        ["ollama", "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )
    atexit.register(proc.terminate)
    return proc

def _pull(model: str):
    subprocess.run(["ollama", "pull", model], check=True)

def ensure_ollama_ready(model: str = DEFAULT_MODEL):
    if not _running():
        print("🔄 Khởi động Ollama daemon…")
        _start_daemon()
        for _ in range(20):
            if _running():
                break
            time.sleep(0.5)
        else:
            raise RuntimeError("❌ Ollama daemon không khởi động!")
    print("✅ Ollama daemon OK")
    print(f"🔍 Kiểm tra model {model} …")
    _pull(model)
    print("✅ Model OK")