import subprocess
import time
import shutil
import atexit
import requests
import os
import sys
from pathlib import Path
from config import CONFIG

HOST = "http://localhost:11434"
DEFAULT_MODEL = CONFIG["OLLAMA_MODELS"][0]

# Fix encoding issues on Windows
if sys.platform.startswith('win'):
    os.environ['PYTHONIOENCODING'] = 'utf-8'

def is_running() -> bool:
    """Kiểm tra Ollama daemon có đang chạy không"""
    try:
        return requests.get(f"{HOST}/api/tags", timeout=2).status_code == 200
    except requests.exceptions.RequestException:
        return False

def start_daemon():
    """Khởi động Ollama daemon"""
    if shutil.which("ollama") is None:
        raise RuntimeError("Lệnh 'ollama' chưa được cài đặt!")
    
    proc = subprocess.Popen(
        ["ollama", "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )
    atexit.register(proc.terminate)
    return proc

def model_exists(model: str) -> bool:
    """Kiểm tra xem model đã tồn tại local chưa"""
    try:
        root = Path.home() / ".ollama" / "models"
        return root.exists() and any(model in p.name for p in root.glob(f"**/{model.replace(':','_')}*"))
    except Exception:
        return False

def pull_model(model: str) -> bool:
    """Pull model với error handling"""
    if model_exists(model):
        print(f"{model} đã có sẵn.")
        return True
    
    print(f"Đang pull {model}...")
    try:
        result = subprocess.run(
            ["ollama", "pull", model], 
            capture_output=True, 
            text=True,
            encoding='utf-8',
            errors='replace',
            timeout=300
        )
        
        if result.returncode == 0:
            print(f"{model} pull thành công!")
            return True
        else:
            error_output = result.stderr if result.stderr else result.stdout
            print(f"Lỗi khi pull {model}: {error_output}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"Timeout khi pull {model}")
        return False
    except Exception as e:
        print(f"Lỗi khi pull {model}: {e}")
        return False

def ensure_ollama_ready(model: str = DEFAULT_MODEL):
    """Đảm bảo Ollama sẵn sàng với model"""
    if not is_running():
        print("Khởi động Ollama daemon...")
        start_daemon()
        for _ in range(20):
            if is_running():
                break
            time.sleep(0.5)
        else:
            raise RuntimeError("Ollama daemon không khởi động được")
    
    print("Ollama daemon đang chạy")
    print(f"Kiểm tra model {model}...")
    
    if not pull_model(model):
        raise RuntimeError(f"Không thể sử dụng model {model}")
    
    print("Model sẵn sàng")