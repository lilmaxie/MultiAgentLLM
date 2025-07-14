import sys
import shutil
import subprocess
import os
from pathlib import Path
from config import CONFIG

# Fix encoding issues on Windows
if sys.platform.startswith('win'):
    os.environ['PYTHONIOENCODING'] = 'utf-8'

def model_exists(model: str) -> bool:
    """Kiểm tra xem model đã tồn tại local chưa"""
    root = Path.home() / ".ollama" / "models"
    return root.exists() and any(model in p.name for p in root.glob(f"**/{model.replace(':','_')}*"))

def pull_model(model: str) -> bool:
    """Pull model với error handling"""
    if model_exists(model):
        print(f"{model} đã có sẵn.")
        return True
    
    print(f"Đang pull {model}...")
    try:
        # Set environment for UTF-8
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        
        result = subprocess.run(
            ["ollama", "pull", model], 
            capture_output=True, 
            text=True,
            encoding='utf-8',
            errors='replace',
            env=env,
            timeout=300
        )
        
        if result.returncode == 0:
            print(f"{model} pull thành công!")
            return True
        else:
            error_output = result.stderr if result.stderr else result.stdout
            
            if "no such host" in error_output or "dial tcp" in error_output:
                print(f"Lỗi kết nối mạng khi pull {model}")
                print("Kiểm tra kết nối internet và thử lại")
            elif "412" in error_output or "newer version" in error_output:
                print(f"Model {model} yêu cầu phiên bản Ollama mới hơn!")
                print("Hãy cập nhật Ollama tại: https://ollama.com/download")
            elif "not found" in error_output:
                print(f"Model {model} không tồn tại. Kiểm tra lại tên model.")
            else:
                print(f"Lỗi khi pull {model}: {error_output}")
            
            return False
            
    except subprocess.TimeoutExpired:
        print(f"Timeout khi pull {model} (>5 phút)")
        return False
    except Exception as e:
        print(f"Lỗi khi pull {model}: {e}")
        return False

def main(models):
    """Main function để pull models"""
    if shutil.which("ollama") is None:
        sys.exit("'ollama' chưa được cài đặt.")
    
    success_count = 0
    for model in models:
        if pull_model(model.strip()):
            success_count += 1
    
    print(f"\nKết quả: {success_count}/{len(models)} model được tải thành công")

if __name__ == "__main__":
    cli = sys.argv[1:]
    models = cli or CONFIG.get("OLLAMA_MODELS", [])
    main(models)