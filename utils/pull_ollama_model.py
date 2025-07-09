import sys, shutil, subprocess
from pathlib import Path
from config import CONFIG

def _exists(model: str) -> bool:
    root = Path.home() / ".ollama" / "models"
    return root.exists() and any(model in p.name for p in root.glob(f"**/{model.replace(':','_')}*"))

def check_ollama_version():
    """Kiểm tra phiên bản Ollama và đưa ra cảnh báo nếu cần"""
    try:
        result = subprocess.run(["ollama", "version"], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"📋 Ollama version: {result.stdout.strip()}")
        else:
            print("⚠️  Không thể kiểm tra phiên bản Ollama")
    except Exception as e:
        print(f"⚠️  Lỗi khi kiểm tra phiên bản: {e}")

def pull(model: str):
    if _exists(model):
        print(f"✅  {model} đã có.")
        return
    
    print(f"⏬  Pull {model} …")
    try:
        result = subprocess.run(["ollama", "pull", model], 
                              capture_output=True, text=True, check=True)
        print(f"✅  {model} OK")
    except subprocess.CalledProcessError as e:
        error_output = e.stderr if e.stderr else e.stdout
        
        # Xử lý các lỗi phổ biến
        if "412" in error_output or "newer version" in error_output:
            print(f"❌  Model {model} yêu cầu phiên bản Ollama mới hơn!")
            print("🔄  Hãy cập nhật Ollama tại: https://ollama.com/download")
            print("💡  Hoặc thử sử dụng model khác tương thích với phiên bản hiện tại")
        elif "not found" in error_output:
            print(f"❌  Model {model} không tồn tại. Kiểm tra lại tên model.")
        else:
            print(f"❌  Lỗi khi pull {model}: {error_output}")
        
        # Không raise exception để tiếp tục với các model khác
        return False
    except Exception as e:
        print(f"❌  Lỗi không xác định khi pull {model}: {e}")
        return False
    
    return True

def main(models):
    if shutil.which("ollama") is None:
        sys.exit("❌ 'ollama' chưa cài.")
    
    # Kiểm tra phiên bản Ollama
    check_ollama_version()
    
    success_count = 0
    for m in models:
        if pull(m.strip()):
            success_count += 1
    
    print(f"\n📊 Kết quả: {success_count}/{len(models)} model được tải thành công")

if __name__ == "__main__":
    cli = sys.argv[1:]
    models = cli or CONFIG.get("OLLAMA_MODELS")
    main(models)