import sys, shutil, subprocess, os
from pathlib import Path
from config import CONFIG

# Fix encoding issues on Windows
if sys.platform.startswith('win'):
    os.environ['PYTHONIOENCODING'] = 'utf-8'

def _exists(model: str) -> bool:
    root = Path.home() / ".ollama" / "models"
    return root.exists() and any(model in p.name for p in root.glob(f"**/{model.replace(':','_')}*"))

def check_ollama_version():
    """Kiểm tra phiên bản Ollama và đưa ra cảnh báo nếu cần"""
    try:
        result = subprocess.run(
            ["ollama", "version"], 
            capture_output=True, 
            text=True,
            encoding='utf-8',
            errors='replace'
        )
        if result.returncode == 0:
            print(f"📋 Ollama version: {result.stdout.strip()}")
        else:
            print("⚠️  Không thể kiểm tra phiên bản Ollama")
    except Exception as e:
        print(f"⚠️  Lỗi khi kiểm tra phiên bản: {e}")

def pull(model: str):
    if _exists(model):
        print(f"✅  {model} đã có.")
        return True
    
    print(f"⏬  Pull {model} …")
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
            check=True
        )
        print(f"✅  {model} OK")
        return True
        
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
        
        return False
        
    except UnicodeDecodeError as e:
        print(f"❌  Lỗi encoding khi pull {model}: {e}")
        print("🔧  Thử với method khác...")
        return pull_with_popen(model)
    except Exception as e:
        print(f"❌  Lỗi không xác định khi pull {model}: {e}")
        return False

def pull_with_popen(model: str):
    """Alternative pull method với real-time output"""
    try:
        print(f"⏬  Pull {model} (method 2)...")
        
        # Set environment for UTF-8
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        
        process = subprocess.Popen(
            ["ollama", "pull", model],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            errors='replace',
            env=env,
            bufsize=1,
            universal_newlines=True
        )
        
        # Đọc output real-time
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                # Clean output for display
                clean_output = output.strip().replace('\r', '').replace('\n', '')
                if clean_output and not clean_output.startswith('pulling'):
                    print(f"  {clean_output}")
        
        return_code = process.wait()
        
        if return_code == 0:
            print(f"✅  {model} OK")
            return True
        else:
            print(f"❌  Pull {model} thất bại với return code: {return_code}")
            return False
            
    except Exception as e:
        print(f"❌  Lỗi trong alternative pull method: {e}")
        return False

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