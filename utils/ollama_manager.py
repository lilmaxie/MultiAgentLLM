import subprocess, time, shutil, atexit, requests, os, sys
from pathlib import Path
from config import CONFIG

HOST = "http://localhost:11434"
DEFAULT_MODEL = CONFIG["OLLAMA_MODELS"][0]

# Fix encoding issues on Windows
if sys.platform.startswith('win'):
    os.environ['PYTHONIOENCODING'] = 'utf-8'

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

def _model_exists(model: str) -> bool:
    """Kiểm tra xem model đã tồn tại local chưa"""
    try:
        root = Path.home() / ".ollama" / "models"
        return root.exists() and any(model in p.name for p in root.glob(f"**/{model.replace(':','_')}*"))
    except Exception:
        return False

def _pull(model: str):
    """Pull model với Unicode handling tốt hơn"""
    # Kiểm tra xem model đã có chưa
    if _model_exists(model):
        print(f"✅ {model} đã có sẵn.")
        return True
    
    print(f"⏬ Đang pull {model}...")
    try:
        # Sử dụng encoding UTF-8 và handle errors
        result = subprocess.run(
            ["ollama", "pull", model], 
            capture_output=True, 
            text=True,
            encoding='utf-8',     # Explicitly set UTF-8
            errors='replace',     # Replace problematic characters
            timeout=300           # 5 phút timeout
        )
        
        if result.returncode == 0:
            print(f"✅ {model} pull thành công!")
            return True
        else:
            error_output = result.stderr if result.stderr else result.stdout
            
            # Xử lý các lỗi phổ biến
            if "no such host" in error_output or "dial tcp" in error_output:
                print(f"❌ Lỗi kết nối mạng khi pull {model}")
                print("🔧 Giải pháp:")
                print("   - Kiểm tra kết nối internet")
                print("   - Kiểm tra DNS settings")
                print("   - Thử lại sau vài phút")
                print("   - Hoặc sử dụng model khác đã có sẵn")
            elif "412" in error_output or "newer version" in error_output:
                print(f"❌ Model {model} yêu cầu phiên bản Ollama mới hơn!")
                print("🔄 Hãy cập nhật Ollama tại: https://ollama.com/download")
            elif "not found" in error_output:
                print(f"❌ Model {model} không tồn tại. Kiểm tra lại tên model.")
            else:
                print(f"❌ Lỗi khi pull {model}: {error_output}")
            
            return False
            
    except subprocess.TimeoutExpired:
        print(f"❌ Timeout khi pull {model} (>5 phút)")
        return False
    except UnicodeDecodeError as e:
        print(f"❌ Lỗi encoding khi pull {model}: {e}")
        print("🔧 Thử với method khác...")
        return _pull_with_popen(model)
    except Exception as e:
        print(f"❌ Lỗi không xác định khi pull {model}: {e}")
        return False

def _pull_with_popen(model: str):
    """Alternative pull method sử dụng Popen với real-time output"""
    try:
        print(f"⏬ Đang pull {model} (method 2)...")
        
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
                if clean_output:
                    print(f"  {clean_output}")
        
        return_code = process.wait()
        
        if return_code == 0:
            print(f"✅ {model} pull thành công!")
            return True
        else:
            print(f"❌ Pull {model} thất bại với return code: {return_code}")
            return False
            
    except Exception as e:
        print(f"❌ Lỗi trong alternative pull method: {e}")
        return False

def ensure_ollama_ready(model: str = DEFAULT_MODEL):
    """Đảm bảo Ollama sẵn sàng với error handling"""
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
    print(f"🔍 Kiểm tra model {model}...")
    
    # Thử pull model, nhưng không fail nếu không thành công
    success = _pull(model)
    if not success:
        print(f"⚠️  Không thể pull {model}, kiểm tra các model có sẵn...")
        
        # Liệt kê các model có sẵn với Unicode handling
        try:
            result = subprocess.run(
                ["ollama", "list"], 
                capture_output=True, 
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=10
            )
            if result.returncode == 0:
                print("📋 Các model có sẵn:")
                print(result.stdout)
            else:
                print("❌ Không thể liệt kê models")
        except Exception as e:
            print(f"❌ Lỗi khi liệt kê models: {e}")
        
        # Đề xuất các model thay thế phổ biến
        print("\n💡 Các model thay thế có thể thử:")
        print("   - llama3.2:1b")
        print("   - llama3.2:3b") 
        print("   - qwen2:1.5b")
        print("   - phi3:mini")
        
        raise RuntimeError(f"❌ Không thể sử dụng model {model}")
    
    print("✅ Model OK")