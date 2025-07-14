from __future__ import annotations
import time
import warnings
from typing import Any

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Try to import from new package first, fallback to old one
try:
    from langchain_ollama import ChatOllama
    print("Sử dụng langchain_ollama package")
except ImportError:
    try:
        from langchain_community.chat_models import ChatOllama
        print("Sử dụng langchain_community.chat_models")
        print("Khuyến nghị cập nhật: pip install -U langchain-ollama")
    except ImportError:
        raise ImportError("Không thể import ChatOllama")

from utils.ollama_manager import ensure_ollama_ready
from config import CONFIG

def get_llm(
    model: str | None = None,
    temperature: float = 0.3,
    timeout: int = 60,
    verbose: bool = False,
) -> ChatOllama:
    """Tạo LLM instance với fallback models"""
    
    model = model or CONFIG["OLLAMA_MODELS"][0]
    
    # Thử với model được yêu cầu
    try:
        ensure_ollama_ready(model)
        return ChatOllama(
            model=model,
            temperature=temperature,
            request_timeout=timeout,
            verbose=verbose,
        )
    except Exception as e:
        print(f"Không thể sử dụng model {model}: {e}")
        
        # Fallback sang các model khác trong config
        fallback_models = [m for m in CONFIG["OLLAMA_MODELS"] if m != model]
        
        for fallback_model in fallback_models:
            try:
                print(f"Thử fallback model: {fallback_model}")
                ensure_ollama_ready(fallback_model)
                return ChatOllama(
                    model=fallback_model,
                    temperature=temperature,
                    request_timeout=timeout,
                    verbose=verbose,
                )
            except Exception as fallback_e:
                print(f"Fallback model {fallback_model} thất bại: {fallback_e}")
                continue
        
        raise RuntimeError(
            f"Không thể khởi tạo bất kỳ model nào.\n"
            f"Kiểm tra:\n"
            f"  1. Kết nối internet\n"
            f"  2. Cài đặt Ollama\n"
            f"  3. Thử pull model thủ công: ollama pull qwen3:1.7b"
        )

def call_llm(llm: ChatOllama, prompt: str, max_retry: int = 2) -> str:
    """Gọi LLM với retry logic"""
    for attempt in range(1, max_retry + 2):
        try:
            response = llm.invoke(prompt)
            if hasattr(response, 'content'):
                return response.content
            else:
                return str(response)
        except Exception as exc:
            print(f"[LLM] Error (lần {attempt}/{max_retry+1}): {exc}")
            if attempt > max_retry:
                return f"Error sau {max_retry+1} lần thử: {exc}"
            
            sleep_time = 1.5 * attempt
            print(f"[LLM] Chờ {sleep_time}s trước khi thử lại...")
            time.sleep(sleep_time)
    
    return "Lỗi không xác định trong retry logic."