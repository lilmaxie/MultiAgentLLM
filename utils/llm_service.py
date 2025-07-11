from __future__ import annotations
import time
import warnings
from typing import Any

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Try to import from new package first, fallback to old one
try:
    from langchain_ollama import ChatOllama
    print("✅ Using new langchain_ollama package")
except ImportError:
    try:
        from langchain_community.chat_models import ChatOllama
        print("⚠️  Using deprecated langchain_community.chat_models")
        print("💡 Consider upgrading: pip install -U langchain-ollama")
    except ImportError:
        raise ImportError("❌ Neither langchain_ollama nor langchain_community.chat_models available")

from utils.ollama_manager import ensure_ollama_ready
from config import CONFIG


def get_llm(
    model: str | None = None,
    temperature: float = 0.3,
    timeout: int = 60,
    verbose: bool = False,
) -> ChatOllama:
    """
    Trả về ChatOllama đã sẵn sàng với fallback models.
    """
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
        print(f"❌ Không thể sử dụng model {model}: {e}")
        
        # Fallback sang các model khác trong config
        fallback_models = [m for m in CONFIG["OLLAMA_MODELS"] if m != model]
        
        for fallback_model in fallback_models:
            try:
                print(f"🔄 Thử fallback model: {fallback_model}")
                ensure_ollama_ready(fallback_model)
                return ChatOllama(
                    model=fallback_model,
                    temperature=temperature,
                    request_timeout=timeout,
                    verbose=verbose,
                )
            except Exception as fallback_e:
                print(f"❌ Fallback model {fallback_model} cũng thất bại: {fallback_e}")
                continue
        
        # Nếu tất cả đều thất bại, thử với các model phổ biến
        common_models = ["llama3.2:1b", "llama3.2:3b", "qwen2:1.5b", "phi3:mini"]
        print("🔄 Thử với các model phổ biến...")
        
        for common_model in common_models:
            try:
                print(f"🔄 Thử model: {common_model}")
                ensure_ollama_ready(common_model)
                return ChatOllama(
                    model=common_model,
                    temperature=temperature,
                    request_timeout=timeout,
                    verbose=verbose,
                )
            except Exception:
                continue
        
        # Nếu vẫn không được, raise lỗi cuối cùng
        raise RuntimeError(
            f"❌ Không thể khởi tạo bất kỳ model nào. "
            f"Hãy kiểm tra:\n"
            f"  1. Kết nối internet\n"
            f"  2. Cài đặt Ollama\n"
            f"  3. Thử pull model thmanually: ollama pull llama3.2:1b"
        )


def call_llm(llm: ChatOllama, prompt: str, max_retry: int = 2) -> str:
    """
    Gọi LLM với retry logic cải thiện.
    """
    for attempt in range(1, max_retry + 2):       # 1 + retries
        try:
            response = llm.invoke(prompt)
            if hasattr(response, 'content'):
                return response.content
            else:
                return str(response)
        except Exception as exc:                  # pylint: disable=broad-except
            print(f"[LLM] Error (try {attempt}/{max_retry+1}): {exc}")
            if attempt > max_retry:
                return f"⚠️ Error generating response after {max_retry+1} attempts: {exc}"
            
            # Tăng thời gian chờ theo số lần thử
            sleep_time = 1.5 * attempt
            print(f"[LLM] Waiting {sleep_time}s before retry...")
            time.sleep(sleep_time)
    
    return "⚠️ Unexpected error in retry logic."