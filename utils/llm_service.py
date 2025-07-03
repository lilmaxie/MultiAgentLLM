from __future__ import annotations
import time
from typing import Any
from langchain_community.chat_models import ChatOllama
from utils.ollama_manager import ensure_ollama_ready
from config import CONFIG


def get_llm(
    model: str | None = None,
    temperature: float = 0.3,
    timeout: int = 60,
    verbose: bool = False,
) -> ChatOllama:
    """
    Trả về ChatOllama đã sẵn sàng (tự gọi ensure_ollama_ready).
    """
    model = model or CONFIG["OLLAMA_MODELS"][0]
    ensure_ollama_ready(model)
    return ChatOllama(
        model=model,
        temperature=temperature,
        request_timeout=timeout,
        verbose=verbose,
    )

def call_llm(llm: ChatOllama, prompt: str, max_retry: int = 2) -> str:
    """
    Gọi LLM; nếu lỗi thì retry `max_retry` lần.
    """
    for attempt in range(1, max_retry + 2):       # 1 + retries
        try:
            return llm.invoke(prompt).content
        except Exception as exc:                  # pylint: disable=broad-except
            print(f"[LLM] Error (try {attempt}/{max_retry+1}): {exc}")
            if attempt > max_retry:
                return "⚠️ Error generating response."
            time.sleep(1.5 * attempt)
