# from functools import lru_cache
# from llama_index.embeddings.openai import OpenAIEmbedding
# from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# from config import CONFIG

# @lru_cache(maxsize=1)
# def get_embed_model():
#     provider = CONFIG.get("EMBED_PROVIDER", "openai").lower()

#     if provider == "openai":
#         model = CONFIG.get("OPENAI_EMBED_MODEL", "text-embedding-3-small")
#         return OpenAIEmbedding(model=model)

#     if provider == "huggingface":
#         model = CONFIG.get("HF_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
#         return HuggingFaceEmbedding(model_name=model)

#     raise ValueError(f"Unsupported EMBED_PROVIDER: {provider}")
