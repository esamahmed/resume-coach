"""
app/core/config.py

Centralised settings — all values come from environment variables or .env file.
Nothing is hardcoded. Pydantic-settings validates types at startup.

LLM_PROVIDER options:
  "ollama"     — Local dev. Free. Run: ollama pull mistral
  "replicate"  — Cloud API. Free tier. Good for prompt experimentation.
  "sagemaker"  — Self-hosted Mistral-7B on AWS SageMaker. Required for graded submission.
  "bedrock"    — AWS Bedrock (Claude Sonnet). Fallback if SageMaker credits run out.
  "openai"     — OpenAI GPT-4o. Fallback / quick testing.
"""
from functools import lru_cache
from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # ── LLM Provider ──────────────────────────────────────────────────────────
    # Switch between all providers with a single env var — no code changes needed
    llm_provider: Literal["ollama", "replicate", "sagemaker", "bedrock", "openai"] = "ollama"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 4096

    # ── Ollama (local dev — free) ─────────────────────────────────────────────
    # Install: https://ollama.ai  |  Pull model: ollama pull mistral
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "mistral"               # or "llama2", "llama2:13b"

    # ── Replicate (cloud API — free tier for experimentation) ─────────────────
    # Sign up: https://replicate.com  |  Copy token from account settings
    replicate_api_token: str = ""
    replicate_model: str = "mistralai/mistral-7b-instruct-v0.2"
    replicate_model_version: str = ""           # leave blank to use latest

    # ── SageMaker (self-hosted Mistral — required for graded submission) ───────
    # Deploy with: python scripts/deploy_sagemaker.py --model mistral
    sagemaker_endpoint_name: str = "resume-coach-mistral-7b"
    sagemaker_region: str = "us-east-1"
    sagemaker_role_arn: str = ""   # arn:aws:iam::<account>:role/<role-name>
    sagemaker_max_new_tokens: int = 1350  # ml.g5.2xlarge: ~25 tok/s → 1350 tokens ≈ 55s, within 60s hard limit

    # ── AWS Bedrock (fallback — Claude Sonnet) ────────────────────────────────
    bedrock_region: str = "us-east-1"
    bedrock_model_id: str = "anthropic.claude-sonnet-4-20250514-v1:0"
    bedrock_embedding_model_id: str = "amazon.titan-embed-text-v2:0"

    # ── OpenAI (fallback — GPT-4o) ────────────────────────────────────────────
    openai_api_key: str = ""
    openai_model_id: str = "gpt-4o"

    # ── Embeddings ────────────────────────────────────────────────────────────
    # Used for FAISS vector store — separate from the chat LLM
    embedding_provider: Literal["ollama", "bedrock", "openai"] = "ollama"
    embedding_model_ollama: str = "nomic-embed-text"   # ollama pull nomic-embed-text
    embedding_model_openai: str = "text-embedding-3-small"

    # ── Langfuse Observability ────────────────────────────────────────────────
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"
    langfuse_enabled: bool = False # enable via .env: LANGFUSE_ENABLED=true

    # ── FAISS Vector Store ────────────────────────────────────────────────────
    faiss_index_path: str = "./data/faiss_index"
    embedding_chunk_size: int = 512
    embedding_chunk_overlap: int = 64
    rag_top_k: int = 5

    # ── Classical ML ──────────────────────────────────────────────────────────
    tfidf_model_path: str = "./data/ml/tfidf_vectorizer.joblib"
    xgb_model_path: str = "./data/ml/xgb_model.ubj"
    xgb_features_path: str = "./data/ml/xgb_features.joblib"
    ml_enabled: bool = True

    # ── Eval ──────────────────────────────────────────────────────────────────
    eval_fixtures_path: str = "./eval/fixtures"
    eval_scores_path: str = "./eval/scores"
    eval_corpus_hash_file: str = "./eval/corpus_hash.txt"

    # ── AWS General ───────────────────────────────────────────────────────────
    aws_region: str = "us-east-1"
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    s3_bucket: str = ""
    s3_eval_prefix: str = "resume-coach/eval/"

    # ── API ───────────────────────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_key: str = "dev-key-change-in-production"
    debug: bool = False


@lru_cache
def get_settings() -> Settings:
    return Settings()
