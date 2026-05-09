"""
app/core/llm.py

LLM factory — returns a LangChain chat model for the selected provider.
Switch providers by setting LLM_PROVIDER in your .env — no code changes needed.

Provider guide:
  ollama     → Free local dev. Run: ollama pull mistral
  replicate  → Free cloud API. Good for prompt experimentation (capstone Week 1).
  sagemaker  → Self-hosted Mistral-7B on AWS. Required for graded submission.
  bedrock    → AWS Bedrock (Claude Sonnet). Fallback.
  openai     → OpenAI GPT-4o. Fallback.

All agents call get_llm() — single entry point, provider is fully transparent.
"""
from __future__ import annotations
import json
import logging
from functools import lru_cache
from typing import Any

from app.core.config import get_settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_llm():
    settings = get_settings()
    provider = settings.llm_provider

    logger.info("Initialising LLM — provider=%s", provider)

    # ── Ollama (local dev — default) ──────────────────────────────────────────
    if provider == "ollama":
        from langchain_ollama import ChatOllama
        logger.info("LLM: Ollama — model=%s url=%s", settings.ollama_model, settings.ollama_base_url)
        return ChatOllama(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            temperature=settings.llm_temperature,
            num_predict=settings.llm_max_tokens,
        )

    # ── Replicate (cloud API — free tier for experimentation) ─────────────────
    if provider == "replicate":
        if not settings.replicate_api_token:
            raise ValueError(
                "REPLICATE_API_TOKEN is not set. "
                "Sign up at https://replicate.com and add your token to .env"
            )
        from langchain_community.llms import Replicate
        import os
        os.environ["REPLICATE_API_TOKEN"] = settings.replicate_api_token

        model_str = settings.replicate_model
        if settings.replicate_model_version:
            model_str = f"{model_str}:{settings.replicate_model_version}"

        logger.info("LLM: Replicate — model=%s", model_str)
        return Replicate(
            model=model_str,
            model_kwargs={
                "temperature": settings.llm_temperature,
                "max_new_tokens": settings.llm_max_tokens,
                "system_prompt": (
                    "You are an expert resume coach and career strategist. "
                    "Always respond with valid JSON when asked to do so."
                ),
            },
        )

    # ── SageMaker (self-hosted Mistral — required for graded submission) ───────
    if provider == "sagemaker":
        if not settings.sagemaker_endpoint_name:
            raise ValueError(
                "SAGEMAKER_ENDPOINT_NAME is not set. "
                "Deploy with: python scripts/deploy_sagemaker.py --model mistral"
            )
        logger.info("LLM: SageMaker — endpoint=%s", settings.sagemaker_endpoint_name)
        return _build_sagemaker_llm(settings)

    # ── AWS Bedrock (fallback — Claude Sonnet) ────────────────────────────────
    if provider == "bedrock":
        from langchain_aws import ChatBedrock
        logger.info("LLM: Bedrock — model=%s", settings.bedrock_model_id)
        return ChatBedrock(
            model_id=settings.bedrock_model_id,
            region_name=settings.bedrock_region,
            model_kwargs={
                "temperature": settings.llm_temperature,
                "max_tokens": settings.llm_max_tokens,
            },
        )

    # ── OpenAI (fallback — GPT-4o) ────────────────────────────────────────────
    if provider == "openai":
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is not set in .env")
        from langchain_openai import ChatOpenAI
        logger.info("LLM: OpenAI — model=%s", settings.openai_model_id)
        return ChatOpenAI(
            model=settings.openai_model_id,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            api_key=settings.openai_api_key,
        )

    raise ValueError(
        f"Unknown LLM_PROVIDER='{provider}'. "
        "Valid options: ollama | replicate | sagemaker | bedrock | openai"
    )


def _build_sagemaker_llm(settings: Any):
    """
    Wraps SageMaker endpoint as a LangChain ChatModel.
    Returns AIMessage with .content — compatible with all agents.
    """
    import boto3
    import json
    from botocore.config import Config as BotocoreConfig
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage, BaseMessage
    from langchain_core.outputs import ChatResult, ChatGeneration
    from typing import Optional, List

    class SageMakerMistralChat(BaseChatModel):
        endpoint_name: str
        region_name: str
        temperature: float
        max_new_tokens: int

        @property
        def _llm_type(self) -> str:
            return "sagemaker_mistral_chat"

        def _generate(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            **kwargs: Any,
        ) -> ChatResult:
            # Convert messages to Mistral instruction format
            prompt = "\n".join(m.content for m in messages)
            formatted = f"<s>[INST] {prompt} [/INST]"

            client = boto3.client(
                "sagemaker-runtime",
                region_name=self.region_name,
                config=BotocoreConfig(
                    read_timeout=300,
                    connect_timeout=10,
                )
            )
            payload = {
                "inputs": formatted,
                "parameters": {
                    "temperature": self.temperature,
                    "max_new_tokens": self.max_new_tokens,
                    "return_full_text": False,
                },
            }

            response = client.invoke_endpoint(
                EndpointName=self.endpoint_name,
                ContentType="application/json",
                Body=json.dumps(payload),
            )
            result = json.loads(response["Body"].read())

            if isinstance(result, list) and result:
                text = result[0].get("generated_text", "").strip()
            else:
                text = str(result)

            return ChatResult(
                generations=[ChatGeneration(message=AIMessage(content=text))]
            )

    return SageMakerMistralChat(
        endpoint_name=settings.sagemaker_endpoint_name,
        region_name=settings.sagemaker_region,
        temperature=settings.llm_temperature,
        max_new_tokens=settings.sagemaker_max_new_tokens,
    )


@lru_cache(maxsize=1)
def get_embeddings():
    """
    Embeddings factory for FAISS vector store.
    Separate from the chat LLM — can use a different provider.
    """
    settings = get_settings()
    provider = settings.embedding_provider

    if provider == "ollama":
        from langchain_ollama import OllamaEmbeddings
        logger.info("Embeddings: Ollama — model=%s", settings.embedding_model_ollama)
        return OllamaEmbeddings(
            model=settings.embedding_model_ollama,
            base_url=settings.ollama_base_url,
        )

    if provider == "bedrock":
        from langchain_aws import BedrockEmbeddings
        logger.info("Embeddings: Bedrock — model=%s", settings.bedrock_embedding_model_id)
        return BedrockEmbeddings(
            model_id=settings.bedrock_embedding_model_id,
            region_name=settings.bedrock_region,
        )

    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        logger.info("Embeddings: OpenAI — model=%s", settings.embedding_model_openai)
        return OpenAIEmbeddings(
            model=settings.embedding_model_openai,
            api_key=settings.openai_api_key,
        )

    raise ValueError(
        f"Unknown EMBEDDING_PROVIDER='{provider}'. "
        "Valid options: ollama | bedrock | openai"
    )
