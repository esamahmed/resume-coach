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
        model_str = settings.replicate_model
        if settings.replicate_model_version:
            model_str = f"{model_str}:{settings.replicate_model_version}"
        logger.info("LLM: Replicate — model=%s", model_str)
        return _build_replicate_llm(settings, model_str)

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


def _build_replicate_llm(settings: Any, model_str: str):
    """
    Wraps the Replicate API as a LangChain chat model.
    langchain_community.llms.Replicate is avoided because it fetches the model's
    OpenAPI schema at init time and errors when the schema is None.
    """
    import replicate as replicate_sdk
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
    from langchain_core.outputs import ChatGeneration, ChatResult
    from typing import Optional, List

    class ReplicateChatModel(BaseChatModel):
        api_token: str
        model: str
        temperature: float
        max_new_tokens: int

        @property
        def _llm_type(self) -> str:
            return "replicate"

        def _generate(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            **kwargs: Any,
        ) -> ChatResult:
            import time

            system = "\n".join(m.content for m in messages if isinstance(m, SystemMessage))
            human  = "\n".join(m.content for m in messages if isinstance(m, HumanMessage))
            if system:
                prompt = f"<s>[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{human} [/INST]"
            else:
                prompt = f"<s>[INST] {human} [/INST]"

            client = replicate_sdk.Client(api_token=self.api_token)
            # Replicate free tier: burst=1, 6 req/min. Retry with backoff on 429.
            wait = 12
            for attempt in range(4):
                try:
                    output = client.run(
                        self.model,
                        input={
                            "prompt": prompt,
                            "temperature": self.temperature,
                            "max_new_tokens": self.max_new_tokens,
                        },
                    )
                    text = "".join(output) if hasattr(output, "__iter__") else str(output)
                    return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])
                except Exception as exc:
                    if attempt < 3 and "429" in str(exc):
                        logger.warning("Replicate 429 — waiting %ds before retry (attempt %d/3)", wait, attempt + 1)
                        time.sleep(wait)
                        wait *= 2
                    else:
                        raise

    return ReplicateChatModel(
        api_token=settings.replicate_api_token,
        model=model_str,
        temperature=settings.llm_temperature,
        max_new_tokens=settings.llm_max_tokens,
    )


def _build_sagemaker_llm(settings: Any):
    """
    Wraps the SageMaker endpoint as a LangChain-compatible LLM.
    Uses a custom LLM class because SageMaker requires direct boto3 invocation
    with Mistral's specific prompt format.
    """
    import boto3
    from langchain.llms.base import LLM
    from typing import Optional, List

    class SageMakerMistralLLM(LLM):
        endpoint_name: str
        region_name: str
        temperature: float
        max_new_tokens: int

        @property
        def _llm_type(self) -> str:
            return "sagemaker_mistral"

        def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            **kwargs: Any,
        ) -> str:
            client = boto3.client("sagemaker-runtime", region_name=self.region_name)

            # Mistral instruction format
            formatted = f"<s>[INST] {prompt} [/INST]"

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

            # SageMaker returns a list: [{"generated_text": "..."}]
            if isinstance(result, list) and result:
                return result[0].get("generated_text", "").strip()
            return str(result)

    return SageMakerMistralLLM(
        endpoint_name=settings.sagemaker_endpoint_name,
        region_name=settings.sagemaker_region,
        temperature=settings.llm_temperature,
        max_new_tokens=settings.llm_max_tokens,
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
