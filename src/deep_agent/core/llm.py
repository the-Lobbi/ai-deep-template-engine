"""LLM and Embeddings Configuration for Deep Agent.

This module provides configuration and factory functions for:
- Claude models via Anthropic (Claude Sonnet 4.5 primary)
- Voyage AI embeddings (voyage-3-large recommended for Claude)
- LangSmith tracing integration
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Dict, List, Optional, Sequence

import structlog
from langchain_anthropic import ChatAnthropic
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = structlog.get_logger(__name__)


class LLMConfig(BaseSettings):
    """Configuration for LLM and embedding models.

    Loads from environment variables with sensible defaults.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Anthropic Claude configuration
    anthropic_api_key: SecretStr = Field(alias="ANTHROPIC_API_KEY")
    default_model: str = Field(default="claude-sonnet-4-5-20250514", alias="DEFAULT_MODEL")
    max_tokens: int = Field(default=4096, alias="MAX_TOKENS")
    temperature: float = Field(default=0.0, alias="TEMPERATURE")

    # Voyage AI embeddings configuration
    voyage_api_key: SecretStr = Field(alias="VOYAGE_API_KEY")
    voyage_model: str = Field(default="voyage-3-large", alias="VOYAGE_MODEL")
    voyage_code_model: str = Field(default="voyage-code-3", alias="VOYAGE_CODE_MODEL")
    voyage_api_base: str = Field(default="https://api.voyageai.com/v1", alias="VOYAGE_API_BASE")

    # LangSmith configuration
    langsmith_api_key: Optional[SecretStr] = Field(default=None, alias="LANGSMITH_API_KEY")
    langsmith_project: str = Field(default="deep-agent", alias="LANGSMITH_PROJECT")
    langsmith_tracing: bool = Field(default=True, alias="LANGSMITH_TRACING")

    # Rate limiting and retry
    max_retries: int = Field(default=3, alias="MAX_RETRIES")
    request_timeout: float = Field(default=60.0, alias="REQUEST_TIMEOUT")


@dataclass
class ModelSpec:
    """Specification for a model with capabilities and costs."""

    name: str
    provider: str
    max_tokens: int
    input_cost_per_1k: float
    output_cost_per_1k: float
    supports_tools: bool = True
    supports_vision: bool = False
    best_for: Sequence[str] = field(default_factory=list)


# Available models with specifications
AVAILABLE_MODELS: Dict[str, ModelSpec] = {
    "claude-sonnet-4-5-20250514": ModelSpec(
        name="claude-sonnet-4-5-20250514",
        provider="anthropic",
        max_tokens=8192,
        input_cost_per_1k=0.003,
        output_cost_per_1k=0.015,
        supports_tools=True,
        supports_vision=True,
        best_for=["general", "coding", "analysis", "tools"],
    ),
    "claude-3-5-haiku-20241022": ModelSpec(
        name="claude-3-5-haiku-20241022",
        provider="anthropic",
        max_tokens=4096,
        input_cost_per_1k=0.00025,
        output_cost_per_1k=0.00125,
        supports_tools=True,
        supports_vision=True,
        best_for=["fast", "simple", "classification"],
    ),
    "claude-opus-4-20250514": ModelSpec(
        name="claude-opus-4-20250514",
        provider="anthropic",
        max_tokens=8192,
        input_cost_per_1k=0.015,
        output_cost_per_1k=0.075,
        supports_tools=True,
        supports_vision=True,
        best_for=["complex", "reasoning", "long-context"],
    ),
}


def create_llm(
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    config: Optional[LLMConfig] = None,
    callbacks: Optional[List[BaseCallbackHandler]] = None,
    **kwargs: Any,
) -> ChatAnthropic:
    """Create a Claude LLM instance.

    Args:
        model: Model name (defaults to claude-sonnet-4-5-20250514)
        temperature: Sampling temperature (0.0 for deterministic)
        max_tokens: Maximum tokens in response
        config: LLMConfig instance (loads from env if not provided)
        callbacks: Optional callback handlers
        **kwargs: Additional arguments to pass to ChatAnthropic

    Returns:
        Configured ChatAnthropic instance

    Example:
        >>> llm = create_llm(model="claude-sonnet-4-5-20250514", temperature=0.0)
        >>> response = await llm.ainvoke("Hello, Claude!")
    """
    if config is None:
        try:
            config = LLMConfig()
        except Exception as e:
            logger.warning("Failed to load LLMConfig from env, using defaults", error=str(e))
            # Create with minimal required settings
            config = LLMConfig(
                anthropic_api_key=SecretStr(os.environ.get("ANTHROPIC_API_KEY", "")),
                voyage_api_key=SecretStr(os.environ.get("VOYAGE_API_KEY", "")),
            )

    model_name = model or config.default_model
    temp = temperature if temperature is not None else config.temperature
    tokens = max_tokens or config.max_tokens

    # Setup LangSmith tracing if configured
    if config.langsmith_tracing and config.langsmith_api_key:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = config.langsmith_api_key.get_secret_value()
        os.environ["LANGCHAIN_PROJECT"] = config.langsmith_project
        logger.info("LangSmith tracing enabled", project=config.langsmith_project)

    llm = ChatAnthropic(
        model=model_name,
        api_key=config.anthropic_api_key.get_secret_value(),
        max_tokens=tokens,
        temperature=temp,
        max_retries=config.max_retries,
        timeout=config.request_timeout,
        callbacks=callbacks or [],
        **kwargs,
    )

    logger.info(
        "LLM created",
        model=model_name,
        temperature=temp,
        max_tokens=tokens,
    )

    return llm


class VoyageAIEmbeddings(Embeddings):
    """Voyage AI embeddings implementation for LangChain.

    Voyage AI is officially recommended by Anthropic for use with Claude.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "voyage-3-large",
        api_base: str = "https://api.voyageai.com/v1",
        batch_size: int = 128,
    ):
        """Initialize Voyage AI embeddings.

        Args:
            api_key: Voyage AI API key
            model: Model name (voyage-3-large, voyage-code-3, etc.)
            api_base: API base URL
            batch_size: Maximum texts per batch request
        """
        self.api_key = api_key
        self.model = model
        self.api_base = api_base
        self.batch_size = batch_size

        import httpx
        self._client = httpx.Client(
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=60.0,
        )
        self._async_client = httpx.AsyncClient(
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=60.0,
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            response = self._client.post(
                f"{self.api_base}/embeddings",
                json={"model": self.model, "input": batch},
            )
            response.raise_for_status()
            result = response.json()
            embeddings.extend([item["embedding"] for item in result["data"]])

        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query text.

        Args:
            text: Query text to embed

        Returns:
            Embedding vector
        """
        return self.embed_documents([text])[0]

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Async embed a list of documents.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            response = await self._async_client.post(
                f"{self.api_base}/embeddings",
                json={"model": self.model, "input": batch},
            )
            response.raise_for_status()
            result = response.json()
            embeddings.extend([item["embedding"] for item in result["data"]])

        return embeddings

    async def aembed_query(self, text: str) -> List[float]:
        """Async embed a single query text.

        Args:
            text: Query text to embed

        Returns:
            Embedding vector
        """
        result = await self.aembed_documents([text])
        return result[0]

    def __del__(self):
        """Clean up HTTP clients."""
        try:
            self._client.close()
        except Exception:
            pass


def create_embeddings(
    model: Optional[str] = None,
    config: Optional[LLMConfig] = None,
    use_code_model: bool = False,
) -> VoyageAIEmbeddings:
    """Create Voyage AI embeddings instance.

    Args:
        model: Model name (overrides config)
        config: LLMConfig instance
        use_code_model: Use voyage-code-3 for code embeddings

    Returns:
        VoyageAIEmbeddings instance
    """
    if config is None:
        try:
            config = LLMConfig()
        except Exception as e:
            logger.warning("Failed to load LLMConfig from env", error=str(e))
            config = LLMConfig(
                anthropic_api_key=SecretStr(os.environ.get("ANTHROPIC_API_KEY", "")),
                voyage_api_key=SecretStr(os.environ.get("VOYAGE_API_KEY", "")),
            )

    model_name = model
    if model_name is None:
        model_name = config.voyage_code_model if use_code_model else config.voyage_model

    embeddings = VoyageAIEmbeddings(
        api_key=config.voyage_api_key.get_secret_value(),
        model=model_name,
        api_base=config.voyage_api_base,
    )

    logger.info("Embeddings created", model=model_name)
    return embeddings


def select_model_for_task(task_type: str) -> str:
    """Select the best model for a given task type.

    Args:
        task_type: Type of task (coding, analysis, fast, complex, etc.)

    Returns:
        Model name string
    """
    task_type = task_type.lower()

    if task_type in ["fast", "simple", "classification", "quick"]:
        return "claude-3-5-haiku-20241022"
    elif task_type in ["complex", "reasoning", "long", "detailed"]:
        return "claude-opus-4-20250514"
    else:
        # Default to Sonnet for most tasks
        return "claude-sonnet-4-5-20250514"


@lru_cache(maxsize=8)
def get_cached_llm(
    model: str,
    temperature: float = 0.0,
    max_tokens: int = 4096,
) -> ChatAnthropic:
    """Get a cached LLM instance.

    Useful for reusing the same model configuration across multiple calls.

    Args:
        model: Model name
        temperature: Sampling temperature
        max_tokens: Maximum tokens

    Returns:
        Cached ChatAnthropic instance
    """
    return create_llm(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )
