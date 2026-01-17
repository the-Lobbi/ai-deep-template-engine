"""RAG (Retrieval-Augmented Generation) Integration for DevOps Multi-Agent System.

This module provides knowledge retrieval capabilities using:
- Pinecone for vector storage
- Voyage AI embeddings (voyage-3-large) - officially recommended by Anthropic for Claude
- Hybrid search combining semantic and keyword retrieval
- Specialized DevOps knowledge base with multiple documentation sources

Environment Variables:
    PINECONE_API_KEY: Pinecone API key
    PINECONE_ENVIRONMENT: Pinecone environment (e.g., 'us-west1-gcp')
    VOYAGE_API_KEY: Voyage AI API key
    DEVOPS_KB_INDEX_NAME: Pinecone index name for DevOps knowledge base
"""

import asyncio
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import httpx
import structlog
from pinecone import Pinecone, ServerlessSpec
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = structlog.get_logger(__name__)


# =============================================================================
# Configuration
# =============================================================================


class RAGConfig(BaseSettings):
    """Configuration for RAG system from environment variables."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    pinecone_api_key: str = Field(alias="PINECONE_API_KEY")
    pinecone_environment: str = Field(default="us-west1-gcp", alias="PINECONE_ENVIRONMENT")
    voyage_api_key: str = Field(alias="VOYAGE_API_KEY")
    devops_kb_index_name: str = Field(default="devops-knowledge-base", alias="DEVOPS_KB_INDEX_NAME")

    # Embedding configuration
    voyage_model: str = Field(default="voyage-3-large")
    voyage_code_model: str = Field(default="voyage-code-3")
    voyage_api_base: str = Field(default="https://api.voyageai.com/v1")

    # Search configuration
    default_top_k: int = Field(default=5)
    max_retries: int = Field(default=3)
    retry_delay: float = Field(default=1.0)


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class Document:
    """Document to be indexed in the knowledge base.

    Attributes:
        id: Unique document identifier
        content: Document text content
        metadata: Additional metadata (source, type, tags, timestamps, etc.)
    """

    id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate document after initialization."""
        if not self.id:
            raise ValueError("Document ID cannot be empty")
        if not self.content:
            raise ValueError("Document content cannot be empty")

    def to_dict(self) -> Dict[str, Any]:
        """Convert document to dictionary for serialization."""
        return {
            "id": self.id,
            "content": self.content,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Document":
        """Create document from dictionary.

        Args:
            data: Dictionary with id, content, and optional metadata

        Returns:
            Document instance
        """
        return cls(
            id=data["id"],
            content=data["content"],
            metadata=data.get("metadata", {}),
        )


@dataclass
class SearchResult:
    """Result from knowledge base search.

    Attributes:
        id: Document identifier
        content: Document content
        score: Similarity score (0-1, higher is better)
        metadata: Document metadata
    """

    id: str
    content: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate search result after initialization."""
        if self.score < 0.0 or self.score > 1.0:
            logger.warning("score_outside_range", score=self.score, expected_range="[0, 1]")

    def to_dict(self) -> Dict[str, Any]:
        """Convert search result to dictionary for serialization."""
        return {
            "id": self.id,
            "content": self.content,
            "score": self.score,
            "metadata": self.metadata,
        }


# =============================================================================
# Voyage AI Embeddings
# =============================================================================


class VoyageEmbeddings:
    """Voyage AI embedding client for text vectorization.

    Voyage AI provides state-of-the-art embeddings officially recommended by
    Anthropic for use with Claude. Supports both general-purpose and code-specific models.

    Models:
        - voyage-3-large: General purpose, best for most use cases
        - voyage-code-3: Optimized for code and technical documentation

    Args:
        api_key: Voyage AI API key
        model: Model name (default: "voyage-3-large")
        api_base: API base URL
        max_retries: Maximum retry attempts for failed requests
        retry_delay: Delay between retries in seconds
    """

    def __init__(
        self,
        api_key: str,
        model: str = "voyage-3-large",
        api_base: str = "https://api.voyageai.com/v1",
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        self.api_key = api_key
        self.model = model
        self.api_base = api_base
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        self.client = httpx.AsyncClient(
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(30.0),
        )

        logger.info("voyage_embeddings_initialized", model=model, api_base=api_base)

    async def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text.

        Args:
            text: Input text to embed

        Returns:
            Embedding vector as list of floats

        Raises:
            httpx.HTTPError: If API request fails after retries
            ValueError: If text is empty
        """
        if not text or not text.strip():
            raise ValueError("Cannot embed empty text")

        embeddings = await self.embed_batch([text])
        return embeddings[0]

    async def embed_batch(
        self, texts: List[str], batch_size: int = 128
    ) -> List[List[float]]:
        """Generate embeddings for multiple texts, batching if necessary.

        Args:
            texts: List of input texts to embed
            batch_size: Maximum number of texts to embed per API call (default: 128)

        Returns:
            List of embedding vectors

        Raises:
            httpx.HTTPError: If API request fails after retries
            ValueError: If texts list is empty
        """
        if not texts:
            raise ValueError("Cannot embed empty list of texts")

        # Filter out empty strings
        valid_texts = [t for t in texts if t and t.strip()]
        if not valid_texts:
            raise ValueError("All texts are empty after filtering")

        # If batch is small enough, process in single request
        if len(valid_texts) <= batch_size:
            return await self._embed_single_batch(valid_texts)

        # Process in multiple batches
        logger.info(
            "embedding_large_batch",
            total_texts=len(valid_texts),
            batch_size=batch_size,
            num_batches=(len(valid_texts) + batch_size - 1) // batch_size,
        )

        all_embeddings = []
        for i in range(0, len(valid_texts), batch_size):
            batch = valid_texts[i : i + batch_size]
            embeddings = await self._embed_single_batch(batch)
            all_embeddings.extend(embeddings)

        return all_embeddings

    @retry(
        retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.RequestError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def _embed_single_batch(self, texts: List[str]) -> List[List[float]]:
        """Internal method to embed a single batch of texts with automatic retry.

        Args:
            texts: List of input texts to embed (must be non-empty)

        Returns:
            List of embedding vectors

        Raises:
            httpx.HTTPError: If API request fails after retries
        """
        url = f"{self.api_base}/embeddings"
        payload = {
            "model": self.model,
            "input": texts,
        }

        try:
            logger.debug(
                "embedding_batch_attempt",
                num_texts=len(texts),
            )

            response = await self.client.post(url, json=payload)
            response.raise_for_status()

            result = response.json()
            embeddings = [item["embedding"] for item in result["data"]]

            logger.debug("embedding_batch_success", num_embeddings=len(embeddings))
            return embeddings

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                logger.warning("rate_limited", status_code=429)
            elif e.response.status_code >= 500:
                logger.warning("server_error", status_code=e.response.status_code)
            else:
                logger.error(
                    "http_client_error",
                    status_code=e.response.status_code,
                    response_text=e.response.text,
                )
            raise

        except httpx.RequestError as e:
            logger.warning("request_error", error=str(e))
            raise

    async def _embed_single_batch_manual_retry(self, texts: List[str]) -> List[List[float]]:
        """Internal method to embed a single batch with manual retry (deprecated).

        This method is kept for compatibility but the decorator-based version is preferred.
        """
        url = f"{self.api_base}/embeddings"
        payload = {
            "model": self.model,
            "input": texts,
        }

        last_error: Optional[Exception] = None

        for attempt in range(self.max_retries):
            try:
                logger.debug(
                    "embedding_batch_attempt",
                    num_texts=len(valid_texts),
                    attempt=attempt + 1,
                    max_retries=self.max_retries,
                )

                response = await self.client.post(url, json=payload)
                response.raise_for_status()

                result = response.json()
                embeddings = [item["embedding"] for item in result["data"]]

                logger.debug("embedding_batch_success", num_embeddings=len(embeddings))
                return embeddings

            except httpx.HTTPStatusError as e:
                last_error = e

                if e.response.status_code == 429:  # Rate limit
                    wait_time = self.retry_delay * (2**attempt)
                    logger.warning(
                        "rate_limited",
                        status_code=429,
                        wait_time=wait_time,
                        attempt=attempt + 1,
                    )
                    await asyncio.sleep(wait_time)
                elif e.response.status_code >= 500:  # Server error
                    logger.warning(
                        "server_error",
                        status_code=e.response.status_code,
                        attempt=attempt + 1,
                    )
                    await asyncio.sleep(self.retry_delay)
                else:
                    # Client error, don't retry
                    logger.error(
                        "http_client_error",
                        status_code=e.response.status_code,
                        response_text=e.response.text,
                    )
                    raise

            except httpx.RequestError as e:
                last_error = e
                logger.warning("request_error", error=str(e), attempt=attempt + 1)
                await asyncio.sleep(self.retry_delay)

        # All retries exhausted
        logger.error("embedding_failed", max_retries=self.max_retries)
        raise last_error or Exception("Embedding failed for unknown reason")

    async def close(self) -> None:
        """Close the HTTP client."""
        await self.client.aclose()

    async def __aenter__(self) -> "VoyageEmbeddings":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()


# =============================================================================
# Pinecone Knowledge Store
# =============================================================================


class PineconeKnowledgeStore:
    """Pinecone vector database client for knowledge storage and retrieval.

    Args:
        index_name: Pinecone index name
        namespace: Namespace for organizing vectors
        embeddings: VoyageEmbeddings instance for text vectorization
        api_key: Pinecone API key
        environment: Pinecone environment
    """

    def __init__(
        self,
        index_name: str,
        namespace: str,
        embeddings: VoyageEmbeddings,
        api_key: str,
        environment: str = "us-west1-gcp",
    ):
        self.index_name = index_name
        self.namespace = namespace
        self.embeddings = embeddings

        # Initialize Pinecone client
        self.pc = Pinecone(api_key=api_key)

        # Connect to index
        try:
            self.index = self.pc.Index(index_name)
            logger.info(
                "pinecone_connected",
                index_name=index_name,
                namespace=namespace,
                environment=environment,
            )
        except Exception as e:
            logger.error(
                "pinecone_connection_failed",
                index_name=index_name,
                error=str(e),
            )
            raise

    async def index_document(self, doc_id: str, text: str, metadata: Dict[str, Any]) -> None:
        """Index a single document in Pinecone.

        Args:
            doc_id: Unique document identifier
            text: Document text content
            metadata: Document metadata

        Raises:
            Exception: If indexing fails
        """
        try:
            # Generate embedding
            embedding = await self.embeddings.embed_text(text)

            # Store full text in metadata
            metadata_with_text = {**metadata, "text": text}

            # Upsert to Pinecone
            self.index.upsert(
                vectors=[(doc_id, embedding, metadata_with_text)],
                namespace=self.namespace,
            )

            logger.debug("document_indexed", doc_id=doc_id, namespace=self.namespace)

        except Exception as e:
            logger.error("document_indexing_failed", doc_id=doc_id, error=str(e))
            raise

    async def index_documents(self, docs: List[Document]) -> None:
        """Index multiple documents in batch.

        Args:
            docs: List of Document objects to index

        Raises:
            Exception: If batch indexing fails
        """
        if not docs:
            logger.warning("no_documents_to_index")
            return

        try:
            # Generate embeddings for all documents
            texts = [doc.content for doc in docs]
            embeddings = await self.embeddings.embed_batch(texts)

            # Prepare vectors for upsert
            vectors = []
            for doc, embedding in zip(docs, embeddings):
                metadata_with_text = {**doc.metadata, "text": doc.content}
                vectors.append((doc.id, embedding, metadata_with_text))

            # Batch upsert (Pinecone recommends batches of 100)
            batch_size = 100
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i : i + batch_size]
                self.index.upsert(vectors=batch, namespace=self.namespace)
                logger.debug(
                    "batch_indexed",
                    batch_num=i // batch_size + 1,
                    batch_size=len(batch),
                )

            logger.info("documents_indexed", total_docs=len(docs), namespace=self.namespace)

        except Exception as e:
            logger.error("batch_indexing_failed", num_docs=len(docs), error=str(e))
            raise

    async def search(
        self,
        query: str,
        top_k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Search for similar documents using semantic similarity.

        Args:
            query: Search query text
            top_k: Number of top results to return
            filter: Optional metadata filter (e.g., {"source": "kubernetes"})

        Returns:
            List of SearchResult objects ordered by similarity

        Raises:
            Exception: If search fails
        """
        try:
            # Generate query embedding
            query_embedding = await self.embeddings.embed_text(query)

            # Query Pinecone
            response = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                namespace=self.namespace,
                filter=filter,
                include_metadata=True,
            )

            # Convert to SearchResult objects
            results = []
            for match in response.matches:
                metadata = match.metadata or {}
                content = metadata.pop("text", "")

                results.append(
                    SearchResult(
                        id=match.id,
                        content=content,
                        score=match.score,
                        metadata=metadata,
                    )
                )

            logger.info(
                "search_completed",
                query=query[:100],
                num_results=len(results),
                top_k=top_k,
                has_filter=filter is not None,
            )
            return results

        except Exception as e:
            logger.error("search_failed", query=query[:100], error=str(e))
            raise

    async def delete_document(self, doc_id: str) -> None:
        """Delete a document from the index.

        Args:
            doc_id: Document identifier to delete

        Raises:
            Exception: If deletion fails
        """
        try:
            self.index.delete(ids=[doc_id], namespace=self.namespace)
            logger.debug("document_deleted", doc_id=doc_id, namespace=self.namespace)
        except Exception as e:
            logger.error("document_deletion_failed", doc_id=doc_id, error=str(e))
            raise

    async def delete_all(self) -> None:
        """Delete all documents in the namespace.

        Warning: This is a destructive operation!

        Raises:
            Exception: If deletion fails
        """
        try:
            self.index.delete(delete_all=True, namespace=self.namespace)
            logger.warning("all_documents_deleted", namespace=self.namespace)
        except Exception as e:
            logger.error("delete_all_failed", namespace=self.namespace, error=str(e))
            raise


# =============================================================================
# DevOps Knowledge Base
# =============================================================================


class DevOpsKnowledgeBase:
    """Specialized knowledge base for DevOps operations.

    Provides semantic search across multiple documentation sources:
    - Kubernetes documentation
    - Harness CI/CD documentation
    - Terraform documentation
    - Internal runbooks and procedures

    Args:
        store: PineconeKnowledgeStore instance
        config: RAGConfig instance
    """

    def __init__(self, store: PineconeKnowledgeStore, config: RAGConfig):
        self.store = store
        self.config = config

        # Documentation source categories
        self.sources = {
            "kubernetes": "Kubernetes documentation and API references",
            "harness": "Harness CI/CD platform documentation",
            "terraform": "Terraform infrastructure as code documentation",
            "runbooks": "Internal runbooks and operational procedures",
            "best_practices": "DevOps best practices and patterns",
        }

        logger.info("devops_knowledge_base_initialized", sources=list(self.sources.keys()))

    async def search_runbooks(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """Search internal runbooks and operational procedures.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of relevant runbook SearchResults
        """
        logger.debug("searching_runbooks", query=query[:100], top_k=top_k)
        return await self.store.search(
            query=query,
            top_k=top_k,
            filter={"source": "runbooks"},
        )

    async def search_k8s_docs(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """Search Kubernetes documentation.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of relevant Kubernetes documentation SearchResults
        """
        logger.debug("searching_kubernetes_docs", query=query[:100], top_k=top_k)
        return await self.store.search(
            query=query,
            top_k=top_k,
            filter={"source": "kubernetes"},
        )

    async def search_harness_docs(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """Search Harness CI/CD documentation.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of relevant Harness documentation SearchResults
        """
        logger.debug("searching_harness_docs", query=query[:100], top_k=top_k)
        return await self.store.search(
            query=query,
            top_k=top_k,
            filter={"source": "harness"},
        )

    async def search_terraform_docs(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """Search Terraform documentation.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of relevant Terraform documentation SearchResults
        """
        logger.debug("searching_terraform_docs", query=query[:100], top_k=top_k)
        return await self.store.search(
            query=query,
            top_k=top_k,
            filter={"source": "terraform"},
        )

    async def search_all(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """Search across all documentation sources.

        Args:
            query: Search query
            top_k: Number of results to return (distributed across sources)

        Returns:
            List of relevant SearchResults from all sources, sorted by score
        """
        logger.debug("searching_all_sources", query=query[:100], top_k=top_k)
        results = await self.store.search(query=query, top_k=top_k)

        # Results are already sorted by score from Pinecone
        return results

    async def search_by_tags(
        self,
        query: str,
        tags: List[str],
        top_k: int = 5,
    ) -> List[SearchResult]:
        """Search documents with specific tags.

        Args:
            query: Search query
            tags: List of tags to filter by
            top_k: Number of results to return

        Returns:
            List of relevant SearchResults matching the tags
        """
        logger.debug("searching_by_tags", query=query[:100], tags=tags, top_k=top_k)

        # Pinecone supports filtering on array fields
        filter_dict = {"tags": {"$in": tags}}

        return await self.store.search(
            query=query,
            top_k=top_k,
            filter=filter_dict,
        )


# =============================================================================
# Hybrid Retriever
# =============================================================================


class HybridRetriever:
    """Hybrid retrieval combining semantic vector search with keyword matching.

    Provides balanced retrieval by combining:
    - Semantic similarity (vector search)
    - Keyword matching (BM25-style ranking)

    Args:
        knowledge_base: DevOpsKnowledgeBase instance
        alpha: Balance between semantic (1.0) and keyword (0.0) search
    """

    def __init__(self, knowledge_base: DevOpsKnowledgeBase, alpha: float = 0.5):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"Alpha must be between 0 and 1, got {alpha}")

        self.knowledge_base = knowledge_base
        self.alpha = alpha

        logger.info("hybrid_retriever_initialized", alpha=alpha)

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        alpha: Optional[float] = None,
    ) -> List[SearchResult]:
        """Retrieve documents using hybrid search.

        Args:
            query: Search query
            top_k: Number of results to return
            alpha: Override default alpha (1.0 = pure semantic, 0.0 = pure keyword)

        Returns:
            List of SearchResults ranked by hybrid score
        """
        alpha = alpha if alpha is not None else self.alpha

        if alpha == 1.0:
            # Pure semantic search
            logger.debug("using_pure_semantic_search", query=query[:100], top_k=top_k)
            return await self.knowledge_base.search_all(query, top_k=top_k)

        # Get more candidates for re-ranking
        semantic_results = await self.knowledge_base.search_all(query, top_k=top_k * 2)

        if alpha == 0.0:
            # Pure keyword search (simple implementation using text matching)
            logger.debug("using_pure_keyword_search", query=query[:100], top_k=top_k)
            return self._keyword_rank(query, semantic_results, top_k)

        # Hybrid: combine semantic and keyword scores
        logger.debug(
            "using_hybrid_search", query=query[:100], alpha=alpha, top_k=top_k
        )
        return self._hybrid_rank(query, semantic_results, top_k, alpha)

    def _keyword_rank(
        self,
        query: str,
        results: List[SearchResult],
        top_k: int,
    ) -> List[SearchResult]:
        """Rank results by keyword matching.

        Simple implementation using term frequency. In production,
        consider using BM25 or other advanced ranking algorithms.
        """
        query_terms = set(query.lower().split())

        # Calculate keyword scores
        scored_results = []
        for result in results:
            content_lower = result.content.lower()

            # Count matching terms
            matches = sum(1 for term in query_terms if term in content_lower)
            keyword_score = matches / len(query_terms) if query_terms else 0.0

            # Create new result with keyword score
            scored_results.append(SearchResult(
                id=result.id,
                content=result.content,
                score=keyword_score,
                metadata=result.metadata,
            ))

        # Sort by keyword score and return top_k
        scored_results.sort(key=lambda x: x.score, reverse=True)
        return scored_results[:top_k]

    def _hybrid_rank(
        self,
        query: str,
        results: List[SearchResult],
        top_k: int,
        alpha: float,
    ) -> List[SearchResult]:
        """Combine semantic and keyword scores.

        Score = alpha * semantic_score + (1 - alpha) * keyword_score
        """
        query_terms = set(query.lower().split())

        # Calculate hybrid scores
        scored_results = []
        for result in results:
            semantic_score = result.score

            # Calculate keyword score
            content_lower = result.content.lower()
            matches = sum(1 for term in query_terms if term in content_lower)
            keyword_score = matches / len(query_terms) if query_terms else 0.0

            # Combine scores
            hybrid_score = alpha * semantic_score + (1 - alpha) * keyword_score

            scored_results.append(SearchResult(
                id=result.id,
                content=result.content,
                score=hybrid_score,
                metadata={
                    **result.metadata,
                    "semantic_score": semantic_score,
                    "keyword_score": keyword_score,
                },
            ))

        # Sort by hybrid score and return top_k
        scored_results.sort(key=lambda x: x.score, reverse=True)
        return scored_results[:top_k]


# =============================================================================
# Factory Function
# =============================================================================


async def create_rag_retriever(
    index_name: Optional[str] = None,
    namespace: str = "devops",
    use_code_embeddings: bool = False,
    alpha: float = 0.5,
    config: Optional[RAGConfig] = None,
) -> tuple[DevOpsKnowledgeBase, HybridRetriever, VoyageEmbeddings]:
    """Factory function to create a configured RAG retrieval system.

    This is the main entry point for using the RAG system. It initializes:
    - Voyage AI embeddings
    - Pinecone knowledge store
    - DevOps knowledge base
    - Hybrid retriever

    Args:
        index_name: Override default Pinecone index name
        namespace: Pinecone namespace (default: "devops")
        use_code_embeddings: Use voyage-code-3 model for code-specific embeddings
        alpha: Balance between semantic (1.0) and keyword (0.0) search
        config: Optional RAGConfig instance (loads from env if not provided)

    Returns:
        Tuple of (DevOpsKnowledgeBase, HybridRetriever, VoyageEmbeddings)

    Raises:
        ValueError: If required environment variables are missing
        Exception: If initialization fails

    Example:
        >>> kb, retriever, embeddings = await create_rag_retriever()
        >>> results = await kb.search_all("How to scale Kubernetes deployment?")
        >>> for result in results:
        ...     print(f"{result.score:.2f}: {result.content[:100]}")
    """
    try:
        # Load configuration
        if config is None:
            config = RAGConfig()

        # Initialize Voyage embeddings
        model = config.voyage_code_model if use_code_embeddings else config.voyage_model
        embeddings = VoyageEmbeddings(
            api_key=config.voyage_api_key,
            model=model,
            api_base=config.voyage_api_base,
            max_retries=config.max_retries,
            retry_delay=config.retry_delay,
        )

        # Initialize Pinecone knowledge store
        actual_index_name = index_name or config.devops_kb_index_name
        store = PineconeKnowledgeStore(
            index_name=actual_index_name,
            namespace=namespace,
            embeddings=embeddings,
            api_key=config.pinecone_api_key,
            environment=config.pinecone_environment,
        )

        # Initialize DevOps knowledge base
        knowledge_base = DevOpsKnowledgeBase(store=store, config=config)

        # Initialize hybrid retriever
        retriever = HybridRetriever(knowledge_base=knowledge_base, alpha=alpha)

        logger.info(
            "rag_retriever_created",
            index_name=actual_index_name,
            namespace=namespace,
            model=model,
            alpha=alpha,
        )
        return knowledge_base, retriever, embeddings

    except Exception as e:
        logger.error("rag_retriever_creation_failed", error=str(e))
        raise


# =============================================================================
# Exports
# =============================================================================


__all__ = [
    "RAGConfig",
    "Document",
    "SearchResult",
    "VoyageEmbeddings",
    "PineconeKnowledgeStore",
    "DevOpsKnowledgeBase",
    "HybridRetriever",
    "create_rag_retriever",
]
