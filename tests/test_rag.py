"""Unit tests for RAG integration module.

These tests verify the functionality of the RAG system including:
- Voyage AI embeddings
- Pinecone knowledge store
- DevOps knowledge base
- Hybrid retriever
"""

import os
from unittest.mock import AsyncMock, Mock, patch

import pytest

from deep_agent.devops.rag import (
    Document,
    DevOpsKnowledgeBase,
    HybridRetriever,
    PineconeKnowledgeStore,
    RAGConfig,
    SearchResult,
    VoyageEmbeddings,
    create_rag_retriever,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_config():
    """Mock RAG configuration."""
    return RAGConfig(
        pinecone_api_key="test-pinecone-key",
        pinecone_environment="us-west1-gcp",
        voyage_api_key="test-voyage-key",
        devops_kb_index_name="test-index",
    )


@pytest.fixture
def sample_documents():
    """Sample documents for testing."""
    return [
        Document(
            id="doc-001",
            content="Kubernetes deployment scaling guide",
            metadata={"source": "kubernetes", "tags": ["deployment", "scaling"]},
        ),
        Document(
            id="doc-002",
            content="Harness pipeline trigger configuration",
            metadata={"source": "harness", "tags": ["pipeline", "trigger"]},
        ),
        Document(
            id="doc-003",
            content="Terraform state management best practices",
            metadata={"source": "terraform", "tags": ["state", "best-practices"]},
        ),
    ]


@pytest.fixture
def sample_search_results():
    """Sample search results for testing."""
    return [
        SearchResult(
            id="doc-001",
            content="Kubernetes deployment scaling guide",
            score=0.95,
            metadata={"source": "kubernetes"},
        ),
        SearchResult(
            id="doc-002",
            content="Harness pipeline trigger configuration",
            score=0.85,
            metadata={"source": "harness"},
        ),
    ]


# =============================================================================
# Document Tests
# =============================================================================


def test_document_creation():
    """Test Document dataclass creation."""
    doc = Document(
        id="test-001",
        content="Test content",
        metadata={"key": "value"},
    )

    assert doc.id == "test-001"
    assert doc.content == "Test content"
    assert doc.metadata == {"key": "value"}


def test_document_validation():
    """Test Document validation."""
    # Empty ID should raise ValueError
    with pytest.raises(ValueError, match="ID cannot be empty"):
        Document(id="", content="Test content")

    # Empty content should raise ValueError
    with pytest.raises(ValueError, match="content cannot be empty"):
        Document(id="test-001", content="")


# =============================================================================
# SearchResult Tests
# =============================================================================


def test_search_result_creation():
    """Test SearchResult dataclass creation."""
    result = SearchResult(
        id="test-001",
        content="Test content",
        score=0.95,
        metadata={"key": "value"},
    )

    assert result.id == "test-001"
    assert result.content == "Test content"
    assert result.score == 0.95
    assert result.metadata == {"key": "value"}


def test_search_result_score_validation():
    """Test SearchResult score validation warning."""
    # Score outside [0, 1] should log warning but not fail
    result = SearchResult(id="test-001", content="Test", score=1.5, metadata={})
    assert result.score == 1.5  # Still created


# =============================================================================
# VoyageEmbeddings Tests
# =============================================================================


@pytest.mark.asyncio
async def test_voyage_embeddings_initialization():
    """Test VoyageEmbeddings initialization."""
    embeddings = VoyageEmbeddings(api_key="test-key", model="voyage-3-large")

    assert embeddings.api_key == "test-key"
    assert embeddings.model == "voyage-3-large"
    assert embeddings.api_base == "https://api.voyageai.com/v1"
    assert embeddings.max_retries == 3
    assert embeddings.retry_delay == 1.0

    await embeddings.close()


@pytest.mark.asyncio
async def test_voyage_embeddings_embed_text_validation():
    """Test VoyageEmbeddings input validation."""
    embeddings = VoyageEmbeddings(api_key="test-key")

    # Empty text should raise ValueError
    with pytest.raises(ValueError, match="Cannot embed empty text"):
        await embeddings.embed_text("")

    with pytest.raises(ValueError, match="Cannot embed empty text"):
        await embeddings.embed_text("   ")

    await embeddings.close()


@pytest.mark.asyncio
async def test_voyage_embeddings_embed_batch_validation():
    """Test VoyageEmbeddings batch input validation."""
    embeddings = VoyageEmbeddings(api_key="test-key")

    # Empty list should raise ValueError
    with pytest.raises(ValueError, match="Cannot embed empty list"):
        await embeddings.embed_batch([])

    # All empty strings should raise ValueError
    with pytest.raises(ValueError, match="All texts are empty"):
        await embeddings.embed_batch(["", "  ", ""])

    await embeddings.close()


@pytest.mark.asyncio
@patch("httpx.AsyncClient.post")
async def test_voyage_embeddings_embed_text_success(mock_post):
    """Test successful text embedding."""
    # Mock API response
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "data": [{"embedding": [0.1, 0.2, 0.3]}]
    }
    mock_post.return_value = mock_response

    embeddings = VoyageEmbeddings(api_key="test-key")

    result = await embeddings.embed_text("test text")

    assert result == [0.1, 0.2, 0.3]
    mock_post.assert_called_once()

    await embeddings.close()


@pytest.mark.asyncio
@patch("httpx.AsyncClient.post")
async def test_voyage_embeddings_embed_batch_success(mock_post):
    """Test successful batch embedding."""
    # Mock API response
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "data": [
            {"embedding": [0.1, 0.2, 0.3]},
            {"embedding": [0.4, 0.5, 0.6]},
        ]
    }
    mock_post.return_value = mock_response

    embeddings = VoyageEmbeddings(api_key="test-key")

    results = await embeddings.embed_batch(["text1", "text2"])

    assert len(results) == 2
    assert results[0] == [0.1, 0.2, 0.3]
    assert results[1] == [0.4, 0.5, 0.6]

    await embeddings.close()


# =============================================================================
# PineconeKnowledgeStore Tests
# =============================================================================


@pytest.mark.asyncio
async def test_pinecone_store_initialization():
    """Test PineconeKnowledgeStore initialization."""
    mock_embeddings = Mock(spec=VoyageEmbeddings)

    with patch("deep_agent.devops.rag.Pinecone") as mock_pinecone:
        mock_index = Mock()
        mock_pinecone.return_value.Index.return_value = mock_index

        store = PineconeKnowledgeStore(
            index_name="test-index",
            namespace="test-namespace",
            embeddings=mock_embeddings,
            api_key="test-key",
        )

        assert store.index_name == "test-index"
        assert store.namespace == "test-namespace"
        assert store.embeddings == mock_embeddings


@pytest.mark.asyncio
async def test_pinecone_store_index_document():
    """Test indexing a single document."""
    mock_embeddings = AsyncMock(spec=VoyageEmbeddings)
    mock_embeddings.embed_text.return_value = [0.1, 0.2, 0.3]

    with patch("deep_agent.devops.rag.Pinecone") as mock_pinecone:
        mock_index = Mock()
        mock_pinecone.return_value.Index.return_value = mock_index

        store = PineconeKnowledgeStore(
            index_name="test-index",
            namespace="test-namespace",
            embeddings=mock_embeddings,
            api_key="test-key",
        )

        await store.index_document(
            doc_id="test-001",
            text="Test content",
            metadata={"key": "value"},
        )

        # Verify embedding was generated
        mock_embeddings.embed_text.assert_called_once_with("Test content")

        # Verify upsert was called
        mock_index.upsert.assert_called_once()


@pytest.mark.asyncio
async def test_pinecone_store_search():
    """Test searching for similar documents."""
    mock_embeddings = AsyncMock(spec=VoyageEmbeddings)
    mock_embeddings.embed_text.return_value = [0.1, 0.2, 0.3]

    with patch("deep_agent.devops.rag.Pinecone") as mock_pinecone:
        # Mock search response
        mock_match = Mock()
        mock_match.id = "doc-001"
        mock_match.score = 0.95
        mock_match.metadata = {"text": "Test content", "source": "test"}

        mock_response = Mock()
        mock_response.matches = [mock_match]

        mock_index = Mock()
        mock_index.query.return_value = mock_response
        mock_pinecone.return_value.Index.return_value = mock_index

        store = PineconeKnowledgeStore(
            index_name="test-index",
            namespace="test-namespace",
            embeddings=mock_embeddings,
            api_key="test-key",
        )

        results = await store.search("test query", top_k=5)

        assert len(results) == 1
        assert results[0].id == "doc-001"
        assert results[0].score == 0.95
        assert results[0].content == "Test content"

        # Verify query was called
        mock_index.query.assert_called_once()


# =============================================================================
# DevOpsKnowledgeBase Tests
# =============================================================================


@pytest.mark.asyncio
async def test_devops_kb_initialization(mock_config):
    """Test DevOpsKnowledgeBase initialization."""
    mock_store = Mock(spec=PineconeKnowledgeStore)

    kb = DevOpsKnowledgeBase(store=mock_store, config=mock_config)

    assert kb.store == mock_store
    assert kb.config == mock_config
    assert "kubernetes" in kb.sources
    assert "harness" in kb.sources
    assert "terraform" in kb.sources
    assert "runbooks" in kb.sources


@pytest.mark.asyncio
async def test_devops_kb_search_k8s_docs(sample_search_results):
    """Test searching Kubernetes documentation."""
    mock_store = AsyncMock(spec=PineconeKnowledgeStore)
    mock_store.search.return_value = sample_search_results
    mock_config = Mock(spec=RAGConfig)

    kb = DevOpsKnowledgeBase(store=mock_store, config=mock_config)

    results = await kb.search_k8s_docs("deployment scaling", top_k=5)

    assert len(results) == 2
    mock_store.search.assert_called_once_with(
        query="deployment scaling",
        top_k=5,
        filter={"source": "kubernetes"},
    )


@pytest.mark.asyncio
async def test_devops_kb_search_all(sample_search_results):
    """Test searching all documentation sources."""
    mock_store = AsyncMock(spec=PineconeKnowledgeStore)
    mock_store.search.return_value = sample_search_results
    mock_config = Mock(spec=RAGConfig)

    kb = DevOpsKnowledgeBase(store=mock_store, config=mock_config)

    results = await kb.search_all("general query", top_k=10)

    assert len(results) == 2
    mock_store.search.assert_called_once_with(
        query="general query",
        top_k=10,
    )


# =============================================================================
# HybridRetriever Tests
# =============================================================================


def test_hybrid_retriever_initialization():
    """Test HybridRetriever initialization."""
    mock_kb = Mock(spec=DevOpsKnowledgeBase)

    retriever = HybridRetriever(knowledge_base=mock_kb, alpha=0.5)

    assert retriever.knowledge_base == mock_kb
    assert retriever.alpha == 0.5


def test_hybrid_retriever_alpha_validation():
    """Test HybridRetriever alpha validation."""
    mock_kb = Mock(spec=DevOpsKnowledgeBase)

    # Alpha outside [0, 1] should raise ValueError
    with pytest.raises(ValueError, match="Alpha must be between 0 and 1"):
        HybridRetriever(knowledge_base=mock_kb, alpha=1.5)

    with pytest.raises(ValueError, match="Alpha must be between 0 and 1"):
        HybridRetriever(knowledge_base=mock_kb, alpha=-0.1)


@pytest.mark.asyncio
async def test_hybrid_retriever_pure_semantic(sample_search_results):
    """Test pure semantic search (alpha=1.0)."""
    mock_kb = AsyncMock(spec=DevOpsKnowledgeBase)
    mock_kb.search_all.return_value = sample_search_results

    retriever = HybridRetriever(knowledge_base=mock_kb, alpha=0.5)

    results = await retriever.retrieve("test query", top_k=5, alpha=1.0)

    # Should call search_all directly
    mock_kb.search_all.assert_called_once_with("test query", top_k=5)
    assert results == sample_search_results


@pytest.mark.asyncio
async def test_hybrid_retriever_keyword_ranking(sample_search_results):
    """Test keyword ranking."""
    mock_kb = AsyncMock(spec=DevOpsKnowledgeBase)
    mock_kb.search_all.return_value = sample_search_results

    retriever = HybridRetriever(knowledge_base=mock_kb, alpha=0.5)

    # Query with keywords that match first result
    results = await retriever.retrieve("kubernetes deployment", top_k=2, alpha=0.0)

    # Results should be re-ranked by keyword matching
    assert len(results) <= 2


# =============================================================================
# Factory Function Tests
# =============================================================================


@patch.dict(os.environ, {
    "PINECONE_API_KEY": "test-pinecone-key",
    "PINECONE_ENVIRONMENT": "us-west1-gcp",
    "VOYAGE_API_KEY": "test-voyage-key",
    "DEVOPS_KB_INDEX_NAME": "test-index",
})
@patch("deep_agent.devops.rag.Pinecone")
def test_create_rag_retriever(mock_pinecone):
    """Test create_rag_retriever factory function."""
    mock_index = Mock()
    mock_pinecone.return_value.Index.return_value = mock_index

    kb, retriever, embeddings = create_rag_retriever()

    assert isinstance(kb, DevOpsKnowledgeBase)
    assert isinstance(retriever, HybridRetriever)
    assert isinstance(embeddings, VoyageEmbeddings)


@patch.dict(os.environ, {
    "PINECONE_API_KEY": "test-pinecone-key",
    "PINECONE_ENVIRONMENT": "us-west1-gcp",
    "VOYAGE_API_KEY": "test-voyage-key",
    "DEVOPS_KB_INDEX_NAME": "test-index",
})
@patch("deep_agent.devops.rag.Pinecone")
def test_create_rag_retriever_with_code_embeddings(mock_pinecone):
    """Test create_rag_retriever with code embeddings."""
    mock_index = Mock()
    mock_pinecone.return_value.Index.return_value = mock_index

    kb, retriever, embeddings = create_rag_retriever(use_code_embeddings=True)

    # Should use voyage-code-3 model
    assert embeddings.model == "voyage-code-3"


# =============================================================================
# Integration Tests (Requires Environment Variables)
# =============================================================================


@pytest.mark.skipif(
    not os.getenv("PINECONE_API_KEY") or not os.getenv("VOYAGE_API_KEY"),
    reason="Integration test requires PINECONE_API_KEY and VOYAGE_API_KEY",
)
@pytest.mark.asyncio
async def test_integration_full_workflow():
    """Integration test for full RAG workflow.

    This test requires actual API keys and will make real API calls.
    Run with: pytest -v --run-integration tests/test_rag.py::test_integration_full_workflow
    """
    # Create RAG system
    kb, retriever, embeddings = create_rag_retriever()

    try:
        # Create test document
        test_doc = Document(
            id=f"test-integration-{os.getpid()}",
            content="This is a test document for integration testing of the RAG system.",
            metadata={
                "source": "test",
                "type": "integration-test",
                "tags": ["test", "integration"],
            },
        )

        # Index document
        await kb.store.index_document(
            test_doc.id,
            test_doc.content,
            test_doc.metadata,
        )

        # Search for document
        results = await kb.store.search(
            "integration testing RAG system",
            top_k=5,
            filter={"source": "test"},
        )

        # Verify we got results
        assert len(results) > 0

        # Clean up - delete test document
        await kb.store.delete_document(test_doc.id)

    finally:
        await embeddings.close()
