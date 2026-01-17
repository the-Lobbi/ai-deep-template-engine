# RAG Integration for DevOps Multi-Agent System

This module provides Retrieval-Augmented Generation (RAG) capabilities for the DevOps multi-agent system, enabling semantic search and knowledge retrieval across multiple documentation sources.

## Features

- **Voyage AI Embeddings**: Uses `voyage-3-large` (general) or `voyage-code-3` (code), officially recommended by Anthropic for Claude
- **Pinecone Vector Storage**: Scalable and efficient vector database for similarity search
- **Hybrid Search**: Combines semantic similarity with keyword matching
- **Source-Specific Search**: Query specific documentation sources (Kubernetes, Harness, Terraform, runbooks)
- **Tag-Based Filtering**: Filter documents by metadata tags
- **Async/Await Support**: Fully async implementation for high performance

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     DevOps Multi-Agent                      │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │                 DevOpsKnowledgeBase                   │  │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐     │  │
│  │  │ Kubernetes │  │  Harness   │  │ Terraform  │ ... │  │
│  │  │    Docs    │  │    Docs    │  │    Docs    │     │  │
│  │  └────────────┘  └────────────┘  └────────────┘     │  │
│  └──────────────────────────────────────────────────────┘  │
│                            │                                │
│                            ▼                                │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              HybridRetriever (α = 0.5)               │  │
│  │    Semantic Search (Vector) + Keyword Matching       │  │
│  └──────────────────────────────────────────────────────┘  │
│                            │                                │
│         ┌──────────────────┴──────────────────┐            │
│         ▼                                      ▼            │
│  ┌──────────────┐                    ┌──────────────┐      │
│  │   Voyage AI   │                    │   Pinecone   │      │
│  │  Embeddings   │◄──────────────────►│    Vector    │      │
│  │ (voyage-3-*) │                    │   Database   │      │
│  └──────────────┘                    └──────────────┘      │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Installation

### 1. Install Dependencies

```bash
# Using pip
pip install pinecone-client httpx

# Or add to pyproject.toml (already included)
```

### 2. Set Environment Variables

Create a `.env` file or export environment variables:

```bash
# Pinecone Configuration
export PINECONE_API_KEY="your-pinecone-api-key"
export PINECONE_ENVIRONMENT="us-west1-gcp"  # or your Pinecone environment
export DEVOPS_KB_INDEX_NAME="devops-knowledge-base"

# Voyage AI Configuration
export VOYAGE_API_KEY="your-voyage-api-key"
```

### 3. Get API Keys

#### Pinecone
1. Sign up at https://www.pinecone.io/
2. Create a new project
3. Copy your API key from the dashboard
4. Create an index with dimensions=1536 for voyage-3-large

#### Voyage AI
1. Sign up at https://www.voyageai.com/
2. Navigate to API keys in your dashboard
3. Generate a new API key

## Quick Start

### Basic Usage

```python
import asyncio
from deep_agent.devops.rag import create_rag_retriever

async def main():
    # Initialize RAG system
    kb, retriever, embeddings = create_rag_retriever()

    try:
        # Search all documentation
        results = await kb.search_all("How to scale Kubernetes deployment?", top_k=5)

        for result in results:
            print(f"Score: {result.score:.3f}")
            print(f"Content: {result.content[:200]}...")
            print("-" * 80)

    finally:
        # Clean up
        await embeddings.close()

asyncio.run(main())
```

### Indexing Documents

```python
from deep_agent.devops.rag import Document, create_rag_retriever

async def index_docs():
    kb, retriever, embeddings = create_rag_retriever()

    try:
        # Create documents
        docs = [
            Document(
                id="k8s-scale-001",
                content="To scale a deployment, use: kubectl scale deployment/name --replicas=5",
                metadata={
                    "source": "kubernetes",
                    "type": "tutorial",
                    "tags": ["scaling", "deployment"],
                }
            ),
            # ... more documents
        ]

        # Index documents
        await kb.store.index_documents(docs)
        print(f"Indexed {len(docs)} documents")

    finally:
        await embeddings.close()

asyncio.run(index_docs())
```

## Usage Examples

### 1. Source-Specific Search

```python
# Search Kubernetes documentation only
k8s_results = await kb.search_k8s_docs("pod troubleshooting", top_k=5)

# Search Harness documentation only
harness_results = await kb.search_harness_docs("pipeline trigger", top_k=5)

# Search Terraform documentation only
terraform_results = await kb.search_terraform_docs("state management", top_k=5)

# Search internal runbooks only
runbook_results = await kb.search_runbooks("incident response", top_k=5)
```

### 2. Hybrid Search

```python
# Balanced hybrid search (50% semantic, 50% keyword)
kb, retriever, embeddings = create_rag_retriever(alpha=0.5)

results = await retriever.retrieve(
    query="kubectl deployment scale replicas",
    top_k=5,
    alpha=0.5  # Can override per-query
)

# Pure semantic search
semantic_results = await retriever.retrieve(query, top_k=5, alpha=1.0)

# Pure keyword search
keyword_results = await retriever.retrieve(query, top_k=5, alpha=0.0)
```

### 3. Tag-Based Filtering

```python
# Search documents with specific tags
results = await kb.search_by_tags(
    query="troubleshooting guide",
    tags=["debugging", "troubleshooting"],
    top_k=5
)
```

### 4. Code-Specific Embeddings

```python
# Use voyage-code-3 model for code documentation
kb, retriever, embeddings = create_rag_retriever(use_code_embeddings=True)

results = await kb.search_all("Python async context manager example", top_k=5)
```

## API Reference

### `create_rag_retriever()`

Factory function to create a configured RAG retrieval system.

**Parameters:**
- `use_code_embeddings` (bool): Use voyage-code-3 for code-specific embeddings (default: False)
- `alpha` (float): Balance between semantic (1.0) and keyword (0.0) search (default: 0.5)
- `config` (RAGConfig): Optional configuration object (loads from env if None)

**Returns:**
- `tuple[DevOpsKnowledgeBase, HybridRetriever, VoyageEmbeddings]`

### `VoyageEmbeddings`

Voyage AI embedding client for text vectorization.

**Methods:**
- `async embed_text(text: str) -> List[float]`: Embed single text
- `async embed_batch(texts: List[str]) -> List[List[float]]`: Embed multiple texts
- `async close()`: Close HTTP client

### `PineconeKnowledgeStore`

Pinecone vector database client.

**Methods:**
- `async index_document(doc_id, text, metadata)`: Index single document
- `async index_documents(docs)`: Index multiple documents in batch
- `async search(query, top_k, filter)`: Search for similar documents
- `async delete_document(doc_id)`: Delete a document
- `async delete_all()`: Delete all documents in namespace (destructive!)

### `DevOpsKnowledgeBase`

Specialized knowledge base for DevOps operations.

**Methods:**
- `async search_all(query, top_k)`: Search across all sources
- `async search_k8s_docs(query, top_k)`: Search Kubernetes docs
- `async search_harness_docs(query, top_k)`: Search Harness docs
- `async search_terraform_docs(query, top_k)`: Search Terraform docs
- `async search_runbooks(query, top_k)`: Search internal runbooks
- `async search_by_tags(query, tags, top_k)`: Search with tag filtering

### `HybridRetriever`

Hybrid retrieval combining semantic and keyword search.

**Methods:**
- `async retrieve(query, top_k, alpha)`: Retrieve with hybrid scoring

### Data Classes

#### `Document`
```python
@dataclass
class Document:
    id: str                      # Unique identifier
    content: str                 # Document text
    metadata: Dict[str, Any]     # Additional metadata
```

#### `SearchResult`
```python
@dataclass
class SearchResult:
    id: str                      # Document identifier
    content: str                 # Document content
    score: float                 # Similarity score (0-1)
    metadata: Dict[str, Any]     # Document metadata
```

## Configuration

### RAGConfig

Configuration loaded from environment variables:

```python
class RAGConfig(BaseSettings):
    pinecone_api_key: str              # PINECONE_API_KEY
    pinecone_environment: str          # PINECONE_ENVIRONMENT
    voyage_api_key: str                # VOYAGE_API_KEY
    devops_kb_index_name: str          # DEVOPS_KB_INDEX_NAME
    voyage_model: str = "voyage-3-large"
    voyage_code_model: str = "voyage-code-3"
    default_top_k: int = 5
    max_retries: int = 3
    retry_delay: float = 1.0
```

## Best Practices

### 1. Document Structure

Organize documents with clear metadata:

```python
Document(
    id="unique-id-with-version",  # e.g., "k8s-scale-v1-001"
    content="Clear, concise content with proper formatting",
    metadata={
        "source": "kubernetes",           # Source category
        "type": "tutorial",               # Document type
        "tags": ["scaling", "deployment"], # Searchable tags
        "url": "https://...",             # Original URL
        "version": "v1.28",               # Version info
        "created_at": "2024-01-15",       # Timestamps
        "updated_at": "2024-01-15",
    }
)
```

### 2. Search Strategy

Choose the right search method for your use case:

- **General queries**: Use `search_all()` for broad searches
- **Domain-specific**: Use source-specific methods (`search_k8s_docs()`, etc.)
- **Exact terms**: Lower alpha (keyword-focused) for technical terms
- **Conceptual**: Higher alpha (semantic-focused) for conceptual queries
- **Balanced**: Default alpha=0.5 works well for most cases

### 3. Error Handling

Always use try/finally for cleanup:

```python
kb, retriever, embeddings = create_rag_retriever()
try:
    results = await kb.search_all(query)
    # Process results
finally:
    await embeddings.close()
```

### 4. Batch Operations

Index documents in batches for better performance:

```python
# Good: Batch indexing
await kb.store.index_documents(all_docs)

# Avoid: Individual indexing in loop
for doc in all_docs:
    await kb.store.index_document(doc.id, doc.content, doc.metadata)
```

### 5. Metadata Filtering

Use metadata filters for efficient searches:

```python
# Search only critical severity runbooks
results = await kb.store.search(
    query="incident response",
    top_k=5,
    filter={"source": "runbooks", "severity": "critical"}
)
```

## Integration with DevOps Workflow

### Using RAG in Agent Tools

```python
from deep_agent.devops.rag import create_rag_retriever
from langchain_core.tools import StructuredTool

# Initialize RAG system
kb, retriever, embeddings = create_rag_retriever()

async def enhanced_documentation_search(query: str, top_k: int = 5):
    """Enhanced documentation search using RAG."""
    results = await kb.search_all(query, top_k=top_k)
    return [
        {
            "content": r.content,
            "score": r.score,
            "source": r.metadata.get("source"),
            "url": r.metadata.get("url"),
        }
        for r in results
    ]

# Create LangChain tool
documentation_tool = StructuredTool.from_function(
    coroutine=enhanced_documentation_search,
    name="enhanced_documentation_search",
    description="Search DevOps documentation using semantic similarity"
)
```

### State Integration

Add RAG results to DevOpsAgentState:

```python
# In your workflow node
async def knowledge_retrieval_node(state: DevOpsAgentState):
    """Retrieve relevant documentation for current task."""
    kb, retriever, embeddings = create_rag_retriever()

    try:
        # Extract query from task context
        query = state["context"].get("error_message", "")

        # Retrieve relevant documents
        results = await kb.search_all(query, top_k=5)

        # Update state with retrieved docs
        return {
            "retrieved_docs": [
                {
                    "id": r.id,
                    "content": r.content,
                    "score": r.score,
                    "metadata": r.metadata,
                }
                for r in results
            ],
            "search_queries": state["search_queries"] + [query],
        }
    finally:
        await embeddings.close()
```

## Performance Considerations

### Embedding Costs

Voyage AI pricing (as of 2024):
- voyage-3-large: ~$0.12 per 1M tokens
- voyage-code-3: ~$0.12 per 1M tokens

Optimize costs by:
- Batching embed requests
- Caching embeddings for frequently searched queries
- Using appropriate context window sizes

### Pinecone Costs

Pinecone pricing varies by tier:
- Starter: Free up to 100K vectors
- Standard: Pay-as-you-go
- Enterprise: Custom pricing

Optimize costs by:
- Using appropriate index dimensions
- Implementing TTL for temporary documents
- Using namespaces for multi-tenancy

### Rate Limits

- **Voyage AI**: Typically 100 requests/minute
- **Pinecone**: Varies by tier (check dashboard)

Handle rate limits with exponential backoff (built-in to `VoyageEmbeddings`).

## Troubleshooting

### Common Issues

#### 1. Empty Search Results

**Problem**: Queries return no results

**Solutions**:
- Verify documents are indexed: Check Pinecone dashboard
- Check namespace matches: Ensure using correct namespace
- Lower top_k: Try fewer results
- Verify index dimensions: Must match embedding model

#### 2. Rate Limiting

**Problem**: HTTP 429 errors from Voyage AI

**Solutions**:
- Increase `retry_delay` in config
- Reduce batch sizes
- Implement request throttling
- Upgrade API tier

#### 3. Poor Search Quality

**Problem**: Irrelevant results returned

**Solutions**:
- Use source-specific search: `search_k8s_docs()` instead of `search_all()`
- Add metadata filters: Filter by tags, source, type
- Adjust alpha: Lower for technical terms, higher for concepts
- Improve document quality: Add better metadata, clearer content
- Try hybrid search: Balance semantic and keyword

#### 4. Connection Errors

**Problem**: Cannot connect to Pinecone

**Solutions**:
- Verify API key is correct
- Check environment matches your Pinecone project
- Ensure index exists and is ready
- Check network connectivity

## Testing

Run the example suite:

```bash
# Set environment variables first
export PINECONE_API_KEY="..."
export VOYAGE_API_KEY="..."

# Run examples
python -m deep_agent.devops.rag_examples
```

Run individual examples:

```python
from deep_agent.devops.rag_examples import (
    example_basic_search,
    example_index_documents,
    example_hybrid_search,
)

asyncio.run(example_basic_search())
```

## Additional Resources

- [Voyage AI Documentation](https://docs.voyageai.com/)
- [Pinecone Documentation](https://docs.pinecone.io/)
- [Anthropic Claude + RAG Guide](https://docs.anthropic.com/claude/docs/retrieval-augmented-generation)
- [LangChain RAG Tutorial](https://python.langchain.com/docs/use_cases/question_answering/)

## License

Apache-2.0

## Contributing

See the main project CONTRIBUTING.md for guidelines.
