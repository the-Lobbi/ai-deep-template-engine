# Server Setup Notes

## Known Issue: Import Error in agents.py

The server.py module depends on agents.py, which has an incorrect import that needs to be fixed:

### Issue

File: `agents.py` (line 31)
```python
from langgraph.graph.graph import CompiledGraph  # INCORRECT
```

### Fix Required

Replace the incorrect import with:
```python
from langgraph.graph.state import CompiledStateGraph as CompiledGraph
```

Or alternatively, use a type alias:
```python
from typing import Any

# Type hint for compiled graph (compatible with LangGraph StateGraph)
CompiledGraph = Any  # Or: from langgraph.graph.state import CompiledStateGraph
```

### Why This Happens

LangGraph's `CompiledGraph` type is not exported from `langgraph.graph.graph`. The correct type for a compiled StateGraph is `CompiledStateGraph` from `langgraph.graph.state`.

### Steps to Fix

1. Open `agents.py`
2. Find line 31: `from langgraph.graph.graph import CompiledGraph`
3. Replace with: `from langgraph.graph.state import CompiledStateGraph as CompiledGraph`
4. Save the file

### Verification

After making the fix, verify the import works:

```bash
cd C:\Users\MarkusAhling\pro\langchang\ai-deep-template-engine
python -c "from src.deep_agent.devops.server import app; print('Server imported successfully')"
```

## Installation

### Install Dependencies

```bash
pip install -r server_requirements.txt
```

### Environment Variables

Set the required environment variables:

```bash
# Required
export ANTHROPIC_API_KEY="your-api-key"

# Optional (for tracing)
export LANGSMITH_API_KEY="your-langsmith-key"

# Optional (for RAG)
export PINECONE_API_KEY="your-pinecone-key"
export PINECONE_ENVIRONMENT="your-environment"
export PINECONE_INDEX_NAME="your-index"
```

### Running the Server

After fixing the import issue and setting environment variables:

```bash
# Development mode with auto-reload
python -m deep_agent.devops.server --reload

# Production mode
python -m deep_agent.devops.server --host 0.0.0.0 --port 8000
```

### Testing

```bash
# Run tests
pytest src/deep_agent/devops/test_server.py -v

# Test health check
curl http://localhost:8000/health

# Test with example client
python src/deep_agent/devops/example_client.py
```

## Quick Fix Script

If you want to automate the fix, you can use this Python script:

```python
import re

# Read the file
file_path = r"C:\Users\MarkusAhling\pro\langchang\ai-deep-template-engine\src\deep_agent\devops\agents.py"

with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Replace the incorrect import
old_import = "from langgraph.graph.graph import CompiledGraph"
new_import = "from langgraph.graph.state import CompiledStateGraph as CompiledGraph"

content = content.replace(old_import, new_import)

# Write back
with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

print("Fixed import in agents.py")
```

Save this as `fix_import.py` and run:
```bash
python fix_import.py
```
