# Secrets Management Quick Reference

> **Full Documentation**: `C:\Users\MarkusAhling\obsidian\Repositories\the-Lobbi\ai-deep-template-engine-secrets-management.md`

## Quick Start

### Basic Usage

```python
from deep_agent.devops.secrets import configure_agent_secrets
import os

# Load all secrets (auto-detects provider)
secrets = await configure_agent_secrets()
os.environ.update(secrets)

# Now all libraries can access secrets
from langchain_anthropic import ChatAnthropic
llm = ChatAnthropic()  # Uses ANTHROPIC_API_KEY from environment
```

### Get Specific Secrets

```python
from deep_agent.devops.secrets import create_secrets_manager

manager = await create_secrets_manager()

# Individual secrets
anthropic_key = await manager.get_anthropic_key()
voyage_key = await manager.get_voyage_key()
pinecone_key = await manager.get_pinecone_key()
harness_token = await manager.get_harness_token()

# Custom secret
custom = await manager.get_secret("MY_CUSTOM_SECRET")
```

## Providers

### Auto-Detection Priority

1. **Azure Key Vault** (if `AZURE_KEYVAULT_URL` set)
2. **Harness Secrets Manager** (if `HARNESS_ACCOUNT_ID` set)
3. **Environment Variables** (fallback)

### Azure Key Vault

```bash
# Configuration
export AZURE_KEYVAULT_URL=https://my-vault.vault.azure.net/

# Optional (for service principal)
export AZURE_CLIENT_ID=...
export AZURE_TENANT_ID=...
export AZURE_CLIENT_SECRET=...
```

```python
# Usage
manager = await create_secrets_manager(provider="azure")
secret = await manager.get_secret("MY_SECRET")
```

**Install Azure dependencies**:
```bash
pip install deep-agent-harness[azure]
```

### Harness Secrets Manager

```bash
# Configuration
export HARNESS_ACCOUNT_ID=abc123
export HARNESS_API_KEY=pat.xyz...
export HARNESS_ORG_ID=default
export HARNESS_PROJECT_ID=my-project
```

```python
# Usage
manager = await create_secrets_manager(provider="harness")

# Project-scoped secret
secret = await manager.get_secret("api-key", scope="project")

# Org-scoped secret
org_secret = await manager.get_secret("shared-token", scope="org")
```

### Environment Variables

```bash
# .env file or environment
export ANTHROPIC_API_KEY=sk-ant-...
export VOYAGE_API_KEY=pa-...
export PINECONE_API_KEY=...
export HARNESS_API_TOKEN=...
```

```python
# Usage (auto-detected if no other provider available)
manager = await create_secrets_manager()

# Or explicit
manager = await create_secrets_manager(provider="env")
```

## Common Patterns

### Pattern 1: Agent Initialization

```python
async def initialize_agent():
    """Initialize agent with secrets."""
    secrets = await configure_agent_secrets()
    os.environ.update(secrets)

    # Now libraries can access secrets
    from langchain_anthropic import ChatAnthropic
    llm = ChatAnthropic(model="claude-sonnet-4-5")
    return llm
```

### Pattern 2: Harness Pipeline

```yaml
steps:
  - step:
      type: Script
      name: Run Agent
      spec:
        shell: Python
        source:
          type: Inline
          spec:
            script: |
              from deep_agent.devops.secrets import configure_agent_secrets
              secrets = await configure_agent_secrets()
              os.environ.update(secrets)

              # Run agent
              from deep_agent.devops.workflow import HarnessWorkflow
              workflow = HarnessWorkflow()
              await workflow.run()

        envVariables:
          HARNESS_ACCOUNT_ID: <+account.identifier>
          HARNESS_API_KEY: <+secrets.getValue("harness_api_key")>
```

### Pattern 3: Azure Function

```python
import azure.functions as func
from deep_agent.devops.secrets import create_secrets_manager

async def main(req: func.HttpRequest) -> func.HttpResponse:
    # Uses Managed Identity automatically
    manager = await create_secrets_manager(
        provider="azure",
        use_managed_identity=True
    )

    secrets = await manager.get_all_secrets()
    # ... agent logic ...

    return func.HttpResponse("Success")
```

### Pattern 4: Local Development

```bash
# .env file
ANTHROPIC_API_KEY=sk-ant-...
VOYAGE_API_KEY=pa-...
PINECONE_API_KEY=...
HARNESS_API_TOKEN=...
```

```python
from dotenv import load_dotenv
from deep_agent.devops.secrets import configure_agent_secrets

load_dotenv()
secrets = await configure_agent_secrets()
# Ready!
```

## Features

### Caching

Secrets are cached for 5 minutes by default:

```python
# Custom TTL
manager = SecretsManager(provider="azure", cache_ttl=600)  # 10 minutes

# Clear cache (e.g., after rotation)
await manager.provider.cache.clear()
```

### Retry Logic

Automatic retry on network errors:
- 3 attempts
- Exponential backoff (2s, 4s, 8s)
- Only retries transient errors

### Error Handling

```python
from deep_agent.devops.secrets import (
    SecretNotFoundError,
    ProviderError,
    ProviderNotAvailableError,
)

try:
    secret = await manager.get_secret("MY_SECRET")
except SecretNotFoundError as e:
    print(f"Secret {e.secret_name} not found in {e.provider}")
except ProviderError as e:
    print(f"Provider error: {e}")
except ProviderNotAvailableError as e:
    print(f"Provider not available: {e}")
```

## Testing

```bash
# Run all tests
pytest src/deep_agent/devops/test_secrets.py -v

# With coverage
pytest src/deep_agent/devops/test_secrets.py --cov=src/deep_agent/devops/secrets

# Specific test
pytest src/deep_agent/devops/test_secrets.py::test_environment_provider_get_secret
```

## Troubleshooting

### Azure Key Vault

**Problem**: `ProviderError: vault_url required`
```bash
export AZURE_KEYVAULT_URL=https://my-vault.vault.azure.net/
```

**Problem**: Authentication fails
```bash
az login
az account show
```

**Problem**: Azure SDK not installed
```bash
pip install deep-agent-harness[azure]
```

### Harness

**Problem**: `ProviderError: account_id required`
```bash
export HARNESS_ACCOUNT_ID=abc123
export HARNESS_API_KEY=pat.xyz...
```

**Problem**: Project secret not found
```python
# Need project_id for project-scoped secrets
manager = await create_secrets_manager(
    provider="harness",
    project_id="my-project"
)
```

### Cache

**Problem**: Stale secret after rotation
```python
# Clear cache
await manager.provider.cache.clear()

# Or use shorter TTL
manager = SecretsManager(cache_ttl=60)  # 1 minute
```

## Security Best Practices

1. **Never hardcode secrets**
   ```python
   # BAD
   api_key = "sk-ant-123456"

   # GOOD
   api_key = await manager.get_anthropic_key()
   ```

2. **Use least privilege**
   - Azure: Grant only `Key Vault Secrets User` role
   - Harness: Use project-scoped secrets when possible

3. **Rotate secrets regularly**
   ```python
   # After rotation, clear cache
   await manager.provider.cache.clear()
   ```

4. **Separate environments**
   ```python
   if env == "production":
       vault = "https://prod-vault.vault.azure.net/"
   else:
       vault = "https://dev-vault.vault.azure.net/"
   ```

## API Reference

### Factory Functions

```python
# Auto-detect provider
manager = await create_secrets_manager()

# Explicit provider
manager = await create_secrets_manager(
    provider="azure",
    vault_url="https://...",
    cache_ttl=300
)

# Load all agent secrets
secrets = await configure_agent_secrets()
```

### SecretsManager Methods

```python
# Get single secret
value = await manager.get_secret("NAME")

# Get multiple secrets
secrets = await manager.get_secrets(["KEY1", "KEY2"])

# List available secrets
names = await manager.list_secrets()

# Convenience methods
anthropic = await manager.get_anthropic_key()
voyage = await manager.get_voyage_key()
pinecone = await manager.get_pinecone_key()
harness = await manager.get_harness_token()

# Get all required secrets
all_secrets = await manager.get_all_secrets()

# Close connections
await manager.close()
```

## Required Secrets

The following secrets are required for the DevOps agent:

| Secret Name | Purpose | Provider |
|-------------|---------|----------|
| `ANTHROPIC_API_KEY` | Claude API access | All |
| `VOYAGE_API_KEY` | Voyage AI embeddings | All |
| `PINECONE_API_KEY` | Vector database | All |
| `HARNESS_API_TOKEN` | Harness API access | All |

## Environment Setup

### Local Development

```bash
# Create .env file
cat > .env << EOF
ANTHROPIC_API_KEY=sk-ant-...
VOYAGE_API_KEY=pa-...
PINECONE_API_KEY=...
HARNESS_API_TOKEN=...
EOF

# Load and use
python -c "
from dotenv import load_dotenv
from deep_agent.devops.secrets import configure_agent_secrets
import asyncio

async def main():
    load_dotenv()
    secrets = await configure_agent_secrets()
    print(f'Loaded {len(secrets)} secrets')

asyncio.run(main())
"
```

### Azure Production

```bash
# Set Key Vault URL
export AZURE_KEYVAULT_URL=https://my-vault.vault.azure.net/

# Secrets auto-loaded from Key Vault
python your_agent.py
```

### Harness Pipeline

```bash
# Environment automatically configured by Harness
export HARNESS_ACCOUNT_ID=<+account.identifier>
export HARNESS_API_KEY=<+secrets.getValue("api_key")>

# Secrets auto-loaded from Harness
python your_agent.py
```

## Links

- **Full Documentation**: `C:\Users\MarkusAhling\obsidian\Repositories\the-Lobbi\ai-deep-template-engine-secrets-management.md`
- **ADR**: `C:\Users\MarkusAhling\obsidian\Repositories\the-Lobbi\Decisions\0003-multi-provider-secrets-management.md`
- **Implementation**: `src/deep_agent/devops/secrets.py`
- **Tests**: `src/deep_agent/devops/test_secrets.py`

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review full documentation in Obsidian vault
3. Check test examples in `test_secrets.py`
4. Contact DevOps team
