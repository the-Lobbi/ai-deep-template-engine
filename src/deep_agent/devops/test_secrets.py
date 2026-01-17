"""
Tests for the secrets management module.

Run with: pytest test_secrets.py -v
"""

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import httpx

from deep_agent.devops.secrets import (
    AzureKeyVaultProvider,
    EnvironmentProvider,
    HarnessSecretsProvider,
    ProviderError,
    ProviderNotAvailableError,
    SecretCache,
    SecretNotFoundError,
    SecretsManager,
    configure_agent_secrets,
    create_secrets_manager,
    AZURE_AVAILABLE,
)


# ============================================================================
# Secret Cache Tests
# ============================================================================


@pytest.mark.asyncio
async def test_secret_cache_basic():
    """Test basic cache operations."""
    cache = SecretCache(ttl_seconds=60)

    # Cache miss
    assert await cache.get("key1") is None

    # Cache set and hit
    await cache.set("key1", "value1")
    assert await cache.get("key1") == "value1"

    # Multiple keys
    await cache.set("key2", "value2")
    assert await cache.get("key1") == "value1"
    assert await cache.get("key2") == "value2"


@pytest.mark.asyncio
async def test_secret_cache_expiration():
    """Test cache TTL expiration."""
    cache = SecretCache(ttl_seconds=0.1)  # 100ms TTL

    await cache.set("key1", "value1")
    assert await cache.get("key1") == "value1"

    # Wait for expiration
    await asyncio.sleep(0.2)
    assert await cache.get("key1") is None


@pytest.mark.asyncio
async def test_secret_cache_clear():
    """Test cache clearing."""
    cache = SecretCache(ttl_seconds=60)

    await cache.set("key1", "value1")
    await cache.set("key2", "value2")

    await cache.clear()

    assert await cache.get("key1") is None
    assert await cache.get("key2") is None


# ============================================================================
# Environment Provider Tests
# ============================================================================


@pytest.mark.asyncio
async def test_environment_provider_get_secret():
    """Test retrieving secrets from environment."""
    os.environ["TEST_SECRET"] = "test_value"

    provider = EnvironmentProvider()
    value = await provider.get_secret("TEST_SECRET")

    assert value == "test_value"

    # Cleanup
    del os.environ["TEST_SECRET"]


@pytest.mark.asyncio
async def test_environment_provider_with_prefix():
    """Test environment provider with prefix."""
    os.environ["SECRET_API_KEY"] = "my_key"

    provider = EnvironmentProvider(prefix="SECRET_")
    value = await provider.get_secret("API_KEY")

    assert value == "my_key"

    # Cleanup
    del os.environ["SECRET_API_KEY"]


@pytest.mark.asyncio
async def test_environment_provider_secret_not_found():
    """Test error when secret not found."""
    provider = EnvironmentProvider()

    with pytest.raises(SecretNotFoundError) as exc_info:
        await provider.get_secret("NONEXISTENT_SECRET")

    assert exc_info.value.secret_name == "NONEXISTENT_SECRET"
    assert exc_info.value.provider == "environment"


@pytest.mark.asyncio
async def test_environment_provider_get_secrets():
    """Test retrieving multiple secrets."""
    os.environ["SECRET1"] = "value1"
    os.environ["SECRET2"] = "value2"

    provider = EnvironmentProvider()
    secrets = await provider.get_secrets(["SECRET1", "SECRET2"])

    assert secrets == {
        "SECRET1": "value1",
        "SECRET2": "value2",
    }

    # Cleanup
    del os.environ["SECRET1"]
    del os.environ["SECRET2"]


@pytest.mark.asyncio
async def test_environment_provider_list_secrets():
    """Test listing secrets with prefix."""
    os.environ["APP_KEY1"] = "value1"
    os.environ["APP_KEY2"] = "value2"
    os.environ["OTHER_KEY"] = "value3"

    provider = EnvironmentProvider(prefix="APP_")
    secrets = await provider.list_secrets()

    assert "KEY1" in secrets
    assert "KEY2" in secrets
    assert "OTHER_KEY" not in secrets

    # Cleanup
    del os.environ["APP_KEY1"]
    del os.environ["APP_KEY2"]
    del os.environ["OTHER_KEY"]


# ============================================================================
# Azure Key Vault Provider Tests (Mocked)
# ============================================================================


@pytest.mark.asyncio
async def test_azure_provider_not_available():
    """Test error when Azure SDK not installed."""
    with patch("deep_agent.devops.secrets.AZURE_AVAILABLE", False):
        with pytest.raises(ProviderNotAvailableError) as exc_info:
            AzureKeyVaultProvider(vault_url="https://test.vault.azure.net/")

        assert exc_info.value.provider == "azure"


@pytest.mark.asyncio
async def test_azure_provider_missing_vault_url():
    """Test error when vault URL not provided."""
    with patch("deep_agent.devops.secrets.AZURE_AVAILABLE", True):
        with pytest.raises(ProviderError) as exc_info:
            AzureKeyVaultProvider()

        assert exc_info.value.provider == "azure"
        assert "vault_url" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_azure_provider_get_secret_mocked():
    """Test Azure provider with mocked client."""
    if not AZURE_AVAILABLE:
        pytest.skip("Azure SDK not available")

    mock_secret = MagicMock()
    mock_secret.value = "test_secret_value"

    mock_client = MagicMock()
    mock_client.get_secret.return_value = mock_secret

    with patch("deep_agent.devops.secrets.SecretClient", return_value=mock_client):
        provider = AzureKeyVaultProvider(vault_url="https://test.vault.azure.net/")
        value = await provider.get_secret("test-secret")

        assert value == "test_secret_value"
        mock_client.get_secret.assert_called_once_with("test-secret")


# ============================================================================
# Harness Secrets Provider Tests (Mocked)
# ============================================================================


@pytest.mark.asyncio
async def test_harness_provider_missing_config():
    """Test error when Harness config missing."""
    with pytest.raises(ProviderError) as exc_info:
        HarnessSecretsProvider()

    assert exc_info.value.provider == "harness"


@pytest.mark.asyncio
async def test_harness_provider_get_secret_mocked():
    """Test Harness provider with mocked HTTP client."""
    mock_response = {
        "data": {
            "secret": {
                "spec": {
                    "value": "test_secret_value"
                }
            }
        }
    }

    async def mock_get(*args, **kwargs):
        response = MagicMock()
        response.json.return_value = mock_response
        response.status_code = 200
        response.raise_for_status = MagicMock()
        return response

    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client = MagicMock()
        mock_client.get = mock_get
        mock_client_class.return_value = mock_client

        provider = HarnessSecretsProvider(
            account_id="test_account",
            api_key="test_key",
            project_id="test_project"
        )

        value = await provider.get_secret("test-secret")
        assert value == "test_secret_value"


@pytest.mark.asyncio
async def test_harness_provider_secret_not_found():
    """Test Harness provider when secret not found."""
    async def mock_get(*args, **kwargs):
        response = MagicMock()
        response.status_code = 404
        response.raise_for_status = MagicMock(
            side_effect=httpx.HTTPStatusError(
                "Not found",
                request=MagicMock(),
                response=response
            )
        )
        return response

    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client = MagicMock()
        mock_client.get = mock_get
        mock_client_class.return_value = mock_client

        provider = HarnessSecretsProvider(
            account_id="test_account",
            api_key="test_key",
            project_id="test_project"
        )

        with pytest.raises(SecretNotFoundError) as exc_info:
            await provider.get_secret("nonexistent-secret")

        assert exc_info.value.secret_name == "nonexistent-secret"
        assert exc_info.value.provider == "harness"


# ============================================================================
# Secrets Manager Tests
# ============================================================================


@pytest.mark.asyncio
async def test_secrets_manager_auto_detect_env():
    """Test auto-detection falls back to environment."""
    # Ensure no Azure/Harness env vars
    env_backup = {}
    for key in ["AZURE_KEYVAULT_URL", "HARNESS_ACCOUNT_ID"]:
        if key in os.environ:
            env_backup[key] = os.environ[key]
            del os.environ[key]

    try:
        manager = SecretsManager(provider="auto")
        assert manager.provider_type == "env"
        assert isinstance(manager.provider, EnvironmentProvider)
    finally:
        # Restore env
        os.environ.update(env_backup)


@pytest.mark.asyncio
async def test_secrets_manager_explicit_provider():
    """Test explicitly specifying provider."""
    manager = SecretsManager(provider="env")
    assert manager.provider_type == "env"
    assert isinstance(manager.provider, EnvironmentProvider)


@pytest.mark.asyncio
async def test_secrets_manager_convenience_methods():
    """Test convenience methods for common secrets."""
    os.environ["ANTHROPIC_API_KEY"] = "anthropic_key"
    os.environ["VOYAGE_API_KEY"] = "voyage_key"
    os.environ["PINECONE_API_KEY"] = "pinecone_key"
    os.environ["HARNESS_API_TOKEN"] = "harness_token"

    try:
        manager = SecretsManager(provider="env")

        assert await manager.get_anthropic_key() == "anthropic_key"
        assert await manager.get_voyage_key() == "voyage_key"
        assert await manager.get_pinecone_key() == "pinecone_key"
        assert await manager.get_harness_token() == "harness_token"

    finally:
        # Cleanup
        for key in ["ANTHROPIC_API_KEY", "VOYAGE_API_KEY", "PINECONE_API_KEY", "HARNESS_API_TOKEN"]:
            if key in os.environ:
                del os.environ[key]


@pytest.mark.asyncio
async def test_secrets_manager_get_all_secrets():
    """Test retrieving all required secrets."""
    os.environ["ANTHROPIC_API_KEY"] = "anthropic_key"
    os.environ["VOYAGE_API_KEY"] = "voyage_key"
    os.environ["PINECONE_API_KEY"] = "pinecone_key"
    os.environ["HARNESS_API_TOKEN"] = "harness_token"

    try:
        manager = SecretsManager(provider="env")
        secrets = await manager.get_all_secrets()

        assert len(secrets) == 4
        assert secrets["ANTHROPIC_API_KEY"] == "anthropic_key"
        assert secrets["VOYAGE_API_KEY"] == "voyage_key"
        assert secrets["PINECONE_API_KEY"] == "pinecone_key"
        assert secrets["HARNESS_API_TOKEN"] == "harness_token"

    finally:
        # Cleanup
        for key in ["ANTHROPIC_API_KEY", "VOYAGE_API_KEY", "PINECONE_API_KEY", "HARNESS_API_TOKEN"]:
            if key in os.environ:
                del os.environ[key]


# ============================================================================
# Factory Function Tests
# ============================================================================


@pytest.mark.asyncio
async def test_create_secrets_manager():
    """Test factory function."""
    manager = await create_secrets_manager(provider="env", validate=False)
    assert isinstance(manager, SecretsManager)
    assert manager.provider_type == "env"


@pytest.mark.asyncio
async def test_configure_agent_secrets():
    """Test agent secrets configuration helper."""
    os.environ["ANTHROPIC_API_KEY"] = "anthropic_key"
    os.environ["VOYAGE_API_KEY"] = "voyage_key"
    os.environ["PINECONE_API_KEY"] = "pinecone_key"
    os.environ["HARNESS_API_TOKEN"] = "harness_token"

    try:
        secrets = await configure_agent_secrets()

        assert len(secrets) == 4
        assert secrets["ANTHROPIC_API_KEY"] == "anthropic_key"

    finally:
        # Cleanup
        for key in ["ANTHROPIC_API_KEY", "VOYAGE_API_KEY", "PINECONE_API_KEY", "HARNESS_API_TOKEN"]:
            if key in os.environ:
                del os.environ[key]


# ============================================================================
# Integration Tests
# ============================================================================


@pytest.mark.asyncio
async def test_caching_reduces_calls():
    """Test that caching reduces provider calls."""
    os.environ["TEST_SECRET"] = "test_value"

    try:
        provider = EnvironmentProvider(cache_ttl=60)

        # First call
        value1 = await provider.get_secret("TEST_SECRET")
        assert value1 == "test_value"

        # Change environment value
        os.environ["TEST_SECRET"] = "new_value"

        # Second call should return cached value
        value2 = await provider.get_secret("TEST_SECRET")
        assert value2 == "test_value"  # Still cached

        # Clear cache
        await provider.cache.clear()

        # Third call should get new value
        value3 = await provider.get_secret("TEST_SECRET")
        assert value3 == "new_value"

    finally:
        del os.environ["TEST_SECRET"]


@pytest.mark.asyncio
async def test_end_to_end_environment_workflow():
    """Test complete workflow with environment provider."""
    # Setup environment
    os.environ["ANTHROPIC_API_KEY"] = "test_anthropic"
    os.environ["VOYAGE_API_KEY"] = "test_voyage"
    os.environ["PINECONE_API_KEY"] = "test_pinecone"
    os.environ["HARNESS_API_TOKEN"] = "test_harness"

    try:
        # Create manager
        manager = await create_secrets_manager(provider="env", validate=False)

        # Get individual secrets
        anthropic = await manager.get_anthropic_key()
        assert anthropic == "test_anthropic"

        # Get all secrets
        all_secrets = await manager.get_all_secrets()
        assert len(all_secrets) == 4

        # List secrets
        secret_names = await manager.list_secrets()
        assert "ANTHROPIC_API_KEY" in secret_names

        # Close manager
        await manager.close()

    finally:
        # Cleanup
        for key in ["ANTHROPIC_API_KEY", "VOYAGE_API_KEY", "PINECONE_API_KEY", "HARNESS_API_TOKEN"]:
            if key in os.environ:
                del os.environ[key]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
