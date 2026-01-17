"""
Secure Secrets Management Module for DevOps Agents.

This module provides a unified interface for retrieving secrets from:
- Azure Key Vault (production)
- Harness Secrets Manager (CI/CD pipelines)
- Environment variables (development/fallback)

Features:
- Protocol-based design for easy provider switching
- Automatic provider detection
- Retry logic with exponential backoff
- Secret caching with TTL
- Comprehensive error handling
- Zero secret value logging

Usage:
    # Auto-detect provider
    manager = await create_secrets_manager()

    # Get specific secrets
    anthropic_key = await manager.get_anthropic_key()

    # Get all required secrets
    secrets = await manager.get_all_secrets()
    os.environ.update(secrets)
"""

import asyncio
import os
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Protocol, runtime_checkable
from functools import wraps
import logging

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

# Optional Azure dependencies
try:
    from azure.identity import DefaultAzureCredential, ManagedIdentityCredential
    from azure.keyvault.secrets import SecretClient
    from azure.core.exceptions import ResourceNotFoundError as AzureResourceNotFoundError

    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False
    DefaultAzureCredential = None
    ManagedIdentityCredential = None
    SecretClient = None
    AzureResourceNotFoundError = Exception


# Configure logging - never log secret values
logger = logging.getLogger(__name__)


# ============================================================================
# Custom Exceptions
# ============================================================================


class SecretError(Exception):
    """Base exception for secrets management errors."""
    pass


class SecretNotFoundError(SecretError):
    """Raised when a secret cannot be found."""

    def __init__(self, secret_name: str, provider: str):
        self.secret_name = secret_name
        self.provider = provider
        super().__init__(
            f"Secret '{secret_name}' not found in provider '{provider}'"
        )


class ProviderError(SecretError):
    """Raised when a provider encounters an error."""

    def __init__(self, provider: str, message: str):
        self.provider = provider
        super().__init__(f"Provider '{provider}' error: {message}")


class ProviderNotAvailableError(SecretError):
    """Raised when a provider is not available (missing dependencies)."""

    def __init__(self, provider: str, reason: str):
        self.provider = provider
        super().__init__(
            f"Provider '{provider}' not available: {reason}"
        )


# ============================================================================
# Secret Provider Protocol
# ============================================================================


@runtime_checkable
class SecretProvider(Protocol):
    """Protocol defining the interface for secret providers.

    All secret providers must implement these methods.
    """

    async def get_secret(self, name: str) -> str:
        """Retrieve a single secret by name.

        Args:
            name: The secret name/identifier

        Returns:
            The secret value as a string

        Raises:
            SecretNotFoundError: If the secret doesn't exist
            ProviderError: If retrieval fails
        """
        ...

    async def get_secrets(self, names: list[str]) -> Dict[str, str]:
        """Retrieve multiple secrets by name.

        Args:
            names: List of secret names to retrieve

        Returns:
            Dictionary mapping secret names to values

        Raises:
            SecretNotFoundError: If any secret doesn't exist
            ProviderError: If retrieval fails
        """
        ...

    async def list_secrets(self) -> list[str]:
        """List all available secret names.

        Returns:
            List of secret names (not values)

        Raises:
            ProviderError: If listing fails
        """
        ...


# ============================================================================
# Secret Caching
# ============================================================================


class SecretCache:
    """Thread-safe cache for secrets with TTL support."""

    def __init__(self, ttl_seconds: int = 300):
        """Initialize cache.

        Args:
            ttl_seconds: Time-to-live for cached secrets (default: 5 minutes)
        """
        self._cache: Dict[str, tuple[str, datetime]] = {}
        self._lock = asyncio.Lock()
        self.ttl = timedelta(seconds=ttl_seconds)

    async def get(self, key: str) -> Optional[str]:
        """Get a cached secret if not expired."""
        async with self._lock:
            if key in self._cache:
                value, expires_at = self._cache[key]
                if datetime.now() < expires_at:
                    logger.debug(f"Cache hit for secret: {key}")
                    return value
                else:
                    logger.debug(f"Cache expired for secret: {key}")
                    del self._cache[key]
        return None

    async def set(self, key: str, value: str) -> None:
        """Cache a secret with TTL."""
        async with self._lock:
            expires_at = datetime.now() + self.ttl
            self._cache[key] = (value, expires_at)
            logger.debug(f"Cached secret: {key} (expires at {expires_at})")

    async def clear(self) -> None:
        """Clear all cached secrets."""
        async with self._lock:
            count = len(self._cache)
            self._cache.clear()
            logger.info(f"Cleared {count} cached secrets")


# ============================================================================
# Azure Key Vault Provider
# ============================================================================


class AzureKeyVaultProvider:
    """Secret provider for Azure Key Vault.

    Supports both DefaultAzureCredential (local dev, MI, service principal)
    and ManagedIdentityCredential (Azure resources).

    Environment variables:
        AZURE_KEYVAULT_URL: Key Vault URL (required)
        AZURE_CLIENT_ID: Service principal client ID (optional)
        AZURE_TENANT_ID: Azure AD tenant ID (optional)
        AZURE_CLIENT_SECRET: Service principal secret (optional)

    Example:
        provider = AzureKeyVaultProvider(
            vault_url="https://my-vault.vault.azure.net/"
        )
        secret = await provider.get_secret("my-secret")
    """

    def __init__(
        self,
        vault_url: Optional[str] = None,
        credential: Any = None,
        use_managed_identity: bool = False,
        cache_ttl: int = 300,
    ):
        """Initialize Azure Key Vault provider.

        Args:
            vault_url: Key Vault URL (e.g., https://vault-name.vault.azure.net/)
            credential: Azure credential object (auto-detected if None)
            use_managed_identity: Use managed identity instead of DefaultAzureCredential
            cache_ttl: Cache time-to-live in seconds

        Raises:
            ProviderNotAvailableError: If Azure SDK is not installed
            ProviderError: If configuration is invalid
        """
        if not AZURE_AVAILABLE:
            raise ProviderNotAvailableError(
                "azure",
                "Install azure-identity and azure-keyvault-secrets packages"
            )

        self.vault_url = vault_url or os.getenv("AZURE_KEYVAULT_URL")
        if not self.vault_url:
            raise ProviderError(
                "azure",
                "vault_url required (set AZURE_KEYVAULT_URL environment variable)"
            )

        # Setup credential
        if credential:
            self.credential = credential
        elif use_managed_identity:
            logger.info("Using Azure Managed Identity")
            self.credential = ManagedIdentityCredential()
        else:
            logger.info("Using Azure DefaultAzureCredential")
            self.credential = DefaultAzureCredential()

        # Create client
        try:
            self.client = SecretClient(
                vault_url=self.vault_url,
                credential=self.credential
            )
        except Exception as e:
            raise ProviderError("azure", f"Failed to create client: {e}")

        self.cache = SecretCache(ttl_seconds=cache_ttl)
        logger.info(f"Initialized Azure Key Vault provider: {self.vault_url}")

    @retry(
        retry=retry_if_exception_type((httpx.NetworkError, httpx.TimeoutException)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    async def get_secret(self, name: str) -> str:
        """Retrieve a secret from Azure Key Vault.

        Args:
            name: Secret name

        Returns:
            Secret value

        Raises:
            SecretNotFoundError: If secret doesn't exist
            ProviderError: If retrieval fails
        """
        # Check cache first
        cached = await self.cache.get(name)
        if cached:
            return cached

        try:
            logger.debug(f"Retrieving secret from Azure Key Vault: {name}")
            # Azure SDK is synchronous, run in thread pool
            loop = asyncio.get_event_loop()
            secret = await loop.run_in_executor(
                None,
                self.client.get_secret,
                name
            )
            value = secret.value

            # Cache the value
            await self.cache.set(name, value)

            logger.info(f"Successfully retrieved secret: {name}")
            return value

        except AzureResourceNotFoundError:
            raise SecretNotFoundError(name, "azure")
        except Exception as e:
            logger.error(f"Failed to retrieve secret '{name}': {e}")
            raise ProviderError("azure", str(e))

    async def get_secrets(self, names: list[str]) -> Dict[str, str]:
        """Retrieve multiple secrets from Azure Key Vault.

        Args:
            names: List of secret names

        Returns:
            Dictionary mapping secret names to values
        """
        results = {}
        for name in names:
            results[name] = await self.get_secret(name)
        return results

    async def list_secrets(self) -> list[str]:
        """List all secret names in the Key Vault.

        Returns:
            List of secret names
        """
        try:
            logger.debug("Listing secrets in Azure Key Vault")
            loop = asyncio.get_event_loop()
            properties = await loop.run_in_executor(
                None,
                lambda: list(self.client.list_properties_of_secrets())
            )
            names = [prop.name for prop in properties]
            logger.info(f"Found {len(names)} secrets in Key Vault")
            return names
        except Exception as e:
            logger.error(f"Failed to list secrets: {e}")
            raise ProviderError("azure", str(e))


# ============================================================================
# Harness Secrets Manager Provider
# ============================================================================


class HarnessSecretsProvider:
    """Secret provider for Harness Secrets Manager.

    Supports account, org, and project-scoped secrets.

    Environment variables:
        HARNESS_ACCOUNT_ID: Harness account ID (required)
        HARNESS_API_KEY: API key for authentication (required)
        HARNESS_ORG_ID: Organization ID (default: "default")
        HARNESS_PROJECT_ID: Project ID (optional)
        HARNESS_API_URL: API base URL (default: https://app.harness.io/gateway)

    Example:
        provider = HarnessSecretsProvider(
            account_id="abc123",
            api_key="pat.xyz"
        )
        secret = await provider.get_secret("my-secret", scope="project")
    """

    def __init__(
        self,
        account_id: Optional[str] = None,
        api_key: Optional[str] = None,
        org_id: Optional[str] = None,
        project_id: Optional[str] = None,
        api_url: str = "https://app.harness.io/gateway",
        cache_ttl: int = 300,
    ):
        """Initialize Harness Secrets Manager provider.

        Args:
            account_id: Harness account ID
            api_key: API key for authentication
            org_id: Organization ID
            project_id: Project ID
            api_url: Harness API base URL
            cache_ttl: Cache time-to-live in seconds

        Raises:
            ProviderError: If configuration is invalid
        """
        self.account_id = account_id or os.getenv("HARNESS_ACCOUNT_ID")
        self.api_key = api_key or os.getenv("HARNESS_API_KEY")
        self.org_id = org_id or os.getenv("HARNESS_ORG_ID", "default")
        self.project_id = project_id or os.getenv("HARNESS_PROJECT_ID")
        self.api_url = api_url.rstrip("/")

        if not self.account_id:
            raise ProviderError(
                "harness",
                "account_id required (set HARNESS_ACCOUNT_ID)"
            )
        if not self.api_key:
            raise ProviderError(
                "harness",
                "api_key required (set HARNESS_API_KEY)"
            )

        # Create HTTP client
        self.client = httpx.AsyncClient(
            base_url=self.api_url,
            headers={
                "x-api-key": self.api_key,
                "Content-Type": "application/json",
            },
            timeout=30.0,
        )

        self.cache = SecretCache(ttl_seconds=cache_ttl)
        logger.info(
            f"Initialized Harness Secrets Manager provider: "
            f"account={self.account_id}, org={self.org_id}"
        )

    def _build_secret_path(self, name: str, scope: str) -> str:
        """Build the API path for a secret based on scope."""
        base = f"/ng/api/v2/secrets/{name}"

        params = [f"accountIdentifier={self.account_id}"]

        if scope in ("org", "project"):
            params.append(f"orgIdentifier={self.org_id}")

        if scope == "project":
            if not self.project_id:
                raise ProviderError(
                    "harness",
                    "project_id required for project-scoped secrets"
                )
            params.append(f"projectIdentifier={self.project_id}")

        return f"{base}?{'&'.join(params)}"

    @retry(
        retry=retry_if_exception_type((httpx.NetworkError, httpx.TimeoutException)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    async def get_secret(self, name: str, scope: str = "project") -> str:
        """Retrieve a secret from Harness Secrets Manager.

        Args:
            name: Secret identifier
            scope: Secret scope ("account", "org", or "project")

        Returns:
            Secret value

        Raises:
            SecretNotFoundError: If secret doesn't exist
            ProviderError: If retrieval fails
        """
        cache_key = f"{scope}:{name}"

        # Check cache first
        cached = await self.cache.get(cache_key)
        if cached:
            return cached

        try:
            path = self._build_secret_path(name, scope)
            logger.debug(f"Retrieving secret from Harness: {name} (scope={scope})")

            response = await self.client.get(path)
            response.raise_for_status()

            data = response.json()

            # Extract secret value from response
            # Harness returns encrypted secrets, need to decrypt
            secret_data = data.get("data", {}).get("secret", {})
            value = secret_data.get("spec", {}).get("value")

            if not value:
                raise SecretNotFoundError(name, "harness")

            # Cache the value
            await self.cache.set(cache_key, value)

            logger.info(f"Successfully retrieved secret: {name}")
            return value

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                raise SecretNotFoundError(name, "harness")
            logger.error(f"HTTP error retrieving secret '{name}': {e}")
            raise ProviderError("harness", str(e))
        except Exception as e:
            logger.error(f"Failed to retrieve secret '{name}': {e}")
            raise ProviderError("harness", str(e))

    async def get_secrets(self, names: list[str], scope: str = "project") -> Dict[str, str]:
        """Retrieve multiple secrets from Harness.

        Args:
            names: List of secret names
            scope: Secret scope

        Returns:
            Dictionary mapping secret names to values
        """
        results = {}
        for name in names:
            results[name] = await self.get_secret(name, scope=scope)
        return results

    async def list_secrets(self, scope: str = "project") -> list[str]:
        """List all secret names in the given scope.

        Args:
            scope: Secret scope ("account", "org", or "project")

        Returns:
            List of secret identifiers
        """
        try:
            path = "/ng/api/v2/secrets"
            params = {"accountIdentifier": self.account_id}

            if scope in ("org", "project"):
                params["orgIdentifier"] = self.org_id

            if scope == "project":
                if not self.project_id:
                    raise ProviderError(
                        "harness",
                        "project_id required for project-scoped secrets"
                    )
                params["projectIdentifier"] = self.project_id

            logger.debug(f"Listing secrets in Harness (scope={scope})")

            response = await self.client.get(path, params=params)
            response.raise_for_status()

            data = response.json()
            secrets = data.get("data", {}).get("content", [])
            names = [s.get("secret", {}).get("identifier") for s in secrets]

            logger.info(f"Found {len(names)} secrets in Harness")
            return names

        except Exception as e:
            logger.error(f"Failed to list secrets: {e}")
            raise ProviderError("harness", str(e))

    async def close(self) -> None:
        """Close the HTTP client."""
        await self.client.aclose()


# ============================================================================
# Environment Variable Provider
# ============================================================================


class EnvironmentProvider:
    """Fallback secret provider using environment variables.

    Useful for local development and testing.

    Example:
        provider = EnvironmentProvider(prefix="SECRET_")
        value = await provider.get_secret("API_KEY")  # reads SECRET_API_KEY
    """

    def __init__(self, prefix: str = "", cache_ttl: int = 300):
        """Initialize environment variable provider.

        Args:
            prefix: Optional prefix for environment variable names
            cache_ttl: Cache time-to-live in seconds
        """
        self.prefix = prefix
        self.cache = SecretCache(ttl_seconds=cache_ttl)
        logger.info(f"Initialized Environment provider (prefix='{prefix}')")

    async def get_secret(self, name: str) -> str:
        """Retrieve a secret from environment variables.

        Args:
            name: Secret name (will be prefixed)

        Returns:
            Secret value

        Raises:
            SecretNotFoundError: If environment variable not set
        """
        # Check cache first
        cached = await self.cache.get(name)
        if cached:
            return cached

        key = f"{self.prefix}{name}" if self.prefix else name
        value = os.getenv(key)

        if value is None:
            raise SecretNotFoundError(name, "environment")

        # Cache the value
        await self.cache.set(name, value)

        logger.info(f"Successfully retrieved secret from environment: {name}")
        return value

    async def get_secrets(self, names: list[str]) -> Dict[str, str]:
        """Retrieve multiple secrets from environment.

        Args:
            names: List of secret names

        Returns:
            Dictionary mapping secret names to values
        """
        results = {}
        for name in names:
            results[name] = await self.get_secret(name)
        return results

    async def list_secrets(self) -> list[str]:
        """List all environment variables matching the prefix.

        Returns:
            List of secret names (with prefix removed)
        """
        all_vars = os.environ.keys()

        if self.prefix:
            # Filter by prefix and remove it
            secrets = [
                var[len(self.prefix):]
                for var in all_vars
                if var.startswith(self.prefix)
            ]
        else:
            secrets = list(all_vars)

        logger.info(f"Found {len(secrets)} secrets in environment")
        return secrets


# ============================================================================
# Unified Secrets Manager
# ============================================================================


class SecretsManager:
    """Unified secrets management interface.

    Automatically detects and uses the appropriate provider based on
    available environment variables and configuration.

    Provider priority:
        1. Explicitly specified provider
        2. Azure Key Vault (if AZURE_KEYVAULT_URL set)
        3. Harness Secrets Manager (if HARNESS_ACCOUNT_ID set)
        4. Environment variables (fallback)

    Example:
        # Auto-detect provider
        manager = SecretsManager()

        # Explicit provider
        manager = SecretsManager(provider="azure", vault_url="https://...")

        # Get secrets
        api_key = await manager.get_anthropic_key()
        all_secrets = await manager.get_all_secrets()
    """

    def __init__(
        self,
        provider: str = "auto",
        cache_ttl: int = 300,
        **provider_kwargs: Any,
    ):
        """Initialize secrets manager.

        Args:
            provider: Provider type ("auto", "azure", "harness", "env")
            cache_ttl: Cache time-to-live in seconds
            **provider_kwargs: Additional provider-specific arguments
        """
        if provider == "auto":
            provider = self._detect_provider()

        self.provider_type = provider
        self.provider = self._create_provider(provider, cache_ttl, **provider_kwargs)

        logger.info(f"Initialized SecretsManager with provider: {provider}")

    def _detect_provider(self) -> str:
        """Auto-detect the best available provider."""
        if os.getenv("AZURE_KEYVAULT_URL"):
            logger.info("Auto-detected Azure Key Vault provider")
            return "azure"
        elif os.getenv("HARNESS_ACCOUNT_ID"):
            logger.info("Auto-detected Harness Secrets Manager provider")
            return "harness"
        else:
            logger.info("Auto-detected Environment provider")
            return "env"

    def _create_provider(
        self,
        provider: str,
        cache_ttl: int,
        **kwargs: Any,
    ) -> SecretProvider:
        """Create the appropriate provider instance."""
        if provider == "azure":
            return AzureKeyVaultProvider(cache_ttl=cache_ttl, **kwargs)
        elif provider == "harness":
            return HarnessSecretsProvider(cache_ttl=cache_ttl, **kwargs)
        elif provider == "env":
            return EnvironmentProvider(cache_ttl=cache_ttl, **kwargs)
        else:
            raise ValueError(
                f"Unknown provider: {provider}. "
                f"Choose from: auto, azure, harness, env"
            )

    async def get_secret(self, name: str, **kwargs: Any) -> str:
        """Get a secret by name.

        Args:
            name: Secret name
            **kwargs: Provider-specific arguments (e.g., scope for Harness)

        Returns:
            Secret value
        """
        return await self.provider.get_secret(name, **kwargs)

    async def get_secrets(self, names: list[str], **kwargs: Any) -> Dict[str, str]:
        """Get multiple secrets.

        Args:
            names: List of secret names
            **kwargs: Provider-specific arguments

        Returns:
            Dictionary mapping secret names to values
        """
        return await self.provider.get_secrets(names, **kwargs)

    async def list_secrets(self, **kwargs: Any) -> list[str]:
        """List all available secret names.

        Args:
            **kwargs: Provider-specific arguments

        Returns:
            List of secret names
        """
        return await self.provider.list_secrets(**kwargs)

    # Convenience methods for common secrets

    async def get_anthropic_key(self) -> str:
        """Get Anthropic API key."""
        return await self.get_secret("ANTHROPIC_API_KEY")

    async def get_voyage_key(self) -> str:
        """Get Voyage AI API key."""
        return await self.get_secret("VOYAGE_API_KEY")

    async def get_pinecone_key(self) -> str:
        """Get Pinecone API key."""
        return await self.get_secret("PINECONE_API_KEY")

    async def get_harness_token(self) -> str:
        """Get Harness API token."""
        return await self.get_secret("HARNESS_API_TOKEN")

    async def get_all_secrets(self) -> Dict[str, str]:
        """Get all required secrets for the DevOps agent.

        Returns:
            Dictionary mapping secret names to values
        """
        names = [
            "ANTHROPIC_API_KEY",
            "VOYAGE_API_KEY",
            "PINECONE_API_KEY",
            "HARNESS_API_TOKEN",
        ]

        logger.info("Retrieving all required secrets")
        return await self.get_secrets(names)

    async def close(self) -> None:
        """Close any open connections."""
        if hasattr(self.provider, "close"):
            await self.provider.close()


# ============================================================================
# Factory Functions
# ============================================================================


async def create_secrets_manager(
    provider: str = "auto",
    validate: bool = True,
    **kwargs: Any,
) -> SecretsManager:
    """Create and initialize a secrets manager.

    Args:
        provider: Provider type ("auto", "azure", "harness", "env")
        validate: Validate connectivity by listing secrets
        **kwargs: Additional provider-specific arguments

    Returns:
        Initialized SecretsManager instance

    Raises:
        ProviderError: If validation fails

    Example:
        manager = await create_secrets_manager(provider="azure")
        api_key = await manager.get_anthropic_key()
    """
    manager = SecretsManager(provider=provider, **kwargs)

    if validate:
        try:
            # Validate connectivity
            await manager.list_secrets()
            logger.info("Secrets manager connectivity validated")
        except Exception as e:
            logger.error(f"Failed to validate secrets manager: {e}")
            raise ProviderError(
                manager.provider_type,
                f"Connectivity validation failed: {e}"
            )

    return manager


async def configure_agent_secrets() -> Dict[str, str]:
    """Load all secrets needed by the DevOps agents.

    This is a convenience function that:
    1. Creates a secrets manager with auto-detected provider
    2. Retrieves all required secrets
    3. Returns them for environment configuration

    Returns:
        Dictionary mapping secret names to values

    Usage:
        secrets = await configure_agent_secrets()
        os.environ.update(secrets)  # Make available to libraries

    Example:
        import asyncio

        async def main():
            secrets = await configure_agent_secrets()
            os.environ.update(secrets)

            # Now all libraries can access secrets via environment
            from langchain_anthropic import ChatAnthropic
            llm = ChatAnthropic()  # Uses ANTHROPIC_API_KEY from env

        asyncio.run(main())
    """
    logger.info("Configuring agent secrets")

    manager = await create_secrets_manager()
    secrets = await manager.get_all_secrets()

    logger.info(f"Loaded {len(secrets)} secrets for agent configuration")
    return secrets


# ============================================================================
# Testing Utilities
# ============================================================================


async def test_provider(provider: str = "auto", **kwargs: Any) -> None:
    """Test a secrets provider by listing and retrieving secrets.

    Args:
        provider: Provider type to test
        **kwargs: Provider-specific arguments

    Example:
        await test_provider("azure", vault_url="https://...")
    """
    logger.info(f"Testing provider: {provider}")

    try:
        manager = await create_secrets_manager(provider=provider, **kwargs)

        # List secrets
        secrets = await manager.list_secrets()
        logger.info(f"Found {len(secrets)} secrets")

        if secrets:
            # Try to retrieve first secret
            first_secret = secrets[0]
            logger.info(f"Testing retrieval of: {first_secret}")
            value = await manager.get_secret(first_secret)
            logger.info(f"Successfully retrieved secret (length: {len(value)})")

        logger.info("Provider test successful!")

    except Exception as e:
        logger.error(f"Provider test failed: {e}")
        raise
    finally:
        if 'manager' in locals():
            await manager.close()


if __name__ == "__main__":
    # Example usage and testing
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    async def main():
        # Test auto-detection
        logger.info("Testing auto-detected provider...")
        try:
            manager = await create_secrets_manager()
            secrets = await manager.get_all_secrets()
            logger.info(f"Successfully retrieved {len(secrets)} secrets")
            await manager.close()
        except Exception as e:
            logger.error(f"Failed: {e}")
            sys.exit(1)

    asyncio.run(main())
