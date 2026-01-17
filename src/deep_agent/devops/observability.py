"""Observability clients for Prometheus, Loki, Elasticsearch, and AlertManager.

This module provides async HTTP clients for querying observability systems
including metrics (Prometheus), logs (Loki/Elasticsearch), and alerts (AlertManager).

Environment variables:
    PROMETHEUS_URL: Prometheus server URL (default: http://prometheus:9090)
    ALERTMANAGER_URL: AlertManager URL (default: http://alertmanager:9093)
    LOKI_URL: Loki server URL (default: http://loki:3100)
    ELASTICSEARCH_URL: Elasticsearch URL (default: http://elasticsearch:9200)
    ELASTICSEARCH_USERNAME: Basic auth username (optional)
    ELASTICSEARCH_PASSWORD: Basic auth password (optional)
    ELASTICSEARCH_INDEX_PATTERN: Index pattern to search (default: logs-*)
    LOG_BACKEND: Log backend to use - 'loki' or 'elasticsearch' (default: loki)
"""

import logging
import os
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Environment Helper
# =============================================================================


def _get_env(key: str, default: str = "") -> str:
    """Get environment variable with fallback."""
    return os.environ.get(key, default)


# =============================================================================
# Prometheus Client
# =============================================================================


class PrometheusClient:
    """Client for querying Prometheus metrics API.

    Supports both instant queries and range queries using PromQL.

    Environment variables:
        PROMETHEUS_URL: Prometheus server URL (default: http://prometheus:9090)
    """

    _instance: Optional["PrometheusClient"] = None

    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: float = 30.0,
    ):
        """Initialize Prometheus client.

        Args:
            base_url: Prometheus server URL (defaults to PROMETHEUS_URL env var)
            timeout: Request timeout in seconds
        """
        self.base_url = (base_url or _get_env("PROMETHEUS_URL", "http://prometheus:9090")).rstrip("/")
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None

    @classmethod
    def get_instance(cls) -> "PrometheusClient":
        """Get or create the singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create async HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout,
                headers={"Accept": "application/json"},
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    @staticmethod
    def parse_time_range(time_range: str) -> timedelta:
        """Parse time range string to timedelta.

        Args:
            time_range: Time range string (e.g., '1h', '30m', '1d', '2w')

        Returns:
            Corresponding timedelta object

        Examples:
            >>> PrometheusClient.parse_time_range('1h')
            timedelta(hours=1)
            >>> PrometheusClient.parse_time_range('30m')
            timedelta(minutes=30)
        """
        pattern = r'^(\d+)([smhdw])$'
        match = re.match(pattern, time_range.lower().strip())

        if not match:
            # Default to 1 hour if parsing fails
            logger.warning(f"Invalid time range format: {time_range}, defaulting to 1h")
            return timedelta(hours=1)

        value = int(match.group(1))
        unit = match.group(2)

        unit_map = {
            's': timedelta(seconds=value),
            'm': timedelta(minutes=value),
            'h': timedelta(hours=value),
            'd': timedelta(days=value),
            'w': timedelta(weeks=value),
        }

        return unit_map.get(unit, timedelta(hours=1))

    @retry(
        retry=retry_if_exception_type((httpx.NetworkError, httpx.TimeoutException)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    async def query_range(
        self,
        promql: str,
        time_range: str = "1h",
        step: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute a PromQL range query.

        Args:
            promql: PromQL query string
            time_range: Time range (e.g., '1h', '30m', '1d')
            step: Query resolution step (e.g., '15s', '1m'). Auto-calculated if not provided.

        Returns:
            Query result with metric data
        """
        client = await self._get_client()

        now = datetime.now()
        duration = self.parse_time_range(time_range)
        start = now - duration

        # Auto-calculate step based on time range for reasonable data points
        if step is None:
            total_seconds = duration.total_seconds()
            if total_seconds <= 3600:  # <= 1 hour
                step = "15s"
            elif total_seconds <= 86400:  # <= 1 day
                step = "1m"
            elif total_seconds <= 604800:  # <= 1 week
                step = "5m"
            else:
                step = "1h"

        params = {
            "query": promql,
            "start": str(int(start.timestamp())),
            "end": str(int(now.timestamp())),
            "step": step,
        }

        try:
            response = await client.get("/api/v1/query_range", params=params)
            response.raise_for_status()
            data = response.json()

            if data.get("status") == "success":
                return {
                    "query": promql,
                    "start": start.isoformat(),
                    "end": now.isoformat(),
                    "step": step,
                    "resultType": data.get("data", {}).get("resultType", "unknown"),
                    "result": data.get("data", {}).get("result", []),
                }
            else:
                error_msg = data.get("error", "Unknown error")
                logger.error(f"Prometheus query failed: {error_msg}")
                return {
                    "query": promql,
                    "error": error_msg,
                    "result": [],
                }

        except httpx.HTTPStatusError as e:
            logger.error(f"Prometheus HTTP error: {e.response.status_code} - {e.response.text}")
            return {
                "query": promql,
                "error": f"HTTP {e.response.status_code}: {e.response.text}",
                "result": [],
            }
        except httpx.RequestError as e:
            logger.error(f"Prometheus request error: {e}")
            return {
                "query": promql,
                "error": f"Request failed: {str(e)}",
                "result": [],
            }

    async def query_instant(self, promql: str) -> Dict[str, Any]:
        """Execute an instant PromQL query.

        Args:
            promql: PromQL query string

        Returns:
            Query result with current metric values
        """
        client = await self._get_client()

        params = {"query": promql}

        try:
            response = await client.get("/api/v1/query", params=params)
            response.raise_for_status()
            data = response.json()

            if data.get("status") == "success":
                return {
                    "query": promql,
                    "timestamp": datetime.now().isoformat(),
                    "resultType": data.get("data", {}).get("resultType", "unknown"),
                    "result": data.get("data", {}).get("result", []),
                }
            else:
                return {
                    "query": promql,
                    "error": data.get("error", "Unknown error"),
                    "result": [],
                }

        except (httpx.HTTPStatusError, httpx.RequestError) as e:
            logger.error(f"Prometheus instant query error: {e}")
            return {
                "query": promql,
                "error": str(e),
                "result": [],
            }


# =============================================================================
# Loki Client
# =============================================================================


class LokiClient:
    """Client for querying Grafana Loki logs API.

    Supports LogQL queries for log aggregation and filtering.

    Environment variables:
        LOKI_URL: Loki server URL (default: http://loki:3100)
    """

    _instance: Optional["LokiClient"] = None

    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: float = 30.0,
    ):
        """Initialize Loki client.

        Args:
            base_url: Loki server URL (defaults to LOKI_URL env var)
            timeout: Request timeout in seconds
        """
        self.base_url = (base_url or _get_env("LOKI_URL", "http://loki:3100")).rstrip("/")
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None

    @classmethod
    def get_instance(cls) -> "LokiClient":
        """Get or create the singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create async HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout,
                headers={"Accept": "application/json"},
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    def _build_logql_query(
        self,
        query: str,
        service: Optional[str] = None,
        level: Optional[str] = None,
    ) -> str:
        """Build LogQL query with optional filters.

        Args:
            query: Base search query or keyword
            service: Service name filter
            level: Log level filter

        Returns:
            Complete LogQL query string
        """
        # Build label selectors
        selectors = []
        if service:
            selectors.append(f'service="{service}"')
        if level:
            selectors.append(f'level="{level}"')

        if selectors:
            label_selector = "{" + ", ".join(selectors) + "}"
        else:
            # Default to matching all streams
            label_selector = '{job=~".+"}'

        # Add line filter for the query
        if query:
            # Escape special regex characters
            escaped_query = re.escape(query)
            return f'{label_selector} |~ "(?i){escaped_query}"'

        return label_selector

    @retry(
        retry=retry_if_exception_type((httpx.NetworkError, httpx.TimeoutException)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    async def query_range(
        self,
        query: str,
        service: Optional[str] = None,
        level: Optional[str] = None,
        time_range: str = "1h",
        limit: int = 1000,
    ) -> List[Dict[str, Any]]:
        """Query logs over a time range.

        Args:
            query: Search query or keyword
            service: Service name filter
            level: Log level filter
            time_range: Time range (e.g., '1h', '30m')
            limit: Maximum number of log entries to return

        Returns:
            List of log entries
        """
        client = await self._get_client()

        now = datetime.now()
        duration = PrometheusClient.parse_time_range(time_range)
        start = now - duration

        logql = self._build_logql_query(query, service, level)

        params = {
            "query": logql,
            "start": str(int(start.timestamp() * 1e9)),  # Nanoseconds
            "end": str(int(now.timestamp() * 1e9)),
            "limit": limit,
            "direction": "backward",  # Most recent first
        }

        try:
            response = await client.get("/loki/api/v1/query_range", params=params)
            response.raise_for_status()
            data = response.json()

            logs = []
            for stream in data.get("data", {}).get("result", []):
                labels = stream.get("stream", {})
                for value in stream.get("values", []):
                    timestamp_ns = int(value[0])
                    message = value[1]
                    logs.append({
                        "timestamp": datetime.fromtimestamp(timestamp_ns / 1e9).isoformat(),
                        "service": labels.get("service", labels.get("job", "unknown")),
                        "level": labels.get("level", "info"),
                        "message": message,
                        "labels": labels,
                    })

            return logs

        except httpx.HTTPStatusError as e:
            logger.error(f"Loki HTTP error: {e.response.status_code} - {e.response.text}")
            return [{"error": f"HTTP {e.response.status_code}: {e.response.text}"}]
        except httpx.RequestError as e:
            logger.error(f"Loki request error: {e}")
            return [{"error": f"Request failed: {str(e)}"}]


# =============================================================================
# Elasticsearch Client
# =============================================================================


class ElasticsearchClient:
    """Client for querying Elasticsearch logs.

    Supports full-text search and structured queries.

    Environment variables:
        ELASTICSEARCH_URL: Elasticsearch URL (default: http://elasticsearch:9200)
        ELASTICSEARCH_USERNAME: Basic auth username (optional)
        ELASTICSEARCH_PASSWORD: Basic auth password (optional)
        ELASTICSEARCH_INDEX_PATTERN: Index pattern to search (default: logs-*)
    """

    _instance: Optional["ElasticsearchClient"] = None

    def __init__(
        self,
        base_url: Optional[str] = None,
        index_pattern: Optional[str] = None,
        timeout: float = 30.0,
        username: Optional[str] = None,
        password: Optional[str] = None,
    ):
        """Initialize Elasticsearch client.

        Args:
            base_url: Elasticsearch URL (defaults to ELASTICSEARCH_URL env var)
            index_pattern: Index pattern to search
            timeout: Request timeout in seconds
            username: Basic auth username
            password: Basic auth password
        """
        self.base_url = (base_url or _get_env("ELASTICSEARCH_URL", "http://elasticsearch:9200")).rstrip("/")
        self.index_pattern = index_pattern or _get_env("ELASTICSEARCH_INDEX_PATTERN", "logs-*")
        self.timeout = timeout
        self.username = username or _get_env("ELASTICSEARCH_USERNAME")
        self.password = password or _get_env("ELASTICSEARCH_PASSWORD")
        self._client: Optional[httpx.AsyncClient] = None

    @classmethod
    def get_instance(cls) -> "ElasticsearchClient":
        """Get or create the singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create async HTTP client."""
        if self._client is None or self._client.is_closed:
            auth = None
            if self.username and self.password:
                auth = (self.username, self.password)

            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout,
                auth=auth,
                headers={
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                },
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    @retry(
        retry=retry_if_exception_type((httpx.NetworkError, httpx.TimeoutException)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    async def search(
        self,
        query: str,
        service: Optional[str] = None,
        level: Optional[str] = None,
        time_range: str = "1h",
        size: int = 100,
    ) -> List[Dict[str, Any]]:
        """Search logs in Elasticsearch.

        Args:
            query: Full-text search query
            service: Service name filter
            level: Log level filter
            time_range: Time range for the search
            size: Maximum number of results

        Returns:
            List of log entries
        """
        client = await self._get_client()

        now = datetime.now()
        duration = PrometheusClient.parse_time_range(time_range)
        start = now - duration

        # Build Elasticsearch query
        must_clauses: List[Dict[str, Any]] = [
            {
                "range": {
                    "@timestamp": {
                        "gte": start.isoformat(),
                        "lte": now.isoformat(),
                    }
                }
            }
        ]

        if query:
            must_clauses.append({
                "query_string": {
                    "query": query,
                    "default_field": "message",
                    "default_operator": "AND",
                }
            })

        if service:
            must_clauses.append({
                "bool": {
                    "should": [
                        {"term": {"service.keyword": service}},
                        {"term": {"kubernetes.pod.name": {"value": service, "case_insensitive": True}}},
                        {"term": {"container.name": {"value": service, "case_insensitive": True}}},
                    ],
                    "minimum_should_match": 1,
                }
            })

        if level:
            must_clauses.append({
                "bool": {
                    "should": [
                        {"term": {"level.keyword": level}},
                        {"term": {"log.level": {"value": level, "case_insensitive": True}}},
                    ],
                    "minimum_should_match": 1,
                }
            })

        search_body = {
            "query": {
                "bool": {
                    "must": must_clauses
                }
            },
            "sort": [{"@timestamp": {"order": "desc"}}],
            "size": size,
        }

        try:
            response = await client.post(
                f"/{self.index_pattern}/_search",
                json=search_body,
            )
            response.raise_for_status()
            data = response.json()

            logs = []
            for hit in data.get("hits", {}).get("hits", []):
                source = hit.get("_source", {})
                logs.append({
                    "timestamp": source.get("@timestamp", datetime.now().isoformat()),
                    "service": (
                        source.get("service") or
                        source.get("kubernetes", {}).get("pod", {}).get("name") or
                        source.get("container", {}).get("name") or
                        "unknown"
                    ),
                    "level": (
                        source.get("level") or
                        source.get("log", {}).get("level") or
                        "info"
                    ),
                    "message": source.get("message", source.get("log", "")),
                    "trace_id": (
                        source.get("trace_id") or
                        source.get("trace", {}).get("id")
                    ),
                })

            return logs

        except httpx.HTTPStatusError as e:
            logger.error(f"Elasticsearch HTTP error: {e.response.status_code} - {e.response.text}")
            return [{"error": f"HTTP {e.response.status_code}: {e.response.text}"}]
        except httpx.RequestError as e:
            logger.error(f"Elasticsearch request error: {e}")
            return [{"error": f"Request failed: {str(e)}"}]


# =============================================================================
# AlertManager Client
# =============================================================================


class AlertManagerClient:
    """Client for querying Prometheus AlertManager API.

    Supports listing and filtering alerts by status.

    Environment variables:
        ALERTMANAGER_URL: AlertManager URL (default: http://alertmanager:9093)
    """

    _instance: Optional["AlertManagerClient"] = None

    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: float = 30.0,
    ):
        """Initialize AlertManager client.

        Args:
            base_url: AlertManager URL (defaults to ALERTMANAGER_URL env var)
            timeout: Request timeout in seconds
        """
        self.base_url = (base_url or _get_env("ALERTMANAGER_URL", "http://alertmanager:9093")).rstrip("/")
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None

    @classmethod
    def get_instance(cls) -> "AlertManagerClient":
        """Get or create the singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create async HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout,
                headers={"Accept": "application/json"},
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    @retry(
        retry=retry_if_exception_type((httpx.NetworkError, httpx.TimeoutException)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    async def list_alerts(
        self,
        status: str = "firing",
        silenced: bool = False,
        inhibited: bool = False,
        active: bool = True,
        filter_labels: Optional[Dict[str, str]] = None,
    ) -> List[Dict[str, Any]]:
        """List alerts from AlertManager.

        Args:
            status: Filter by alert status (firing, pending, resolved, all)
            silenced: Include silenced alerts
            inhibited: Include inhibited alerts
            active: Include active alerts
            filter_labels: Additional label filters

        Returns:
            List of alerts matching the criteria
        """
        client = await self._get_client()

        # Build filter parameters
        params: Dict[str, Union[str, List[str]]] = {
            "silenced": str(silenced).lower(),
            "inhibited": str(inhibited).lower(),
            "active": str(active).lower(),
        }

        # Add label filters
        if filter_labels:
            filter_expressions = [f'{k}="{v}"' for k, v in filter_labels.items()]
            params["filter"] = filter_expressions

        try:
            response = await client.get("/api/v2/alerts", params=params)
            response.raise_for_status()
            data = response.json()

            alerts = []
            for alert in data:
                alert_status = alert.get("status", {}).get("state", "unknown")

                # Filter by status if specified
                if status != "all":
                    if status == "firing" and alert_status not in ["firing", "active"]:
                        continue
                    elif status == "pending" and alert_status != "pending":
                        continue
                    elif status == "resolved" and alert_status not in ["resolved", "inactive"]:
                        continue

                labels = alert.get("labels", {})
                annotations = alert.get("annotations", {})

                alerts.append({
                    "name": labels.get("alertname", "unknown"),
                    "severity": labels.get("severity", "info"),
                    "status": alert_status,
                    "started_at": alert.get("startsAt"),
                    "ended_at": alert.get("endsAt"),
                    "labels": labels,
                    "annotations": {
                        "summary": annotations.get("summary", ""),
                        "description": annotations.get("description", ""),
                        "runbook_url": annotations.get("runbook_url", ""),
                    },
                    "fingerprint": alert.get("fingerprint"),
                    "generator_url": alert.get("generatorURL"),
                })

            return alerts

        except httpx.HTTPStatusError as e:
            logger.error(f"AlertManager HTTP error: {e.response.status_code} - {e.response.text}")
            return [{"error": f"HTTP {e.response.status_code}: {e.response.text}"}]
        except httpx.RequestError as e:
            logger.error(f"AlertManager request error: {e}")
            return [{"error": f"Request failed: {str(e)}"}]

    async def get_alert_groups(self) -> List[Dict[str, Any]]:
        """Get alert groups from AlertManager.

        Returns:
            List of alert groups with their alerts
        """
        client = await self._get_client()

        try:
            response = await client.get("/api/v2/alerts/groups")
            response.raise_for_status()
            return response.json()

        except (httpx.HTTPStatusError, httpx.RequestError) as e:
            logger.error(f"AlertManager groups error: {e}")
            return [{"error": str(e)}]


# =============================================================================
# Singleton Accessors
# =============================================================================


# Global client instances (lazy initialization)
_prometheus_client: Optional[PrometheusClient] = None
_loki_client: Optional[LokiClient] = None
_elasticsearch_client: Optional[ElasticsearchClient] = None
_alertmanager_client: Optional[AlertManagerClient] = None


def get_prometheus_client() -> PrometheusClient:
    """Get the singleton Prometheus client instance."""
    global _prometheus_client
    if _prometheus_client is None:
        _prometheus_client = PrometheusClient.get_instance()
    return _prometheus_client


def get_loki_client() -> LokiClient:
    """Get the singleton Loki client instance."""
    global _loki_client
    if _loki_client is None:
        _loki_client = LokiClient.get_instance()
    return _loki_client


def get_elasticsearch_client() -> ElasticsearchClient:
    """Get the singleton Elasticsearch client instance."""
    global _elasticsearch_client
    if _elasticsearch_client is None:
        _elasticsearch_client = ElasticsearchClient.get_instance()
    return _elasticsearch_client


def get_alertmanager_client() -> AlertManagerClient:
    """Get the singleton AlertManager client instance."""
    global _alertmanager_client
    if _alertmanager_client is None:
        _alertmanager_client = AlertManagerClient.get_instance()
    return _alertmanager_client


def get_log_backend() -> str:
    """Get configured log backend (loki or elasticsearch)."""
    return _get_env("LOG_BACKEND", "loki").lower()


# =============================================================================
# Convenience Functions for Tool Implementations
# =============================================================================


async def query_metrics(promql: str, time_range: str = "1h") -> Dict[str, Any]:
    """Query Prometheus metrics using the singleton client.

    Args:
        promql: PromQL query string
        time_range: Time range (e.g., '1h', '30m', '1d')

    Returns:
        Query result with metric data
    """
    client = get_prometheus_client()
    return await client.query_range(promql, time_range)


async def search_logs(
    query: str,
    service: Optional[str] = None,
    level: Optional[str] = None,
    time_range: str = "1h",
) -> List[Dict[str, Any]]:
    """Search logs using the configured backend (Loki or Elasticsearch).

    Args:
        query: Search query or keyword
        service: Service name filter
        level: Log level filter
        time_range: Time range (e.g., '1h', '30m')

    Returns:
        List of log entries
    """
    backend = get_log_backend()

    if backend == "elasticsearch":
        client = get_elasticsearch_client()
        return await client.search(query, service, level, time_range)
    else:
        # Default to Loki
        client = get_loki_client()
        return await client.query_range(query, service, level, time_range)


async def list_alerts(status: str = "firing") -> List[Dict[str, Any]]:
    """List alerts from AlertManager using the singleton client.

    Args:
        status: Alert status filter (firing, pending, resolved, all)

    Returns:
        List of alerts matching the criteria
    """
    client = get_alertmanager_client()
    return await client.list_alerts(status=status)


__all__ = [
    # Clients
    "PrometheusClient",
    "LokiClient",
    "ElasticsearchClient",
    "AlertManagerClient",
    # Singleton accessors
    "get_prometheus_client",
    "get_loki_client",
    "get_elasticsearch_client",
    "get_alertmanager_client",
    "get_log_backend",
    # Convenience functions
    "query_metrics",
    "search_logs",
    "list_alerts",
]
