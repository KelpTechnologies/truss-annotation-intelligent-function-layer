"""
Cloud SQL PostgreSQL client for taxonomy lookups.

Mirrors the truss-data-service-layer JS implementation:
- Cloud SQL Connector with IAM auth (public IP, no VPC needed)
- Same GCP service account used for BigQuery
- Connection pooling via pg8000
- Knowledge table result caching

Instance: truss-data-science:europe-west2:truss-api-postgres
Database: truss-api
Schemas: api_Dev (dev), api_staging (staging), api (prod)
"""

import hashlib
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Cloud SQL config (matches DSL JS implementation)
INSTANCE_CONNECTION_NAME = os.getenv(
    "CLOUD_SQL_INSTANCE", "truss-data-science:europe-west2:truss-api-postgres"
)
DATABASE_NAME = os.getenv("POSTGRES_DATABASE", "truss-api")
MAX_RETRIES = int(os.getenv("POSTGRES_MAX_RETRIES", "2"))
RETRY_BASE_DELAY_S = 1.0
CACHE_TTL_S = int(os.getenv("POSTGRES_KNOWLEDGE_CACHE_TTL_S", "3600"))  # 1 hour

# Singletons
_connector = None
_pool = None

# In-memory cache: key -> (data, timestamp)
_cache: Dict[str, Tuple[Any, float]] = {}


def _get_api_schema() -> str:
    """Map STAGE env var to Postgres schema name (matches BQ dataset names)."""
    stage = os.getenv("STAGE", "dev").lower()
    mapping = {"dev": "api_dev", "staging": "api", "prod": "api"}
    return mapping.get(stage, "api_dev")


def _cache_key(sql: str, params: list) -> str:
    raw = sql + json.dumps(params, default=str)
    return hashlib.md5(raw.encode()).hexdigest()


def _get_pool():
    """Create or return singleton pg8000 connection pool via Cloud SQL Connector."""
    global _connector, _pool

    if _pool is not None:
        return _pool

    try:
        from google.cloud.sql.connector import Connector
        import pg8000
    except ImportError:
        logger.warning("[Postgres] cloud-sql-python-connector or pg8000 not installed")
        return None

    try:
        # ensure_gcp_adc sets GOOGLE_APPLICATION_CREDENTIALS
        from core.utils.credentials import ensure_gcp_adc
        ensure_gcp_adc()
    except Exception as e:
        logger.warning(f"[Postgres] Could not set up GCP credentials: {e}")
        return None

    try:
        _connector = Connector()

        def _getconn():
            return _connector.connect(
                INSTANCE_CONNECTION_NAME,
                "pg8000",
                user=_iam_user(),
                db=DATABASE_NAME,
                enable_iam_auth=True,
                ip_type="PUBLIC",
            )

        # pg8000 doesn't have a built-in pool, so we use sqlalchemy or manual.
        # For simplicity, cache a single connection (Lambda = single-concurrency).
        conn = _getconn()
        _pool = conn
        logger.info(f"[Postgres] Connected to {INSTANCE_CONNECTION_NAME}/{DATABASE_NAME}")
        return _pool
    except Exception as e:
        logger.warning(f"[Postgres] Connection failed: {e}")
        _pool = None
        return None


def _iam_user() -> str:
    """Extract IAM user from GCP credentials (email minus domain)."""
    creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "/tmp/gcp_sa.json")
    try:
        with open(creds_path, "r") as f:
            creds = json.load(f)
        email = creds.get("client_email", "")
        return email.replace(".gserviceaccount.com", "")
    except Exception:
        return ""


def _reconnect():
    """Force reconnect on stale connection."""
    global _pool
    _pool = None
    return _get_pool()


def query(sql: str, params: Optional[list] = None, use_cache: bool = True) -> Optional[List[Dict[str, Any]]]:
    """
    Execute a Postgres query with retry and optional caching.

    Args:
        sql: SQL query with %s placeholders for pg8000
        params: Query parameter values
        use_cache: Cache results for knowledge table queries

    Returns:
        List of row dicts, or None if Postgres unavailable
    """
    params = params or []

    # Check cache
    if use_cache:
        ck = _cache_key(sql, params)
        cached = _cache.get(ck)
        if cached and (time.time() - cached[1]) < CACHE_TTL_S:
            logger.debug("[Postgres] Cache hit")
            return cached[0]

    conn = _get_pool()
    if conn is None:
        return None

    for attempt in range(MAX_RETRIES + 1):
        try:
            cursor = conn.cursor()
            cursor.execute(sql, params)
            columns = [desc[0] for desc in cursor.description]
            rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
            conn.commit()

            # Cache result
            if use_cache:
                _cache[_cache_key(sql, params)] = (rows, time.time())

            logger.debug(f"[Postgres] Query OK: {len(rows)} rows")
            return rows

        except Exception as e:
            err_msg = str(e)
            is_retryable = any(x in err_msg.lower() for x in [
                "connection", "timeout", "reset", "broken pipe", "eof"
            ])

            if is_retryable and attempt < MAX_RETRIES:
                delay = RETRY_BASE_DELAY_S * (2 ** attempt)
                logger.warning(f"[Postgres] Retryable error (attempt {attempt + 1}), retrying in {delay}s: {e}")
                time.sleep(delay)
                conn = _reconnect()
                if conn is None:
                    return None
                continue
            else:
                logger.warning(f"[Postgres] Query failed: {e}")
                return None

    return None


def close():
    """Close connection and connector."""
    global _pool, _connector
    if _pool:
        try:
            _pool.close()
        except Exception:
            pass
        _pool = None
    if _connector:
        try:
            _connector.close()
        except Exception:
            pass
        _connector = None


def clear_cache():
    """Clear the in-memory result cache."""
    _cache.clear()
