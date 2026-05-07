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
import threading
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
_connector_lock = threading.Lock()

# Per-thread connection (TRS-471): the previous singleton conn was shared across
# all worker threads, so a single failed query left the conn in "transaction
# aborted" state and poisoned every subsequent query on every thread until
# process exit. Pg8000 connections are not thread-safe; give each thread its own.
_thread_local = threading.local()

# In-memory cache: key -> (data, timestamp)
_cache: Dict[str, Tuple[Any, float]] = {}


def _get_api_schema() -> str:
    """Map DB_STAGE env var to Postgres schema name (matches BQ dataset names)."""
    stage = (os.getenv("DB_STAGE") or os.getenv("STAGE", "dev")).lower()
    mapping = {"dev": "api_staging", "staging": "api", "prod": "api"}
    return mapping.get(stage, "api")


def _cache_key(sql: str, params: list) -> str:
    raw = sql + json.dumps(params, default=str)
    return hashlib.md5(raw.encode()).hexdigest()


def _get_connector():
    """Create or return the process-wide Cloud SQL Connector. Thread-safe init."""
    global _connector
    if _connector is not None:
        return _connector
    with _connector_lock:
        if _connector is not None:
            return _connector
        try:
            from google.cloud.sql.connector import Connector
            import pg8000  # noqa: F401
        except ImportError:
            logger.warning("[Postgres] cloud-sql-python-connector or pg8000 not installed")
            return None
        try:
            from core.utils.credentials import ensure_gcp_adc
            ensure_gcp_adc()
        except Exception as e:
            logger.warning(f"[Postgres] Could not set up GCP credentials: {e}")
            return None
        try:
            _connector = Connector()
            return _connector
        except Exception as e:
            logger.warning(f"[Postgres] Connector init failed: {e}")
            return None


def _open_conn():
    """Open a new pg8000 connection via the shared Cloud SQL Connector."""
    connector = _get_connector()
    if connector is None:
        return None
    try:
        return connector.connect(
            INSTANCE_CONNECTION_NAME,
            "pg8000",
            user=_iam_user(),
            db=DATABASE_NAME,
            enable_iam_auth=True,
            ip_type="PUBLIC",
        )
    except Exception as e:
        logger.warning(f"[Postgres] Connection failed: {e}")
        return None


def _get_pool():
    """Return this thread's pg8000 connection, opening one lazily.

    Pg8000 connections are not thread-safe and a failed query poisons the
    connection's transaction state. Per-thread isolation prevents one thread's
    error from cascading into every other thread.
    """
    conn = getattr(_thread_local, "conn", None)
    if conn is not None:
        return conn
    conn = _open_conn()
    if conn is None:
        return None
    _thread_local.conn = conn
    logger.info(
        f"[Postgres] Connected to {INSTANCE_CONNECTION_NAME}/{DATABASE_NAME} "
        f"(thread={threading.current_thread().name})"
    )
    return conn


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
    """Force reconnect this thread's connection on stale/poisoned state."""
    old = getattr(_thread_local, "conn", None)
    if old is not None:
        try:
            old.close()
        except Exception:
            pass
    _thread_local.conn = None
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
            err_msg = str(e).lower()
            # "aborted" catches "current transaction is aborted, commands ignored
            # until end of transaction block" — left over from a prior failed query
            # on the same connection. Reconnect and retry.
            is_retryable = any(x in err_msg for x in [
                "connection", "timeout", "reset", "broken pipe", "eof", "aborted",
                "closed", "ssl"
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
                # Roll back to clear any aborted transaction state on this conn
                try:
                    conn.rollback()
                except Exception:
                    pass
                return None

    return None


def close():
    """Close this thread's connection and the process-wide connector.

    Note: only closes the calling thread's connection. Other threads will close
    their own connections lazily when they exit (or via process exit).
    """
    global _connector
    conn = getattr(_thread_local, "conn", None)
    if conn is not None:
        try:
            conn.close()
        except Exception:
            pass
        _thread_local.conn = None
    if _connector:
        try:
            _connector.close()
        except Exception:
            pass
        _connector = None


def clear_cache():
    """Clear the in-memory result cache."""
    _cache.clear()
