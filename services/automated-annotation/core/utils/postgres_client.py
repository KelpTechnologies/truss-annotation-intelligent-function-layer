"""
Azure PostgreSQL client for taxonomy lookups (TP-461, 2026-10-19).

Replaces the Cloud SQL Connector + IAM path: Cloud SQL `truss-api-postgres` is being
retired, the display layer now lives on the Azure Flexible Server `truss-client-pg`
(one schema, `api`, for every stage — there is no api_dev / api_staging copy there).

Mirrors truss-data-service-layer/services/utils/postgres-database.js (azure-postgres):
- plain TCP + TLS (certificate verified against the system CA bundle), pg8000 driver
- credentials from the consolidated `truss-platform-secrets` blob, key
  AZURE_PG_SECRET_KEY (default `azure_database` = role svc_dsl); the box firewall is
  an IP allow-list, so the Lambda must sit in the VPC and egress via the NAT EIP
- port AZURE_PG_PORT (default 6432 = the server's built-in PgBouncer, transaction
  pooling; this module holds no session state so that is safe)
- application_name = aifl-<service>-<stage> so sessions are attributable in
  pg_stat_activity
- single cached connection per process, serialised by a lock (TRS-471), reconnect
  on any error, result cache for knowledge tables; callers get None on failure and
  fall back to BigQuery, exactly as before.

Env overrides (local runs): AZURE_PG_HOST / AZURE_PG_USER / AZURE_PG_PASSWORD /
AZURE_PG_DATABASE bypass Secrets Manager; POSTGRES_SCHEMA overrides the schema.
"""

import hashlib
import json
import logging
import os
import ssl
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

AZURE_PG_PORT = int(os.getenv("AZURE_PG_PORT", "6432"))
AZURE_PG_SECRET_KEY = os.getenv("AZURE_PG_SECRET_KEY", "azure_database")
AZURE_PG_CONNECT_TIMEOUT_S = int(os.getenv("AZURE_PG_CONNECT_TIMEOUT_S", "8"))
MAX_RETRIES = int(os.getenv("POSTGRES_MAX_RETRIES", "2"))
RETRY_BASE_DELAY_S = 1.0
CACHE_TTL_S = int(os.getenv("POSTGRES_KNOWLEDGE_CACHE_TTL_S", "3600"))  # 1 hour

# Singleton connection (Lambda = single-concurrency per container)
_pool = None

# Serialise query()/reconnect across threads so non-Lambda callers (CLI, batch
# extractors) don't corrupt the single shared connection by interleaving cursor
# operations. Reentrant so query() can call _reconnect() within the same
# critical section.
_pool_lock = threading.RLock()

# In-memory cache: key -> (data, timestamp)
_cache: Dict[str, Tuple[Any, float]] = {}


def _get_api_schema() -> str:
    """Postgres schema for the display layer. The Azure box carries ONE display
    layer, `api`, for every stage (the Cloud SQL api_dev / api_staging copies were
    never loaded there). POSTGRES_SCHEMA overrides for ad-hoc runs."""
    return os.getenv("POSTGRES_SCHEMA", "api")


def _cache_key(sql: str, params: list) -> str:
    raw = sql + json.dumps(params, default=str)
    return hashlib.md5(raw.encode()).hexdigest()


def _connection_params() -> Dict[str, Any]:
    """host/user/password/database for the Azure box. Env overrides win; otherwise
    the `AZURE_PG_SECRET_KEY` entry of truss-platform-secrets ({host, name, user,
    password}) — the same blob the BigQuery credentials come from."""
    env = {
        "host": os.getenv("AZURE_PG_HOST"),
        "user": os.getenv("AZURE_PG_USER"),
        "password": os.getenv("AZURE_PG_PASSWORD"),
        "database": os.getenv("AZURE_PG_DATABASE"),
    }
    if all(env.values()):
        return env

    from core.utils.credentials import get_azure_database_credentials
    creds = get_azure_database_credentials(AZURE_PG_SECRET_KEY)
    params = {
        "host": env["host"] or creds.get("host"),
        "user": env["user"] or creds.get("user"),
        "password": env["password"] or creds.get("password"),
        "database": env["database"] or creds.get("name") or creds.get("database"),
    }
    missing = [k for k, v in params.items() if not v]
    if missing:
        raise ValueError(
            f"Azure Postgres credentials incomplete (missing {missing}); set them in "
            f"truss-platform-secrets under `{AZURE_PG_SECRET_KEY}` or via AZURE_PG_* env"
        )
    return params


def _application_name() -> str:
    service = os.getenv("SERVICE_NAME", "automated-annotation")
    stage = os.getenv("STAGE", "local")
    return f"aifl-{service}-{stage}"


def _get_pool():
    """Create or return the singleton pg8000 connection to the Azure box."""
    global _pool

    if _pool is not None:
        return _pool

    try:
        import pg8000.dbapi as pg
    except ImportError:
        logger.warning("[Postgres] pg8000 not installed")
        return None

    try:
        params = _connection_params()
    except Exception as e:
        logger.warning(f"[Postgres] Could not load Azure credentials: {e}")
        return None

    try:
        ssl_context = ssl.create_default_context()
        if os.getenv("AZURE_PG_SSL_NO_VERIFY") == "true":
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
        conn = pg.connect(
            host=params["host"],
            port=AZURE_PG_PORT,
            user=params["user"],
            password=params["password"],
            database=params["database"],
            ssl_context=ssl_context,
            timeout=AZURE_PG_CONNECT_TIMEOUT_S,
            application_name=_application_name(),
        )
        _pool = conn
        logger.info(
            f"[Postgres] Connected to Azure {params['host']}:{AZURE_PG_PORT}/{params['database']} "
            f"as {params['user']} (schema {_get_api_schema()}, app {_application_name()})"
        )
        return _pool
    except Exception as e:
        logger.warning(f"[Postgres] Connection failed: {e}")
        _pool = None
        return None


def _reconnect():
    """Force reconnect on stale connection."""
    global _pool
    if _pool is not None:
        try:
            _pool.close()
        except Exception:
            pass
    _pool = None
    return _get_pool()


def _drop_pool():
    """Drop the singleton without trying to reconnect. Next call to _get_pool
    will create a fresh connection."""
    global _pool
    if _pool is not None:
        try:
            _pool.close()
        except Exception:
            pass
    _pool = None


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

    # Check cache (lock-free; _cache dict ops are atomic enough for our use).
    if use_cache:
        ck = _cache_key(sql, params)
        cached = _cache.get(ck)
        if cached and (time.time() - cached[1]) < CACHE_TTL_S:
            logger.debug("[Postgres] Cache hit")
            return cached[0]

    # Serialise the rest: the singleton connection is not thread-safe; multiple
    # concurrent callers interleaving cursor.execute / commit corrupt it (the
    # error then masquerades as a non-retryable failure and the singleton stays
    # dead until process restart).
    with _pool_lock:
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
                # Any error forces a reconnect on the next attempt (see TRS-471):
                # a narrow keyword list left the singleton stuck on
                # InternalError / InterfaceError / "cursor already closed".
                if attempt < MAX_RETRIES:
                    delay = RETRY_BASE_DELAY_S * (2 ** attempt)
                    logger.warning(f"[Postgres] Error (attempt {attempt + 1}), reconnecting in {delay}s: {e}")
                    time.sleep(delay)
                    conn = _reconnect()
                    if conn is None:
                        return None
                    continue
                else:
                    logger.warning(f"[Postgres] Query failed after {MAX_RETRIES} retries: {e}")
                    # Drop the conn so the next caller reconnects rather than
                    # inheriting a broken singleton.
                    _drop_pool()
                    return None

        return None


def close():
    """Close the connection."""
    global _pool
    if _pool:
        try:
            _pool.close()
        except Exception:
            pass
        _pool = None


def clear_cache():
    """Clear the in-memory result cache."""
    _cache.clear()
