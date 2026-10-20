"""TP-461: postgres_client connects to the Azure box (plain TLS, pg8000, platform
secret), reads schema `api` for every stage, serialises the singleton, and returns
None on failure so callers fall back to BigQuery. Stubs pg8000 and Secrets Manager —
nothing touches the network.

    python3 tests/test_postgres_client_azure.py
"""
import importlib
import os
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "services" / "automated-annotation"))

# --- stubs -------------------------------------------------------------------
connects = []

class FakeCursor:
    description = [("id",), ("brand",)]
    def execute(self, sql, params): self.sql, self.params = sql, params
    def fetchall(self): return [(1, "Chanel"), (2, "Hermès")]

class FakeConn:
    def __init__(self, **kw): connects.append(kw); self.closed = False
    def cursor(self): return FakeCursor()
    def commit(self): pass
    def close(self): self.closed = True

fake_pg = types.ModuleType("pg8000"); fake_dbapi = types.ModuleType("pg8000.dbapi")
fake_dbapi.connect = lambda **kw: FakeConn(**kw)
fake_pg.dbapi = fake_dbapi
sys.modules["pg8000"] = fake_pg; sys.modules["pg8000.dbapi"] = fake_dbapi

fake_boto3 = types.ModuleType("boto3")
class _SM:
    def get_secret_value(self, SecretId):
        return {"SecretString": '{"bigquery": {}, "azure_database": {"host": "truss-client-pg.postgres.database.azure.com", "name": "truss-api", "user": "svc_dsl", "password": "pw"}}'}
fake_boto3.client = lambda *a, **k: _SM()
sys.modules["boto3"] = fake_boto3

os.environ["TRUSS_SECRETS_ARN"] = "arn:aws:secretsmanager:eu-west-2:1:secret:test"
os.environ["STAGE"] = "staging"; os.environ["DB_STAGE"] = "prod"; os.environ["SERVICE_NAME"] = "automated-annotation"
for k in ("AZURE_PG_HOST", "AZURE_PG_USER", "AZURE_PG_PASSWORD", "AZURE_PG_DATABASE", "POSTGRES_SCHEMA", "AZURE_PG_PORT"):
    os.environ.pop(k, None)

pc = importlib.import_module("core.utils.postgres_client")

def run(name, fn):
    try:
        fn(); print(f"  PASS: {name}")
    except AssertionError as e:
        print(f"  FAIL: {name}: {e}"); sys.exit(1)

print("postgres_client (Azure)")

def t_schema():
    for st in ("dev", "staging", "prod"):
        os.environ["STAGE"] = st
        assert pc._get_api_schema() == "api", f"{st} must read api, got {pc._get_api_schema()}"
    os.environ["POSTGRES_SCHEMA"] = "api_legacy"
    assert pc._get_api_schema() == "api_legacy"
    os.environ.pop("POSTGRES_SCHEMA")
run("every stage reads schema api; POSTGRES_SCHEMA overrides", t_schema)

def t_connect():
    os.environ["STAGE"] = "staging"
    rows = pc.query("SELECT id, brand FROM api.brand_knowledge_display WHERE LOWER(brand) LIKE %s", ["%chanel%"], use_cache=False)
    assert rows == [{"id": 1, "brand": "Chanel"}, {"id": 2, "brand": "Hermès"}], rows
    assert len(connects) == 1, f"one connection expected, got {len(connects)}"
    kw = connects[0]
    assert kw["host"] == "truss-client-pg.postgres.database.azure.com" and kw["user"] == "svc_dsl" and kw["database"] == "truss-api" and kw["password"] == "pw"
    assert kw["port"] == 6432, f"PgBouncer port expected, got {kw['port']}"
    assert kw["ssl_context"] is not None and kw["ssl_context"].check_hostname is True, "TLS must verify"
    assert kw["application_name"] == "aifl-automated-annotation-staging", kw["application_name"]
    assert kw["timeout"] == 8
run("connects once to Azure via platform secret: 6432, TLS verified, application_name", t_connect)

def t_reuse():
    pc.query("SELECT 1", [], use_cache=False)
    assert len(connects) == 1, "singleton reused"
run("second query reuses the connection", t_reuse)

def t_cache():
    n = len(connects)
    a = pc.query("SELECT id, brand FROM api.brand_knowledge_display", [])
    b = pc.query("SELECT id, brand FROM api.brand_knowledge_display", [])
    assert a == b and len(connects) == n
run("knowledge result cache hit", t_cache)

def t_failure_returns_none():
    pc.close(); pc.clear_cache()
    def boom(**kw): raise RuntimeError("connection refused")
    fake_dbapi.connect = boom
    assert pc.query("SELECT 1", [], use_cache=False) is None, "unavailable Postgres -> None (BigQuery fallback)"
    fake_dbapi.connect = lambda **kw: FakeConn(**kw)
run("connection failure returns None, never raises", t_failure_returns_none)

def t_env_override():
    pc.close()
    os.environ.update({"AZURE_PG_HOST": "localhost", "AZURE_PG_USER": "u", "AZURE_PG_PASSWORD": "p", "AZURE_PG_DATABASE": "d"})
    before = len(connects)
    pc.query("SELECT 1", [], use_cache=False)
    assert connects[before]["host"] == "localhost", "env override must bypass Secrets Manager"
    for k in ("AZURE_PG_HOST", "AZURE_PG_USER", "AZURE_PG_PASSWORD", "AZURE_PG_DATABASE"): os.environ.pop(k)
run("AZURE_PG_* env overrides bypass the secret", t_env_override)

def t_no_cloudsql():
    src = Path(pc.__file__).read_text()
    assert "cloud.sql" not in src and "INSTANCE_CONNECTION_NAME" not in src, "Cloud SQL connector must be gone"
run("no Cloud SQL connector left in the module", t_no_cloudsql)

print("all postgres_client Azure tests passed")
