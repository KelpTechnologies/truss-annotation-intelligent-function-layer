/**
 * PostgreSQL database connection via Google Cloud SQL Connector
 * Provides Postgres integration for AWS Lambda functions
 *
 * Uses Cloud SQL Connector with IAM authentication.
 * Service account credentials fetched from AWS Secrets Manager.
 */

let Pool, Connector, IpAddressTypes, AuthTypes;
try {
  Pool = require("pg").Pool;
  const cloudSqlConnector = require("@google-cloud/cloud-sql-connector");
  Connector = cloudSqlConnector.Connector;
  IpAddressTypes = cloudSqlConnector.IpAddressTypes;
  AuthTypes = cloudSqlConnector.AuthTypes;
} catch (err) {
  // Dependencies not available - will fail at runtime if postgres is used
  console.warn("Postgres dependencies not installed:", err.message);
}
const { writeFileSync, mkdirSync, unlinkSync } = require("fs");
const { tmpdir } = require("os");
const { join } = require("path");
const crypto = require("crypto");
const { getBigQueryCredentials } = require("./secrets-manager");
const { createLogger } = require("./structured-logger");

const initLogger = createLogger({ serviceName: "postgres-init", layer: "dsl" });
const initContext = {
  requestId: "init",
  layer: "dsl",
  serviceName: "postgres-init",
  route: "MODULE_INIT",
  routeNormalized: "MODULE_INIT",
};

initLogger.debug("Postgres module loaded successfully");

// Cloud SQL instance connection name
const INSTANCE_CONNECTION_NAME =
  process.env.CLOUD_SQL_INSTANCE || "truss-data-science:europe-west2:truss-api-postgres";
const DATABASE_NAME = process.env.POSTGRES_DATABASE || "truss-api";

let pool = null;
let connector = null;
let tmpKeyPath = null;

/**
 * Knowledge query result cache (same pattern as bigquery-database.js)
 */
const knowledgeCache = new Map();
const KNOWLEDGE_CACHE_TTL_MS = parseInt(
  process.env.POSTGRES_KNOWLEDGE_CACHE_TTL_MS || "3600000",
  10
); // Default: 1 hour

const pendingQueries = new Map();

const MAX_RETRIES = parseInt(process.env.POSTGRES_MAX_RETRIES || "3", 10);
const RETRY_BASE_DELAY_MS = 1000;

const KNOWLEDGE_TABLES = [
  "brand_knowledge_display",
  "model_knowledge_display",
  "material_knowledge_display",
  "colour_knowledge_display",
  "type_knowledge_display",
  "size_knowledge_display",
  "condition_knowledge_display",
  "vendor_knowledge_display",
  "gender_knowledge_display",
  "location_knowledge_display",
  "hardware_knowledge_display",
  "model_size_knowledge_display",
  "display_product_listings",
];

function isKnowledgeQuery(sql) {
  return KNOWLEDGE_TABLES.some(
    (table) => sql.includes(table) && !sql.includes("display_product_listings")
  );
}

function generateCacheKey(sql, args) {
  const keyString = sql + JSON.stringify(args);
  return crypto.createHash("md5").update(keyString).digest("hex");
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * Translates BigQuery-style SQL to Postgres-compatible SQL
 * - Converts @param_N (0-based) to $N+1 (1-based positional)
 * - Converts CAST(x AS INT64) to CAST(x AS INTEGER)
 * - Converts CAST(x AS FLOAT64) to CAST(x AS DOUBLE PRECISION)
 * - Converts DATE_SUB(CURRENT_DATE(), INTERVAL N MONTH) to CURRENT_DATE - INTERVAL 'N months'
 * - Strips backtick quoting from table names
 */
function translateSql(sql) {
  let translated = sql;

  // @param_N → $N+1 (BigQuery 0-based named → Postgres 1-based positional)
  translated = translated.replace(/@param_(\d+)/g, (_, n) => `$${parseInt(n, 10) + 1}`);

  // CAST(x AS INT64) → CAST(x AS INTEGER)
  translated = translated.replace(/\bAS\s+INT64\b/gi, "AS INTEGER");

  // CAST(x AS FLOAT64) → CAST(x AS DOUBLE PRECISION)
  translated = translated.replace(/\bAS\s+FLOAT64\b/gi, "AS DOUBLE PRECISION");

  // DATE_SUB(CURRENT_DATE(), INTERVAL N MONTH) → CURRENT_DATE - INTERVAL 'N months'
  translated = translated.replace(
    /DATE_SUB\s*\(\s*CURRENT_DATE\s*\(\s*\)\s*,\s*INTERVAL\s+(\d+)\s+MONTH\s*\)/gi,
    (_, n) => `(CURRENT_DATE - INTERVAL '${n} months')`
  );

  // Strip backtick quoting (BigQuery-style)
  translated = translated.replace(/`/g, "");

  // TRUE/FALSE: BigQuery uses TRUE/FALSE which Postgres also supports, so no change needed
  // BOOL type: both support it

  return translated;
}

function isRetryableError(error) {
  return (
    error.code === "ECONNRESET" ||
    error.code === "ECONNREFUSED" ||
    error.code === "ETIMEDOUT" ||
    error.code === "57P01" || // admin_shutdown
    error.code === "57P03" || // cannot_connect_now
    error.message?.includes("Connection terminated") ||
    error.message?.includes("timeout")
  );
}

/**
 * Pre-warm credentials at Lambda init
 */
const warmupPromise = (async () => {
  try {
    initLogger.debug("Pre-warming Postgres credentials at Lambda init");
    await getBigQueryCredentials(); // Same SA used for Cloud SQL
    initLogger.debug("Postgres credentials pre-warmed successfully");
  } catch (error) {
    initLogger.logWarning(
      initContext,
      "Postgres credentials pre-warm failed (will retry on first request)",
      { error: error.message }
    );
  }
})();

/**
 * Creates and returns a pg.Pool connected via Cloud SQL Connector with IAM auth
 */
async function getPool() {
  if (pool) return pool;

  const saJson = await getBigQueryCredentials();

  // Write SA key to temp file for ADC
  const dir = join(tmpdir(), "cloudsql-pg");
  mkdirSync(dir, { recursive: true });
  tmpKeyPath = join(dir, "sa-key.json");

  // Reconstruct full SA JSON (secrets-manager may only store partial fields)
  const fullSaJson = {
    type: "service_account",
    project_id: saJson.project_id,
    private_key_id: saJson.private_key_id,
    private_key: (saJson.private_key || "").replace(/\\n/g, "\n"),
    client_email: saJson.client_email,
    client_id: saJson.client_id,
    auth_uri: "https://accounts.google.com/o/oauth2/auth",
    token_uri: "https://oauth2.googleapis.com/token",
    auth_provider_x509_cert_url: "https://www.googleapis.com/oauth2/v1/certs",
    client_x509_cert_url: `https://www.googleapis.com/robot/v1/metadata/x509/${saJson.client_email}`,
  };
  writeFileSync(tmpKeyPath, JSON.stringify(fullSaJson), { mode: 0o600 });
  process.env.GOOGLE_APPLICATION_CREDENTIALS = tmpKeyPath;

  // IAM user = SA email without .gserviceaccount.com
  const iamUser = saJson.client_email.replace(".gserviceaccount.com", "");

  connector = new Connector();
  const clientOpts = await connector.getOptions({
    instanceConnectionName: INSTANCE_CONNECTION_NAME,
    ipType: IpAddressTypes.PUBLIC,
    authType: AuthTypes.IAM,
  });

  pool = new Pool({
    ...clientOpts,
    user: iamUser,
    database: DATABASE_NAME,
    max: 5,
    min: 1,
  });

  initLogger.debug("Postgres pool created successfully", {}, initContext);
  return pool;
}

/**
 * Execute a parameterised Postgres query
 * @param {string} sql - SQL with $1, $2, ... placeholders
 * @param {Array} args - Parameter values
 * @param {object} config - Service config (for logging)
 * @param {object} options - Query options (limit, skipCache, timeout)
 * @returns {Promise<Array>} Array of row objects
 */
async function executeQuery(sql, args, config, options) {
  const p = await getPool();
  const startTime = Date.now();
  const serviceName = config?.service?.name || "unknown";
  const logger = createLogger({ serviceName, layer: "dsl" });
  const queryContext = {
    requestId: null,
    serviceName,
    layer: "dsl",
    route: "POSTGRES_QUERY",
    routeNormalized: "POSTGRES_QUERY",
  };

  // Translate BigQuery SQL syntax to Postgres
  const pgSql = translateSql(sql);

  logger.debug(
    "Executing Postgres query",
    {
      query: pgSql.substring(0, 200) + (pgSql.length > 200 ? "..." : ""),
      parameters: args.length,
    },
    queryContext
  );

  const result = await p.query(pgSql, args);
  const queryTime = Date.now() - startTime;

  logger.debug(
    `Postgres query completed: ${result.rows.length} rows, ${queryTime}ms`,
    { rowCount: result.rows.length, durationMs: queryTime },
    queryContext
  );

  // Apply limit if specified
  if (options.limit && Number(options.limit) > 0) {
    return result.rows.slice(0, Number(options.limit));
  }

  return result.rows;
}

/**
 * Main query function - matches bigquery-database.js interface
 * @param {string} sql - SQL query with $1, $2, ... placeholders
 * @param {Array} args - Parameter values
 * @param {object} config - Service configuration
 * @param {object} options - Query options
 * @returns {Promise<Array>} Query results
 */
async function query(sql, args = [], config, options = {}) {
  const cacheKey = generateCacheKey(sql, args);
  const isKnowledge = isKnowledgeQuery(sql);

  // Check cache
  if (isKnowledge && !options.skipCache) {
    const cached = knowledgeCache.get(cacheKey);
    if (cached && Date.now() - cached.timestamp < KNOWLEDGE_CACHE_TTL_MS) {
      return cached.data;
    }
  }

  // Deduplicate in-flight
  if (pendingQueries.has(cacheKey)) {
    return pendingQueries.get(cacheKey);
  }

  const executeWithRetry = async () => {
    for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
      try {
        const results = await executeQuery(sql, args, config, options);

        if (isKnowledge && !options.skipCache) {
          knowledgeCache.set(cacheKey, {
            data: results,
            timestamp: Date.now(),
          });
        }

        return results;
      } catch (error) {
        if (isRetryableError(error) && attempt < MAX_RETRIES) {
          const delay = RETRY_BASE_DELAY_MS * Math.pow(2, attempt);
          const serviceName = config?.service?.name || "unknown";
          const logger = createLogger({ serviceName, layer: "dsl" });
          logger.logWarning(
            {
              requestId: null,
              serviceName,
              layer: "dsl",
              route: "POSTGRES_QUERY",
              routeNormalized: "POSTGRES_QUERY",
            },
            `Postgres error (attempt ${attempt + 1}/${MAX_RETRIES + 1}), retrying in ${delay}ms`,
            { attempt: attempt + 1, errorCode: error.code }
          );
          await sleep(delay);
          continue;
        }
        throw error;
      }
    }
  };

  const promise = executeWithRetry().finally(() => {
    pendingQueries.delete(cacheKey);
  });

  pendingQueries.set(cacheKey, promise);
  return promise;
}

/**
 * Check Postgres connection health
 */
async function checkConnectionHealth(config) {
  try {
    const p = await getPool();
    const result = await p.query("SELECT 1 as test_value");
    return {
      status: "healthy",
      connectionType: "postgres",
      testQuery: "success",
      timestamp: new Date().toISOString(),
    };
  } catch (error) {
    return {
      status: "unhealthy",
      connectionType: "postgres",
      error: error.message,
      timestamp: new Date().toISOString(),
    };
  }
}

/**
 * Close pool, connector, clean up temp files
 */
async function closeConnection() {
  if (pool) {
    await pool.end();
    pool = null;
  }
  if (connector) {
    connector.close();
    connector = null;
  }
  if (tmpKeyPath) {
    try {
      unlinkSync(tmpKeyPath);
    } catch {}
    tmpKeyPath = null;
  }
  initLogger.debug("Postgres connection closed", {}, initContext);
}

module.exports = {
  query,
  checkConnectionHealth,
  closeConnection,
  getPool,
};
