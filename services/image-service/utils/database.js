/**
 * Database connection pool (initialized once)
 */
let pool = null;
let mysql = null;
let poolInitPromise = null; // Track initialization promise for async pool creation

// Import BigQuery module
const bigqueryDb = require("./bigquery-database");
// Import Postgres module
const postgresDb = require("./postgres-database");

// Import secrets manager for database credentials
const { getDatabaseCredentials } = require("./secrets-manager");

/**
 * Lazy load mysql2 module only when needed
 */
function getMysql() {
  if (!mysql) {
    try {
      mysql = require("mysql2");
    } catch (error) {
      console.warn("MySQL2 module not available:", error.message);
      return null;
    }
  }
  return mysql;
}

/**
 * Initializes the database connection based on the service configuration.
 * Supports both MySQL and BigQuery connection types.
 * For MySQL, credentials are fetched from AWS Secrets Manager.
 * @param {object} config - Service configuration object
 * @returns {Promise<object|null>} The database connection object or null if database is not required.
 * @throws {Error} If database is required but connection details are missing.
 */
async function initDatabase(config) {
  if (!config.database.required) {
    console.log("Database not required for this service.");
    return null;
  }

  // Handle BigQuery connection type
  if (config.database.connection_type === "bigquery") {
    console.log("BigQuery connection type detected - using BigQuery client");
    return bigqueryDb.initBigQueryClient(config);
  }

  // Handle Postgres connection type (Cloud SQL via IAM)
  if (config.database.connection_type === "postgres") {
    console.log("Postgres connection type detected - using Cloud SQL Connector");
    return postgresDb.getPool();
  }

  // Default to MySQL for backward compatibility
  console.log("MySQL connection type detected - using MySQL connection pool");

  // Use existing pool if already initialized
  if (pool) {
    return pool;
  }

  // If pool is being initialized, wait for it
  if (poolInitPromise) {
    return poolInitPromise;
  }

  // Start initialization
  poolInitPromise = (async () => {
    try {
      // Get mysql module
      const mysqlModule = getMysql();
      if (!mysqlModule) {
        throw new Error(
          "MySQL2 module not available. Please install mysql2 package."
        );
      }

      // Get credentials - prefer environment variables (injected at deploy time via CloudFormation)
      // Fall back to Secrets Manager if env vars not set
      let user, password, host, databaseName;

      if (process.env.DB_USER && process.env.DB_PASSWORD) {
        console.log("Using database credentials from environment variables");
        user = process.env.DB_USER;
        password = process.env.DB_PASSWORD;
        host = process.env.DB_HOST || process.env.RDS_PROXY_ENDPOINT;
        databaseName =
          process.env.DB_NAME || config.database.name || "api_staging";
      } else {
        // Fallback to Secrets Manager (requires VPC endpoint or NAT)
        console.log("Fetching database credentials from Secrets Manager...");
        const dbCreds = await getDatabaseCredentials();
        user = dbCreds.user;
        password = dbCreds.password;
        host = dbCreds.host;
        databaseName =
          config.database.name ||
          process.env.DB_NAME ||
          dbCreds.name ||
          "api_staging";
      }

      if (!host) {
        throw new Error(
          "Database host not available (checked DB_HOST, RDS_PROXY_ENDPOINT env vars, and Secrets Manager)"
        );
      }

      // Create the database connection pool with optimized settings for Lambda cold starts
      // Note: mysql2 has different options than mysql - removed invalid: acquireTimeout, timeout, reconnect, idleTimeout
      pool = mysqlModule.createPool({
        host: host,
        user: user,
        password: password,
        database: databaseName,
        // Pool settings
        connectionLimit: 3, // Reduced for Lambda - fewer concurrent connections needed
        queueLimit: 0, // No limit on queued connection requests
        waitForConnections: true, // Wait for available connection instead of erroring
        // Connection settings
        connectTimeout: 10000, // 10 seconds to establish initial connection
        ssl: false,
        // Data handling
        multipleStatements: false, // Security best practice
        supportBigNumbers: true,
        bigNumberStrings: true,
        // Keep-alive settings for connection reuse
        enableKeepAlive: true,
        keepAliveInitialDelay: 0,
      });

      console.log(
        `Database connection pool initialized for ${databaseName} on ${host}`
      );

      // Pre-warm the connection pool for faster cold starts
      preWarmConnection(pool);

      return pool;
    } catch (error) {
      poolInitPromise = null; // Reset so retry is possible
      throw error;
    }
  })();

  return poolInitPromise;
}

/**
 * Pre-warms the database connection pool by establishing an initial connection
 * @param {object} pool - The MySQL connection pool
 */
function preWarmConnection(pool) {
  const startTime = Date.now();

  pool.getConnection((err, connection) => {
    const connectionTime = Date.now() - startTime;

    if (err) {
      console.warn("Pre-warm connection failed:", err.message);
      return;
    }

    // Test the connection with a simple query
    connection.query("SELECT 1 as test", (queryErr) => {
      const totalTime = Date.now() - startTime;

      if (queryErr) {
        console.warn("Pre-warm query failed:", queryErr.message);
      } else {
        console.log(
          `[OK] Database pre-warmed successfully (${totalTime}ms total, ${connectionTime}ms connection)`
        );
      }

      connection.release();
    });
  });
}

/**
 * Executes a SQL query against the database with retry logic for cold starts
 * Supports both MySQL and BigQuery connection types.
 * @param {string} sql - The SQL query string.
 * @param {Array} args - An array of values to escape and insert into the query.
 * @param {object} config - Service configuration object
 * @param {object} options - Query options (for BigQuery)
 * @param {number} maxRetries - Maximum number of retry attempts (default: 3)
 * @returns {Promise<Array>} A promise that resolves with the query results.
 * @throws {Error} If the database is not configured or a query error occurs.
 */
async function query(sql, args = [], config, options = {}, maxRetries = 3) {
  // Handle BigQuery connection type
  if (config.database.connection_type === "bigquery") {
    return bigqueryDb.query(sql, args, config, options);
  }

  // Handle Postgres connection type
  if (config.database.connection_type === "postgres") {
    return postgresDb.query(sql, args, config, options);
  }

  // Default to MySQL for backward compatibility - await since initDatabase is now async
  const connectionPool = await initDatabase(config);
  if (!connectionPool) {
    throw new Error("Database not configured for query execution.");
  }

  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    try {
      const startTime = Date.now();
      const results = await executeQuery(connectionPool, sql, args);
      const queryTime = Date.now() - startTime;

      if (attempt > 1) {
        console.log(
          `[OK] Query succeeded on attempt ${attempt} (${queryTime}ms)`
        );
      }

      return results;
    } catch (error) {
      const isLastAttempt = attempt === maxRetries;
      const isTimeoutError =
        error.code === "PROTOCOL_CONNECTION_LOST" ||
        error.code === "ECONNRESET" ||
        error.message.includes("timeout") ||
        error.message.includes("ETIMEDOUT");

      console.warn(`Query attempt ${attempt} failed:`, {
        error: error.message,
        code: error.code,
        isTimeout: isTimeoutError,
        isLastAttempt,
      });

      if (isLastAttempt) {
        throw error;
      }

      // Only retry on connection/timeout errors
      if (isTimeoutError) {
        const delay = Math.min(1000 * Math.pow(2, attempt - 1), 5000); // Exponential backoff, max 5s
        console.log(`Retrying in ${delay}ms...`);
        await new Promise((resolve) => setTimeout(resolve, delay));
      } else {
        // Don't retry on non-connection errors
        throw error;
      }
    }
  }
}

/**
 * Executes a single database query
 * @param {object} connectionPool - The MySQL connection pool
 * @param {string} sql - The SQL query string
 * @param {Array} args - Query parameters
 * @returns {Promise<Array>} Query results
 */
function executeQuery(connectionPool, sql, args) {
  return new Promise((resolve, reject) => {
    connectionPool.getConnection((err, connection) => {
      if (err) {
        console.error("Error getting database connection:", err);
        return reject(err);
      }

      connection.query(sql, args, (queryErr, results) => {
        connection.release(); // Release the connection back to the pool
        if (queryErr) {
          console.error("Error executing query:", queryErr);
          return reject(queryErr);
        }
        resolve(results);
      });
    });
  });
}

/**
 * Checks the health of the database connection
 * Supports both MySQL and BigQuery connection types.
 * @param {object} config - Service configuration object
 * @returns {Promise<object>} Health status object
 */
async function checkConnectionHealth(config) {
  // Handle BigQuery connection type
  if (config.database.connection_type === "bigquery") {
    return bigqueryDb.checkConnectionHealth(config);
  }

  // Default to MySQL for backward compatibility - await since initDatabase is now async
  const connectionPool = await initDatabase(config);
  if (!connectionPool) {
    return { healthy: false, error: "Database not configured" };
  }

  try {
    const startTime = Date.now();
    const results = await executeQuery(
      connectionPool,
      "SELECT 1 as health_check",
      []
    );
    const responseTime = Date.now() - startTime;

    return {
      healthy: true,
      responseTime: responseTime,
      poolStats: {
        totalConnections: connectionPool._allConnections?.length || 0,
        freeConnections: connectionPool._freeConnections?.length || 0,
        acquiringConnections: connectionPool._acquiringConnections?.length || 0,
      },
    };
  } catch (error) {
    return {
      healthy: false,
      error: error.message,
      code: error.code,
    };
  }
}

/**
 * Closes the database connection
 * Supports both MySQL and BigQuery connection types.
 */
function closePool() {
  // Handle BigQuery connection type
  if (bigqueryDb) {
    bigqueryDb.closeConnection();
  }

  // Handle MySQL connection type
  if (pool) {
    pool.end();
    pool = null;
    console.log("Database connection pool closed.");
  }
}

module.exports = {
  initDatabase,
  query,
  checkConnectionHealth,
  closePool,
  // Export BigQuery specific functions
  bigquery: bigqueryDb,
};
