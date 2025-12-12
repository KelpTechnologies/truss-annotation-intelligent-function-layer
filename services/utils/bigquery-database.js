/**
 * BigQuery database connection and query execution module
 * Provides BigQuery integration for AWS Lambda functions
 */

const AWS = require("aws-sdk");
const { BigQuery } = require("@google-cloud/bigquery");
const { createLogger } = require("./structured-logger");

// Create fallback logger for initialization (no request context)
const initLogger = createLogger({
  serviceName: "bigquery-init",
  layer: "annotation-ifl",
});
const initContext = {
  requestId: "init",
  layer: "annotation-ifl",
  serviceName: "bigquery-init",
  route: "MODULE_INIT",
  routeNormalized: "MODULE_INIT",
};

initLogger.debug("AWS SDK version", { version: AWS.VERSION }, initContext);
initLogger.debug("BigQuery module loaded successfully", {}, initContext);

let secretsClient = null;

/**
 * Get BigQuery module (already loaded at startup)
 */
function getBigQuery() {
  return BigQuery;
}

/**
 * Lazy load AWS Secrets Manager client
 */
function getSecretsClient() {
  if (!secretsClient) {
    try {
      initLogger.debug(
        "Creating AWS Secrets Manager client",
        {
          region: process.env.AWS_REGION || "us-east-1",
        },
        initContext
      );

      secretsClient = new AWS.SecretsManager({
        region: process.env.AWS_REGION || "us-east-1",
      });
      initLogger.debug(
        "AWS Secrets Manager client created successfully",
        {},
        initContext
      );
    } catch (error) {
      initLogger.logError(initContext, error, {
        statusCode: 500,
      });
      return null;
    }
  }
  return secretsClient;
}

/**
 * Retrieves GCP credentials from AWS Secrets Manager and creates BigQuery client
 * @param {string} secretName - Name of the secret containing GCP service account credentials
 * @returns {Promise<object>} BigQuery client instance
 */
async function getBigQueryClient(secretName = "bigquery-service-account") {
  try {
    initLogger.debug("Getting AWS Secrets Manager client", {}, initContext);
    const secretsClient = getSecretsClient();
    if (!secretsClient) {
      throw new Error("AWS Secrets Manager client not available");
    }
    initLogger.debug("AWS Secrets Manager client obtained", {}, initContext);

    initLogger.debug(
      "Retrieving secret from AWS Secrets Manager",
      {
        secretId: secretName,
        region: process.env.AWS_REGION || "us-east-1",
      },
      initContext
    );

    const response = await secretsClient
      .getSecretValue({ SecretId: secretName })
      .promise();
    initLogger.debug(
      "Secret retrieved from AWS Secrets Manager",
      {
        arn: response.ARN,
        name: response.Name,
        versionId: response.VersionId,
      },
      initContext
    );

    initLogger.debug("Parsing credentials JSON", {}, initContext);
    const credentialsJson = JSON.parse(response.SecretString);
    initLogger.debug("Credentials JSON parsed successfully", {}, initContext);

    initLogger.debug("Creating credentials object", {}, initContext);
    const credentials = {
      type: "service_account",
      project_id: credentialsJson.project_id,
      private_key_id: credentialsJson.private_key_id,
      private_key: credentialsJson.private_key.replace(/\\n/g, "\n"),
      client_email: credentialsJson.client_email,
      client_id: credentialsJson.client_id,
      auth_uri: "https://accounts.google.com/o/oauth2/auth",
      token_uri: "https://oauth2.googleapis.com/token",
      auth_provider_x509_cert_url: "https://www.googleapis.com/oauth2/v1/certs",
      client_x509_cert_url: `https://www.googleapis.com/robot/v1/metadata/x509/${credentialsJson.client_email}`,
    };
    initLogger.debug("Credentials object created", {}, initContext);

    initLogger.debug("Creating BigQuery client", {}, initContext);
    const client = new BigQuery({
      credentials: credentials,
      projectId: credentialsJson.project_id,
      scopes: ["https://www.googleapis.com/auth/bigquery"],
      // Add timeout settings
      timeout: 30000, // 30 seconds
    });
    initLogger.debug("BigQuery client created", {}, initContext);

    initLogger.debug(
      "BigQuery client initialized successfully",
      {},
      initContext
    );
    return client;
  } catch (error) {
    initLogger.logError(initContext, error, {
      statusCode: 500,
    });
    throw error;
  }
}

/**
 * BigQuery client cache to avoid re-initialization
 */
let clientCache = null;

/**
 * Gets or creates a BigQuery client instance
 * @param {object} config - Service configuration object
 * @returns {Promise<object>} BigQuery client instance
 */
async function initBigQueryClient(config) {
  if (clientCache) {
    initLogger.debug("Using cached BigQuery client", {}, initContext);
    return clientCache;
  }

  initLogger.debug("Initializing new BigQuery client", {}, initContext);
  // Extract secret name from ARN if it's a full ARN
  let secretName =
    process.env.BIGQUERY_SECRET_ARN ||
    process.env.GCP_SECRET_NAME ||
    "bigquery-service-account";

  // If it's a full ARN, extract just the secret name
  if (secretName.startsWith("arn:aws:secretsmanager:")) {
    const arnParts = secretName.split(":");
    secretName = arnParts[arnParts.length - 1].replace(/-\w+$/, ""); // Remove the random suffix
  }

  initLogger.debug(
    "Secret name extracted",
    {
      secretName,
      bigquerySecretArn: process.env.BIGQUERY_SECRET_ARN,
      gcpSecretName: process.env.GCP_SECRET_NAME,
      awsRegion: process.env.AWS_REGION,
    },
    initContext
  );

  try {
    clientCache = await getBigQueryClient(secretName);
    initLogger.debug(
      "BigQuery client initialized successfully",
      {},
      initContext
    );
    return clientCache;
  } catch (error) {
    initLogger.logError(initContext, error, {
      statusCode: 500,
    });
    throw error;
  }
}

/**
 * Executes a BigQuery SQL query with retry logic
 * @param {string} sql - The SQL query string
 * @param {Array} args - Query parameters (for parameterized queries)
 * @param {object} config - Service configuration object
 * @param {object} options - Query options
 * @param {number} maxRetries - Maximum number of retry attempts (default: 3)
 * @returns {Promise<Array>} A promise that resolves with the query results
 * @throws {Error} If the database is not configured or a query error occurs
 */
async function query(sql, args = [], config, options = {}, maxRetries = 3) {
  if (!config.database.required) {
    throw new Error("Database not required for this service");
  }

  if (config.database.connection_type !== "bigquery") {
    throw new Error("BigQuery connection type not configured");
  }

  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    try {
      const client = await initBigQueryClient(config);
      const startTime = Date.now();

      // Configure query job
      // Normalize limit: convert empty strings, null, undefined to undefined
      // Only include maxResults if it's a valid positive number
      const normalizedLimit =
        options.limit !== undefined &&
        options.limit !== null &&
        options.limit !== "" &&
        !isNaN(options.limit) &&
        Number(options.limit) > 0
          ? Number(options.limit)
          : undefined;

      const queryConfig = {
        query: sql,
        useQueryCache: options.useCache !== false,
        useLegacySql: false,
        dryRun: options.dryRun || false,
        jobTimeoutMs: options.timeout || 30000,
      };

      // Only include maxResults if we have a valid positive number
      if (normalizedLimit !== undefined) {
        queryConfig.maxResults = normalizedLimit;
      }

      // Add parameterized query support if args are provided
      if (args && args.length > 0) {
        queryConfig.queryParameters = args.map((arg, index) => ({
          name: `param_${index}`,
          parameterType: { type: getBigQueryType(arg) },
          parameterValue: { value: String(arg) },
        }));
      }

      // Log query execution (debug level)
      const serviceName = config?.service?.name || "unknown";
      const logger = createLogger({ serviceName, layer: "annotation-ifl" });
      const queryContext = {
        requestId: null,
        serviceName,
        layer: "annotation-ifl",
        route: "BIGQUERY_QUERY",
        routeNormalized: "BIGQUERY_QUERY",
      };
      logger.debug(
        "Executing BigQuery query",
        {
          query: sql.substring(0, 200) + (sql.length > 200 ? "..." : ""),
          parameters: args.length,
          attempt: attempt,
        },
        queryContext
      );

      const [job] = await client.createQueryJob(queryConfig);
      const [rows] = await job.getQueryResults();

      const queryTime = Date.now() - startTime;
      const results = rows.map((row) => {
        const result = {};
        for (const [key, value] of Object.entries(row)) {
          result[key] = value;
        }
        return result;
      });

      if (attempt > 1) {
        logger.debug(
          `BigQuery succeeded on attempt ${attempt} (${queryTime}ms)`,
          {
            attempt,
            durationMs: queryTime,
          },
          queryContext
        );
      }

      logger.debug(
        `BigQuery query completed: ${results.length} rows, ${queryTime}ms`,
        {
          rowCount: results.length,
          durationMs: queryTime,
        },
        queryContext
      );
      return results;
    } catch (error) {
      const isLastAttempt = attempt === maxRetries;
      const isRetryableError = isRetryableBigQueryError(error);

      const serviceName = config?.service?.name || "unknown";
      const logger = createLogger({ serviceName, layer: "annotation-ifl" });
      const errorContext = {
        requestId: null,
        serviceName,
        layer: "annotation-ifl",
        route: "BIGQUERY_QUERY",
        routeNormalized: "BIGQUERY_QUERY",
      };

      if (isLastAttempt || !isRetryableError) {
        // Log error for final attempt or non-retryable errors
        logger.logError(errorContext, error, {
          statusCode: 500,
          bigquery: {
            queryPreview:
              sql.substring(0, 200) + (sql.length > 200 ? "..." : ""),
            parametersCount: args.length,
            errorCode: error.code,
            attempt,
            isRetryable: isRetryableError,
            isLastAttempt,
          },
        });
        throw error;
      }

      // Log warning for retryable errors
      const delay = Math.min(1000 * Math.pow(2, attempt - 1), 10000);
      logger.logWarning(
        errorContext,
        `BigQuery attempt ${attempt} failed, retrying in ${delay}ms`,
        {
          attempt,
          maxRetries,
          retryDelayMs: delay,
          errorCode: error.code,
          isRetryable: isRetryableError,
        }
      );
      await new Promise((resolve) => setTimeout(resolve, delay));
    }
  }
}

/**
 * Determines if a BigQuery error is retryable
 * @param {Error} error - The error to check
 * @returns {boolean} True if the error is retryable
 */
function isRetryableBigQueryError(error) {
  const retryableCodes = [
    "RATE_LIMIT_EXCEEDED",
    "QUERY_TIMEOUT",
    "INTERNAL_ERROR",
    "SERVICE_UNAVAILABLE",
    "DEADLINE_EXCEEDED",
  ];

  const retryableMessages = [
    "timeout",
    "rate limit",
    "quota exceeded",
    "service unavailable",
    "internal error",
    "deadline exceeded",
  ];

  return (
    retryableCodes.includes(error.code) ||
    retryableMessages.some((msg) => error.message.toLowerCase().includes(msg))
  );
}

/**
 * Gets BigQuery parameter type for a value
 * @param {any} value - The value to get type for
 * @returns {string} BigQuery type string
 */
function getBigQueryType(value) {
  if (typeof value === "number") {
    return Number.isInteger(value) ? "INT64" : "FLOAT64";
  }
  if (typeof value === "boolean") {
    return "BOOL";
  }
  if (value instanceof Date) {
    return "TIMESTAMP";
  }
  return "STRING";
}

/**
 * Lists datasets in the BigQuery project
 * @param {object} config - Service configuration object
 * @returns {Promise<Array>} List of datasets
 */
async function listDatasets(config) {
  const client = await initBigQueryClient(config);
  const [datasets] = await client.getDatasets();

  return datasets.map((dataset) => ({
    datasetId: dataset.id,
    fullId: dataset.metadata.fullId,
    location: dataset.metadata.location,
    created: dataset.metadata.created,
    modified: dataset.metadata.modified,
  }));
}

/**
 * Lists tables in a BigQuery dataset
 * @param {string} datasetId - The dataset ID
 * @param {object} config - Service configuration object
 * @returns {Promise<Array>} List of tables
 */
async function listTables(datasetId, config) {
  const client = await initBigQueryClient(config);
  const dataset = client.dataset(datasetId);
  const [tables] = await dataset.getTables();

  return tables.map((table) => ({
    tableId: table.id,
    fullId: table.metadata.fullId,
    type: table.metadata.type,
    created: table.metadata.created,
    modified: table.metadata.modified,
    numRows: table.metadata.numRows,
    numBytes: table.metadata.numBytes,
  }));
}

/**
 * Gets table schema information
 * @param {string} datasetId - The dataset ID
 * @param {string} tableId - The table ID
 * @param {object} config - Service configuration object
 * @returns {Promise<object>} Table schema and metadata
 */
async function getTableSchema(datasetId, tableId, config) {
  const client = await initBigQueryClient(config);
  const table = client.dataset(datasetId).table(tableId);
  const [metadata] = await table.getMetadata();

  return {
    schema: metadata.schema.fields.map((field) => ({
      name: field.name,
      type: field.type,
      mode: field.mode,
      description: field.description,
    })),
    numRows: metadata.numRows,
    numBytes: metadata.numBytes,
    created: metadata.creationTime,
    modified: metadata.lastModifiedTime,
    location: metadata.location,
    type: metadata.type,
  };
}

/**
 * Checks the health of the BigQuery connection
 * @param {object} config - Service configuration object
 * @returns {Promise<object>} Health status object
 */
async function checkConnectionHealth(config) {
  const serviceName = config?.service?.name || "unknown";
  const logger = createLogger({ serviceName, layer: "annotation-ifl" });
  const healthContext = {
    requestId: null,
    serviceName,
    layer: "annotation-ifl",
    route: "BIGQUERY_HEALTH_CHECK",
    routeNormalized: "BIGQUERY_HEALTH_CHECK",
  };

  try {
    logger.debug("Starting BigQuery health check", {}, healthContext);
    const client = await initBigQueryClient(config);
    logger.debug(
      "BigQuery client obtained for health check",
      {},
      healthContext
    );

    // Test connection with a simple query
    logger.debug("Executing test query", {}, healthContext);
    const testQuery = "SELECT 1 as test_value";
    const [rows] = await client.query(testQuery);
    logger.debug("Test query executed successfully", {}, healthContext);

    return {
      status: "healthy",
      connectionType: "bigquery",
      projectId: client.projectId,
      testQuery: "success",
      timestamp: new Date().toISOString(),
    };
  } catch (error) {
    logger.logError(healthContext, error, {
      statusCode: 500,
    });
    return {
      status: "unhealthy",
      connectionType: "bigquery",
      error: error.message,
      timestamp: new Date().toISOString(),
    };
  }
}

/**
 * Closes the BigQuery client connection
 * Note: BigQuery client doesn't require explicit closing, but we clear the cache
 */
function closeConnection() {
  clientCache = null;
  initLogger.debug("BigQuery client cache cleared", {}, initContext);
}

module.exports = {
  initBigQueryClient,
  query,
  listDatasets,
  listTables,
  getTableSchema,
  checkConnectionHealth,
  closeConnection,
  getBigQueryType,
  isRetryableBigQueryError,
};
