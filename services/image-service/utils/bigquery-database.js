/**
 * BigQuery database connection and query execution module
 * Provides BigQuery integration for AWS Lambda functions
 */

const AWS = require("aws-sdk");
console.log("🔍 AWS SDK version:", AWS.VERSION);
console.log("🔍 AWS SDK available:", typeof AWS !== "undefined");

// Import BigQuery at the top - this will fail fast if not available
const { BigQuery } = require("@google-cloud/bigquery");
console.log("✅ BigQuery module loaded successfully");

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
      console.log("🔍 Creating AWS Secrets Manager client...");
      console.log("🔍 AWS Region:", process.env.AWS_REGION || "us-east-1");
      console.log("🔍 AWS SDK available:", typeof AWS !== "undefined");

      secretsClient = new AWS.SecretsManager({
        region: process.env.AWS_REGION || "us-east-1",
      });
      console.log("✅ AWS Secrets Manager client created successfully");
    } catch (error) {
      console.error("❌ AWS Secrets Manager not available:", error.message);
      console.error("❌ Error name:", error.name);
      console.error("❌ Error code:", error.code);
      console.error("❌ Full error:", JSON.stringify(error, null, 2));
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
    console.log("🔍 Getting AWS Secrets Manager client...");
    const secretsClient = getSecretsClient();
    if (!secretsClient) {
      throw new Error("AWS Secrets Manager client not available");
    }
    console.log("✅ AWS Secrets Manager client obtained");

    console.log("🔍 Retrieving secret from AWS Secrets Manager...");
    console.log("🔍 Secret ID:", secretName);
    console.log("🔍 AWS Region:", process.env.AWS_REGION || "us-east-1");

    const response = await secretsClient
      .getSecretValue({ SecretId: secretName })
      .promise();
    console.log("✅ Secret retrieved from AWS Secrets Manager");
    console.log("🔍 Secret ARN:", response.ARN);
    console.log("🔍 Secret name:", response.Name);
    console.log("🔍 Secret version ID:", response.VersionId);

    console.log("🔍 Parsing credentials JSON...");
    const credentialsJson = JSON.parse(response.SecretString);
    console.log("✅ Credentials JSON parsed successfully");

    console.log("🔍 Getting BigQuery module...");
    // BigQuery is already imported at the top of the file
    console.log("✅ BigQuery module obtained");

    console.log("🔍 Creating credentials object...");
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
    console.log("✅ Credentials object created");

    console.log("🔍 Creating BigQuery client...");
    const client = new BigQuery({
      credentials: credentials,
      projectId: credentialsJson.project_id,
      scopes: ["https://www.googleapis.com/auth/bigquery"],
      // Add timeout settings
      timeout: 30000, // 30 seconds
    });
    console.log("✅ BigQuery client created");

    console.log("BigQuery client initialized successfully");
    return client;
  } catch (error) {
    console.error("❌ Error initializing BigQuery client:", error);
    console.error("❌ Error name:", error.name);
    console.error("❌ Error message:", error.message);
    console.error("❌ Error code:", error.code);
    console.error("❌ Error stack:", error.stack);
    console.error("❌ Full error object:", JSON.stringify(error, null, 2));
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
    console.log("🔍 Using cached BigQuery client");
    return clientCache;
  }

  console.log("🔍 Initializing new BigQuery client...");
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

  console.log("🔍 Secret name:", secretName);
  console.log("🔍 Environment variables:", {
    BIGQUERY_SECRET_ARN: process.env.BIGQUERY_SECRET_ARN,
    GCP_SECRET_NAME: process.env.GCP_SECRET_NAME,
    AWS_REGION: process.env.AWS_REGION,
  });

  try {
    clientCache = await getBigQueryClient(secretName);
    console.log("✅ BigQuery client initialized successfully");
    return clientCache;
  } catch (error) {
    console.error("❌ Failed to initialize BigQuery client:", error);
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
      const queryConfig = {
        query: sql,
        useQueryCache: options.useCache !== false,
        useLegacySql: false,
        dryRun: options.dryRun || false,
        maxResults: options.limit || null,
        jobTimeoutMs: options.timeout || 30000,
      };

      // Add parameterized query support if args are provided
      if (args && args.length > 0) {
        queryConfig.queryParameters = args.map((arg, index) => ({
          name: `param_${index}`,
          parameterType: { type: getBigQueryType(arg) },
          parameterValue: { value: String(arg) },
        }));
      }

      console.log("Executing BigQuery:", {
        query: sql.substring(0, 200) + (sql.length > 200 ? "..." : ""),
        parameters: args.length,
        attempt: attempt,
      });

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
        console.log(
          `✅ BigQuery succeeded on attempt ${attempt} (${queryTime}ms)`
        );
      }

      console.log(
        `BigQuery query completed: ${results.length} rows, ${queryTime}ms`
      );
      return results;
    } catch (error) {
      const isLastAttempt = attempt === maxRetries;
      const isRetryableError = isRetryableBigQueryError(error);

      console.warn(`BigQuery attempt ${attempt} failed:`, {
        error: error.message,
        code: error.code,
        isRetryable: isRetryableError,
        isLastAttempt,
      });

      if (isLastAttempt || !isRetryableError) {
        throw error;
      }

      // Exponential backoff for retryable errors
      const delay = Math.min(1000 * Math.pow(2, attempt - 1), 10000);
      console.log(`Retrying BigQuery in ${delay}ms...`);
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
  try {
    console.log("🔍 Starting BigQuery health check...");
    const client = await initBigQueryClient(config);
    console.log("✅ BigQuery client obtained for health check");

    // Test connection with a simple query
    console.log("🔍 Executing test query...");
    const testQuery = "SELECT 1 as test_value";
    const [rows] = await client.query(testQuery);
    console.log("✅ Test query executed successfully");

    return {
      status: "healthy",
      connectionType: "bigquery",
      projectId: client.projectId,
      testQuery: "success",
      timestamp: new Date().toISOString(),
    };
  } catch (error) {
    console.error("❌ BigQuery health check failed:", error);
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
  console.log("BigQuery client cache cleared");
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
