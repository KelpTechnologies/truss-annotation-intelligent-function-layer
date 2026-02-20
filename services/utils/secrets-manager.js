/**
 * Centralized Secrets Manager for Truss Platform
 * Retrieves and caches secrets from AWS Secrets Manager
 *
 * All secrets are stored in a single consolidated secret:
 * arn:aws:secretsmanager:eu-west-2:193757560043:secret:truss-platform-secrets-yVuz1R
 *
 * Secret structure:
 * {
 *   "bigquery": { project_id, private_key_id, private_key, client_email, client_id },
 *   "openai": { api_key },
 *   "pinecone": { api_key },
 *   "database": { host, user, password, name },
 *   "internal_apis": { dsl_api_key, annotation_dsl_api_key, vectorization_api_key }
 * }
 */

const { SecretsManagerClient, GetSecretValueCommand } = require("@aws-sdk/client-secrets-manager");

// Default ARN - can be overridden via environment variable
const DEFAULT_SECRET_ARN =
  "arn:aws:secretsmanager:eu-west-2:193757560043:secret:truss-platform-secrets-yVuz1R";

let secretsClient = null;
let cachedSecrets = null;
let cacheTimestamp = null;
const CACHE_TTL_MS = 5 * 60 * 1000; // 5 minutes cache TTL

/**
 * Get or create AWS Secrets Manager client
 * @returns {SecretsManagerClient} Secrets Manager client
 */
function getSecretsClient() {
  if (!secretsClient) {
    secretsClient = new SecretsManagerClient({
      region: process.env.AWS_REGION || "eu-west-2",
    });
  }
  return secretsClient;
}

/**
 * Check if cached secrets are still valid
 * @returns {boolean} True if cache is valid
 */
function isCacheValid() {
  if (!cachedSecrets || !cacheTimestamp) {
    return false;
  }
  return Date.now() - cacheTimestamp < CACHE_TTL_MS;
}

/**
 * Retrieve all secrets from AWS Secrets Manager
 * Results are cached for performance
 *
 * @returns {Promise<object>} Parsed secrets object
 */
async function getSecrets() {
  // Return cached secrets if still valid
  if (isCacheValid()) {
    return cachedSecrets;
  }

  const secretArn = process.env.TRUSS_SECRETS_ARN || DEFAULT_SECRET_ARN;
  const client = getSecretsClient();

  try {
    console.log("🔐 Retrieving secrets from AWS Secrets Manager...");
    const response = await client.send(
      new GetSecretValueCommand({ SecretId: secretArn })
    );

    cachedSecrets = JSON.parse(response.SecretString);
    cacheTimestamp = Date.now();

    console.log("✅ Secrets retrieved and cached successfully");
    return cachedSecrets;
  } catch (error) {
    console.error("❌ Failed to retrieve secrets:", error.message);
    throw new Error(
      `Failed to retrieve secrets from Secrets Manager: ${error.message}`
    );
  }
}

/**
 * Get BigQuery credentials from secrets
 * @returns {Promise<object>} BigQuery service account credentials
 */
async function getBigQueryCredentials() {
  const secrets = await getSecrets();
  if (!secrets.bigquery) {
    throw new Error("BigQuery credentials not found in secrets");
  }
  return secrets.bigquery;
}

/**
 * Get OpenAI API key from secrets
 * @returns {Promise<string>} OpenAI API key
 */
async function getOpenAIApiKey() {
  const secrets = await getSecrets();
  if (!secrets.openai?.api_key) {
    throw new Error("OpenAI API key not found in secrets");
  }
  return secrets.openai.api_key;
}

/**
 * Get Pinecone API key from secrets
 * @returns {Promise<string>} Pinecone API key
 */
async function getPineconeApiKey() {
  const secrets = await getSecrets();
  if (!secrets.pinecone?.api_key) {
    throw new Error("Pinecone API key not found in secrets");
  }
  return secrets.pinecone.api_key;
}

/**
 * Get database credentials from secrets
 * @returns {Promise<object>} Database credentials { host, user, password, name }
 */
async function getDatabaseCredentials() {
  const secrets = await getSecrets();
  if (!secrets.database) {
    throw new Error("Database credentials not found in secrets");
  }
  return secrets.database;
}

/**
 * Get internal API keys from secrets
 * @returns {Promise<object>} Internal API keys
 */
async function getInternalApiKeys() {
  const secrets = await getSecrets();
  return secrets.internal_apis || {};
}

/**
 * Clear the secrets cache (useful for testing or forced refresh)
 */
function clearCache() {
  cachedSecrets = null;
  cacheTimestamp = null;
  console.log("🔄 Secrets cache cleared");
}

module.exports = {
  getSecrets,
  getBigQueryCredentials,
  getOpenAIApiKey,
  getPineconeApiKey,
  getDatabaseCredentials,
  getInternalApiKeys,
  clearCache,
};
