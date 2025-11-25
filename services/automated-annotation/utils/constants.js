/**
 * Common constants used across all services
 */

// Database configuration
const DATABASE_CONFIG = {
  NAME: "api_staging",
  CONNECTION_LIMIT: 10,
  ACQUIRE_TIMEOUT: 60000,
  TIMEOUT: 60000,
  SSL: false,
};

// Pagination defaults
const PAGINATION = {
  DEFAULT_LIMIT: 500,
  MAX_LIMIT: 1000,
  DEFAULT_OFFSET: 0,
};

// Sort orders
const SORT_ORDERS = {
  ASC: "ASC",
  DESC: "DESC",
};

// Query modes for confidence metrics
const QUERY_MODES = {
  BASIC: "basic",
  BASIC_DIAGNOSTICS: "basic_diagnostics",
};

// Confidence metrics configuration
const CONFIDENCE_METRICS = {
  DEFAULT_MODE: "basic",
  Z_SCORE_95: 1.96,
  Z_SCORE_99: 2.58,
  CONFIDENCE_LEVEL_95: 0.95,
  CONFIDENCE_LEVEL_99: 0.99,
};

// Valid group by fields mapping
const GROUP_BY_FIELDS = {
  brand: "brand",
  type: "type",
  material: "material",
  color: "colour",
  condition: "condition",
  size: "size",
  vendor: "vendor",
  gender: "gender",
  model: "model",
  decade: "decade",
  location: "sold_location",
  hardware: "hardware",
  monthly: "listed_date",
};
const FIELD_MAPPINGS = {
  brand: "brand",
  type: "type",
  material: "material",
  color: "colour",
  condition: "condition",
  size: "size",
  vendor: "vendor",
  gender: "gender",
  model: "model",
  decade: "decade",
  location: "sold_location",
  hardware: "hardware",
  key_word: "listing_title", // maps to listing_title for text search
  monthly: "listed_date",
};

// Item types for filtering
const ITEM_TYPES = {
  SOLD: "sold",
  LISTED: "listed",
  ALL: "all",
};

// Price fields
const PRICE_FIELDS = {
  SOLD_PRICE: "sold_price",
  LISTED_PRICE: "listed_price",
};

// Common aggregation functions
const AGGREGATION_FUNCTIONS = {
  SUM_SOLD_PRICE: "SUM(sold_price) AS value",
  COUNT_ALL: "COUNT(*) AS value",
  AVG_SOLD_PRICE: "AVG(sold_price) AS value",
  AVG_LISTED_PRICE: "AVG(listed_price) AS value",
  AVG_DISCOUNT: "AVG(discount) AS value",
};

// Confidence metrics aggregation functions
const CONFIDENCE_AGGREGATION_FUNCTIONS = {
  BASIC: "AVG(x) AS value, COUNT(*) AS count",
  COMPREHENSIVE: `
    AVG(x) AS final_value,
    COUNT(*) AS n,
    VAR_POP(x) AS variance,
    STDDEV_POP(x) AS stddev,
    STDDEV_POP(x) / SQRT(COUNT(*)) AS sem,
    AVG(x) - 1.96 * STDDEV_POP(x) / SQRT(COUNT(*)) AS ci95_lower,
    AVG(x) + 1.96 * STDDEV_POP(x) / SQRT(COUNT(*)) AS ci95_upper,
    MIN(x) AS min_val,
    MAX(x) AS max_val,
    (MAX(x) - MIN(x)) AS range_val,
    CASE WHEN AVG(x) = 0 THEN NULL ELSE STDDEV_POP(x) / AVG(x) END AS cv,
    CASE WHEN STDDEV_POP(x) = 0 THEN NULL ELSE
      (AVG(POWER(x,3)) - 3*AVG(x)*AVG(POWER(x,2)) + 2*POWER(AVG(x),3))
      / POWER(STDDEV_POP(x), 3)
    END AS skewness,
    CASE WHEN STDDEV_POP(x) = 0 THEN NULL ELSE
      (AVG(POWER(x,4))
        - 4*AVG(x)*AVG(POWER(x,3))
        + 6*POWER(AVG(x),2)*AVG(POWER(x,2))
        - 3*POWER(AVG(x),4))
      / POWER(STDDEV_POP(x), 4) - 3
    END AS kurtosis_excess
  `
    .replace(/\s+/g, " ")
    .trim(),
};

// HTTP status codes
const HTTP_STATUS = {
  OK: 200,
  BAD_REQUEST: 400,
  NOT_FOUND: 404,
  INTERNAL_SERVER_ERROR: 500,
};

// Error messages
const ERROR_MESSAGES = {
  INVALID_LIMIT: "Invalid limit parameter. Must be a positive integer.",
  INVALID_OFFSET: "Invalid offset parameter. Must be a non-negative integer.",
  INVALID_ORDER: "Invalid order parameter. Must be 'ASC' or 'DESC'.",
  INVALID_QUERY_MODE:
    "Invalid query_mode parameter. Must be 'basic' or 'basic_diagnostics'.",
  DATABASE_NOT_CONFIGURED: "Database not configured for query execution.",
  DATABASE_HOST_MISSING:
    "Database host (RDS_PROXY_ENDPOINT or DB_HOST) environment variable is not set.",
  DATABASE_USER_MISSING:
    "Database user (DB_USER) environment variable is not set.",
  DATABASE_PASSWORD_MISSING:
    "Database password (DB_PASSWORD) environment variable is not set.",
  ENDPOINT_NOT_FOUND: "Endpoint not found",
  INTERNAL_SERVER_ERROR: "Internal server error",
  BAD_REQUEST: "Bad Request",
};

// Component types for responses
const COMPONENT_TYPES = {
  GMV_ANALYSIS: "gmv_analysis",
  MARKET_SHARE_ANALYSIS: "market_share_analysis",
  SOLD_COUNT_ANALYSIS: "sold_count_analysis",
  LISTED_COUNT_ANALYSIS: "listed_count_analysis",
  SELL_THROUGH_ANALYSIS: "sell_through_analysis",
  DISCOUNT_ANALYSIS: "discount_analysis",
  AVERAGE_SOLD_PRICE_ANALYSIS: "average_sold_price_analysis",
  AVERAGE_LISTED_PRICE_ANALYSIS: "average_listed_price_analysis",
  DAYS_TO_SELL_ANALYSIS: "days_to_sell_analysis",
  SOLD_COUNT_MONTHLY_ANALYSIS: "sold_count_monthly_analysis",
  LISTED_COUNT_MONTHLY_ANALYSIS: "listed_count_monthly_analysis",
  SELL_THROUGH_MONTHLY_ANALYSIS: "sell_through_monthly_analysis",
};

// Data filtering thresholds
const FILTER_THRESHOLDS = {
  DEFAULT_PERCENTAGE: 5,
  MIN_PERCENTAGE: 1,
  MAX_PERCENTAGE: 50,
};

// Data processing options
const DATA_PROCESSING = {
  DEFAULT_RETURN_NULLS: true,
};

// CORS headers
const CORS_HEADERS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers":
    "Content-Type,X-Amz-Date,Authorization,X-Api-Key",
  "Access-Control-Allow-Methods": "GET,OPTIONS",
  "Content-Type": "application/json",
};

module.exports = {
  DATABASE_CONFIG,
  PAGINATION,
  SORT_ORDERS,
  QUERY_MODES,
  CONFIDENCE_METRICS,
  GROUP_BY_FIELDS,
  ITEM_TYPES,
  PRICE_FIELDS,
  AGGREGATION_FUNCTIONS,
  CONFIDENCE_AGGREGATION_FUNCTIONS,
  HTTP_STATUS,
  ERROR_MESSAGES,
  COMPONENT_TYPES,
  FILTER_THRESHOLDS,
  DATA_PROCESSING,
  CORS_HEADERS,
  FIELD_MAPPINGS,
};
