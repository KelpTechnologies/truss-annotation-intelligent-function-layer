/**
 * AUTO-GENERATED FILE - DO NOT EDIT DIRECTLY
 * 
 * This file is managed by truss-api-platform/logging-toolkit
 * To update, run: npm run copy:toolkit from truss-api-platform
 * 
 * Source: truss-api-platform/logging-toolkit/structured-logger.js
 * Generated: 2025-12-03T20:38:57.984Z
 */

/**
 * Structured Logger - Standardized logging toolkit for all Truss services
 * Emits logs in schema v2 format for downstream processing by the log monitor
 *
 * @version 2.0.0
 */

const LOG_SCHEMA_VERSION = 2;

/**
 * Log types
 */
const LOG_TYPES = {
  REQUEST: "REQUEST",
  RESPONSE: "RESPONSE",
  ERROR: "ERROR",
  METRIC: "METRIC",
  DEBUG: "DEBUG",
};

/**
 * Log levels
 */
const LOG_LEVELS = {
  ERROR: 0,
  WARN: 1,
  INFO: 2,
  DEBUG: 3,
};

/**
 * Layer detection mapping
 */
const LAYER_MAP = {
  "/automations/annotation": "annotation-ifl",
  "/automations/pricing": "pricing-ifl",
  "/images": "annotation-ifl",
  "/knowledge": "annotation-dsl",
  "/visual-classifier": "annotation-dsl",
  "/footwear": "dsl",
  "/bags": "dsl",
  "/apparel": "dsl",
  "/analytics": "dsl",
  "/prices": "dsl",
  "/products": "dsl",
  "/brands": "dsl",
  "/beta": "dsl",
  "/discounts": "dsl",
  "/pricing": "pricing-ifl",
};

const SORTED_PREFIXES = Object.keys(LAYER_MAP).sort(
  (a, b) => b.length - a.length
);

/**
 * Default fields to redact from logs
 */
const DEFAULT_REDACT_FIELDS = [
  "password",
  "apiKey",
  "api_key",
  "x-api-key",
  "authorization",
  "token",
  "secret",
  "creditCard",
  "ssn",
  "image", // Often contains base64 or URLs
];

/**
 * Current log level from environment
 */
const currentLogLevel = process.env.LOG_LEVEL
  ? LOG_LEVELS[process.env.LOG_LEVEL.toUpperCase()] || LOG_LEVELS.INFO
  : LOG_LEVELS.INFO;

const MAX_RESPONSE_SIZE = parseInt(
  process.env.LOG_MAX_RESPONSE_SIZE || "10000",
  10
);
const REDACT_RESPONSE =
  (process.env.LOG_REDACT_RESPONSE || "true").toLowerCase() === "true";

/**
 * Detect layer from resource path
 */
function detectLayer(resourcePath) {
  if (!resourcePath || typeof resourcePath !== "string") {
    return "unknown";
  }

  const normalizedPath = resourcePath.startsWith("/")
    ? resourcePath
    : `/${resourcePath}`;

  for (const prefix of SORTED_PREFIXES) {
    if (
      normalizedPath.startsWith(prefix) &&
      (normalizedPath.length === prefix.length ||
        normalizedPath[prefix.length] === "/")
    ) {
      return LAYER_MAP[prefix];
    }
  }

  return "unknown";
}

/**
 * Normalize route by replacing dynamic segments with placeholders
 */
function normalizeRoute(method, path, pathParameters = {}) {
  if (!path) {
    return method ? `${method} /` : "/";
  }

  let normalized = path;

  // Replace known path parameter values
  Object.entries(pathParameters).forEach(([key, value]) => {
    if (value) {
      const escaped = value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
      normalized = normalized.replace(
        new RegExp(`/${escaped}(?=/|$)`, "g"),
        `/{${key}}`
      );
    }
  });

  // Common patterns to normalize
  const patterns = [
    // UUIDs
    {
      regex: /\/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}/gi,
      replacement: "/{id}",
    },
    // Numeric IDs
    { regex: /\/\d+(?=\/|$)/g, replacement: "/{id}" },
    // Processing IDs
    { regex: /\/proc_[a-zA-Z0-9]+/g, replacement: "/{processingId}" },
  ];

  patterns.forEach(({ regex, replacement }) => {
    normalized = normalized.replace(regex, replacement);
  });

  return method ? `${method} ${normalized}` : normalized;
}

/**
 * Deep redact sensitive fields from an object
 */
function redactSensitiveFields(obj, fieldsToRedact = DEFAULT_REDACT_FIELDS) {
  if (!obj || typeof obj !== "object") {
    return obj;
  }

  if (Array.isArray(obj)) {
    return obj.map((item) => redactSensitiveFields(item, fieldsToRedact));
  }

  const redacted = {};
  for (const [key, value] of Object.entries(obj)) {
    const lowerKey = key.toLowerCase();

    // Check if field should be redacted
    if (fieldsToRedact.some((field) => lowerKey.includes(field.toLowerCase()))) {
      redacted[key] = "[REDACTED]";
      continue;
    }

    // Check for URL patterns (likely image URLs)
    if (
      typeof value === "string" &&
      (value.startsWith("http://") ||
        value.startsWith("https://") ||
        value.startsWith("s3://"))
    ) {
      redacted[key] = "[REDACTED_URL]";
      continue;
    }

    // Check for base64 content
    if (
      typeof value === "string" &&
      value.length > 100 &&
      /^[A-Za-z0-9+/=]+$/.test(value)
    ) {
      redacted[key] = "[REDACTED_BASE64]";
      continue;
    }

    // Recurse into nested objects
    if (typeof value === "object" && value !== null) {
      redacted[key] = redactSensitiveFields(value, fieldsToRedact);
    } else {
      redacted[key] = value;
    }
  }

  return redacted;
}

/**
 * Truncate large objects for logging
 */
function truncateForLogging(obj, maxSize = MAX_RESPONSE_SIZE) {
  if (!obj) return obj;

  const stringified = JSON.stringify(obj);
  if (stringified.length <= maxSize) {
    return obj;
  }

  return {
    _truncated: true,
    _originalSize: stringified.length,
    _preview: stringified.substring(0, 500) + "...",
  };
}

/**
 * Generate unique request ID
 */
function generateRequestId() {
  return `req_${Date.now()}_${Math.random().toString(36).substring(2, 11)}`;
}

/**
 * Extract user context from Lambda event
 */
function extractUserContext(event) {
  const authorizer = event.requestContext?.authorizer || {};

  return {
    userId: authorizer.keyId || authorizer.userId || authorizer.principalId || null,
    tenantId: authorizer.tenantId || null,
    authType: authorizer.authType || "unknown",
  };
}

/**
 * StructuredLogger class for consistent logging
 */
class StructuredLogger {
  constructor(options = {}) {
    this.layer = options.layer || null;
    this.serviceName = options.serviceName || null;
    this.redactFields = [...DEFAULT_REDACT_FIELDS, ...(options.redactFields || [])];
    this.requestTimings = new Map();
  }

  /**
   * Start tracking a request
   * @param {object} event - Lambda event
   * @returns {object} Request context for subsequent logs
   */
  startRequest(event) {
    const requestId =
      event.requestContext?.requestId ||
      event.headers?.["x-request-id"] ||
      generateRequestId();

    const method = event.httpMethod || event.requestContext?.http?.method || "";
    const path = event.path || event.requestContext?.http?.path || "";
    const pathParams = event.pathParameters || {};

    const userContext = extractUserContext(event);
    const layer = this.layer || detectLayer(path);

    const context = {
      requestId,
      method,
      path,
      route: `${method} ${path}`,
      routeNormalized: normalizeRoute(method, path, pathParams),
      layer,
      serviceName: this.serviceName,
      userId: userContext.userId,
      tenantId: userContext.tenantId,
      authType: userContext.authType,
      startTime: Date.now(),
      queryParams: event.queryStringParameters || {},
      pathParams,
    };

    this.requestTimings.set(requestId, context.startTime);

    // Emit REQUEST log
    this._emit(LOG_TYPES.REQUEST, context, {
      queryParams: context.queryParams,
      pathParams: context.pathParams,
    });

    return context;
  }

  /**
   * Log a successful response
   */
  logResponse(requestContext, options = {}) {
    const durationMs = Date.now() - (requestContext.startTime || Date.now());
    this.requestTimings.delete(requestContext.requestId);

    let responseData = options.response;
    if (REDACT_RESPONSE && responseData) {
      responseData = redactSensitiveFields(responseData, this.redactFields);
      responseData = truncateForLogging(responseData);
    }

    this._emit(LOG_TYPES.RESPONSE, requestContext, {
      statusCode: options.statusCode || 200,
      durationMs,
      count: options.count || null,
      queryComplexity: options.queryComplexity || null,
      response: responseData,
    });
  }

  /**
   * Log an error
   */
  logError(requestContext, error, options = {}) {
    const durationMs = Date.now() - (requestContext.startTime || Date.now());
    this.requestTimings.delete(requestContext.requestId);

    this._emit(LOG_TYPES.ERROR, requestContext, {
      statusCode: options.statusCode || 500,
      durationMs,
      error: {
        message: error.message || String(error),
        name: error.name || "Error",
        stack: error.stack?.substring(0, 1000) || null,
        code: error.code || null,
      },
    });
  }

  /**
   * Log a custom metric
   */
  logMetric(requestContext, metricName, value, unit = "Count") {
    this._emit(LOG_TYPES.METRIC, requestContext, {
      metricName,
      value,
      unit,
    });
  }

  /**
   * Log debug information (only if LOG_LEVEL is DEBUG)
   */
  debug(message, data = {}, requestContext = null) {
    if (currentLogLevel < LOG_LEVELS.DEBUG) return;

    this._emit(
      LOG_TYPES.DEBUG,
      requestContext || { requestId: null },
      {
        message,
        ...data,
      }
    );
  }

  /**
   * Internal emit function
   */
  _emit(logType, requestContext, additionalFields = {}) {
    const payload = {
      schemaVersion: LOG_SCHEMA_VERSION,
      logType,
      ts: new Date().toISOString(),
      requestId: requestContext.requestId || null,
      layer: requestContext.layer || this.layer || "unknown",
      serviceName: requestContext.serviceName || this.serviceName || null,
      route: requestContext.route || null,
      routeNormalized: requestContext.routeNormalized || null,
      userId: requestContext.userId || null,
      tenantId: requestContext.tenantId || null,
      authType: requestContext.authType || "unknown",
      ...additionalFields,
    };

    try {
      console.log(JSON.stringify(payload));
    } catch (err) {
      console.error("Failed to emit structured log:", err.message);
    }
  }
}

/**
 * Calculate query complexity based on parameters
 */
function calculateQueryComplexity(params) {
  let complexity = 1;

  const filterParams = [
    "brands",
    "types",
    "materials",
    "colors",
    "conditions",
    "sizes",
    "vendors",
    "models",
    "genders",
  ];

  filterParams.forEach((param) => {
    if (params[param]) {
      let values;
      if (Array.isArray(params[param])) {
        values = params[param].length;
      } else if (typeof params[param] === "string") {
        values = params[param].split(",").length;
      } else {
        values = 1;
      }
      complexity += Math.min(values, 3);
    }
  });

  if (params.group_by) {
    let groups;
    if (Array.isArray(params.group_by)) {
      groups = params.group_by.length;
    } else if (typeof params.group_by === "string") {
      groups = params.group_by.split(",").length;
    } else {
      groups = 1;
    }
    complexity += groups * 2;
  }

  if (complexity <= 3) return "low";
  if (complexity <= 6) return "medium";
  if (complexity <= 10) return "high";
  return "very_high";
}

/**
 * Create a logger instance with service context
 */
function createLogger(options = {}) {
  return new StructuredLogger(options);
}

module.exports = {
  LOG_SCHEMA_VERSION,
  LOG_TYPES,
  LOG_LEVELS,
  LAYER_MAP,
  StructuredLogger,
  createLogger,
  detectLayer,
  normalizeRoute,
  redactSensitiveFields,
  truncateForLogging,
  generateRequestId,
  extractUserContext,
  calculateQueryComplexity,
};

