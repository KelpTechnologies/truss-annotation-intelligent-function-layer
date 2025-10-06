/**
 * Enhanced response utilities for robust API responses
 * Based on the API conceptual guide response structure requirements
 */

// Security headers for enhanced protection
const SECURITY_HEADERS = {
  "X-Content-Type-Options": "nosniff",
  "X-Frame-Options": "DENY",
  "X-XSS-Protection": "1; mode=block",
  "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
  "Content-Security-Policy": "default-src 'self'",
  "Referrer-Policy": "strict-origin-when-cross-origin",
};

// CORS configuration
const CORS_CONFIG = {
  allowedOrigins: process.env.ALLOWED_ORIGINS
    ? process.env.ALLOWED_ORIGINS.split(",")
    : ["*"],
  allowedMethods: ["GET", "OPTIONS"],
  allowedHeaders: [
    "Content-Type",
    "X-Amz-Date",
    "Authorization",
    "X-Api-Key",
    "X-Requested-With",
  ],
  maxAge: 86400, // 24 hours
};

/**
 * Generates comprehensive CORS headers with security.
 * @param {string} origin - Request origin
 * @returns {object} An object containing CORS headers.
 */
function getCorsHeaders(origin = "*") {
  const headers = {
    "Access-Control-Allow-Methods": CORS_CONFIG.allowedMethods.join(","),
    "Access-Control-Allow-Headers": CORS_CONFIG.allowedHeaders.join(","),
    "Access-Control-Max-Age": CORS_CONFIG.maxAge.toString(),
    "Content-Type": "application/json; charset=utf-8",
    ...SECURITY_HEADERS,
  };

  // Handle CORS origin
  if (CORS_CONFIG.allowedOrigins.includes("*")) {
    headers["Access-Control-Allow-Origin"] = "*";
  } else if (origin && CORS_CONFIG.allowedOrigins.includes(origin)) {
    headers["Access-Control-Allow-Origin"] = origin;
  } else {
    headers["Access-Control-Allow-Origin"] = CORS_CONFIG.allowedOrigins[0];
  }

  return headers;
}

/**
 * Creates a comprehensive success response with enhanced metadata
 * @param {object} data - Response data
 * @param {object} config - Service configuration
 * @param {string} endpoint - Endpoint name
 * @param {object} queryParams - Query parameters
 * @param {object} pagination - Pagination info
 * @param {string} componentType - Component type for the response
 * @param {object} additionalMetadata - Additional metadata to include
 * @returns {object} Formatted response object
 */
function createSuccessResponse(
  data,
  config,
  endpoint,
  queryParams,
  pagination = null,
  componentType = null,
  additionalMetadata = {}
) {
  const headers = getCorsHeaders();
  const timestamp = new Date().toISOString();

  // Build comprehensive metadata
  const metadata = {
    query_params: sanitizeMetadata(queryParams),
    service: config.service.name,
    endpoint: endpoint,
    generated_at: timestamp,
    version: config.service.version || "1.0.0",
    request_id: generateRequestId(),
    ...additionalMetadata,
  };

  // Build response body
  const responseBody = {
    data: data,
    metadata: metadata,
  };

  // Add component type if provided
  if (componentType) {
    responseBody.component_type = componentType;
  }

  // Add pagination if provided
  if (pagination) {
    responseBody.pagination = {
      limit: pagination.limit,
      offset: pagination.offset,
      total: pagination.total || null,
      has_more: pagination.has_more || false,
      next_offset: pagination.has_more
        ? pagination.offset + pagination.limit
        : null,
    };
  }

  // Add performance metrics if available
  if (additionalMetadata.execution_time_ms) {
    responseBody.metadata.execution_time_ms =
      additionalMetadata.execution_time_ms;
  }

  return {
    statusCode: 200,
    headers,
    body: JSON.stringify(responseBody, null, 2),
  };
}

/**
 * Creates a comprehensive error response with detailed error information
 * @param {Error} error - Error object
 * @param {object} config - Service configuration
 * @param {number} statusCode - HTTP status code (default: 500)
 * @param {object} additionalInfo - Additional error information
 * @returns {object} Formatted error response object
 */
function createErrorResponse(
  error,
  config,
  statusCode = 500,
  additionalInfo = {}
) {
  const headers = getCorsHeaders();
  const timestamp = new Date().toISOString();

  // Determine error type and status code
  const errorInfo = classifyError(error, statusCode);
  const finalStatusCode = errorInfo.statusCode;

  // Build comprehensive error response
  const errorResponse = {
    error: {
      type: errorInfo.type,
      message: error.message,
      code: errorInfo.code,
      details: errorInfo.details || null,
      timestamp: timestamp,
      request_id: generateRequestId(),
      service: config.service.name,
      endpoint: additionalInfo.endpoint || "unknown",
      ...additionalInfo,
    },
    metadata: {
      service: config.service.name,
      version: config.service.version || "1.0.0",
      generated_at: timestamp,
    },
  };

  // Add validation errors if available
  if (error.validationErrors) {
    errorResponse.error.validation_errors = error.validationErrors;
  }

  // Add stack trace in development
  if (process.env.NODE_ENV === "development" && error.stack) {
    errorResponse.error.stack = error.stack;
  }

  return {
    statusCode: finalStatusCode,
    headers,
    body: JSON.stringify(errorResponse, null, 2),
  };
}

/**
 * Creates a comprehensive not found response
 * @param {string} path - Requested path
 * @param {object} config - Service configuration
 * @param {object} additionalInfo - Additional information
 * @returns {object} Formatted not found response object
 */
function createNotFoundResponse(path, config, additionalInfo = {}) {
  const headers = getCorsHeaders();
  const timestamp = new Date().toISOString();

  const notFoundResponse = {
    error: {
      type: "NOT_FOUND",
      message: `Endpoint /${path} not found for ${config.api.base_path}`,
      code: "ENDPOINT_NOT_FOUND",
      timestamp: timestamp,
      request_id: generateRequestId(),
      service: config.service.name,
      requested_path: path,
      ...additionalInfo,
    },
    metadata: {
      service: config.service.name,
      version: config.service.version || "1.0.0",
      generated_at: timestamp,
      available_endpoints: config.api.endpoints.map(
        (e) => `${e.method} ${config.api.base_path}${e.path}`
      ),
    },
  };

  return {
    statusCode: 404,
    headers,
    body: JSON.stringify(notFoundResponse, null, 2),
  };
}

/**
 * Creates a CORS preflight response
 * @param {string} origin - Request origin
 * @returns {object} CORS preflight response object
 */
function createCorsResponse(origin = "*") {
  return {
    statusCode: 200,
    headers: getCorsHeaders(origin),
    body: JSON.stringify({ status: "ok" }),
  };
}

/**
 * Creates a rate limit exceeded response
 * @param {object} config - Service configuration
 * @param {object} additionalInfo - Additional information
 * @returns {object} Rate limit response object
 */
function createRateLimitResponse(config, additionalInfo = {}) {
  const headers = getCorsHeaders();
  const timestamp = new Date().toISOString();

  const rateLimitResponse = {
    error: {
      type: "RATE_LIMIT_EXCEEDED",
      message: "Too many requests. Please try again later.",
      code: "RATE_LIMIT_EXCEEDED",
      timestamp: timestamp,
      request_id: generateRequestId(),
      service: config.service.name,
      retry_after: additionalInfo.retryAfter || 60,
      ...additionalInfo,
    },
    metadata: {
      service: config.service.name,
      version: config.service.version || "1.0.0",
      generated_at: timestamp,
    },
  };

  return {
    statusCode: 429,
    headers: {
      ...headers,
      "Retry-After": (additionalInfo.retryAfter || 60).toString(),
    },
    body: JSON.stringify(rateLimitResponse, null, 2),
  };
}

/**
 * Creates a service unavailable response
 * @param {object} config - Service configuration
 * @param {object} additionalInfo - Additional information
 * @returns {object} Service unavailable response object
 */
function createServiceUnavailableResponse(config, additionalInfo = {}) {
  const headers = getCorsHeaders();
  const timestamp = new Date().toISOString();

  const serviceUnavailableResponse = {
    error: {
      type: "SERVICE_UNAVAILABLE",
      message: "Service temporarily unavailable. Please try again later.",
      code: "SERVICE_UNAVAILABLE",
      timestamp: timestamp,
      request_id: generateRequestId(),
      service: config.service.name,
      ...additionalInfo,
    },
    metadata: {
      service: config.service.name,
      version: config.service.version || "1.0.0",
      generated_at: timestamp,
    },
  };

  return {
    statusCode: 503,
    headers: {
      ...headers,
      "Retry-After": (additionalInfo.retryAfter || 30).toString(),
    },
    body: JSON.stringify(serviceUnavailableResponse, null, 2),
  };
}

/**
 * Classifies errors and determines appropriate status codes
 * @param {Error} error - Error object
 * @param {number} defaultStatusCode - Default status code
 * @returns {object} Error classification information
 */
function classifyError(error, defaultStatusCode) {
  const message = error.message.toLowerCase();

  // Validation errors
  if (
    message.includes("invalid") ||
    message.includes("validation") ||
    message.includes("parameter")
  ) {
    return {
      type: "VALIDATION_ERROR",
      statusCode: 400,
      code: "INVALID_PARAMETER",
      details: "One or more parameters are invalid",
    };
  }

  // Authentication errors
  if (
    message.includes("unauthorized") ||
    message.includes("authentication") ||
    message.includes("auth")
  ) {
    return {
      type: "AUTHENTICATION_ERROR",
      statusCode: 401,
      code: "UNAUTHORIZED",
      details: "Authentication required",
    };
  }

  // Authorization errors
  if (
    message.includes("forbidden") ||
    message.includes("permission") ||
    message.includes("access")
  ) {
    return {
      type: "AUTHORIZATION_ERROR",
      statusCode: 403,
      code: "FORBIDDEN",
      details: "Insufficient permissions",
    };
  }

  // Not found errors
  if (message.includes("not found") || message.includes("does not exist")) {
    return {
      type: "NOT_FOUND",
      statusCode: 404,
      code: "RESOURCE_NOT_FOUND",
      details: "The requested resource was not found",
    };
  }

  // Rate limiting errors
  if (message.includes("rate limit") || message.includes("too many requests")) {
    return {
      type: "RATE_LIMIT_ERROR",
      statusCode: 429,
      code: "RATE_LIMIT_EXCEEDED",
      details: "Too many requests",
    };
  }

  // Database errors
  if (
    message.includes("database") ||
    message.includes("connection") ||
    message.includes("timeout")
  ) {
    return {
      type: "DATABASE_ERROR",
      statusCode: 503,
      code: "SERVICE_UNAVAILABLE",
      details: "Database connection issue",
    };
  }

  // Default to internal server error
  return {
    type: "INTERNAL_SERVER_ERROR",
    statusCode: defaultStatusCode || 500,
    code: "INTERNAL_ERROR",
    details: "An unexpected error occurred",
  };
}

/**
 * Sanitizes metadata to remove sensitive information
 * @param {object} metadata - Metadata object
 * @returns {object} Sanitized metadata
 */
function sanitizeMetadata(metadata) {
  if (!metadata || typeof metadata !== "object") {
    return metadata;
  }

  const sanitized = { ...metadata };
  const sensitiveKeys = ["password", "token", "key", "secret", "authorization"];

  // Remove sensitive information
  sensitiveKeys.forEach((key) => {
    if (sanitized[key]) {
      sanitized[key] = "[REDACTED]";
    }
  });

  return sanitized;
}

/**
 * Generates a unique request ID
 * @returns {string} Unique request ID
 */
function generateRequestId() {
  return `req_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
}

/**
 * Creates a health check response
 * @param {object} config - Service configuration
 * @param {object} healthInfo - Health information
 * @returns {object} Health check response object
 */
function createHealthResponse(config, healthInfo = {}) {
  const headers = getCorsHeaders();
  const timestamp = new Date().toISOString();

  const healthResponse = {
    status: "healthy",
    service: config.service.name,
    version: config.service.version || "1.0.0",
    timestamp: timestamp,
    uptime: process.uptime(),
    memory: process.memoryUsage(),
    ...healthInfo,
  };

  return {
    statusCode: 200,
    headers,
    body: JSON.stringify(healthResponse, null, 2),
  };
}

/**
 * Creates a response with custom status code and body
 * @param {number} statusCode - HTTP status code
 * @param {object} body - Response body
 * @param {object} config - Service configuration
 * @param {object} additionalHeaders - Additional headers
 * @returns {object} Custom response object
 */
function createCustomResponse(
  statusCode,
  body,
  config,
  additionalHeaders = {}
) {
  const headers = {
    ...getCorsHeaders(),
    ...additionalHeaders,
  };

  return {
    statusCode,
    headers,
    body: JSON.stringify(body, null, 2),
  };
}

module.exports = {
  getCorsHeaders,
  createSuccessResponse,
  createErrorResponse,
  createNotFoundResponse,
  createCorsResponse,
  createRateLimitResponse,
  createServiceUnavailableResponse,
  createHealthResponse,
  createCustomResponse,
  classifyError,
  sanitizeMetadata,
  generateRequestId,
  SECURITY_HEADERS,
  CORS_CONFIG,
};
