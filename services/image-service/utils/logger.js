/**
 * Shared logging utility for consistent logging across all services
 */

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
 * Current log level (can be set via environment variable)
 */
const currentLogLevel = process.env.LOG_LEVEL
  ? LOG_LEVELS[process.env.LOG_LEVEL.toUpperCase()] || LOG_LEVELS.INFO
  : LOG_LEVELS.INFO;

/**
 * Formats a log message with timestamp and service info
 * @param {string} level - Log level
 * @param {string} message - Log message
 * @param {object} data - Additional data to log
 * @param {string} serviceName - Service name
 * @returns {string} Formatted log message
 */
function formatLogMessage(level, message, data = null, serviceName = null) {
  const timestamp = new Date().toISOString();
  const serviceInfo = serviceName ? `[${serviceName}]` : "";
  const dataString = data ? ` | ${JSON.stringify(data)}` : "";

  return `${timestamp} ${level} ${serviceInfo} ${message}${dataString}`;
}

/**
 * Logs an error message
 * @param {string} message - Error message
 * @param {Error|object} error - Error object or additional data
 * @param {string} serviceName - Service name
 */
function error(message, error = null, serviceName = null) {
  if (currentLogLevel >= LOG_LEVELS.ERROR) {
    const errorData =
      error instanceof Error
        ? {
            message: error.message,
            stack: error.stack,
            name: error.name,
          }
        : error;

    console.error(formatLogMessage("ERROR", message, errorData, serviceName));
  }
}

/**
 * Logs a warning message
 * @param {string} message - Warning message
 * @param {object} data - Additional data to log
 * @param {string} serviceName - Service name
 */
function warn(message, data = null, serviceName = null) {
  if (currentLogLevel >= LOG_LEVELS.WARN) {
    console.warn(formatLogMessage("WARN", message, data, serviceName));
  }
}

/**
 * Logs an info message
 * @param {string} message - Info message
 * @param {object} data - Additional data to log
 * @param {string} serviceName - Service name
 */
function info(message, data = null, serviceName = null) {
  if (currentLogLevel >= LOG_LEVELS.INFO) {
    console.log(formatLogMessage("INFO", message, data, serviceName));
  }
}

/**
 * Logs a debug message
 * @param {string} message - Debug message
 * @param {object} data - Additional data to log
 * @param {string} serviceName - Service name
 */
function debug(message, data = null, serviceName = null) {
  if (currentLogLevel >= LOG_LEVELS.DEBUG) {
    console.log(formatLogMessage("DEBUG", message, data, serviceName));
  }
}

/**
 * Logs service request information
 * @param {string} serviceName - Service name
 * @param {object} event - Lambda event
 */
function logServiceRequest(serviceName, event) {
  info(
    `${serviceName} service request`,
    {
      path: event.path,
      method: event.httpMethod,
      queryParams: event.queryStringParameters || {},
      sourceIp: event.requestContext?.identity?.sourceIp,
      userAgent: event.requestContext?.identity?.userAgent,
    },
    serviceName
  );
}

/**
 * Logs database query information
 * @param {string} sql - SQL query
 * @param {Array} args - Query arguments
 * @param {string} serviceName - Service name
 */
function logDatabaseQuery(sql, args = [], serviceName = null) {
  debug(
    "Database query",
    {
      sql: sql ? sql.trim() : "undefined",
      args: args,
      argCount: args.length,
    },
    serviceName
  );
}

/**
 * Logs database query results
 * @param {Array} results - Query results
 * @param {string} serviceName - Service name
 */
function logDatabaseResults(results, serviceName = null) {
  debug(
    "Database results",
    {
      rowCount: results.length,
      firstRow: results.length > 0 ? Object.keys(results[0]) : [],
    },
    serviceName
  );
}

/**
 * Logs service response information
 * @param {string} serviceName - Service name
 * @param {string} endpoint - Endpoint name
 * @param {number} statusCode - HTTP status code
 * @param {number} dataCount - Number of data items returned
 */
function logServiceResponse(
  serviceName,
  endpoint,
  statusCode,
  dataCount = null
) {
  info(
    `Service response`,
    {
      service: serviceName,
      endpoint: endpoint,
      statusCode: statusCode,
      dataCount: dataCount,
    },
    serviceName
  );
}

/**
 * Logs service error information
 * @param {string} serviceName - Service name
 * @param {string} endpoint - Endpoint name
 * @param {Error} errorObj - Error object
 */
function logServiceError(serviceName, endpoint, errorObj) {
  error(`Service error in ${endpoint}`, errorObj, serviceName);
}

/**
 * Generates a unique request ID for tracking requests across the system
 * @returns {string} Unique request identifier
 */
function generateRequestId() {
  return `req_${Date.now()}_${Math.random().toString(36).substring(2, 11)}`;
}

/**
 * Calculates query complexity based on parameters to help with performance monitoring
 * @param {object} params - Query parameters (can be raw strings or validated arrays)
 * @returns {string} Complexity level (low, medium, high, very_high)
 */
function calculateQueryComplexity(params) {
  let complexity = 0;

  // Base complexity
  complexity += 1;

  // Add complexity for each filter parameter
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
        // Handle validated parameters (arrays)
        values = params[param].length;
      } else if (typeof params[param] === "string") {
        // Handle raw parameters (strings)
        values = params[param].split(",").length;
      } else {
        // Handle other types (numbers, etc.)
        values = 1;
      }
      complexity += Math.min(values, 3); // Cap at 3 per parameter
    }
  });

  // Add complexity for grouping
  if (params.group_by) {
    let groups;
    if (Array.isArray(params.group_by)) {
      // Handle validated parameters (arrays)
      groups = params.group_by.length;
    } else if (typeof params.group_by === "string") {
      // Handle raw parameters (strings)
      groups = params.group_by.split(",").length;
    } else {
      // Handle other types
      groups = 1;
    }
    complexity += groups * 2;
  }

  // Add complexity for temporal grouping
  if (params.group_by) {
    const groupByValue = Array.isArray(params.group_by)
      ? params.group_by.join(",")
      : params.group_by;
    if (groupByValue.includes("monthly") || groupByValue.includes("weekly")) {
      complexity += 3;
    }
  }

  // Determine complexity level
  if (complexity <= 3) return "low";
  if (complexity <= 6) return "medium";
  if (complexity <= 10) return "high";
  return "very_high";
}

/**
 * Tests database connection and returns health status using optimized connection health check
 * @param {object} config - Service configuration
 * @returns {Promise<object>} Database health status
 */
async function testDatabaseConnection(config) {
  // For BigQuery, use a simple health check since module availability is checked at startup
  if (config.database?.connection_type === "bigquery") {
    console.log("🔍 Performing BigQuery health check...");

    // Since we fail fast on module loading, if we get here the module is available
    return {
      status: "healthy",
      response_time_ms: null,
      timestamp: new Date().toISOString(),
      details: {
        connection: "module_available",
        query_execution: "skipped_for_health_check",
        connection_type: "bigquery",
        project_id: config.database?.project_id || "unknown",
        note: "BigQuery module available (verified at startup)",
      },
    };
  }

  // For MySQL, use the original health check
  try {
    const { checkConnectionHealth } = require("./database");
    const healthStatus = await checkConnectionHealth(config);

    return {
      status: healthStatus.healthy ? "healthy" : "unhealthy",
      response_time_ms: healthStatus.responseTime || null,
      timestamp: new Date().toISOString(),
      details: {
        connection: healthStatus.healthy ? "established" : "failed",
        query_execution: healthStatus.healthy ? "successful" : "failed",
        pool_stats: healthStatus.poolStats || null,
        error: healthStatus.error || null,
        error_code: healthStatus.code || null,
      },
    };
  } catch (error) {
    return {
      status: "unhealthy",
      error: error.message,
      timestamp: new Date().toISOString(),
      details: {
        connection: "failed",
        query_execution: "failed",
        error_type: error.name,
      },
    };
  }
}

module.exports = {
  LOG_LEVELS,
  error,
  warn,
  info,
  debug,
  logServiceRequest,
  logDatabaseQuery,
  logDatabaseResults,
  logServiceResponse,
  logServiceError,
  generateRequestId,
  calculateQueryComplexity,
  testDatabaseConnection,
};
