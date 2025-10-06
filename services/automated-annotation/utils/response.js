/**
 * Response utility functions for LLM Agent service
 */

/**
 * Create a successful response
 */
function createSuccessResponse(
  data,
  config,
  endpoint,
  queryParams = {},
  pagination = null,
  componentType = null
) {
  const response = {
    success: true,
    data: data,
    metadata: {
      service: config.service.name,
      version: config.service.version,
      endpoint: endpoint,
      timestamp: new Date().toISOString(),
      component_type: componentType,
    },
  };

  // Add pagination if provided
  if (pagination) {
    response.metadata.pagination = pagination;
  }

  // Add query parameters if provided
  if (queryParams && Object.keys(queryParams).length > 0) {
    response.metadata.query_params = queryParams;
  }

  return {
    statusCode: 200,
    headers: {
      "Content-Type": "application/json",
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Headers":
        "Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token",
      "Access-Control-Allow-Methods": "GET,POST,PUT,DELETE,OPTIONS",
    },
    body: JSON.stringify(response),
  };
}

/**
 * Create an error response
 */
function createErrorResponse(error, config, statusCode = 500, endpoint = null) {
  const response = {
    success: false,
    error: {
      message: error.message || "Internal server error",
      type: error.name || "Error",
      service: config.service.name,
      version: config.service.version,
      timestamp: new Date().toISOString(),
    },
  };

  if (endpoint) {
    response.error.endpoint = endpoint;
  }

  // Add stack trace in development
  if (process.env.NODE_ENV === "development") {
    response.error.stack = error.stack;
  }

  return {
    statusCode: statusCode,
    headers: {
      "Content-Type": "application/json",
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Headers":
        "Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token",
      "Access-Control-Allow-Methods": "GET,POST,PUT,DELETE,OPTIONS",
    },
    body: JSON.stringify(response),
  };
}

/**
 * Create a CORS response for preflight requests
 */
function createCorsResponse() {
  return {
    statusCode: 200,
    headers: {
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Headers":
        "Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token",
      "Access-Control-Allow-Methods": "GET,POST,PUT,DELETE,OPTIONS",
    },
    body: "",
  };
}

module.exports = {
  createSuccessResponse,
  createErrorResponse,
  createCorsResponse,
};
