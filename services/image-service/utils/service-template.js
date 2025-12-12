/**
 * Generic Service Template for Confidence Metrics Integration
 * This template provides a standardized way for any service to implement confidence metrics
 * with minimal code duplication and maximum reusability.
 */

const {
  validateAllParams,
  VALID_ENTITIES,
  VALID_TEMPORAL_GROUPINGS,
} = require("./validation");
const {
  validateAndProcessQueryMode,
  processConfidenceMetrics,
  addConfidenceMetadata,
  getOrderByField,
  postAggregationCalcs,
} = require("./data-parser");
const {
  buildCompleteQuery,
  buildConfidenceMetricsSQL,
  buildMarketShareQuery,
} = require("./query-builder");
const {
  filterInternalCalculationFields,
  mapAllEntities,
} = require("./data-parser");
const { createSuccessResponse, createErrorResponse } = require("./response");
const {
  generateRequestId,
  logServiceRequest,
  logDatabaseQuery,
  logDatabaseResults,
  logServiceResponse,
  logServiceError,
  testDatabaseConnection,
} = require("./logger");
const { createLogger } = require("./structured-logger");

/**
 * Generic confidence metrics service handler template
 * @param {object} event - Lambda event
 * @param {object} context - Lambda context
 * @param {object} serviceConfig - Service configuration
 * @param {object} options - Service-specific options
 * @returns {object} Lambda response
 */
async function handleConfidenceMetricsRequest(
  event,
  context,
  serviceConfig,
  options = {},
  queryFunction,
  partitionKey = null
) {
  // Create structured logger for this request
  const structuredLogger = createLogger({
    layer: "annotation-ifl",
    serviceName: serviceConfig.service.name,
  });
  const requestContext = structuredLogger.startRequest(event);

  const requestId = generateRequestId();
  const startTime = Date.now();

  try {
    logServiceRequest(serviceConfig.service.name, event);

    // Extract and validate query parameters
    const queryParams = event.queryStringParameters || {};
    console.log(
      "🔍 Raw query parameters:",
      JSON.stringify(queryParams, null, 2)
    );

    const validatedParams = validateAllParams(
      queryParams,
      serviceConfig.service.name,
      options.endpoint || "default"
    );
    console.log(
      "✅ Validated parameters:",
      JSON.stringify(validatedParams, null, 2)
    );

    // Validate and process query_mode parameter
    const queryMode = validateAndProcessQueryMode(queryParams, serviceConfig, {
      allowFallback: options.allowFallback !== false,
    });
    console.log("🔧 Query mode:", queryMode);

    // Determine endpoint-level metrics policy from service config (if provided)
    let endpointMetrics = null;
    try {
      const endpointPath = `/${options.endpoint || "default"}`;
      const apiEndpoint = (serviceConfig.api?.endpoints || []).find(
        (e) => e.path === endpointPath
      );
      if (apiEndpoint?.metrics?.[queryMode]) {
        endpointMetrics = apiEndpoint.metrics[queryMode];
      }
    } catch (e) {
      console.log("ℹ️ No endpoint-level metrics override found", e?.message);
    }
    console.log("🔧 Endpoint metrics override:", endpointMetrics);

    // Build the SQL query using the generic query builder
    console.log(
      "🔧 Building SQL query with options:",
      JSON.stringify(options, null, 2)
    );
    // Ensure valueField is provided - this is required for proper service configuration
    if (
      !options.valueField &&
      typeof options.customQueryBuilder !== "function"
    ) {
      throw new Error(
        `valueField is required but not provided in service options. Service configuration must specify the aggregation field (e.g., "SUM(sold_price) AS value", "COUNT(*) AS value", "AVG(discount) AS value"). Check service configuration for endpoint: ${
          options.endpoint || "unknown"
        }`
      );
    }

    // Determine which query builder to use based on database connection type
    const isBigQuery = serviceConfig.database?.connection_type === "bigquery";
    const queryBuilder = isBigQuery
      ? require("./bigquery-query-builder")
      : require("./query-builder");

    console.log("🔧 Using query builder:", isBigQuery ? "BigQuery" : "MySQL");
    console.log(
      "🔧 Database connection type:",
      serviceConfig.database?.connection_type
    );

    const queryResult =
      typeof options.customQueryBuilder === "function"
        ? await options.customQueryBuilder(validatedParams, serviceConfig, {
            order: validatedParams.order || "DESC",
            limit: validatedParams.limit || 500,
            offset: validatedParams.offset || 0,
            route: partitionKey || route, // Pass the detected route to custom query builders
            ...options,
          })
        : queryBuilder.buildCompleteQuery(
            validatedParams,
            options.valueField,
            options.itemType || "sold",
            options.priceField || "sold_price",
            validatedParams.order || "DESC",
            validatedParams.limit || 500,
            validatedParams.offset || 0,
            options.temporalDateField || null,
            serviceConfig,
            options.queryOptions || {},
            options.route || null
          );
    console.log("🔧 Built SQL query:", queryResult.sql);
    console.log("🔧 SQL arguments:", queryResult.args);

    // Execute the query
    logDatabaseQuery(
      queryResult.sql,
      queryResult.args,
      serviceConfig.service.name
    );
    const results = await queryFunction(
      queryResult.sql,
      queryResult.args,
      serviceConfig
    );
    console.log(
      "📊 Raw SQL results (first 5):",
      JSON.stringify(results.slice(0, 5), null, 2)
    );
    console.log("📊 Raw SQL results length:", results.length);
    if (results.length > 0) {
      console.log("📊 Raw SQL results structure:", Object.keys(results[0]));
    }
    logDatabaseResults(results, serviceConfig.service.name);

    // Process the results with confidence metrics
    console.log("🔧 Processing results with confidence metrics...");
    let processedData = processConfidenceMetrics(
      results,
      queryMode,
      serviceConfig,
      {
        fastModeFields: options.fastModeFields || [],
        slowModeFields: options.slowModeFields || [],
      }
    );
    // Optional per-route post-processing hook
    if (typeof options.postProcessData === "function") {
      try {
        processedData = options.postProcessData(
          processedData,
          validatedParams,
          queryMode
        );
      } catch (hookErr) {
        structuredLogger.logWarning(
          requestContext,
          "postProcessData hook failed",
          {
            error: hookErr?.message,
            hookName: "postProcessData",
          }
        );
      }
    }
    console.log(
      "📊 Processed data (first 5):",
      JSON.stringify(processedData.slice(0, 5), null, 2)
    );
    console.log("📊 Processed data length:", processedData.length);

    // Apply calc (e.g., percentage_change) if requested
    let finalData = processedData;
    const calc = validatedParams.calc;
    if (calc) {
      const groupByStr = Array.isArray(validatedParams.group_by)
        ? validatedParams.group_by.join(",")
        : validatedParams.group_by || "";
      console.log("🔧 Applying calc:", calc, "group_by:", groupByStr);
      finalData = postAggregationCalcs(
        finalData,
        calc,
        groupByStr,
        validatedParams,
        queryParams
      );
      console.log(
        "📊 After calc processing (first 5):",
        JSON.stringify(finalData.slice(0, 5), null, 2)
      );
      console.log("📊 After calc processing length:", finalData.length);
    }

    // Map database field names to API field names (e.g., material_parent -> material)
    console.log("🔧 Mapping database fields to API fields...");
    const beforeMapping = [...finalData];
    finalData = mapAllEntities(finalData);
    console.log(
      "📊 Before field mapping (first 3):",
      JSON.stringify(beforeMapping.slice(0, 3), null, 2)
    );
    console.log(
      "📊 After field mapping (first 3):",
      JSON.stringify(finalData.slice(0, 3), null, 2)
    );
    console.log("📊 After field mapping length:", finalData.length);

    // Determine allowed metrics list for filtering based on endpoint metrics override
    let allowedMetrics = null;
    if (endpointMetrics && Array.isArray(endpointMetrics)) {
      allowedMetrics = endpointMetrics;
    }

    // Filter out internal calculation fields, honoring endpoint-defined metric set
    console.log("🔧 Filtering internal calculation fields...");
    const beforeFiltering = [...finalData];
    finalData = filterInternalCalculationFields(
      finalData,
      serviceConfig.service.name,
      options.endpoint || "default",
      allowedMetrics
    );
    // Optional finalization hook to tweak/reshape response data
    if (typeof options.finalizeData === "function") {
      try {
        finalData = options.finalizeData(finalData, {
          validatedParams,
          queryParams,
          queryMode,
          requestId,
        });
      } catch (hookErr) {
        structuredLogger.logWarning(
          requestContext,
          "finalizeData hook failed",
          {
            error: hookErr?.message,
            hookName: "finalizeData",
          }
        );
      }
    }
    console.log(
      "📊 Before internal field filtering (first 3):",
      JSON.stringify(beforeFiltering.slice(0, 3), null, 2)
    );
    console.log(
      "📊 After internal field filtering (first 3):",
      JSON.stringify(finalData.slice(0, 3), null, 2)
    );
    console.log("📊 After internal field filtering length:", finalData.length);

    // Build response metadata with confidence information and reflect endpoint metrics
    const metadata = {
      query_params: validatedParams,
      service: serviceConfig.service.name,
      endpoint: options.endpoint || "default",
      generated_at: new Date().toISOString(),
      request_id: requestId,
      execution_time_ms: Date.now() - startTime,
      query_mode: queryMode,
      confidence_enabled: serviceConfig.confidence_metrics?.enabled || false,
    };

    // Add confidence metrics metadata with per-endpoint override
    const enhancedMetadata = addConfidenceMetadata(
      metadata,
      queryMode,
      serviceConfig,
      {
        fastModeMetadata: options.fastModeMetadata || {},
        slowModeMetadata: options.slowModeMetadata || {},
        availableMetricsOverride: endpointMetrics
          ? { [queryMode]: endpointMetrics }
          : undefined,
      }
    );

    // Create pagination info if needed
    let pagination = null;
    if (options.includePagination) {
      pagination = {
        limit: validatedParams.limit || 500,
        offset: validatedParams.offset || 0,
        total: finalData.length,
        has_more: finalData.length >= (validatedParams.limit || 500),
        next_offset:
          (validatedParams.offset || 0) + (validatedParams.limit || 500),
      };
    }

    // Log successful response
    logServiceResponse(
      serviceConfig.service.name,
      options.endpoint || "default",
      200,
      finalData.length
    );

    console.log("📊 Final data length:", finalData.length);
    console.log(
      "📊 Final data (first 3):",
      JSON.stringify(finalData.slice(0, 3), null, 2)
    );

    return createSuccessResponse(
      finalData,
      serviceConfig,
      options.endpoint || "default",
      validatedParams,
      pagination,
      options.componentType || "data_analysis",
      enhancedMetadata
    );
  } catch (error) {
    structuredLogger.logError(requestContext, error, {
      statusCode: error.statusCode || 500,
    });
    logServiceError(
      serviceConfig.service.name,
      options.endpoint || "default",
      error
    );
    return createErrorResponse(error, serviceConfig);
  }
}

/**
 * Generic aggregation service handler template
 * @param {object} event - Lambda event
 * @param {object} context - Lambda context
 * @param {object} serviceConfig - Service configuration
 * @param {object} options - Service-specific options
 * @returns {object} Lambda response
 */
async function handleAggregationRequest(
  event,
  context,
  serviceConfig,
  options = {},
  queryFunction,
  partitionKey = null
) {
  // Create structured logger for this request
  const structuredLogger = createLogger({
    layer: "annotation-ifl",
    serviceName: serviceConfig.service.name,
  });
  const requestContext = structuredLogger.startRequest(event);

  const requestId = generateRequestId();
  const startTime = Date.now();

  try {
    logServiceRequest(serviceConfig.service.name, event);

    // Extract and validate query parameters
    const queryParams = event.queryStringParameters || {};
    console.log(
      "🔍 Raw query parameters:",
      JSON.stringify(queryParams, null, 2)
    );

    const validatedParams = validateAllParams(
      queryParams,
      serviceConfig.service.name,
      options.endpoint || "default"
    );
    console.log(
      "✅ Validated parameters:",
      JSON.stringify(validatedParams, null, 2)
    );

    // Validate and process query_mode parameter
    const queryMode = validateAndProcessQueryMode(queryParams, serviceConfig, {
      allowFallback: options.allowFallback !== false,
    });
    console.log("🔧 Query mode:", queryMode);

    // Determine endpoint-level metrics policy from service config (if provided)
    let endpointMetrics = null;
    try {
      const endpointPath = `/${options.endpoint || "default"}`;
      const apiEndpoint = (serviceConfig.api?.endpoints || []).find(
        (e) => e.path === endpointPath
      );
      if (apiEndpoint?.metrics?.[queryMode]) {
        endpointMetrics = apiEndpoint.metrics[queryMode];
      }
    } catch (e) {
      console.log("ℹ️ No endpoint-level metrics override found", e?.message);
    }
    console.log("🔧 Endpoint metrics override:", endpointMetrics);

    // Determine aggregation type and field from valueField
    let aggregationType = "avg";
    let aggregationField = "sold_price";

    if (options.valueField) {
      if (options.valueField.includes("SUM(")) {
        aggregationType = "sum";
        const match = options.valueField.match(/SUM\(([^)]+)\)/);
        aggregationField = match ? match[1] : "sold_price";
      } else if (options.valueField.includes("COUNT(")) {
        aggregationType = "count";
        aggregationField = "*";
      } else if (options.valueField.includes("AVG(")) {
        aggregationType = "avg";
        const match = options.valueField.match(/AVG\(([^)]+)\)/);
        aggregationField = match ? match[1] : "sold_price";
      }
    }

    const aggregationConfig = {
      type: aggregationType,
      field: aggregationField,
      alias: "value",
    };
    console.log(
      "🔧 Aggregation config:",
      JSON.stringify(aggregationConfig, null, 2)
    );

    // Use the service's valueField directly - don't override it with confidence metrics
    console.log("🔧 Using service valueField:", options.valueField);

    // Build the complete SQL query
    console.log("🔧 Building complete SQL query...");
    const useMarketShare =
      options.marketShare === true ||
      (options.endpoint || "") === "market-share";
    console.log("🔍 Checking for custom query builder...");
    console.log(
      "🔍 options.customQueryBuilder type:",
      typeof options.customQueryBuilder
    );
    console.log("🔍 options.customQueryBuilder:", options.customQueryBuilder);
    console.log("🔍 useMarketShare:", useMarketShare);

    // Determine which query builder to use based on database connection type
    const isBigQuery = serviceConfig.database?.connection_type === "bigquery";
    const queryBuilder = isBigQuery
      ? require("./bigquery-query-builder")
      : require("./query-builder");

    console.log("🔧 Using query builder:", isBigQuery ? "BigQuery" : "MySQL");
    console.log(
      "🔧 Database connection type:",
      serviceConfig.database?.connection_type
    );

    const queryResult =
      typeof options.customQueryBuilder === "function"
        ? await options.customQueryBuilder(validatedParams, serviceConfig, {
            order: validatedParams.order || "DESC",
            limit: validatedParams.limit || 500,
            offset: validatedParams.offset || 0,
            route: partitionKey || route, // Pass the detected route to custom query builders
            ...options,
          })
        : useMarketShare
        ? queryBuilder.buildMarketShareQuery(
            validatedParams,
            options.itemType || "sold",
            options.priceField || "sold_price",
            options.temporalDateField || null,
            serviceConfig,
            options.queryOptions || {},
            options.route || null
          )
        : queryBuilder.buildCompleteQuery(
            validatedParams,
            options.valueField, // Use the service's valueField directly
            options.itemType || "sold",
            options.priceField || "sold_price",
            validatedParams.order || "DESC",
            validatedParams.limit || 500,
            validatedParams.offset || 0,
            options.temporalDateField || null,
            serviceConfig,
            options.queryOptions || {},
            options.route || null
          );
    console.log("🔧 Built SQL query:", queryResult.sql);
    console.log("🔧 SQL arguments:", queryResult.args);

    // Execute the query
    logDatabaseQuery(
      queryResult.sql,
      queryResult.args,
      serviceConfig.service.name
    );
    const results = await queryFunction(
      queryResult.sql,
      queryResult.args,
      serviceConfig
    );
    console.log(
      "📊 Raw SQL results (first 5):",
      JSON.stringify(results.slice(0, 5), null, 2)
    );
    console.log("📊 Raw SQL results length:", results.length);
    if (results.length > 0) {
      console.log("📊 Raw SQL results structure:", Object.keys(results[0]));
    }
    logDatabaseResults(results, serviceConfig.service.name);

    // Process the results with confidence metrics
    console.log("🔧 Processing results with confidence metrics...");
    let processedData = processConfidenceMetrics(
      results,
      queryMode,
      serviceConfig,
      {
        fastModeFields: options.fastModeFields || [],
        slowModeFields: options.slowModeFields || [],
      }
    );
    // Optional per-route post-processing hook
    if (typeof options.postProcessData === "function") {
      try {
        processedData = options.postProcessData(
          processedData,
          validatedParams,
          queryMode
        );
      } catch (hookErr) {
        structuredLogger.logWarning(
          requestContext,
          "postProcessData hook failed",
          {
            error: hookErr?.message,
            hookName: "postProcessData",
          }
        );
      }
    }
    console.log(
      "📊 Processed data (first 5):",
      JSON.stringify(processedData.slice(0, 5), null, 2)
    );
    console.log("📊 Processed data length:", processedData.length);

    // Apply calc (e.g., percentage_change) if requested
    let finalData = processedData;
    const calc = validatedParams.calc;
    if (calc) {
      const groupByStr = Array.isArray(validatedParams.group_by)
        ? validatedParams.group_by.join(",")
        : validatedParams.group_by || "";
      console.log("🔧 Applying calc:", calc, "group_by:", groupByStr);
      finalData = postAggregationCalcs(
        finalData,
        calc,
        groupByStr,
        validatedParams,
        queryParams
      );
      console.log(
        "📊 After calc processing (first 5):",
        JSON.stringify(finalData.slice(0, 5), null, 2)
      );
      console.log("📊 After calc processing length:", finalData.length);
    }

    // Map database field names to API field names (e.g., material_parent -> material)
    console.log("🔧 Mapping database fields to API fields...");
    const beforeMapping = [...finalData];
    finalData = mapAllEntities(finalData);
    console.log(
      "📊 Before field mapping (first 3):",
      JSON.stringify(beforeMapping.slice(0, 3), null, 2)
    );
    console.log(
      "📊 After field mapping (first 3):",
      JSON.stringify(finalData.slice(0, 3), null, 2)
    );
    console.log("📊 After field mapping length:", finalData.length);

    // Determine allowed metrics list for filtering based on endpoint metrics override
    let allowedMetrics = null;
    if (endpointMetrics && Array.isArray(endpointMetrics)) {
      allowedMetrics = endpointMetrics;
    }

    // Filter out internal calculation fields, honoring endpoint-defined metric set
    console.log("🔧 Filtering internal calculation fields...");
    const beforeFiltering = [...finalData];
    finalData = filterInternalCalculationFields(
      finalData,
      serviceConfig.service.name,
      options.endpoint || "default",
      allowedMetrics
    );
    // Optional finalization hook to tweak/reshape response data
    if (typeof options.finalizeData === "function") {
      try {
        finalData = options.finalizeData(finalData, {
          validatedParams,
          queryParams,
          queryMode,
          requestId,
        });
      } catch (hookErr) {
        structuredLogger.logWarning(
          requestContext,
          "finalizeData hook failed",
          {
            error: hookErr?.message,
            hookName: "finalizeData",
          }
        );
      }
    }
    console.log(
      "📊 Before internal field filtering (first 3):",
      JSON.stringify(beforeFiltering.slice(0, 3), null, 2)
    );
    console.log(
      "📊 After internal field filtering (first 3):",
      JSON.stringify(finalData.slice(0, 3), null, 2)
    );
    console.log("📊 After internal field filtering length:", finalData.length);

    // Build response metadata
    const metadata = {
      query_params: validatedParams,
      service: serviceConfig.service.name,
      endpoint: options.endpoint || "default",
      generated_at: new Date().toISOString(),
      request_id: requestId,
      execution_time_ms: Date.now() - startTime,
      query_mode: queryMode,
      confidence_enabled: serviceConfig.confidence_metrics?.enabled || false,
      aggregation_type: aggregationConfig.type,
      aggregation_field: aggregationConfig.field,
    };

    // Add confidence metrics metadata
    const enhancedMetadata = addConfidenceMetadata(
      metadata,
      queryMode,
      serviceConfig,
      {
        fastModeMetadata: options.fastModeMetadata || {},
        slowModeMetadata: options.slowModeMetadata || {},
      }
    );

    // Create pagination info if needed
    let pagination = null;
    if (options.includePagination) {
      pagination = {
        limit: validatedParams.limit || 500,
        offset: validatedParams.offset || 0,
        total: finalData.length,
        has_more: finalData.length >= (validatedParams.limit || 500),
        next_offset:
          (validatedParams.offset || 0) + (validatedParams.limit || 500),
      };
    }

    // Log successful response
    logServiceResponse(
      serviceConfig.service.name,
      options.endpoint || "default",
      200,
      finalData.length
    );

    console.log("📊 Final data length:", finalData.length);
    console.log(
      "📊 Final data (first 3):",
      JSON.stringify(finalData.slice(0, 3), null, 2)
    );

    return createSuccessResponse(
      finalData,
      serviceConfig,
      options.endpoint || "default",
      validatedParams,
      pagination,
      options.componentType || "data_analysis",
      enhancedMetadata
    );
  } catch (error) {
    structuredLogger.logError(requestContext, error, {
      statusCode: error.statusCode || 500,
    });
    logServiceError(
      serviceConfig.service.name,
      options.endpoint || "default",
      error
    );
    return createErrorResponse(error, serviceConfig);
  }
}

/**
 * Generic health check handler template
 * @param {object} event - Lambda event
 * @param {object} context - Lambda context
 * @param {object} serviceConfig - Service configuration
 * @returns {object} Lambda response
 */
async function handleHealthCheck(event, context, serviceConfig) {
  const requestId = generateRequestId();
  const startTime = Date.now();

  try {
    logServiceRequest(serviceConfig.service.name, event);

    // Test database connection
    const dbHealth = await testDatabaseConnection(serviceConfig);

    const healthInfo = {
      service: serviceConfig.service.name,
      status: "healthy",
      timestamp: new Date().toISOString(),
      database: dbHealth,
      confidence_metrics: {
        enabled: serviceConfig.confidence_metrics?.enabled || false,
        modes: serviceConfig.confidence_metrics?.modes || {},
        default_mode: serviceConfig.confidence_metrics?.default_mode || "basic",
      },
      execution_time_ms: Date.now() - startTime,
    };

    logServiceResponse(serviceConfig.service.name, "health", 200);

    return {
      statusCode: 200,
      headers: {
        "Content-Type": "application/json",
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Headers":
          "Content-Type,X-Amz-Date,Authorization,X-Api-Key",
        "Access-Control-Allow-Methods": "GET,OPTIONS",
      },
      body: JSON.stringify(healthInfo),
    };
  } catch (error) {
    logServiceError(serviceConfig.service.name, "health", error);

    return {
      statusCode: 500,
      headers: {
        "Content-Type": "application/json",
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Headers":
          "Content-Type,X-Amz-Date,Authorization,X-Api-Key",
        "Access-Control-Allow-Methods": "GET,OPTIONS",
      },
      body: JSON.stringify({
        error: "Service unhealthy",
        service: serviceConfig.service.name,
        timestamp: new Date().toISOString(),
        details: error.message,
      }),
    };
  }
}

/**
 * Generic CORS handler template
 * @param {object} event - Lambda event
 * @param {object} context - Lambda context
 * @returns {object} Lambda response
 */
function handleCorsRequest(event, context) {
  return {
    statusCode: 200,
    headers: {
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Headers":
        "Content-Type,X-Amz-Date,Authorization,X-Api-Key",
      "Access-Control-Allow-Methods": "GET,OPTIONS",
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ message: "CORS preflight successful" }),
  };
}

/**
 * Generic route handler that automatically routes requests to appropriate handlers
 * @param {object} event - Lambda event
 * @param {object} context - Lambda context
 * @param {object} serviceConfig - Service configuration
 * @param {object} routeConfig - Route configuration
 * @returns {object} Lambda response
 */
async function handleGenericRoute(event, context, serviceConfig, routeConfig) {
  const { httpMethod, path } = event;
  const route = path.split("/").pop() || "default";

  // Partition-aware query param augmentation (root_type auto-apply)
  let partitionKey = null;
  try {
    const partitions = serviceConfig?.api?.partitions;
    if (partitions && partitions.routes) {
      // Find matching partition by base_path prefix (e.g., /bags/analytics, /all/analytics)
      console.log("🔍 Looking for partition match for path:", path);
      console.log("🔍 Available partitions:", Object.keys(partitions.routes));
      const partitionEntry = Object.entries(partitions.routes).find(
        ([, cfg]) => {
          const matches =
            typeof cfg?.base_path === "string" &&
            path.startsWith(cfg.base_path);
          console.log(`🔍 Checking partition ${cfg?.base_path}: ${matches}`);
          return matches;
        }
      );

      if (partitionEntry) {
        const [partKey, partitionCfg] = partitionEntry;
        partitionKey = partKey;
        console.log(
          "🔍 Found matching partition:",
          partKey,
          "with config:",
          partitionCfg
        );
        const qs = { ...(event.queryStringParameters || {}) };

        // Partition-specific diffs: restrict filters/group_by to allowed sets
        try {
          const endpointRoute = path.split("/").pop() || "";
          const endpointPath = `/${endpointRoute}`;
          const epDef = (serviceConfig.api.endpoints || []).find(
            (e) => e.path === endpointPath
          );
          const baseFilters = Array.isArray(epDef?.parameters?.filters)
            ? epDef.parameters.filters.slice()
            : [];
          const baseGroupBy = Array.isArray(epDef?.parameters?.group_by)
            ? epDef.parameters.group_by.slice()
            : [];

          const modFilters = partitionCfg.modified_filters || {
            add: [],
            remove: [],
          };
          const modGroupBys = partitionCfg.modified_group_bys || {
            add: [],
            remove: [],
          };

          const allowedFilters = new Set([
            ...baseFilters.filter(
              (f) => !(modFilters.remove || []).includes(f)
            ),
            ...(modFilters.add || []),
          ]);

          const allowedGroupBy = new Set([
            ...baseGroupBy.filter(
              (g) => !(modGroupBys.remove || []).includes(g)
            ),
            ...(modGroupBys.add || []),
          ]);

          // Enforce allowlists automatically when endpoint defines filters/group_by in config.
          // Otherwise, pass through client-supplied filters/group_by.
          const enforceAllowlist =
            (Array.isArray(baseFilters) && baseFilters.length > 0) ||
            (Array.isArray(baseGroupBy) && baseGroupBy.length > 0);

          if (enforceAllowlist) {
            const reservedParams = new Set([
              "limit",
              "offset",
              "order",
              "group_by",
              "calc",
              "query_mode",
              "monthly",
              "weekly",
              "root_type",
              "root_type_id",
              "min_price",
              "max_price",
              "listed_after",
              "listed_before",
              "sold_after",
              "sold_before",
              "is_sold",
              "page",
              "sort",
              "search",
            ]);

            const allowedParamKeys = new Set([...allowedFilters].map((f) => f));

            // Remove any param not allowed (except reserved/system)
            Object.keys(qs).forEach((key) => {
              if (!reservedParams.has(key) && !allowedParamKeys.has(key)) {
                delete qs[key];
              }
            });

            if (qs.group_by) {
              const gb = String(qs.group_by)
                .split(",")
                .map((s) => s.trim())
                .filter((s) => s && allowedGroupBy.has(s));
              qs.group_by = gb.join(",");
              if (!qs.group_by) delete qs.group_by;
            }
          }
        } catch (ppErr) {
          console.log("Partition param filtering skipped:", ppErr?.message);
        }

        // Apply default_filters from partition configuration
        if (partitionCfg.default_filters) {
          console.log(
            "🔍 Applying default_filters:",
            partitionCfg.default_filters
          );
          Object.entries(partitionCfg.default_filters).forEach(
            ([key, value]) => {
              if (qs[key] === undefined) {
                console.log(`🔍 Adding default filter: ${key} = ${value}`);
                qs[key] = value;
              } else {
                console.log(
                  `🔍 Default filter already exists: ${key} = ${qs[key]}`
                );
              }
            }
          );
          console.log("🔍 Query params after applying default filters:", qs);
        } else {
          console.log("🔍 No default_filters found in partition config");
        }

        // Attach back into event for downstream handlers
        event.queryStringParameters = qs;
        event.partition = partitionKey; // optional for logging/debug
      }
    }
  } catch (e) {
    console.log("Partition handling skipped due to error:", e?.message);
  }

  // Handle CORS preflight
  if (httpMethod === "OPTIONS") {
    return handleCorsRequest(event, context);
  }

  // Route to appropriate handler based on configuration
  if (route === "health" || route === "health-check") {
    return handleHealthCheck(event, context, serviceConfig);
  }

  // Check if this is an aggregation endpoint
  if (
    routeConfig.aggregationEndpoints &&
    routeConfig.aggregationEndpoints.includes(route)
  ) {
    console.log("🔧 Routing to aggregation handler for route:", route);
    console.log(
      "🔧 routeConfig.queryFunction:",
      typeof routeConfig.queryFunction
    );
    console.log(
      "🔧 routeConfig.queryFunction value:",
      routeConfig.queryFunction
    );

    const endpointConfig = routeConfig.endpoints?.[route] || {};
    return handleAggregationRequest(
      event,
      context,
      serviceConfig,
      {
        endpoint: route,
        route: partitionKey || route,
        ...routeConfig.defaultOptions,
        ...endpointConfig, // endpointConfig should override defaultOptions
      },
      routeConfig.queryFunction,
      partitionKey
    );
  }

  // Check if this is a confidence metrics endpoint
  if (
    routeConfig.confidenceEndpoints &&
    routeConfig.confidenceEndpoints.includes(route)
  ) {
    const endpointConfig = routeConfig.endpoints?.[route] || {};
    return handleConfidenceMetricsRequest(
      event,
      context,
      serviceConfig,
      {
        endpoint: route,
        route: partitionKey || route,
        ...routeConfig.defaultOptions,
        ...endpointConfig, // endpointConfig should override defaultOptions
      },
      routeConfig.queryFunction,
      partitionKey
    );
  }

  // Default to confidence metrics request
  const endpointConfig =
    routeConfig.endpoints?.[route] || routeConfig.defaultEndpoint || {};
  return handleConfidenceMetricsRequest(
    event,
    context,
    serviceConfig,
    {
      endpoint: route,
      route: partitionKey || route,
      ...routeConfig.defaultOptions,
      ...endpointConfig, // endpointConfig should override defaultOptions
    },
    routeConfig.queryFunction,
    partitionKey
  );
}

/**
 * Helper function to create a complete route configuration
 * @param {object} options - Configuration options
 * @returns {object} Complete route configuration
 */
function createRouteConfig(options = {}) {
  console.log("🔧 createRouteConfig called with options:", {
    aggregationEndpoints: options.aggregationEndpoints,
    confidenceEndpoints: options.confidenceEndpoints,
    queryFunction: typeof options.queryFunction,
    queryFunctionValue: options.queryFunction,
  });

  const config = {
    // Endpoints that should use aggregation handler
    aggregationEndpoints: options.aggregationEndpoints || [],

    // Endpoints that should use confidence metrics handler
    confidenceEndpoints: options.confidenceEndpoints || [],

    // Default endpoint configuration
    defaultEndpoint: options.defaultEndpoint || {},

    // Default options for all endpoints
    defaultOptions: {
      itemType: "sold",
      priceField: "sold_price",
      // valueField is required and must be provided by each service
      temporalDateField: null,
      includePagination: true,
      allowFallback: true,
      basicModeFields: [],
      basicDiagnosticsModeFields: [],
      basicModeMetadata: {},
      basicDiagnosticsModeMetadata: {},
      componentType: "data_analysis",
      ...options.defaultOptions,
    },

    // Individual endpoint configurations
    endpoints: options.endpoints || {},

    // Query options
    queryOptions: options.queryOptions || {},

    // Confidence options
    confidenceOptions: options.confidenceOptions || {},

    // Query function for database operations
    queryFunction: options.queryFunction || null,
  };

  console.log("🔧 createRouteConfig returning config:", {
    queryFunction: typeof config.queryFunction,
    queryFunctionValue: config.queryFunction,
  });

  return config;
}

module.exports = {
  handleConfidenceMetricsRequest,
  handleAggregationRequest,
  handleHealthCheck,
  handleCorsRequest,
  handleGenericRoute,
  createRouteConfig,
};
