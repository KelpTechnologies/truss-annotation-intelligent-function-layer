/**
 * BigQuery query builder utilities
 * Provides BigQuery-specific query building functions
 */

// Column mappings for BigQuery table
const COLUMN_MAPPINGS = {
  // Entity mappings
  brand: "brand",
  type: "type",
  material: "material",
  color: "colour",
  condition: "condition",
  size: "size",
  vendor: "vendor",
  // gender: "gender", // Column does not exist in BigQuery table
  model: "model",
  models: "model",
  // decade: "decade", // Column does not exist in BigQuery table
  location: "sold_location",
  hardware: "hardware_material",
  key_word: "listing_title", // for text search filtering

  // Date fields
  monthly: "listed_date",
  weekly: "listed_date",
  sold_date: "sold_date",
  listed_date: "listed_date",

  // Price fields
  sold_price: "sold_price",
  listed_price: "listed_price",

  // Other fields
  is_sold: "is_sold",
};

// Root type mappings
const ROOT_TYPE_MAPPINGS = {
  5: "Footwear",
  6: "Accessories",
  29: "Headwear",
  30: "Bags",
  0: "Unknown",
  113: "All",
  45: "Eyewear",
  114: "Clothing",
};

/**
 * Builds a BigQuery WHERE clause from query parameters
 * @param {object} params - Query parameters
 * @param {string} itemType - Type of items (sold, listed, etc.)
 * @param {string} priceField - Price field name
 * @param {string} temporalDateField - Temporal date field name
 * @param {object} config - Service configuration
 * @param {string} route - Route name
 * @returns {object} Object containing WHERE clause and query arguments
 */
function buildWhereClause(
  params,
  itemType = "sold",
  priceField = "sold_price",
  temporalDateField = "sold_date",
  config = null,
  route = null
) {
  console.log(
    "🔍 buildWhereClause called with params:",
    JSON.stringify(params, null, 2)
  );
  console.log("🔍 Route:", route);
  console.log("🔍 Config:", config ? "provided" : "not provided");

  const conditions = [];
  const args = [];

  // Add item type condition
  if (itemType === "sold") {
    conditions.push(`${COLUMN_MAPPINGS.sold_date} IS NOT NULL`);
  } else if (itemType === "listed") {
    conditions.push(`${COLUMN_MAPPINGS.listed_date} IS NOT NULL`);
  }

  // Add root_type filter - check both params and route configuration
  let rootTypeToUse = null;

  // First check if root_type is explicitly provided in params
  if (params.root_type) {
    console.log("🔍 root_type found in params:", params.root_type);
    rootTypeToUse = params.root_type;
  }
  // If not in params, check route configuration for default_filters
  else if (
    route &&
    config &&
    config.api &&
    config.api.partitions &&
    config.api.partitions.routes
  ) {
    const routeConfig = config.api.partitions.routes[route];
    if (
      routeConfig &&
      routeConfig.default_filters &&
      routeConfig.default_filters.root_type
    ) {
      console.log(
        "🔍 root_type found in route config:",
        routeConfig.default_filters.root_type
      );
      rootTypeToUse = routeConfig.default_filters.root_type;
    }
    // Check if root_type should be suppressed for this route
    else if (routeConfig && routeConfig.suppress_root_type) {
      console.log("🔍 root_type suppressed for route:", route);
    }
  }

  // Apply root_type filter if we have one
  if (rootTypeToUse) {
    const rootTypes = Array.isArray(rootTypeToUse)
      ? rootTypeToUse
      : rootTypeToUse.split(",");
    const rootTypePlaceholders = rootTypes
      .map((_, index) => `@param_${args.length + index}`)
      .join(", ");
    conditions.push(`root_type IN (${rootTypePlaceholders})`);

    // Use root type values directly (no mapping needed)
    rootTypes.forEach((rootType) => {
      args.push(rootType.trim());
    });

    console.log(
      "🔍 Added root_type condition:",
      `root_type IN (${rootTypePlaceholders})`
    );
  } else {
    console.log("🔍 No root_type filter applied");
  }

  // Brand filter
  if (params.brands) {
    const brands = Array.isArray(params.brands)
      ? params.brands
      : params.brands.split(",");
    const brandPlaceholders = brands
      .map((_, index) => `@param_${args.length + index}`)
      .join(", ");
    conditions.push(`${COLUMN_MAPPINGS.brand} IN (${brandPlaceholders})`);
    brands.forEach((brand) => args.push(brand.trim()));
  }

  // Model filter
  if (params.models) {
    const models = Array.isArray(params.models)
      ? params.models
      : params.models.split(",");
    const modelPlaceholders = models
      .map((_, index) => `@param_${args.length + index}`)
      .join(", ");
    conditions.push(`${COLUMN_MAPPINGS.model} IN (${modelPlaceholders})`);
    models.forEach((model) => args.push(model.trim()));
  }

  // Color filter
  if (params.colors) {
    const colors = Array.isArray(params.colors)
      ? params.colors
      : params.colors.split(",");
    const colorPlaceholders = colors
      .map((_, index) => `@param_${args.length + index}`)
      .join(", ");
    conditions.push(`${COLUMN_MAPPINGS.color} IN (${colorPlaceholders})`);
    colors.forEach((color) => args.push(color.trim()));
  }

  // Material filter
  if (params.materials) {
    const materials = Array.isArray(params.materials)
      ? params.materials
      : params.materials.split(",");
    const materialPlaceholders = materials
      .map((_, index) => `@param_${args.length + index}`)
      .join(", ");
    conditions.push(`${COLUMN_MAPPINGS.material} IN (${materialPlaceholders})`);
    materials.forEach((material) => args.push(material.trim()));
  }

  // Hardware filter
  if (params.hardwares) {
    const hardwares = Array.isArray(params.hardwares)
      ? params.hardwares
      : params.hardwares.split(",");
    const hardwarePlaceholders = hardwares
      .map((_, index) => `@param_${args.length + index}`)
      .join(", ");
    conditions.push(`${COLUMN_MAPPINGS.hardware} IN (${hardwarePlaceholders})`);
    hardwares.forEach((hardware) => args.push(hardware.trim()));
  }

  // Type filter
  if (params.types) {
    const types = Array.isArray(params.types)
      ? params.types
      : params.types.split(",");
    const typePlaceholders = types
      .map((_, index) => `@param_${args.length + index}`)
      .join(", ");
    conditions.push(`${COLUMN_MAPPINGS.type} IN (${typePlaceholders})`);
    types.forEach((type) => args.push(type.trim()));
  }

  // Shape filter (mapped to type in BigQuery)
  if (params.shapes) {
    const shapes = Array.isArray(params.shapes)
      ? params.shapes
      : params.shapes.split(",");
    const shapePlaceholders = shapes
      .map((_, index) => `@param_${args.length + index}`)
      .join(", ");
    conditions.push(`${COLUMN_MAPPINGS.type} IN (${shapePlaceholders})`);
    shapes.forEach((shape) => args.push(shape.trim()));
  }

  // Size filter
  if (params.sizes) {
    const sizes = Array.isArray(params.sizes)
      ? params.sizes
      : params.sizes.split(",");
    const sizePlaceholders = sizes
      .map((_, index) => `@param_${args.length + index}`)
      .join(", ");
    conditions.push(`${COLUMN_MAPPINGS.size} IN (${sizePlaceholders})`);
    sizes.forEach((size) => args.push(size.trim()));
  }

  // Condition filter
  if (params.conditions) {
    const conditions_list = Array.isArray(params.conditions)
      ? params.conditions
      : params.conditions.split(",");
    const conditionPlaceholders = conditions_list
      .map((_, index) => `@param_${args.length + index}`)
      .join(", ");
    conditions.push(
      `${COLUMN_MAPPINGS.condition} IN (${conditionPlaceholders})`
    );
    conditions_list.forEach((condition) => args.push(condition.trim()));
  }

  // Location filter
  if (params.locations) {
    const locations = Array.isArray(params.locations)
      ? params.locations
      : params.locations.split(",");
    const locationPlaceholders = locations
      .map((_, index) => `@param_${args.length + index}`)
      .join(", ");
    conditions.push(`${COLUMN_MAPPINGS.location} IN (${locationPlaceholders})`);
    locations.forEach((location) => args.push(location.trim()));
  }

  // Key word filter (text search using BigQuery SEARCH function)
  if (params.key_words) {
    const keyWords = Array.isArray(params.key_words)
      ? params.key_words
      : params.key_words.split(",");

    // For each key word, create a SEARCH condition
    const searchConditions = keyWords.map((keyword, index) => {
      args.push(keyword.trim());
      return `SEARCH(${COLUMN_MAPPINGS.key_word}, @param_${args.length - 1})`;
    });

    // Combine all search conditions with OR
    if (searchConditions.length > 0) {
      conditions.push(`(${searchConditions.join(" OR ")})`);
    }
  }

  // Price range filter
  if (params.sold_price_min) {
    conditions.push(`${priceField} >= @param_${args.length}`);
    args.push(parseFloat(params.sold_price_min));
  }
  if (params.sold_price_max) {
    conditions.push(`${priceField} <= @param_${args.length}`);
    args.push(parseFloat(params.sold_price_max));
  }

  // Date range filters
  if (params.sold_date_range) {
    const dateCondition = buildDateRangeCondition(
      params.sold_date_range,
      temporalDateField
    );
    if (dateCondition.condition) {
      conditions.push(dateCondition.condition);
      args.push(...dateCondition.args);
    }
  }

  if (params.listed_date_range) {
    const dateCondition = buildDateRangeCondition(
      params.listed_date_range,
      "listed_date"
    );
    if (dateCondition.condition) {
      conditions.push(dateCondition.condition);
      args.push(...dateCondition.args);
    }
  }

  // Monthly filter - always use listed_date for monthly grouping
  if (params.monthlys || params.monthly) {
    const months = params.monthlys || params.monthly;
    const monthArray = Array.isArray(months) ? months : months.split(",");
    const monthConditions = monthArray.map((month, index) => {
      args.push(month.trim());
      return `FORMAT_DATE('%Y-%m', ${COLUMN_MAPPINGS.monthly}) = @param_${
        args.length - 1
      }`;
    });
    conditions.push(`(${monthConditions.join(" OR ")})`);
  }

  // Weekly filter - always use listed_date for weekly grouping
  if (params.weeklys || params.weekly) {
    const weeks = params.weeklys || params.weekly;
    const weekArray = Array.isArray(weeks) ? weeks : weeks.split(",");
    const weekConditions = weekArray.map((week, index) => {
      args.push(week.trim());
      return `FORMAT_DATE('%Y-%W', ${COLUMN_MAPPINGS.weekly}) = @param_${
        args.length - 1
      }`;
    });
    conditions.push(`(${weekConditions.join(" OR ")})`);
  }

  const whereClause =
    conditions.length > 0 ? `WHERE ${conditions.join(" AND ")}` : "";

  return {
    whereClause,
    queryArgs: args,
  };
}

/**
 * Builds date range condition for BigQuery
 * @param {string} dateRange - Date range string
 * @param {string} dateField - Date field name
 * @returns {object} Object containing condition and arguments
 */
function buildDateRangeCondition(dateRange, dateField) {
  const conditions = [];
  const args = [];

  switch (dateRange.toLowerCase()) {
    case "last month":
      conditions.push(
        `${dateField} >= DATE_SUB(CURRENT_DATE(), INTERVAL 1 MONTH)`
      );
      conditions.push(`${dateField} < CURRENT_DATE()`);
      break;

    case "last 3 months":
      conditions.push(
        `${dateField} >= DATE_SUB(CURRENT_DATE(), INTERVAL 3 MONTH)`
      );
      conditions.push(`${dateField} < CURRENT_DATE()`);
      break;

    case "last 6 months":
      conditions.push(
        `${dateField} >= DATE_SUB(CURRENT_DATE(), INTERVAL 6 MONTH)`
      );
      conditions.push(`${dateField} < CURRENT_DATE()`);
      break;

    case "last year":
      conditions.push(
        `${dateField} >= DATE_SUB(CURRENT_DATE(), INTERVAL 1 YEAR)`
      );
      conditions.push(`${dateField} < CURRENT_DATE()`);
      break;

    case "this month":
      conditions.push(`${dateField} >= DATE_TRUNC(CURRENT_DATE(), MONTH)`);
      conditions.push(`${dateField} < CURRENT_DATE()`);
      break;

    case "this year":
      conditions.push(`${dateField} >= DATE_TRUNC(CURRENT_DATE(), YEAR)`);
      conditions.push(`${dateField} < CURRENT_DATE()`);
      break;

    default:
      // Handle custom date ranges or specific months like "Feb 2025"
      if (dateRange.includes("2024") || dateRange.includes("2025")) {
        // Parse month/year format
        const monthYear = dateRange.match(/(\w+)\s+(\d{4})/);
        if (monthYear) {
          const month = monthYear[1];
          const year = monthYear[2];
          const monthNumber = getMonthNumber(month);
          if (monthNumber) {
            conditions.push(
              `EXTRACT(YEAR FROM ${dateField}) = @param_${args.length}`
            );
            args.push(parseInt(year));
            conditions.push(
              `EXTRACT(MONTH FROM ${dateField}) = @param_${args.length}`
            );
            args.push(monthNumber);
          }
        }
      }
      break;
  }

  return {
    condition: conditions.length > 0 ? conditions.join(" AND ") : null,
    args,
  };
}

/**
 * Gets month number from month name
 * @param {string} monthName - Month name
 * @returns {number|null} Month number (1-12) or null
 */
function getMonthNumber(monthName) {
  const months = {
    january: 1,
    jan: 1,
    february: 2,
    feb: 2,
    march: 3,
    mar: 3,
    april: 4,
    apr: 4,
    may: 5,
    june: 6,
    jun: 6,
    july: 7,
    jul: 7,
    august: 8,
    aug: 8,
    september: 9,
    sep: 9,
    october: 10,
    oct: 10,
    november: 11,
    nov: 11,
    december: 12,
    dec: 12,
  };

  return months[monthName.toLowerCase()] || null;
}

/**
 * Builds a BigQuery GROUP BY clause from query parameters
 * @param {object} queryParams - Query parameters
 * @param {string} itemType - Type of items
 * @param {string} priceField - Price field name
 * @param {string} temporalDateField - Temporal date field name
 * @returns {string} GROUP BY clause
 */
function buildGroupByClause(
  queryParams,
  itemType = "sold",
  priceField = "sold_price",
  temporalDateField = "sold_date"
) {
  console.log("🔍 buildGroupByClause called - NEW VERSION v2");
  console.log("🔍 queryParams:", JSON.stringify(queryParams));

  if (!queryParams.group_by) {
    return "";
  }

  const groupByFields = queryParams.group_by
    .split(",")
    .map((field) => field.trim());
  const groupByClauses = [];

  groupByFields.forEach((field) => {
    const columnName = COLUMN_MAPPINGS[field] || field;

    switch (field) {
      case "monthly":
        // Always use listed_date for monthly grouping
        groupByClauses.push(`DATE_TRUNC(${COLUMN_MAPPINGS.monthly}, MONTH)`);
        break;
      case "weekly":
        // Always use listed_date for weekly grouping
        groupByClauses.push(`DATE_TRUNC(${COLUMN_MAPPINGS.weekly}, WEEK)`);
        break;
      case "yearly":
        groupByClauses.push(`EXTRACT(YEAR FROM ${COLUMN_MAPPINGS.sold_date})`);
        break;
      case "brand":
      case "brands":
        groupByClauses.push(`${COLUMN_MAPPINGS.brand}`);
        break;
      case "model":
      case "models":
        groupByClauses.push(`${COLUMN_MAPPINGS.model}`);
        break;
      case "color":
      case "colors":
        groupByClauses.push(`${COLUMN_MAPPINGS.color}`);
        break;
      case "material":
      case "materials":
        groupByClauses.push(`${COLUMN_MAPPINGS.material}`);
        break;
      case "hardware":
      case "hardwares":
        groupByClauses.push(`${COLUMN_MAPPINGS.hardware}`);
        break;
      case "shape":
      case "shapes":
        groupByClauses.push(`${COLUMN_MAPPINGS.type}`);
        break;
      case "size":
      case "sizes":
        groupByClauses.push(`${COLUMN_MAPPINGS.size}`);
        break;
      case "condition":
      case "conditions":
        groupByClauses.push(`${COLUMN_MAPPINGS.condition}`);
        break;
      case "location":
      case "locations":
        groupByClauses.push(`${COLUMN_MAPPINGS.location}`);
        break;
      case "vendor":
      case "vendors":
        groupByClauses.push(`${COLUMN_MAPPINGS.vendor}`);
        break;
      case "gender":
        groupByClauses.push(`${COLUMN_MAPPINGS.gender}`);
        break;
      case "decade":
      case "decades":
        groupByClauses.push(`${COLUMN_MAPPINGS.decade}`);
        break;
      case "type":
      case "types":
        groupByClauses.push(`${COLUMN_MAPPINGS.type}`);
        break;
      default:
        // For any other field, use mapped column name
        groupByClauses.push(`${columnName}`);
        break;
    }
  });

  // Return GROUP BY clause without AS aliases for BigQuery compatibility
  const result =
    groupByClauses.length > 0 ? `GROUP BY ${groupByClauses.join(", ")}` : "";
  console.log("🔍 buildGroupByClause result:", result);
  console.log("🔍 groupByClauses array:", groupByClauses);
  return result;
}

/**
 * Builds a BigQuery SELECT clause with aggregation
 * @param {object} queryParams - Query parameters
 * @param {string} valueField - Value field expression
 * @param {string} itemType - Type of items
 * @param {string} priceField - Price field name
 * @param {string} temporalDateField - Temporal date field name
 * @param {object} config - Service configuration
 * @param {object} options - Query options
 * @returns {string} SELECT clause
 */
function buildSelectClause(
  queryParams,
  valueField,
  itemType = "sold",
  priceField = "sold_price",
  temporalDateField = "sold_date",
  config = null,
  options = {}
) {
  const selectFields = [];

  // Add grouping fields
  if (queryParams.group_by) {
    const groupByFields = queryParams.group_by
      .split(",")
      .map((field) => field.trim());

    groupByFields.forEach((field) => {
      const columnName = COLUMN_MAPPINGS[field] || field;

      switch (field) {
        case "monthly":
          // Always use listed_date for monthly grouping
          selectFields.push(
            `DATE_TRUNC(${COLUMN_MAPPINGS.monthly}, MONTH) AS monthly`
          );
          break;
        case "weekly":
          // Always use listed_date for weekly grouping
          selectFields.push(
            `DATE_TRUNC(${COLUMN_MAPPINGS.weekly}, WEEK) AS weekly`
          );
          break;
        case "yearly":
          selectFields.push(
            `EXTRACT(YEAR FROM ${COLUMN_MAPPINGS.sold_date}) AS yearly`
          );
          break;
        case "brand":
        case "brands":
          selectFields.push(`${COLUMN_MAPPINGS.brand} AS brand`);
          break;
        case "model":
        case "models":
          selectFields.push(`${COLUMN_MAPPINGS.model} AS model`);
          break;
        case "color":
        case "colors":
          selectFields.push(`${COLUMN_MAPPINGS.color} AS color`);
          break;
        case "material":
        case "materials":
          selectFields.push(`${COLUMN_MAPPINGS.material} AS material`);
          break;
        case "hardware":
        case "hardwares":
          selectFields.push(`${COLUMN_MAPPINGS.hardware} AS hardware`);
          break;
        case "shape":
        case "shapes":
          selectFields.push(`${COLUMN_MAPPINGS.type} AS shape`);
          break;
        case "size":
        case "sizes":
          selectFields.push(`${COLUMN_MAPPINGS.size} AS size`);
          break;
        case "condition":
        case "conditions":
          selectFields.push(`${COLUMN_MAPPINGS.condition} AS condition`);
          break;
        case "location":
        case "locations":
          selectFields.push(`${COLUMN_MAPPINGS.location} AS location`);
          break;
        case "vendor":
        case "vendors":
          selectFields.push(`${COLUMN_MAPPINGS.vendor} AS vendor`);
          break;
        case "gender":
          selectFields.push(`${COLUMN_MAPPINGS.gender} AS gender`);
          break;
        case "decade":
        case "decades":
          selectFields.push(`${COLUMN_MAPPINGS.decade} AS decade`);
          break;
        case "type":
        case "types":
          selectFields.push(`${COLUMN_MAPPINGS.type} AS type`);
          break;
        default:
          // For any other field, use mapped column name
          selectFields.push(`${columnName} AS ${field}`);
          break;
      }
    });
  }

  // Add value field
  selectFields.push(valueField);

  // Add count field
  selectFields.push("COUNT(*) AS count");

  return selectFields.join(", ");
}

/**
 * Builds a complete BigQuery query
 * @param {object} queryParams - Query parameters
 * @param {string} valueField - Value field expression
 * @param {string} itemType - Type of items
 * @param {string} priceField - Price field name
 * @param {string} sortOrder - Sort order (ASC/DESC)
 * @param {number} limit - Result limit
 * @param {number} offset - Result offset
 * @param {string} temporalDateField - Temporal date field name
 * @param {object} config - Service configuration
 * @param {object} options - Query options
 * @param {string} route - Route name
 * @returns {object} Object containing SQL query and arguments
 */
function buildCompleteQuery(
  queryParams,
  valueField,
  itemType = "sold",
  priceField = "sold_price",
  sortOrder = "DESC",
  limit = 500,
  offset = 0,
  temporalDateField = "sold_date",
  config = null,
  options = {},
  route = null
) {
  // Get table name
  const tableName = getTableName(config, route);

  // Build WHERE clause
  const { whereClause, queryArgs } = buildWhereClause(
    queryParams,
    itemType,
    priceField,
    temporalDateField,
    config,
    route
  );

  // Build SELECT clause
  const selectClause = buildSelectClause(
    queryParams,
    valueField,
    itemType,
    priceField,
    temporalDateField,
    config,
    options
  );

  // Build GROUP BY clause
  const groupByClause = buildGroupByClause(
    queryParams,
    itemType,
    priceField,
    temporalDateField
  );

  // Build ORDER BY clause
  const orderByField = queryParams.order_by || "value";
  const orderByClause = `ORDER BY ${orderByField} ${sortOrder}`;

  // Build LIMIT and OFFSET
  const limitClause = `LIMIT ${limit} OFFSET ${offset}`;

  // Construct the complete query (note: GROUP BY uses column names, not aliases)
  const sql = `
      SELECT ${selectClause}
      FROM \`${tableName}\`
      ${whereClause}
      ${groupByClause}
      ${orderByClause}
      ${limitClause}
    `.trim();

  return {
    sql: sql.replace(/\s+/g, " "),
    args: queryArgs,
  };
}

/**
 * Gets the table name for BigQuery
 * @param {object} config - Service configuration
 * @param {string} route - Route name
 * @returns {string} Table name
 */
function getTableName(config, route = null) {
  if (!config || !config.database) {
    throw new Error("Database configuration not found");
  }

  // For BigQuery, we need the full table reference
  const projectId = process.env.GCP_PROJECT_ID || "truss-data-science";
  const datasetId = config.database.dataset || "api";

  // Default table name
  let tableName = config.database.table || "display_product_listings";

  // Check if we need to filter by root_type based on route
  if (
    route &&
    config.api &&
    config.api.partitions &&
    config.api.partitions.routes
  ) {
    const routeConfig = config.api.partitions.routes[route];
    if (routeConfig && routeConfig.root_type) {
      // Add root_type filter to WHERE clause instead of using different tables
      // All data is in the same table with different root_type values
    }
  }

  return `${projectId}.${datasetId}.${tableName}`;
}

/**
 * Builds a BigQuery market share query
 * @param {object} queryParams - Query parameters
 * @param {string} itemType - Type of items
 * @param {string} priceField - Price field name
 * @param {string} temporalDateField - Temporal date field name
 * @param {object} config - Service configuration
 * @param {object} options - Query options
 * @param {string} route - Route name
 * @returns {object} Object containing SQL query and arguments
 */
function buildHavingClause(queryParams) {
  const havingConditions = [];
  const havingArgs = [];

  console.log(
    "🔧 buildHavingClause called with:",
    JSON.stringify(queryParams, null, 2)
  );
  console.log("🔧 min_samples value:", queryParams.min_samples);
  console.log("🔧 query_mode:", queryParams.query_mode);

  // Add min_samples filtering if specified
  if (queryParams.min_samples !== undefined && queryParams.min_samples > 0) {
    // For basic mode, filter on count field
    if (queryParams.query_mode === "basic_diagnostics") {
      havingConditions.push("n >= @param_" + havingArgs.length);
      console.log(
        "🔧 Adding HAVING n >= @param_" +
          havingArgs.length +
          " for basic_diagnostics mode"
      );
    } else {
      havingConditions.push("count >= @param_" + havingArgs.length);
      console.log(
        "🔧 Adding HAVING count >= @param_" +
          havingArgs.length +
          " for basic mode"
      );
    }
    havingArgs.push(queryParams.min_samples);
    console.log("🔧 HAVING args:", havingArgs);
  } else {
    console.log("🔧 No min_samples filtering applied");
  }

  const havingClause =
    havingConditions.length > 0
      ? `HAVING ${havingConditions.join(" AND ")}`
      : "";

  console.log("🔧 Final HAVING clause:", havingClause);
  console.log("🔧 Final HAVING args:", havingArgs);

  return { havingClause, havingArgs };
}

function buildMarketShareQuery(
  queryParams,
  itemType = "sold",
  priceField = "sold_price",
  temporalDateField = "sold_date",
  config = null,
  options = {},
  route = null
) {
  const tableName = getTableName(config, route);
  const { whereClause, queryArgs } = buildWhereClause(
    queryParams,
    itemType,
    priceField,
    temporalDateField,
    config,
    route
  );
  const groupByClause = buildGroupByClause(
    queryParams,
    itemType,
    priceField,
    temporalDateField
  );

  // Build SELECT clause for market share
  const selectFields = [];

  if (queryParams.group_by) {
    const groupByFields = queryParams.group_by
      .split(",")
      .map((field) => field.trim());

    groupByFields.forEach((field) => {
      const columnName = COLUMN_MAPPINGS[field] || field;

      switch (field) {
        case "monthly":
          // Always use listed_date for monthly grouping
          selectFields.push(
            `DATE_TRUNC(${COLUMN_MAPPINGS.monthly}, MONTH) AS monthly`
          );
          break;
        case "weekly":
          // Always use listed_date for weekly grouping
          selectFields.push(
            `DATE_TRUNC(${COLUMN_MAPPINGS.weekly}, WEEK) AS weekly`
          );
          break;
        case "brand":
        case "brands":
          selectFields.push(`${COLUMN_MAPPINGS.brand} AS brand`);
          break;
        case "model":
        case "models":
          selectFields.push(`${COLUMN_MAPPINGS.model} AS model`);
          break;
        case "color":
        case "colors":
          selectFields.push(`${COLUMN_MAPPINGS.color} AS color`);
          break;
        case "material":
        case "materials":
          selectFields.push(`${COLUMN_MAPPINGS.material} AS material`);
          break;
        case "hardware":
        case "hardwares":
          selectFields.push(`${COLUMN_MAPPINGS.hardware} AS hardware`);
          break;
        case "shape":
        case "shapes":
          selectFields.push(`${COLUMN_MAPPINGS.type} AS shape`);
          break;
        case "size":
        case "sizes":
          selectFields.push(`${COLUMN_MAPPINGS.size} AS size`);
          break;
        case "condition":
        case "conditions":
          selectFields.push(`${COLUMN_MAPPINGS.condition} AS condition`);
          break;
        case "location":
        case "locations":
          selectFields.push(`${COLUMN_MAPPINGS.location} AS location`);
          break;
        case "vendor":
        case "vendors":
          selectFields.push(`${COLUMN_MAPPINGS.vendor} AS vendor`);
          break;
        case "gender":
          selectFields.push(`${COLUMN_MAPPINGS.gender} AS gender`);
          break;
        case "decade":
        case "decades":
          selectFields.push(`${COLUMN_MAPPINGS.decade} AS decade`);
          break;
        case "type":
        case "types":
          selectFields.push(`${COLUMN_MAPPINGS.type} AS type`);
          break;
        default:
          // For any other field, use mapped column name
          selectFields.push(`${columnName} AS ${field}`);
          break;
      }
    });
  }

  selectFields.push(`SUM(${COLUMN_MAPPINGS.sold_price}) AS entity_gmv`);
  selectFields.push("COUNT(*) AS count");
  selectFields.push(
    `(SUM(${COLUMN_MAPPINGS.sold_price}) / SUM(SUM(${COLUMN_MAPPINGS.sold_price})) OVER ()) AS value`
  );

  const selectClause = selectFields.join(", ");
  const orderByField = queryParams.order_by || "value";
  const sortOrder = queryParams.order || "DESC";
  const limit = queryParams.limit || 500;
  const offset = queryParams.offset || 0;

  const sql = `
      SELECT ${selectClause}
      FROM \`${tableName}\`
      ${whereClause}
      ${groupByClause}
      ORDER BY ${orderByField} ${sortOrder}
      LIMIT ${limit} OFFSET ${offset}
    `.trim();

  return {
    sql: sql.replace(/\s+/g, " "),
    args: queryArgs,
  };
}

module.exports = {
  COLUMN_MAPPINGS,
  buildWhereClause,
  buildGroupByClause,
  buildSelectClause,
  buildHavingClause,
  buildCompleteQuery,
  buildMarketShareQuery,
  buildDateRangeCondition,
  getTableName,
  getMonthNumber,
};
