const { validateArrayParam } = require("./validation");
const { FIELD_MAPPINGS } = require("./constants");

/**
 * Gets the table name from the route configuration
 * @param {object} config - Service configuration
 * @param {string} route - Route name (e.g., "bags", "api", "apparel", "footwear")
 * @returns {string} Table name or default table name
 */
function getTableName(config, route = null) {
  // Default table name for backward compatibility
  const DEFAULT_TABLE = "display_product_listings";

  // If no config or route provided, return default
  if (!config || !route) {
    return DEFAULT_TABLE;
  }

  // Try to get table name from route configuration
  const routeConfig = config?.api?.partitions?.routes?.[route];
  if (routeConfig && routeConfig.table) {
    return routeConfig.table;
  }

  // Try to find route by base_path matching
  if (config?.api?.partitions?.routes) {
    for (const [routeKey, routeCfg] of Object.entries(
      config.api.partitions.routes
    )) {
      if (routeCfg?.base_path && routeCfg.table) {
        // If route matches or route is a substring match (e.g., "gmv" in "/bags/gmv")
        if (route === routeKey || route.includes(routeKey)) {
          return routeCfg.table;
        }
      }
    }
  }

  return DEFAULT_TABLE;
}

/**
 * Builds the WHERE clause for filtering data based on query parameters.
 * @param {object} params - Query parameters from the request.
 * @param {string} itemType - Type of items to filter ("sold", "listed", "all")
 * @param {string} priceField - Price field to filter for ("sold_price" or "listed_price")
 * @returns {object} Object containing the WHERE clause and query arguments.
 */
function buildWhereClause(
  params,
  itemType = "sold",
  priceField = "sold_price",
  temporalDateField = null
) {
  const conditions = [];
  const queryArgs = [];

  // Partition-based root_type_id filter (if injected by router mapping)
  if (params && params.root_type_id) {
    conditions.push("root_type_id = ?");
    queryArgs.push(parseInt(params.root_type_id, 10));
  }

  // Add item type specific filters
  if (itemType === "sold") {
    conditions.push("is_sold = 1");
  }

  // Add price field specific filters
  if (priceField) {
    conditions.push(`${priceField} IS NOT NULL`);
    if (priceField === "listed_price") {
      conditions.push("listed_price > 0");
    }
  }

  // Filter by brands
  if (params.brands) {
    const brands = validateArrayParam(params.brands);
    if (brands && brands.length > 0) {
      const placeholders = brands.map(() => "?").join(", ");
      conditions.push(`${FIELD_MAPPINGS.brand} IN (${placeholders})`);
      queryArgs.push(...brands);
    }
  }

  // Filter by types
  if (params.types) {
    const types = validateArrayParam(params.types);
    if (types && types.length > 0) {
      const placeholders = types.map(() => "?").join(", ");
      conditions.push(`${FIELD_MAPPINGS.type} IN (${placeholders})`);
      queryArgs.push(...types);
    }
  }

  // Filter by materials
  if (params.materials) {
    const materials = validateArrayParam(params.materials);
    if (materials && materials.length > 0) {
      const placeholders = materials.map(() => "?").join(", ");
      conditions.push(`${FIELD_MAPPINGS.material} IN (${placeholders})`);
      queryArgs.push(...materials);
    }
  }

  // Filter by colors
  if (params.colors) {
    const colors = validateArrayParam(params.colors);
    if (colors && colors.length > 0) {
      const placeholders = colors.map(() => "?").join(", ");
      conditions.push(`colour IN (${placeholders})`);
      queryArgs.push(...colors);
    }
  }

  // Filter by conditions
  if (params.conditions) {
    const conditions_list = validateArrayParam(params.conditions);
    if (conditions_list && conditions_list.length > 0) {
      const placeholders = conditions_list.map(() => "?").join(", ");
      // Only backtick the alias/keyword; DB column comes from mapping
      const conditionCol = FIELD_MAPPINGS.condition || "`condition`";
      conditions.push(`${conditionCol} IN (${placeholders})`);
      queryArgs.push(...conditions_list);
    }
  }

  // Filter by sizes
  if (params.sizes) {
    const sizes = validateArrayParam(params.sizes);
    if (sizes && sizes.length > 0) {
      const placeholders = sizes.map(() => "?").join(", ");
      conditions.push(`${FIELD_MAPPINGS.size} IN (${placeholders})`);
      queryArgs.push(...sizes);
    }
  }

  // Filter by locations
  if (params.locations) {
    const locations = validateArrayParam(params.locations);
    if (locations && locations.length > 0) {
      const placeholders = locations.map(() => "?").join(", ");
      conditions.push(`${FIELD_MAPPINGS.location} IN (${placeholders})`);
      queryArgs.push(...locations);
    }
  }

  // Filter by hardware
  if (params.hardwares) {
    const hardwares = validateArrayParam(params.hardwares);
    if (hardwares && hardwares.length > 0) {
      const placeholders = hardwares.map(() => "?").join(", ");
      conditions.push(`${FIELD_MAPPINGS.hardware} IN (${placeholders})`);
      queryArgs.push(...hardwares);
    }
  }

  // Filter by vendors
  if (params.vendors) {
    const vendors = validateArrayParam(params.vendors);
    if (vendors && vendors.length > 0) {
      const placeholders = vendors.map(() => "?").join(", ");
      conditions.push(`${FIELD_MAPPINGS.vendor} IN (${placeholders})`);
      queryArgs.push(...vendors);
    }
  }

  // Filter by genders
  if (params.genders) {
    const genders = validateArrayParam(params.genders);
    if (genders && genders.length > 0) {
      const placeholders = genders.map(() => "?").join(", ");
      conditions.push(`${FIELD_MAPPINGS.gender} IN (${placeholders})`);
      queryArgs.push(...genders);
    }
  }

  // Filter by models
  if (params.models) {
    const models = validateArrayParam(params.models);
    if (models && models.length > 0) {
      const placeholders = models.map(() => "?").join(", ");
      conditions.push(`${FIELD_MAPPINGS.model} IN (${placeholders})`);
      queryArgs.push(...models);
    }
  }

  // Filter by decades
  if (params.decades) {
    const decades = validateArrayParam(params.decades);
    if (decades && decades.length > 0) {
      const placeholders = decades.map(() => "?").join(", ");
      conditions.push(`${FIELD_MAPPINGS.decade} IN (${placeholders})`);
      queryArgs.push(...decades);
    }
  }

  // Filter by key words (text search in listing_title)
  if (params.key_words) {
    const keyWords = validateArrayParam(params.key_words);
    if (keyWords && keyWords.length > 0) {
      // For MySQL, use LIKE with wildcards for text search
      const searchConditions = keyWords.map(() => `${FIELD_MAPPINGS.key_word} LIKE ?`);
      conditions.push(`(${searchConditions.join(" OR ")})`);
      // Add wildcards around each keyword for partial matching
      queryArgs.push(...keyWords.map(keyword => `%${keyword}%`));
    }
  }

  // Determine which date field to use for temporal filtering
  let temporalField = temporalDateField || "listed_date";
  if (!temporalDateField) {
    // Fallback logic for backward compatibility
    if (itemType === "sold" || priceField === "sold_price") {
      temporalField = "sold_date";
    }
  }
  // Choose the correct month column for monthly filters
  const monthlyColumn =
    temporalField === "sold_date" ? "sold_month" : "listed_month";

  // Filter by monthly (use precomputed month columns)
  if (params.monthlys) {
    const monthlys = validateArrayParam(params.monthlys);
    if (monthlys && monthlys.length > 0) {
      const placeholders = monthlys.map(() => "?").join(", ");
      // Convert YYYY-MM format to YYYY-MM-01 for date comparison
      const dateValues = monthlys.map((month) => month + "-01");
      conditions.push(`${monthlyColumn} IN (${placeholders})`);
      queryArgs.push(...dateValues);
    }
  }

  // Filter by weekly
  if (params.weeklys) {
    const weeklys = validateArrayParam(params.weeklys);
    if (weeklys && weeklys.length > 0) {
      const placeholders = weeklys.map(() => "?").join(", ");
      conditions.push(
        `DATE_FORMAT(${temporalField}, '%Y-%u') IN (${placeholders})`
      );
      queryArgs.push(...weeklys);
    }
  }

  // Price range filters
  if (params.min_price) {
    conditions.push(`${priceField} >= ?`);
    queryArgs.push(parseFloat(params.min_price));
  }

  if (params.max_price) {
    conditions.push(`${priceField} <= ?`);
    queryArgs.push(parseFloat(params.max_price));
  }

  // Date range filters
  if (params.listed_after) {
    conditions.push("listed_date >= ?");
    queryArgs.push(params.listed_after);
  }

  if (params.listed_before) {
    conditions.push("listed_date <= ?");
    queryArgs.push(params.listed_before);
  }

  if (params.sold_after) {
    conditions.push("sold_date >= ?");
    queryArgs.push(params.sold_after);
  }

  if (params.sold_before) {
    conditions.push("sold_date <= ?");
    queryArgs.push(params.sold_before);
  }

  // Text search
  if (params.search) {
    const brandCol = FIELD_MAPPINGS.brand;
    const typeCol = FIELD_MAPPINGS.type;
    const materialCol = FIELD_MAPPINGS.material;
    const colorCol = FIELD_MAPPINGS.color;
    const modelCol = FIELD_MAPPINGS.model;
    conditions.push(
      `(${brandCol} LIKE ? OR ${typeCol} LIKE ? OR ${materialCol} LIKE ? OR ${colorCol} LIKE ? OR ${modelCol} LIKE ? OR model_child LIKE ?)`
    );
    const searchTerm = `%${params.search}%`;
    queryArgs.push(
      searchTerm,
      searchTerm,
      searchTerm,
      searchTerm,
      searchTerm,
      searchTerm
    );
  }

  // --- GROUP BY non-null enforcement ---
  // If group_by is present, require all group-by fields to be non-null
  const groupByParams = require("./validation").validateArrayParam(
    params.group_by
  );
  if (groupByParams && groupByParams.length > 0) {
    // Map API group_by fields to DB columns
    groupByParams.forEach((field) => {
      const dbCol = FIELD_MAPPINGS[field.trim()];
      if (dbCol) {
        conditions.push(`${dbCol} IS NOT NULL`);
      }
    });
  }
  // --- END GROUP BY non-null enforcement ---

  const whereClause =
    conditions.length > 0 ? `WHERE ${conditions.join(" AND ")}` : "";

  return { whereClause, queryArgs };
}

/**
 * Builds the WHERE clause for knowledge service entity endpoints with additional conditions.
 * @param {object} params - Query parameters from the request.
 * @param {Array} additionalConditions - Additional WHERE conditions to include.
 * @returns {object} Object containing the WHERE clause and query arguments.
 */
function buildEntityWhereClause(params, additionalConditions = []) {
  const conditions = [...additionalConditions];
  const args = [];

  // Filter by brands
  const brands = validateArrayParam(params.brands);
  if (brands && brands.length > 0) {
    conditions.push(
      `${FIELD_MAPPINGS.brand} IN (${brands.map(() => "?").join(",")})`
    );
    args.push(...brands);
  }

  // Filter by types
  const types = validateArrayParam(params.types);
  if (types && types.length > 0) {
    conditions.push(
      `${FIELD_MAPPINGS.type} IN (${types.map(() => "?").join(",")})`
    );
    args.push(...types);
  }

  // Filter by materials
  const materials = validateArrayParam(params.materials);
  if (materials && materials.length > 0) {
    conditions.push(
      `${FIELD_MAPPINGS.material} IN (${materials.map(() => "?").join(",")})`
    );
    args.push(...materials);
  }

  // Filter by colors
  const colors = validateArrayParam(params.colors);
  if (colors && colors.length > 0) {
    conditions.push(`colour IN (${colors.map(() => "?").join(",")})`);
    args.push(...colors);
  }

  // Filter by conditions
  const conditions_list = validateArrayParam(params.conditions);
  if (conditions_list && conditions_list.length > 0) {
    conditions.push(
      `${FIELD_MAPPINGS.condition} IN (${conditions_list
        .map(() => "?")
        .join(",")})`
    );
    args.push(...conditions_list);
  }

  // Filter by sizes
  const sizes = validateArrayParam(params.sizes);
  if (sizes && sizes.length > 0) {
    conditions.push(
      `${FIELD_MAPPINGS.size} IN (${sizes.map(() => "?").join(",")})`
    );
    args.push(...sizes);
  }

  // Filter by locations
  const locations = validateArrayParam(params.locations);
  if (locations && locations.length > 0) {
    conditions.push(
      `${FIELD_MAPPINGS.location} IN (${locations.map(() => "?").join(",")})`
    );
    args.push(...locations);
  }

  // Filter by hardware
  const hardwares = validateArrayParam(params.hardwares);
  if (hardwares && hardwares.length > 0) {
    conditions.push(
      `${FIELD_MAPPINGS.hardware} IN (${hardwares.map(() => "?").join(",")})`
    );
    args.push(...hardwares);
  }

  // Filter by vendors
  const vendors = validateArrayParam(params.vendors);
  if (vendors && vendors.length > 0) {
    conditions.push(
      `${FIELD_MAPPINGS.vendor} IN (${vendors.map(() => "?").join(",")})`
    );
    args.push(...vendors);
  }

  // Filter by genders
  const genders = validateArrayParam(params.genders);
  if (genders && genders.length > 0) {
    conditions.push(
      `${FIELD_MAPPINGS.gender} IN (${genders.map(() => "?").join(",")})`
    );
    args.push(...genders);
  }

  // Filter by models
  const models = validateArrayParam(params.models);
  if (models && models.length > 0) {
    conditions.push(
      `${FIELD_MAPPINGS.model} IN (${models.map(() => "?").join(",")})`
    );
    args.push(...models);
  }

  // Filter by decades
  const decades = validateArrayParam(params.decades);
  if (decades && decades.length > 0) {
    conditions.push(`decade IN (${decades.map(() => "?").join(",")})`);
    args.push(...decades);
  }

  // Filter by key words (text search in listing_title)
  const keyWords = validateArrayParam(params.key_words);
  if (keyWords && keyWords.length > 0) {
    // For MySQL, use LIKE with wildcards for text search
    const searchConditions = keyWords.map(() => `${FIELD_MAPPINGS.key_word} LIKE ?`);
    conditions.push(`(${searchConditions.join(" OR ")})`);
    // Add wildcards around each keyword for partial matching
    args.push(...keyWords.map(keyword => `%${keyword}%`));
  }

  const whereClause =
    conditions.length > 0 ? `WHERE ${conditions.join(" AND ")}` : "";

  return { whereClause, queryArgs: args };
}

/**
 * Builds the GROUP BY clause based on group_by parameters.
 * @param {object} queryParams - Query parameters from the request.
 * @param {string} itemType - Type of items to filter ("sold", "listed", "all")
 * @param {string} priceField - Price field to filter for ("sold_price" or "listed_price")
 * @param {string} temporalDateField - Date field to use for temporal groupings ("sold_date" or "listed_date")
 * @returns {string} The GROUP BY clause.
 */
function buildGroupByClause(
  queryParams,
  itemType = "sold",
  priceField = "sold_price",
  temporalDateField = null
) {
  const groupByParams = validateArrayParam(queryParams.group_by);
  if (!groupByParams || groupByParams.length === 0) {
    return "";
  }

  // Determine which month column to use for temporal grouping
  const monthlyColumn =
    (temporalDateField || itemType === "listed" || priceField === "listed_price"
      ? temporalDateField || "listed_date"
      : "sold_date") === "sold_date"
      ? "sold_month"
      : "listed_month";
  // Keep a date field for weekly formatting
  let monthlyField = temporalDateField || "sold_date";
  if (!temporalDateField) {
    // Fallback logic for backward compatibility
    if (itemType === "listed" || priceField === "listed_price") {
      monthlyField = "listed_date";
    }
  }

  const validFields = FIELD_MAPPINGS;

  const validGroupBys = groupByParams
    .map((field) => {
      const trimmed = field.trim();
      if (trimmed === "monthly") return monthlyColumn;
      return validFields[trimmed];
    })
    .filter(Boolean);

  return validGroupBys.length > 0 ? `GROUP BY ${validGroupBys.join(", ")}` : "";
}

/**
 * Generic function to build confidence metrics SQL based on query mode and field type
 * @param {string} queryMode - Query mode (fast or slow)
 * @param {string} valueField - Field to calculate metrics on (e.g., 'sold_price', 'discount', etc.)
 * @param {string} aggregationType - Type of aggregation (avg, sum, count, etc.)
 * @param {object} config - Service configuration
 * @param {object} options - Additional options for specific calculations
 * @returns {string} SQL aggregation function
 */
function buildConfidenceMetricsSQL(
  queryMode,
  valueField,
  aggregationType = "avg",
  config = null,
  options = {}
) {
  if (queryMode === "basic") {
    // Basic mode: return basic metrics only
    switch (aggregationType) {
      case "avg":
        return `AVG(${valueField}) AS value, COUNT(*) AS count`;
      case "sum":
        return `SUM(${valueField}) AS value, COUNT(*) AS count`;
      case "count":
        return `COUNT(*) AS value, COUNT(*) AS count`;
      case "custom":
        return (
          options.customBasicSQL ||
          `AVG(${valueField}) AS value, COUNT(*) AS count`
        );
      default:
        return `AVG(${valueField}) AS value, COUNT(*) AS count`;
    }
  }

  // Basic diagnostics mode: comprehensive statistical analysis
  const zScore =
    config?.confidence_metrics?.modes?.basic_diagnostics?.z_score || 1.96;

  // For complex expressions like DATEDIFF, we need to use a subquery approach
  const isComplexExpression =
    valueField.includes("(") && valueField.includes(")");

  if (isComplexExpression) {
    // Use subquery approach for complex expressions
    return `
      AVG(${valueField}) AS final_value,
      COUNT(*) AS n,
      VAR_POP(${valueField}) AS variance,
      STDDEV_POP(${valueField}) AS stddev,
      STDDEV_POP(${valueField}) / SQRT(COUNT(*)) AS sem,
      AVG(${valueField}) - ${zScore} * STDDEV_POP(${valueField}) / SQRT(COUNT(*)) AS ci95_lower,
      AVG(${valueField}) + ${zScore} * STDDEV_POP(${valueField}) / SQRT(COUNT(*)) AS ci95_upper,
      MIN(${valueField}) AS min_val,
      MAX(${valueField}) AS max_val,
      (MAX(${valueField}) - MIN(${valueField})) AS range_val,
      CASE WHEN AVG(${valueField}) = 0 THEN NULL ELSE STDDEV_POP(${valueField}) / AVG(${valueField}) END AS cv,
      CASE WHEN STDDEV_POP(${valueField}) = 0 THEN NULL ELSE
        (AVG(POWER(${valueField},3)) - 3*AVG(${valueField})*AVG(POWER(${valueField},2)) + 2*POWER(AVG(${valueField}),3))
        / POWER(STDDEV_POP(${valueField}), 3)
      END AS skewness,
      CASE WHEN STDDEV_POP(${valueField}) = 0 THEN NULL ELSE
        (AVG(POWER(${valueField},4))
          - 4*AVG(${valueField})*AVG(POWER(${valueField},3))
          + 6*POWER(AVG(${valueField}),2)*AVG(POWER(${valueField},2))
          - 3*POWER(AVG(${valueField}),4))
        / POWER(STDDEV_POP(${valueField}), 4) - 3
      END AS kurtosis_excess
    `
      .replace(/\s+/g, " ")
      .trim();
  }

  switch (aggregationType) {
    case "avg":
      return `
        AVG(${valueField}) AS final_value,
        COUNT(*) AS n,
        VAR_POP(${valueField}) AS variance,
        STDDEV_POP(${valueField}) AS stddev,
        STDDEV_POP(${valueField}) / SQRT(COUNT(*)) AS sem,
        AVG(${valueField}) - ${zScore} * STDDEV_POP(${valueField}) / SQRT(COUNT(*)) AS ci95_lower,
        AVG(${valueField}) + ${zScore} * STDDEV_POP(${valueField}) / SQRT(COUNT(*)) AS ci95_upper,
        MIN(${valueField}) AS min_val,
        MAX(${valueField}) AS max_val,
        (MAX(${valueField}) - MIN(${valueField})) AS range_val,
        CASE WHEN AVG(${valueField}) = 0 THEN NULL ELSE STDDEV_POP(${valueField}) / AVG(${valueField}) END AS cv,
        CASE WHEN STDDEV_POP(${valueField}) = 0 THEN NULL ELSE
          (AVG(POWER(${valueField},3)) - 3*AVG(${valueField})*AVG(POWER(${valueField},2)) + 2*POWER(AVG(${valueField}),3))
          / POWER(STDDEV_POP(${valueField}), 3)
        END AS skewness,
        CASE WHEN STDDEV_POP(${valueField}) = 0 THEN NULL ELSE
          (AVG(POWER(${valueField},4))
            - 4*AVG(${valueField})*AVG(POWER(${valueField},3))
            + 6*POWER(AVG(${valueField}),2)*AVG(POWER(${valueField},2))
            - 3*POWER(AVG(${valueField}),4))
          / POWER(STDDEV_POP(${valueField}), 4) - 3
        END AS kurtosis_excess
      `
        .replace(/\s+/g, " ")
        .trim();

    case "sum":
      return `
        SUM(${valueField}) AS final_value,
        COUNT(*) AS n,
        VAR_POP(${valueField}) AS variance,
        STDDEV_POP(${valueField}) AS stddev,
        STDDEV_POP(${valueField}) / SQRT(COUNT(*)) AS sem,
        SUM(${valueField}) - ${zScore} * STDDEV_POP(${valueField}) / SQRT(COUNT(*)) AS ci95_lower,
        SUM(${valueField}) + ${zScore} * STDDEV_POP(${valueField}) / SQRT(COUNT(*)) AS ci95_upper,
        MIN(${valueField}) AS min_val,
        MAX(${valueField}) AS max_val,
        (MAX(${valueField}) - MIN(${valueField})) AS range_val,
        CASE WHEN SUM(${valueField}) = 0 THEN NULL ELSE STDDEV_POP(${valueField}) / SUM(${valueField}) END AS cv,
        CASE WHEN STDDEV_POP(${valueField}) = 0 THEN NULL ELSE
          (AVG(POWER(${valueField},3)) - 3*AVG(${valueField})*AVG(POWER(${valueField},2)) + 2*POWER(AVG(${valueField}),3))
          / POWER(STDDEV_POP(${valueField}), 3)
        END AS skewness,
        CASE WHEN STDDEV_POP(${valueField}) = 0 THEN NULL ELSE
          (AVG(POWER(${valueField},4))
            - 4*AVG(${valueField})*AVG(POWER(${valueField},3))
            + 6*POWER(AVG(${valueField}),2)*AVG(POWER(${valueField},2))
            - 3*POWER(AVG(${valueField}),4))
          / POWER(STDDEV_POP(${valueField}), 4) - 3
        END AS kurtosis_excess
      `
        .replace(/\s+/g, " ")
        .trim();

    case "count":
      return `
        COUNT(*) AS final_value,
        COUNT(*) AS n,
        0 AS variance,
        0 AS stddev,
        0 AS sem,
        COUNT(*) AS ci95_lower,
        COUNT(*) AS ci95_upper,
        COUNT(*) AS min_val,
        COUNT(*) AS max_val,
        0 AS range_val,
        0 AS cv,
        0 AS skewness,
        0 AS kurtosis_excess
      `
        .replace(/\s+/g, " ")
        .trim();

    case "custom":
      return (
        options.customBasicDiagnosticsSQL ||
        buildConfidenceMetricsSQL(queryMode, valueField, "avg", config)
      );

    default:
      return buildConfidenceMetricsSQL(queryMode, valueField, "avg", config);
  }
}

/**
 * Helper function to build SELECT clause with confidence metrics support
 * @param {object} queryParams - Query parameters
 * @param {string} valueField - Field to aggregate (e.g., "SUM(sold_price)", "COUNT(*)", "AVG(discount)")
 * @param {string} itemType - Type of items to filter ("sold", "listed", "all")
 * @param {string} priceField - Price field to filter for ("sold_price" or "listed_price")
 * @param {string} temporalDateField - Date field to use for temporal groupings ("sold_date" or "listed_date")
 * @param {object} config - Service configuration for confidence metrics
 * @param {object} options - Additional options for specific calculations
 * @returns {string} The SELECT clause
 */
function buildSelectClause(
  queryParams,
  valueField,
  itemType = "sold",
  priceField = "sold_price",
  temporalDateField = null,
  config = null,
  options = {}
) {
  console.log("🔧 buildSelectClause called");
  console.log("🔧 Query params:", JSON.stringify(queryParams, null, 2));
  console.log("🔧 Value field:", valueField);
  console.log("🔧 Item type:", itemType);
  console.log("🔧 Price field:", priceField);
  console.log("🔧 Temporal date field:", temporalDateField);

  const groupByParams = queryParams.group_by
    ? queryParams.group_by.split(",").map((f) => f.trim())
    : [];
  console.log("🔧 Group by params:", groupByParams);

  const validFields = FIELD_MAPPINGS;
  console.log("🔧 Valid fields mapping:", JSON.stringify(validFields, null, 2));

  // Check if we're using BigQuery
  const isBigQuery = config?.database?.connection_type === "bigquery";

  const monthlySelectColumn = isBigQuery
    ? temporalDateField ||
      itemType === "listed" ||
      priceField === "listed_price"
      ? temporalDateField === "sold_date"
        ? "sold_date"
        : "listed_date"
      : "sold_date"
    : temporalDateField ||
      itemType === "listed" ||
      priceField === "listed_price"
    ? temporalDateField === "sold_date"
      ? "sold_month"
      : "listed_month"
    : "sold_month";

  const selectFields = groupByParams
    .map((field) => {
      if (field === "monthly") {
        if (isBigQuery) {
          return `FORMAT_DATE('%Y-%m', ${monthlySelectColumn}) AS monthly`;
        } else {
          return `DATE_FORMAT(${monthlySelectColumn}, '%Y-%m') AS monthly`;
        }
      }
      if (field === "weekly") {
        if (isBigQuery) {
          return `FORMAT_DATE('%Y-%W', ${
            temporalDateField || "sold_date"
          }) AS weekly`;
        } else {
          return `DATE_FORMAT(${
            temporalDateField || "sold_date"
          }, '%Y-%u') AS weekly`;
        }
      }
      const dbCol = validFields[field];
      if (field === "condition" && dbCol) return `${dbCol} AS \`condition\``;
      if (dbCol === "root_model") return "root_model AS model";
      if (dbCol) return `${dbCol} AS ${field}`;
      return null;
    })
    .filter(Boolean);
  console.log("🔧 Select fields:", selectFields);

  const selectList =
    selectFields.length > 0 ? selectFields.join(", ") + "," : "";
  console.log("🔧 Select list:", selectList);

  // Check if confidence metrics are enabled and what mode to use
  const queryMode = queryParams.query_mode || "basic";
  const confidenceEnabled = config?.confidence_metrics?.enabled;
  console.log("🔧 Query mode:", queryMode);
  console.log("🔧 Confidence enabled:", confidenceEnabled);

  // Ensure we have a valueField - this is required for proper service configuration
  if (!valueField) {
    throw new Error(
      `valueField is required but not provided. Service configuration must specify the aggregation field (e.g., "SUM(sold_price) AS value", "COUNT(*) AS value", "AVG(discount) AS value"). Check service configuration for endpoint: ${
        options.endpoint || "unknown"
      }`
    );
  }

  let valueCalculation;
  if (confidenceEnabled && queryMode === "basic_diagnostics") {
    // Determine aggregation type from valueField
    let aggregationType = "avg";
    if (valueField.includes("SUM(")) aggregationType = "sum";
    else if (valueField.includes("COUNT(")) aggregationType = "count";
    else if (valueField.includes("AVG(")) aggregationType = "avg";

    // Extract the actual field name from the aggregation function
    const fieldMatch = valueField.match(/\((.*?)\)/);
    const actualField = fieldMatch ? fieldMatch[1] : priceField;
    console.log("🔧 Aggregation type:", aggregationType);
    console.log("🔧 Actual field:", actualField);

    // Check if this is a complex expression (like DATEDIFF)
    const isComplexExpression =
      actualField.includes("(") && actualField.includes(")");

    if (isComplexExpression) {
      // For complex expressions like DATEDIFF, we need to use a different approach
      // We'll create a CTE-like structure using a subquery in the FROM clause
      console.log("🔧 Complex expression detected, using CTE approach");

      // For complex expressions, we'll use a simpler approach that works with MySQL
      // We'll calculate the basic metrics and let the application handle complex statistics
      valueCalculation = `
        AVG(${actualField}) AS final_value,
        COUNT(*) AS n,
        NULL AS variance,
        NULL AS stddev,
        NULL AS sem,
        NULL AS ci95_lower,
        NULL AS ci95_upper,
        MIN(${actualField}) AS min_val,
        MAX(${actualField}) AS max_val,
        (MAX(${actualField}) - MIN(${actualField})) AS range_val,
        NULL AS cv,
        NULL AS skewness,
        NULL AS kurtosis_excess
      `
        .replace(/\s+/g, " ")
        .trim();
    } else {
      // Use the standard confidence metrics approach for simple fields
      valueCalculation = buildConfidenceMetricsSQL(
        queryMode,
        actualField,
        aggregationType,
        config,
        options
      );
    }
  } else {
    // Use basic metrics (fast mode or confidence disabled)
    valueCalculation = valueField;
  }
  console.log("🔧 Value calculation:", valueCalculation);

  // Always expose a count column in basic mode to populate processConfidenceMetrics
  let finalSelect;
  if (queryMode !== "basic_diagnostics") {
    finalSelect = `${selectList} ${valueCalculation}, COUNT(*) AS count`;
  } else {
    finalSelect = `${selectList} ${valueCalculation}`;
  }
  console.log("🔧 Final SELECT clause:", finalSelect);

  return finalSelect;
}

/**
 * Generic function to build complete SQL query with confidence metrics support
 * @param {object} queryParams - Query parameters
 * @param {string} valueField - Field to aggregate
 * @param {string} itemType - Type of items to filter
 * @param {string} priceField - Price field to filter
 * @param {string} sortOrder - Sort order (ASC/DESC)
 * @param {number} limit - Limit for results
 * @param {number} offset - Offset for pagination
 * @param {string} temporalDateField - Date field to use for temporal groupings ("sold_date" or "listed_date")
 * @param {object} config - Service configuration for confidence metrics
 * @param {object} options - Additional options for specific calculations
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
  temporalDateField = null,
  config = null,
  options = {},
  route = null
) {
  console.log("🔧 buildCompleteQuery called");
  console.log("🔧 Query params:", JSON.stringify(queryParams, null, 2));
  console.log("🔧 Value field:", valueField);
  console.log("🔧 Item type:", itemType);
  console.log("🔧 Price field:", priceField);
  console.log("🔧 Sort order:", sortOrder);
  console.log("🔧 Limit:", limit);
  console.log("🔧 Offset:", offset);
  console.log("🔧 Temporal date field:", temporalDateField);
  console.log("🔧 Route:", route);

  // Get table name from route configuration
  const tableName = getTableName(config, route);
  console.log("🔧 Using table name:", tableName);

  // Build WHERE clause
  const { whereClause, queryArgs } = buildWhereClause(
    queryParams,
    itemType,
    priceField,
    temporalDateField
  );
  console.log("🔧 WHERE clause:", whereClause);
  console.log("🔧 Query args:", queryArgs);

  // Build GROUP BY clause
  const groupByClause = buildGroupByClause(
    queryParams,
    itemType,
    priceField,
    temporalDateField
  );
  console.log("🔧 GROUP BY clause:", groupByClause);

  // Build SELECT clause with confidence metrics support
  const selectClause = buildSelectClause(
    queryParams,
    valueField,
    itemType,
    priceField,
    temporalDateField,
    config,
    options
  );
  console.log("🔧 SELECT clause:", selectClause);

  // Build HAVING clause for min_samples filtering
  const { havingClause, havingArgs } = buildHavingClause(queryParams);
  console.log("🔧 HAVING clause:", havingClause);
  console.log("🔧 HAVING args:", havingArgs);

  // Determine the correct ORDER BY field based on query mode
  const orderByField =
    options.orderByField ||
    (queryParams.query_mode === "basic_diagnostics" ? "final_value" : "value");
  console.log("🔧 ORDER BY field:", orderByField);

  // Build complete SQL
  const sql = `
    SELECT ${selectClause}
    FROM ${tableName}
    ${whereClause}
    ${groupByClause}
    ${havingClause}
    ORDER BY ${orderByField} ${sortOrder}
    LIMIT ? OFFSET ?
  `
    .replace(/\s+/g, " ")
    .trim();
  console.log("🔧 Complete SQL:", sql);

  // Calculate dynamic limit for temporal groupings
  let dynamicLimit = limit;
  let temporalMultiplier = 1;

  if (queryParams.group_by) {
    const groupByFields = queryParams.group_by.split(",").map((f) => f.trim());
    const hasTemporal = groupByFields.some(
      (f) => f === "monthly" || f === "weekly"
    );

    if (hasTemporal && !queryParams.calc) {
      // For temporal groupings without calc, we need to limit unique entity combinations
      temporalMultiplier = 2; // Conservative multiplier
      dynamicLimit = Math.floor(limit / temporalMultiplier);
    }
  }
  console.log("🔧 Dynamic limit:", dynamicLimit);
  console.log("🔧 Temporal multiplier:", temporalMultiplier);

  const result = {
    sql,
    args: [...queryArgs, ...havingArgs, dynamicLimit, offset],
    dynamicLimit,
    temporalMultiplier,
  };
  console.log("🔧 Final query result:", JSON.stringify(result, null, 2));

  return result;
}

/**
 * Specialized function for building market share queries with confidence metrics
 * @param {object} queryParams - Query parameters
 * @param {string} itemType - Type of items to filter
 * @param {string} priceField - Price field to filter
 * @param {string} temporalDateField - Date field for temporal groupings
 * @param {object} config - Service configuration
 * @param {object} options - Additional options
 * @returns {object} Object containing SQL query and arguments
 */
function buildMarketShareQuery(
  queryParams,
  itemType = "sold",
  priceField = "sold_price",
  temporalDateField = null,
  config = null,
  options = {},
  route = null
) {
  const { whereClause, queryArgs } = buildWhereClause(
    queryParams,
    itemType,
    priceField,
    temporalDateField
  );

  const groupByClause = buildGroupByClause(
    queryParams,
    itemType,
    priceField,
    temporalDateField
  );

  // Get table name from route configuration
  const tableName = getTableName(config, route);
  console.log("🔧 buildMarketShareQuery using table name:", tableName);

  const groupByParams = queryParams.group_by
    ? queryParams.group_by.split(",").map((f) => f.trim())
    : [];

  const validFields = FIELD_MAPPINGS;

  const monthlySelectColumnMs =
    temporalDateField || itemType === "listed" || priceField === "listed_price"
      ? temporalDateField === "sold_date"
        ? "sold_month"
        : "listed_month"
      : "sold_month";

  const selectFields = groupByParams
    .map((field) => {
      if (field === "monthly")
        return `DATE_FORMAT(${monthlySelectColumnMs}, '%Y-%m') AS monthly`;
      if (field === "weekly")
        return `DATE_FORMAT(${
          temporalDateField || "sold_date"
        }, '%Y-%u') AS weekly`;
      const dbCol = validFields[field];
      if (dbCol === "root_model") return "root_model AS model";
      if (dbCol) return `${dbCol} AS ${field}`;
      return null;
    })
    .filter(Boolean);

  const selectList =
    selectFields.length > 0 ? selectFields.join(", ") + "," : "";

  const hasTemporalGrouping = groupByParams.some(
    (field) => field === "monthly" || field === "weekly"
  );

  // Build market share calculation with confidence metrics support
  const queryMode = queryParams.query_mode || "basic";
  const confidenceEnabled = config?.confidence_metrics?.enabled;

  let valueCalculation;
  if (confidenceEnabled && queryMode === "basic_diagnostics") {
    const zScore =
      config.confidence_metrics?.modes?.basic_diagnostics?.z_score || 1.96;
    const monthlyPartitionCol =
      temporalDateField ||
      itemType === "listed" ||
      priceField === "listed_price"
        ? temporalDateField === "sold_date"
          ? "sold_month"
          : "listed_month"
        : "sold_month";
    valueCalculation = `
      SUM(${priceField}) AS entity_gmv,
      COUNT(*) AS n,
      ROUND((SUM(${priceField}) / SUM(SUM(${priceField})) OVER (PARTITION BY ${groupByParams
      .filter((field) => field === "monthly" || field === "weekly")
      .map((field) => {
        if (field === "monthly") return monthlyPartitionCol;
        if (field === "weekly")
          return (
            "DATE_FORMAT(" + (temporalDateField || "sold_date") + ", '%Y-%u')"
          );
        return null;
      })
      .filter(Boolean)
      .join(", ")})) * 100, 2) AS final_value,
      VAR_POP(${priceField}) AS variance,
      STDDEV_POP(${priceField}) AS stddev,
      STDDEV_POP(${priceField}) / SQRT(COUNT(*)) AS sem,
      ROUND((SUM(${priceField}) / SUM(SUM(${priceField})) OVER (PARTITION BY ${groupByParams
      .filter((field) => field === "monthly" || field === "weekly")
      .map((field) => {
        if (field === "monthly") return monthlyPartitionCol;
        if (field === "weekly")
          return (
            "DATE_FORMAT(" + (temporalDateField || "sold_date") + ", '%Y-%u')"
          );
        return null;
      })
      .filter(Boolean)
      .join(
        ", "
      )})) * 100, 2) - ${zScore} * STDDEV_POP(${priceField}) / SQRT(COUNT(*)) AS ci95_lower,
      ROUND((SUM(${priceField}) / SUM(SUM(${priceField})) OVER (PARTITION BY ${groupByParams
      .filter((field) => field === "monthly" || field === "weekly")
      .map((field) => {
        if (field === "monthly") return monthlyPartitionCol;
        if (field === "weekly")
          return (
            "DATE_FORMAT(" + (temporalDateField || "sold_date") + ", '%Y-%u')"
          );
        return null;
      })
      .filter(Boolean)
      .join(
        ", "
      )})) * 100, 2) + ${zScore} * STDDEV_POP(${priceField}) / SQRT(COUNT(*)) AS ci95_upper,
      MIN(${priceField}) AS min_val,
      MAX(${priceField}) AS max_val,
      (MAX(${priceField}) - MIN(${priceField})) AS range_val,
      CASE WHEN AVG(${priceField}) = 0 THEN NULL ELSE STDDEV_POP(${priceField}) / AVG(${priceField}) END AS cv,
      CASE WHEN STDDEV_POP(${priceField}) = 0 THEN NULL ELSE
        (AVG(POWER(${priceField},3)) - 3*AVG(${priceField})*AVG(POWER(${priceField},2)) + 2*POWER(AVG(${priceField}),3))
        / POWER(STDDEV_POP(${priceField}), 3)
      END AS skewness,
      CASE WHEN STDDEV_POP(${priceField}) = 0 THEN NULL ELSE
        (AVG(POWER(${priceField},4))
          - 4*AVG(${priceField})*AVG(POWER(${priceField},3))
          + 6*POWER(AVG(${priceField}),2)*AVG(POWER(${priceField},2))
          - 3*POWER(AVG(${priceField}),4))
        / POWER(STDDEV_POP(${priceField}), 4) - 3
      END AS kurtosis_excess
    `
      .replace(/\s+/g, " ")
      .trim();
  } else {
    const monthlyPartitionCol =
      temporalDateField ||
      itemType === "listed" ||
      priceField === "listed_price"
        ? temporalDateField === "sold_date"
          ? "sold_month"
          : "listed_month"
        : "sold_month";
    valueCalculation = `
      SUM(${priceField}) AS entity_gmv,
      COUNT(*) AS count,
      ROUND((SUM(${priceField}) / SUM(SUM(${priceField})) OVER (PARTITION BY ${groupByParams
      .filter((field) => field === "monthly" || field === "weekly")
      .map((field) => {
        if (field === "monthly") return monthlyPartitionCol;
        if (field === "weekly")
          return (
            "DATE_FORMAT(" + (temporalDateField || "sold_date") + ", '%Y-%u')"
          );
        return null;
      })
      .filter(Boolean)
      .join(", ")})) * 100, 2) AS value
    `
      .replace(/\s+/g, " ")
      .trim();
  }

  // Build HAVING clause for min_samples filtering
  const { havingClause, havingArgs } = buildHavingClause(queryParams);

  const orderByField =
    options.orderByField ||
    (queryMode === "basic_diagnostics" ? "final_value" : "value");

  const sql = `
    SELECT
      ${selectList}
      ${valueCalculation}
    FROM ${tableName}
    ${whereClause}
    ${groupByClause}
    ${havingClause}
    ORDER BY ${orderByField} ${queryParams.order || "DESC"}
    LIMIT ? OFFSET ?
  `
    .replace(/\s+/g, " ")
    .trim();

  return {
    sql,
    args: [
      ...queryArgs,
      ...havingArgs,
      queryParams.limit || 500,
      queryParams.offset || 0,
    ],
    hasTemporalGrouping,
  };
}

/**
 * Builds the HAVING clause for filtering groups based on minimum sample count
 * @param {object} queryParams - Query parameters from the request
 * @returns {object} Object containing the HAVING clause and additional query arguments
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
      havingConditions.push("n >= ?");
      console.log("🔧 Adding HAVING n >= ? for basic_diagnostics mode");
    } else {
      havingConditions.push("count >= ?");
      console.log("🔧 Adding HAVING count >= ? for basic mode");
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

  return { havingClause, havingArgs };
}

module.exports = {
  buildWhereClause,
  buildEntityWhereClause,
  buildGroupByClause,
  buildSelectClause,
  buildCompleteQuery,
  buildConfidenceMetricsSQL,
  buildMarketShareQuery,
  buildHavingClause,
  getTableName,
};
