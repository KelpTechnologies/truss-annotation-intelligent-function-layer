/**
 * Enhanced validation utilities for robust API parameter handling
 * Based on the API conceptual guide dimensions and requirements
 */

// Valid entity fields according to the conceptual guide
const VALID_ENTITIES = [
  "brand",
  "type",
  "material",
  "color",
  "condition",
  "size",
  "vendor",
  "model",
  "gender",
  "decade",
  "location",
  "hardware",
  "key_word", // text search filter only (not groupable)
  "monthly", // allow monthly as filter
  "weekly", // allow weekly as filter
];

// Valid temporal groupings
const VALID_TEMPORAL_GROUPINGS = ["monthly", "weekly"];

// Valid metric endpoints by service
const VALID_METRICS = {
  analytics: ["gmv"],
  counts: ["sold", "listed", "sell-through"],
  prices: ["average-sold", "average-listed"],
  discounts: ["average"],
  knowledge: [
    "brands",
    "types",
    "materials",
    "colors",
    "sizes",
    "conditions",
    "models",
    "vendors",
    "summary",
  ],
  listings: [""],
  "meta-entities": [""],
  "meta-routes": ["", "endpoints", "health"],
};

// Add valid calcs for post-aggregation
const VALID_CALCS = ["percentage_change"];

// Map of which calcs require a temporal grouping
const TEMPORAL_REQUIRED = {
  percentage_change: true,
  // Add future calcs here, e.g. rolling_avg: true, some_non_temporal_calc: false
};

// Security and performance limits
const LIMITS = {
  MAX_LIMIT: 1000,
  MAX_OFFSET: 10000,
  MAX_FILTER_VALUES: 50,
  MAX_GROUP_BY_FIELDS: 5,
  MAX_STRING_LENGTH: 1000,
  MIN_NUMERIC_VALUE: 0,
  MAX_NUMERIC_VALUE: 999999999,
};

/**
 * Validates the query_mode parameter
 * @param {string} queryMode - The query mode to validate
 * @returns {string} The validated query mode
 */
function validateQueryMode(queryMode) {
  if (!queryMode) {
    return "basic"; // Default to basic mode
  }

  const validModes = ["basic", "basic_diagnostics"];
  const normalizedMode = queryMode.toLowerCase().trim();

  if (!validModes.includes(normalizedMode)) {
    throw new Error(
      `Invalid query_mode parameter. Must be one of: ${validModes.join(", ")}`
    );
  }

  return normalizedMode;
}

/**
 * Comprehensive parameter validation with security checks
 * @param {object} queryParams - Query parameters
 * @param {string} service - Service name
 * @param {string} endpoint - Endpoint name
 * @returns {object} Validated and sanitized parameters
 * @throws {Error} If parameters are invalid
 */
function validateAllParams(queryParams, service, endpoint) {
  const validated = {};
  const errors = [];

  try {
    // Validate pagination
    const pagination = validatePaginationParams(queryParams);
    validated.limit = pagination.limit;
    validated.offset = pagination.offset;

    // Validate sorting
    validated.order = validateSortOrder(queryParams.order);

    // Validate query mode
    validated.query_mode = validateQueryMode(queryParams.query_mode);

    // Validate entity filters
    VALID_ENTITIES.forEach((entity) => {
      const paramName = `${entity}s`; // Plural form
      if (queryParams[paramName] !== undefined) {
        const result = validateEntityFilter(queryParams[paramName], entity);
        if (result.error) {
          errors.push(result.error);
        } else {
          validated[paramName] = result.value;
        }
      }
    });

    // Pass-through partition root_type_id (numeric mapping via base path)
    if (queryParams.root_type_id !== undefined) {
      const rootId = validateNumericParam(
        queryParams.root_type_id,
        "root_type_id",
        0,
        LIMITS.MAX_NUMERIC_VALUE
      );
      if (rootId !== null) validated.root_type_id = Math.floor(rootId);
    }

    // Accept singular 'monthly' and 'weekly' as filter params and normalize to plural
    if (queryParams.monthly !== undefined) {
      validated.monthlys = validateArrayParam(queryParams.monthly);
    }
    if (queryParams.weekly !== undefined) {
      validated.weeklys = validateArrayParam(queryParams.weekly);
    }

    // Validate group_by parameter
    if (queryParams.group_by !== undefined) {
      const result = validateGroupBy(queryParams.group_by);
      if (result.error) {
        errors.push(result.error);
      } else {
        validated.group_by = result.value;
      }
    }

    // Validate numeric filters
    if (queryParams.min_price !== undefined) {
      const result = validateNumericParam(
        queryParams.min_price,
        "min_price",
        LIMITS.MIN_NUMERIC_VALUE,
        LIMITS.MAX_NUMERIC_VALUE
      );
      if (result === null) {
        errors.push("Invalid min_price parameter");
      } else {
        validated.min_price = result;
      }
    }

    if (queryParams.max_price !== undefined) {
      const result = validateNumericParam(
        queryParams.max_price,
        "max_price",
        LIMITS.MIN_NUMERIC_VALUE,
        LIMITS.MAX_NUMERIC_VALUE
      );
      if (result === null) {
        errors.push("Invalid max_price parameter");
      } else {
        validated.max_price = result;
      }
    }

    // Validate date parameters
    if (queryParams.start_date !== undefined) {
      const result = validateDateParam(queryParams.start_date, "start_date");
      if (result === null) {
        errors.push("Invalid start_date parameter");
      } else {
        validated.start_date = result;
      }
    }

    if (queryParams.end_date !== undefined) {
      const result = validateDateParam(queryParams.end_date, "end_date");
      if (result === null) {
        errors.push("Invalid end_date parameter");
      } else {
        validated.end_date = result;
      }
    }

    // Validate boolean parameters
    if (queryParams.sold !== undefined) {
      const result = validateBooleanParam(queryParams.sold, "sold");
      if (result === null) {
        errors.push("Invalid sold parameter");
      } else {
        validated.sold = result;
      }
    }

    // Validate return_nulls parameter
    if (queryParams.return_nulls !== undefined) {
      const result = validateBooleanParam(
        queryParams.return_nulls,
        "return_nulls"
      );
      if (result === null) {
        errors.push("Invalid return_nulls parameter");
      } else {
        validated.return_nulls = result;
      }
    } else {
      // Default to true if not provided
      validated.return_nulls = true;
    }

    // Validate min_samples parameter (integer)
    if (queryParams.min_samples !== undefined) {
      const result = validateNumericParam(
        queryParams.min_samples,
        "min_samples",
        0,
        LIMITS.MAX_NUMERIC_VALUE
      );
      if (result === null) {
        errors.push("Invalid min_samples parameter");
      } else {
        validated.min_samples = Math.floor(result); // Ensure it's an integer
      }
    } else {
      // Default to 0 if not provided
      validated.min_samples = 0;
    }

    // Validate sample parameter (boolean)
    if (queryParams.sample !== undefined) {
      const result = validateBooleanParam(queryParams.sample, "sample");
      if (result === null) {
        errors.push("Invalid sample parameter");
      } else {
        validated.sample = result;
      }
    } else {
      validated.sample = false;
    }

    // Validate calc parameter (for analytics/counts/prices/discounts/forecasts only)
    if (queryParams.calc !== undefined) {
      if (VALID_CALCS.includes(queryParams.calc)) {
        validated.calc = queryParams.calc;
      } else {
        errors.push(
          `Invalid calc parameter. Valid options: ${VALID_CALCS.join(", ")}`
        );
      }
    }

    // Always add data_points: true to signal count should be returned for every metric
    validated.data_points = true;

    // Security validations
    const securityErrors = validateSecurity(queryParams);
    errors.push(...securityErrors);

    // Business logic validations
    const businessErrors = validateBusinessLogic(validated, service, endpoint);
    errors.push(...businessErrors);

    // --- Expand temporal filters for calc if needed (was expandTemporalFiltersForCalc) ---
    if (validated.calc) {
      // Expand monthlys
      if (
        validated.monthlys &&
        Array.isArray(validated.monthlys) &&
        validated.monthlys.length > 0
      ) {
        const sorted = Array.from(new Set(validated.monthlys)).sort();
        const first = sorted[0];
        const [year, month] = first.split("-").map(Number);
        let prevYear = year;
        let prevMonth = month - 1;
        if (prevMonth === 0) {
          prevMonth = 12;
          prevYear -= 1;
        }
        const prev = `${prevYear}-${String(prevMonth).padStart(2, "0")}`;
        if (!sorted.includes(prev)) {
          validated.monthlys = [prev, ...sorted];
        } else {
          validated.monthlys = sorted;
        }
      }
      // Expand weeklys
      if (
        validated.weeklys &&
        Array.isArray(validated.weeklys) &&
        validated.weeklys.length > 0
      ) {
        const sorted = Array.from(new Set(validated.weeklys)).sort();
        const first = sorted[0];
        const [yearStr, weekStr] = first.split("-W");
        let year = Number(yearStr);
        let week = Number(weekStr);
        let prevYear = year;
        let prevWeek = week - 1;
        if (prevWeek === 0) {
          prevYear -= 1;
          prevWeek = 52;
        }
        const prev = `${prevYear}-W${String(prevWeek).padStart(2, "0")}`;
        if (!sorted.includes(prev)) {
          validated.weeklys = [prev, ...sorted];
        } else {
          validated.weeklys = sorted;
        }
      }
    }
    // --- END expand temporal filters ---
  } catch (error) {
    errors.push(error.message);
  }

  if (errors.length > 0) {
    throw new Error(`Validation failed: ${errors.join("; ")}`);
  }

  return validated;
}

/**
 * Validates pagination parameters with enhanced security
 * @param {object} queryParams - Query parameters
 * @returns {object} Validated pagination parameters
 * @throws {Error} If parameters are invalid
 */
function validatePaginationParams(queryParams) {
  const limit = parseInt(queryParams.limit);
  const offset = parseInt(queryParams.offset);

  if (
    queryParams.limit !== undefined &&
    (isNaN(limit) || limit < 1 || limit > LIMITS.MAX_LIMIT)
  ) {
    throw new Error(
      `Invalid limit parameter. Must be between 1 and ${LIMITS.MAX_LIMIT}.`
    );
  }

  if (
    queryParams.offset !== undefined &&
    (isNaN(offset) || offset < 0 || offset > LIMITS.MAX_OFFSET)
  ) {
    throw new Error(
      `Invalid offset parameter. Must be between 0 and ${LIMITS.MAX_OFFSET}.`
    );
  }

  return {
    limit: Math.min(limit || 500, LIMITS.MAX_LIMIT),
    offset: offset || 0,
  };
}

/**
 * Validates sort order parameter
 * @param {string} order - Sort order parameter
 * @returns {string} Validated sort order
 * @throws {Error} If parameter is invalid
 */
function validateSortOrder(order) {
  if (order === undefined || order === null || order === "") {
    return "DESC";
  }

  const upperOrder = order.toUpperCase();
  if (!["ASC", "DESC"].includes(upperOrder)) {
    throw new Error("Invalid order parameter. Must be 'ASC' or 'DESC'.");
  }

  return upperOrder;
}

/**
 * Validates entity filter parameters with enhanced security
 * @param {string|Array} param - Parameter value
 * @param {string} entityName - Entity name for validation
 * @returns {object} Validation result with value or error
 */
function validateEntityFilter(param, entityName) {
  if (param === undefined || param === null || param === "") {
    return { value: null };
  }

  let values;
  if (Array.isArray(param)) {
    values = param;
  } else if (typeof param === "string") {
    values = param
      .split(",")
      .map((item) => item.trim())
      .filter((item) => item !== "");
  } else {
    return { error: `Invalid ${entityName} parameter format` };
  }

  // Security checks
  if (values.length > LIMITS.MAX_FILTER_VALUES) {
    return {
      error: `Too many ${entityName} values. Maximum allowed: ${LIMITS.MAX_FILTER_VALUES}`,
    };
  }

  // Validate each value
  const validatedValues = [];
  for (const value of values) {
    if (typeof value !== "string" || value.length > LIMITS.MAX_STRING_LENGTH) {
      return { error: `Invalid ${entityName} value: ${value}` };
    }

    // Basic sanitization - remove potentially dangerous characters
    const sanitized = value.replace(/[<>\"'&]/g, "").trim();
    if (sanitized && sanitized.length > 0) {
      validatedValues.push(sanitized);
    }
  }

  return { value: validatedValues.length > 0 ? validatedValues : null };
}

/**
 * Validates group_by parameter according to conceptual guide
 * @param {string} groupBy - Group by parameter
 * @returns {object} Validation result with value or error
 */
function validateGroupBy(groupBy) {
  if (!groupBy || typeof groupBy !== "string") {
    return { error: "Invalid group_by parameter format" };
  }

  const fields = groupBy
    .split(",")
    .map((field) => field.trim())
    .filter((field) => field !== "");

  if (fields.length === 0) {
    return { error: "group_by parameter cannot be empty" };
  }

  if (fields.length > LIMITS.MAX_GROUP_BY_FIELDS) {
    return {
      error: `Too many group_by fields. Maximum allowed: ${LIMITS.MAX_GROUP_BY_FIELDS}`,
    };
  }

  const validatedFields = [];
  for (const field of fields) {
    // Check if it's a valid entity
    if (VALID_ENTITIES.includes(field)) {
      validatedFields.push(field);
    }
    // Check if it's a valid temporal grouping
    else if (VALID_TEMPORAL_GROUPINGS.includes(field)) {
      validatedFields.push(field);
    } else {
      return {
        error: `Invalid group_by field: ${field}. Valid fields: ${VALID_ENTITIES.join(
          ", "
        )}, ${VALID_TEMPORAL_GROUPINGS.join(", ")}`,
      };
    }
  }

  return { value: validatedFields.join(",") };
}

/**
 * Validates and normalizes array parameters (brands, types, etc.)
 * @param {string|Array} param - Parameter value
 * @returns {Array} Normalized array
 */
function validateArrayParam(param) {
  if (param === undefined || param === null || param === "") {
    return null;
  }

  if (Array.isArray(param)) {
    return param.filter(
      (item) => item && typeof item === "string" && item.trim() !== ""
    );
  }

  if (typeof param === "string") {
    return param
      .split(",")
      .map((item) => item.trim())
      .filter((item) => item !== "");
  }

  return null;
}

/**
 * Validates numeric parameters with enhanced bounds checking
 * @param {string|number} param - Parameter value
 * @param {string} paramName - Parameter name for error messages
 * @param {number} min - Minimum value (optional)
 * @param {number} max - Maximum value (optional)
 * @returns {number|null} Validated number or null
 * @throws {Error} If parameter is invalid
 */
function validateNumericParam(param, paramName, min = null, max = null) {
  if (param === undefined || param === null || param === "") {
    return null;
  }

  const num = parseFloat(param);
  if (isNaN(num)) {
    throw new Error(`Invalid ${paramName} parameter. Must be a valid number.`);
  }

  if (min !== null && num < min) {
    throw new Error(`Invalid ${paramName} parameter. Must be at least ${min}.`);
  }

  if (max !== null && num > max) {
    throw new Error(`Invalid ${paramName} parameter. Must be at most ${max}.`);
  }

  return num;
}

/**
 * Validates date parameters with enhanced format checking
 * @param {string} param - Date parameter value
 * @param {string} paramName - Parameter name for error messages
 * @returns {string|null} Validated date string or null
 * @throws {Error} If parameter is invalid
 */
function validateDateParam(param, paramName) {
  if (!param) return null;

  // Check for ISO date format or common date formats
  const dateRegex = /^\d{4}-\d{2}-\d{2}$/;
  if (!dateRegex.test(param)) {
    throw new Error(
      `Invalid ${paramName} parameter. Must be in YYYY-MM-DD format.`
    );
  }

  const date = new Date(param);
  if (isNaN(date.getTime())) {
    throw new Error(`Invalid ${paramName} parameter. Must be a valid date.`);
  }

  // Check if date is not in the future (for business logic)
  if (date > new Date()) {
    throw new Error(
      `Invalid ${paramName} parameter. Date cannot be in the future.`
    );
  }

  return param;
}

/**
 * Validates boolean parameters
 * @param {string} param - Boolean parameter value
 * @param {string} paramName - Parameter name for error messages
 * @returns {boolean|null} Validated boolean or null
 * @throws {Error} If parameter is invalid
 */
function validateBooleanParam(param, paramName) {
  if (param === undefined || param === null || param === "") {
    return null;
  }

  if (typeof param === "boolean") {
    return param;
  }

  if (typeof param === "string") {
    const lowerParam = param.toLowerCase();
    if (lowerParam === "true" || lowerParam === "1") {
      return true;
    }
    if (lowerParam === "false" || lowerParam === "0") {
      return false;
    }
  }

  throw new Error(
    `Invalid ${paramName} parameter. Must be 'true', 'false', '1', or '0'.`
  );
}

/**
 * Security validation checks
 * @param {object} queryParams - Query parameters
 * @returns {Array} Array of security error messages
 */
function validateSecurity(queryParams) {
  const errors = [];

  // Check for SQL injection patterns
  const sqlPatterns = [
    /(\b(union|select|insert|update|delete|drop|create|alter)\b)/i,
    /(--|\/\*|\*\/|;)/,
    /(\b(exec|execute|script|javascript|vbscript)\b)/i,
  ];

  for (const [key, value] of Object.entries(queryParams)) {
    if (typeof value === "string") {
      for (const pattern of sqlPatterns) {
        if (pattern.test(value)) {
          errors.push(`Potential security issue in parameter ${key}`);
          break;
        }
      }
    }
  }

  // Check for excessive parameter values
  for (const [key, value] of Object.entries(queryParams)) {
    if (typeof value === "string" && value.length > LIMITS.MAX_STRING_LENGTH) {
      errors.push(`Parameter ${key} value too long`);
    }
  }

  return errors;
}

/**
 * Business logic validation (relaxed: does NOT check service/endpoint)
 * @param {object} validatedParams - Validated parameters
 * @param {string} service - Service name (unused)
 * @param {string} endpoint - Endpoint name (unused)
 * @returns {Array} Array of business logic error messages
 */
function validateBusinessLogic(validatedParams, service, endpoint) {
  const errors = [];

  // Validate price range logic
  if (
    validatedParams.min_price !== undefined &&
    validatedParams.max_price !== undefined
  ) {
    if (validatedParams.min_price > validatedParams.max_price) {
      errors.push("min_price cannot be greater than max_price");
    }
  }

  // Validate date range logic
  if (validatedParams.start_date && validatedParams.end_date) {
    const startDate = new Date(validatedParams.start_date);
    const endDate = new Date(validatedParams.end_date);
    if (startDate > endDate) {
      errors.push("start_date cannot be after end_date");
    }
  }

  // Validate group_by with temporal grouping
  if (validatedParams.group_by) {
    const fields = validatedParams.group_by.split(",");
    // Use the top-level VALID_ENTITIES (which includes 'decade')
    const hasTemporal = fields.some((field) =>
      VALID_TEMPORAL_GROUPINGS.includes(field)
    );
    const hasEntity = fields.some((field) => VALID_ENTITIES.includes(field));

    if (hasTemporal && !hasEntity) {
      // Temporal grouping without entity grouping is valid
    } else if (!hasTemporal && !hasEntity) {
      errors.push(
        "group_by must include at least one valid entity or temporal grouping"
      );
    }
  }

  // Enforce: if calc=percentage_change, group_by must include a valid temporal grouping
  if (validatedParams.calc === "percentage_change") {
    const temporalGroupings = VALID_TEMPORAL_GROUPINGS;
    const groupByFields = (validatedParams.group_by || "").split(",");
    const hasTemporal = groupByFields.some((f) =>
      temporalGroupings.includes(f)
    );
    if (!hasTemporal) {
      errors.push(
        "The 'percentage_change' calc requires group_by to include a valid temporal grouping (e.g., 'monthly')."
      );
    }
  }

  // Enforce: if calc requires temporal grouping, group_by must include a valid temporal grouping
  if (validatedParams.calc && TEMPORAL_REQUIRED[validatedParams.calc]) {
    const groupByFields = (validatedParams.group_by || "").split(",");
    const hasTemporal = groupByFields.some((f) =>
      VALID_TEMPORAL_GROUPINGS.includes(f)
    );
    if (!hasTemporal) {
      errors.push(
        `The '${
          validatedParams.calc
        }' calc requires group_by to include a valid temporal grouping (e.g., '${VALID_TEMPORAL_GROUPINGS.join(
          ", "
        )}').`
      );
    }
  }

  return errors;
}

/**
 * Sanitizes input strings to prevent injection attacks
 * @param {string} input - Input string
 * @returns {string} Sanitized string
 */
function sanitizeString(input) {
  if (typeof input !== "string") {
    return input;
  }

  // Remove potentially dangerous characters
  return input
    .replace(/[<>\"'&]/g, "")
    .replace(/\b(union|select|insert|update|delete|drop|create|alter)\b/gi, "")
    .trim();
}

/**
 * Validates and sanitizes all string parameters
 * @param {object} params - Parameters object
 * @returns {object} Sanitized parameters
 */
function sanitizeParams(params) {
  const sanitized = {};

  for (const [key, value] of Object.entries(params)) {
    if (typeof value === "string") {
      sanitized[key] = sanitizeString(value);
    } else if (Array.isArray(value)) {
      sanitized[key] = value.map((item) =>
        typeof item === "string" ? sanitizeString(item) : item
      );
    } else {
      sanitized[key] = value;
    }
  }

  return sanitized;
}

module.exports = {
  validateAllParams,
  validatePaginationParams,
  validateSortOrder,
  validateArrayParam,
  validateNumericParam,
  validateDateParam,
  validateBooleanParam,
  validateEntityFilter,
  validateGroupBy,
  validateSecurity,
  validateBusinessLogic,
  sanitizeString,
  sanitizeParams,
  VALID_ENTITIES,
  VALID_TEMPORAL_GROUPINGS,
  VALID_METRICS,
  LIMITS,
  VALID_CALCS,
  TEMPORAL_REQUIRED,
  validateQueryMode,
};
