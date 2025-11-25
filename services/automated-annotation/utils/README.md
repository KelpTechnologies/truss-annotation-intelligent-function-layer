# Shared Utilities Toolkit

This directory contains shared utilities that are automatically copied to all services during deployment. These utilities provide common functionality, reduce code duplication, and ensure consistency across all services.

## 📁 Structure

```
utils/
├── index.js           # Main export file (imports all utilities)
├── database.js        # Database connection and query utilities
├── response.js        # HTTP response formatting utilities
├── validation.js      # Parameter validation utilities
├── query-builder.js   # SQL query construction utilities
├── data-parser.js     # Data filtering and parsing utilities
├── constants.js       # Shared constants and configuration
├── logger.js          # Logging utilities
└── README.md          # This file
```

## 🚀 Usage

### Importing Utilities

You can import utilities in several ways:

```javascript
// Import everything
const utils = require("./utils");

// Import specific modules
const {
  query,
  createSuccessResponse,
  validatePaginationParams,
} = require("./utils");

// Import individual modules
const database = require("./utils/database");
const response = require("./utils/response");
```

### Basic Service Setup

Here's how to set up a service using the shared utilities:

```javascript
const config = require("./config.json");
const {
  query,
  createSuccessResponse,
  createErrorResponse,
  createNotFoundResponse,
  createCorsResponse,
  validatePaginationParams,
  validateSortOrder,
  parseAndFilterData,
  logServiceRequest,
  logServiceError,
} = require("./utils");

// Main Lambda handler
exports.handler = async (event) => {
  logServiceRequest(config.service.name, event);

  // Handle CORS preflight
  if (event.httpMethod === "OPTIONS") {
    return createCorsResponse();
  }

  try {
    const path =
      event.path?.replace(config.api.base_path, "").replace(/^\/+/, "") || "";
    const result = await routeRequest(path, event);
    return result;
  } catch (error) {
    logServiceError(config.service.name, "handler", error);
    return createErrorResponse(error, config);
  }
};

// Route handler example
async function getData(queryParams, headers) {
  try {
    const { limit, offset } = validatePaginationParams(queryParams);
    const sortOrder = validateSortOrder(queryParams.order);

    const sql = `SELECT * FROM table LIMIT ? OFFSET ?`;
    const results = await query(sql, [limit, offset], config);

    // Filter data using the parser
    const filteredResults = parseAndFilterData(results);

    return createSuccessResponse(
      filteredResults,
      config,
      "data",
      queryParams,
      { limit, offset, has_more: results.length === limit },
      "data_analysis"
    );
  } catch (error) {
    logServiceError(config.service.name, "getData", error);
    return createErrorResponse(error, config);
  }
}
```

## 📚 Module Reference

### Database Utilities (`database.js`)

Database connection and query management.

```javascript
const { query, initDatabase, closePool } = require("./utils/database");

// Execute a query
const results = await query(sql, args, config);

// Close pool (usually not needed in Lambda)
closePool();
```

### Response Utilities (`response.js`)

Standardized HTTP response formatting.

```javascript
const {
  createSuccessResponse,
  createErrorResponse,
  createNotFoundResponse,
  createCorsResponse,
  getCorsHeaders,
} = require("./utils/response");

// Success response
return createSuccessResponse(
  data,
  config,
  endpoint,
  queryParams,
  pagination,
  componentType
);

// Error response
return createErrorResponse(error, config, statusCode);

// Not found response
return createNotFoundResponse(path, config);

// CORS response
return createCorsResponse();
```

### Validation Utilities (`validation.js`)

Parameter validation and normalization.

```javascript
const {
  validatePaginationParams,
  validateSortOrder,
  validateArrayParam,
  validateNumericParam,
  validateDateParam,
  validateBooleanParam,
} = require("./utils/validation");

// Validate pagination
const { limit, offset } = validatePaginationParams(queryParams);

// Validate sort order
const sortOrder = validateSortOrder(queryParams.order);

// Validate array parameters
const brands = validateArrayParam(queryParams.brands);

// Validate numeric parameters
const minPrice = validateNumericParam(queryParams.min_price, "min_price", 0);

// Validate date parameters
const startDate = validateDateParam(queryParams.start_date, "start_date");

// Validate boolean parameters
const isSold = validateBooleanParam(queryParams.is_sold, "is_sold");
```

### Query Builder Utilities (`query-builder.js`)

SQL query construction helpers.

```javascript
const {
  buildWhereClause,
  buildGroupByClause,
  buildSelectClause,
  buildCompleteQuery,
} = require("./utils/query-builder");

// Build WHERE clause
const { whereClause, queryArgs } = buildWhereClause(
  params,
  "sold",
  "sold_price"
);

// Build GROUP BY clause
const groupByClause = buildGroupByClause(queryParams);

// Build SELECT clause
const selectClause = buildSelectClause(queryParams, "SUM(sold_price) AS value");

// Build complete query
const { sql, args } = buildCompleteQuery(
  queryParams,
  "SUM(sold_price) AS value",
  "sold",
  "sold_price",
  "DESC",
  500,
  0
);
```

### Data Parser Utilities (`data-parser.js`)

Data filtering and parsing functions.

```javascript
const {
  parseAndFilterData,
  parseAndRemoveNulls,
  filterByThreshold,
  mapDbConditionToApi,
  mapDbColourToColor,
  mapDbHardwareToApi,
  mapDbLocationToApi,
  mapDbMaterialToApi,
  mapAllEntities,
  postAggregationCalcs,
  processDataWithNullsOption,
  mapDbTypeToShape,
} = require("./utils/data-parser");

// Remove nulls and filter by threshold (default 5%)
const filteredData = parseAndFilterData(results);

// Remove nulls and filter by threshold, but keep nulls if returnNulls is true
const filteredDataWithNulls = parseAndFilterData(results, 5, true);

// Remove nulls only
const cleanData = parseAndRemoveNulls(results);

// Filter by custom threshold (10%)
const thresholdData = filterByThreshold(results, 10);

// Individual entity field mappings
const conditionMapped = mapDbConditionToApi(results); // _condition → condition
const colorMapped = mapDbColourToColor(results); // colour → color
const hardwareMapped = mapDbHardwareToApi(results); // primary_hardware_materials → hardware
const locationMapped = mapDbLocationToApi(results); // sold_location → location
const materialMapped = mapDbMaterialToApi(results); // material_parent → material
const typeMapped = mapDbTypeToShape(results); // type → shape

// ✨ Consolidated mapping function (recommended)
const allMapped = mapAllEntities(results); // Applies all entity mappings

// Process data with optional null filtering based on return_nulls parameter
const processedData = processDataWithNullsOption(results, false); // Remove nulls
const processedDataWithNulls = processDataWithNullsOption(results, true); // Keep nulls

// Filter out internal calculation fields before sending to client
const cleanData = filterInternalCalculationFields(
  results,
  "analytics",
  "market-share"
);
```

#### **Entity Field Mappings**

The `mapAllEntities` function consolidates all entity field mappings to provide consistent API responses:

| Database Field               | API Field   | Description                                         |
| ---------------------------- | ----------- | --------------------------------------------------- |
| `_condition`                 | `condition` | Item condition (e.g., "Excellent", "Good")          |
| `colour`                     | `color`     | American spelling standardization                   |
| `primary_hardware_materials` | `hardware`  | Hardware material type (e.g., "Gold", "Silver")     |
| `sold_location`              | `location`  | Location where item was sold (e.g., "US", "UK")     |
| `material_parent`            | `material`  | Primary material type (e.g., "Lambskin", "Canvas")  |
| `type`                       | `shape`     | Item type (e.g., "Bag", "Shoe")                     |
| `listing_title`              | `key_word`  | Text search in product listing titles (filter only) |

**Usage Pattern:**

```javascript
// ❌ Old approach (individual mappings)
let results = queryResults;
results = mapDbConditionToApi(results);
results = mapDbColourToColor(results);
// ... more individual mappings

// ✅ New approach (consolidated)
const results = mapAllEntities(queryResults);
```

#### **Internal Calculation Field Filtering**

The `filterInternalCalculationFields` function removes internal calculation fields that shouldn't be exposed to API clients using a **blanket filtering approach**:

```javascript
// Blanket filter - removes all non-client fields across all services
const cleanData = filterInternalCalculationFields(
  results,
  "analytics",
  "market-share"
);
```

**How It Works:**

- **Client-Allowed Columns:** Only fields defined in the product listing schema and entity definitions are allowed
- **Entity Fields:** `brand`, `type`, `material`, `color`, `condition`, `size`, `vendor`, `gender`, `model`, `decade`, `location`, `hardware`, `key_word`
- **Temporal Fields:** `monthly`, `weekly`
- **Value Fields:** `value`, `count`, `percentage_change`
- **Product Fields:** All fields from product listing schema including dates, prices, dimensions, etc.
- **Automatic Filtering:** Any field not in the allowed list is automatically removed

**Applied to All Services:**

- ✅ `knowledge` (brands, types, materials, colors, etc.)

This ensures that internal calculation values and working fields are never exposed in API responses.

### Constants (`constants.js`)

Shared constants and configuration values.

```javascript
const {
  DATABASE_CONFIG,
  PAGINATION,
  SORT_ORDERS,
  GROUP_BY_FIELDS,
  ITEM_TYPES,
  PRICE_FIELDS,
  AGGREGATION_FUNCTIONS,
  HTTP_STATUS,
  ERROR_MESSAGES,
  COMPONENT_TYPES,
  FILTER_THRESHOLDS,
  DATA_PROCESSING,
  CORS_HEADERS,
} = require("./utils/constants");

// Use constants
const limit = Math.min(userLimit, PAGINATION.MAX_LIMIT);
const sortOrder = SORT_ORDERS.DESC;
const componentType = COMPONENT_TYPES.GMV_ANALYSIS;
const returnNulls = DATA_PROCESSING.DEFAULT_RETURN_NULLS;
```

### Logger Utilities (`logger.js`)

Structured logging with different levels.

```javascript
const {
  error,
  warn,
  info,
  debug,
  logServiceRequest,
  logDatabaseQuery,
  logDatabaseResults,
  logServiceResponse,
  logServiceError,
} = require("./utils/logger");

// Log levels
error("Error message", errorObject, serviceName);
warn("Warning message", data, serviceName);
info("Info message", data, serviceName);
debug("Debug message", data, serviceName);

// Specialized logging
logServiceRequest(serviceName, event);
logDatabaseQuery(sql, args, serviceName);
logDatabaseResults(results, serviceName);
logServiceResponse(serviceName, endpoint, statusCode, dataCount);
logServiceError(serviceName, endpoint, error);
```

## 🔧 Configuration

### Environment Variables

- `LOG_LEVEL`: Set logging level (`ERROR`, `WARN`, `INFO`, `DEBUG`)
- `RDS_PROXY_ENDPOINT`: Database host endpoint
- `DB_HOST`: Alternative database host
- `DB_USER`: Database username
- `DB_PASSWORD`: Database password

### Service Configuration

The utilities expect a service configuration object with this structure:

```json
{
  "service": {
    "name": "service-name"
  },
  "api": {
    "base_path": "/api/service-name",
    "endpoints": [...]
  },
  "database": {
    "required": true
  }
}
```

## 🚀 Deployment

The utils folder is automatically copied to all services during deployment using the `copy-utils.js` script. This happens:

1. **Before deployment**: When running `deploy-all-services.js`
2. **During packaging**: When packaging individual services
3. **Manually**: By running `node scripts/copy-utils.js`

### Manual Copy

```bash
# Copy to all services
node scripts/copy-utils.js

# Copy to specific service
node scripts/copy-utils.js knowledge
```

## 📝 Best Practices

1. **Use the shared utilities** instead of duplicating code
2. **Import specific functions** rather than entire modules when possible
3. **Use the logger** for consistent logging across services
4. **Validate parameters** using the validation utilities
5. **Use constants** for magic numbers and strings
6. **Handle errors** using the error response utilities
7. **Filter data** using the parser utilities for consistent results

## 🔄 Updates

When updating utilities:

1. Modify the files in `services/utils/`
2. Run `node scripts/copy-utils.js` to copy to all services
3. Deploy services to get the updated utilities

## 🐛 Troubleshooting

### Common Issues

1. **Module not found**: Ensure utils folder exists in your service
2. **Database connection errors**: Check environment variables
3. **Validation errors**: Use proper parameter validation
4. **Logging not working**: Check LOG_LEVEL environment variable

### Debug Mode

Enable debug logging by setting the environment variable:

```bash
export LOG_LEVEL=DEBUG
```

This will show detailed information about database queries, parameter validation, and data processing.

## 🧮 Dynamic Post-Aggregation Calculations (calc parameter)

All aggregation endpoints now support a `calc` query parameter for dynamic post-aggregation insights.

- **Supported calcs:**
  - `percentage_change`: Month-on-month and week-on-week change as a percentage (decimal, e.g., 0.12 for +12%)
- **How it works:**
  - Add `calc=percentage_change` to your query (must include `group_by=monthly` or `group_by=weekly`)
  - The API will add a `value` field to each row (except the first period for each group), representing the percentage change from the previous period.
  - The schema and flat parameter model are preserved.

## 🚫 Null Value Filtering (return_nulls parameter)

All endpoints now support a `return_nulls` query parameter to control whether null values are included in the response.

- **Parameter:** `return_nulls` (boolean)
- **Default:** `true` (return null values)
- **Values:** `true`, `false`, `1`, `0`
- **How it works:**
  - When `return_nulls=true` (default): All data is returned, including null values
  - When `return_nulls=false`: Objects with null values are filtered out after data fetching and calculations
  - This filtering happens after all database queries and post-aggregation calculations are applied

**Example Usage:**

```javascript
// Keep null values (default behavior)
const dataWithNulls = await getData({ return_nulls: true });

// Filter out null values
const dataWithoutNulls = await getData({ return_nulls: false });

// In service implementation
const processedData = processDataWithNullsOption(
  results,
  queryParams.return_nulls
);
```

### Example

Request:

```
GET /api/knowledge/brands?group_by=monthly&calc=percentage_change
```

Response:

```json
[
  { "brand": "Gucci", "monthly": "2024-05", "count": 1000, "percentage_change": null },
  { "brand": "Gucci", "monthly": "2024-06", "count": 1200, "percentage_change": 0.2 },
  { "brand": "Gucci", "monthly": "2024-07", "count": 1100, "percentage_change": -0.0833 },
  ...
]
```

Weekly request:

```
GET /api/knowledge/brands?group_by=weekly&calc=percentage_change
```

### Utility function

```javascript
const { postAggregationCalcs } = require("./utils/data-parser");
const resultWithCalcs = postAggregationCalcs(
  data,
  "percentage_change",
  "brand,weekly"
);
```

- More calcs (rolling averages, yoy, etc.) can be added in the future.
