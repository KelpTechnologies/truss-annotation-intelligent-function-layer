#!/usr/bin/env node

/**
 * Script to consolidate service registry and aggregated OpenAPI YAML into a Swagger JSON template
 * This creates a comprehensive API documentation template for Swagger UI
 */

const fs = require("fs");
const path = require("path");
const yaml = require("js-yaml");

function stripRootSuffix(name) {
  return typeof name === "string" ? name.replace(/-(bags|all)$/i, "") : name;
}

/**
 * Main function to generate Swagger template
 */
function generateSwaggerTemplate(options = {}) {
  const {
    serviceRegistryPath = "api-gateway/service-registry.json",
    openApiPath = "api-gateway/aggregated-openapi.yaml",
    outputPath = "swagger-template.json",
    baseUrl = "https://lealvbo928.execute-api.eu-west-2.amazonaws.com/external-prod",
    root = null,
  } = options;

  console.log("🔍 Loading service registry and OpenAPI specification...");

  // Load service registry
  let serviceRegistry;
  try {
    serviceRegistry = JSON.parse(fs.readFileSync(serviceRegistryPath, "utf8"));
    console.log(
      `✅ Loaded service registry with ${serviceRegistry.services.length} services`
    );
  } catch (error) {
    throw new Error(`Failed to load service registry: ${error.message}`);
  }

  // Load aggregated OpenAPI YAML
  let openApiSpec;
  try {
    const yamlContent = fs.readFileSync(openApiPath, "utf8");
    openApiSpec = yaml.load(yamlContent);
    console.log(
      `✅ Loaded OpenAPI specification with ${
        Object.keys(openApiSpec.paths || {}).length
      } paths`
    );
  } catch (error) {
    throw new Error(`Failed to load OpenAPI specification: ${error.message}`);
  }

  // Generate comprehensive Swagger template using OpenAPI as the base for path structure
  const swaggerTemplate = buildSwaggerTemplate(
    serviceRegistry,
    openApiSpec,
    baseUrl,
    root
  );

  // Write output
  try {
    fs.writeFileSync(outputPath, JSON.stringify(swaggerTemplate, null, 2));
    console.log(`✅ Swagger template written to: ${outputPath}`);
  } catch (error) {
    throw new Error(`Failed to write swagger template: ${error.message}`);
  }

  return {
    outputPath,
    servicesCount: serviceRegistry.services.length,
    pathsCount: Object.keys(swaggerTemplate.paths).length,
    tagsCount: swaggerTemplate.tags.length,
  };
}

/**
 * Build Swagger template from OpenAPI (structure) + Service Registry (parameters)
 */
function buildSwaggerTemplate(
  serviceRegistry,
  openApiSpec,
  baseUrl,
  root = null
) {
  const title = root
    ? `Truss ${root.charAt(0).toUpperCase() + root.slice(1)} Market Data API`
    : "TRUSS Market Data API";

  // Base object
  const swagger = {
    openapi: "3.0.1",
    info: {
      title,
      version: "1.1.0",
      description: buildApiDescription(serviceRegistry, root),
      contact: {
        name: "TRUSS Market Data API Team",
        email: "api-support@truss.com",
      },
      license: {
        name: "Proprietary",
        url: "https://truss.com/license",
      },
    },
    servers: [
      {
        url: "https://api.trussarchive.io",
        description: "Truss Data Service API",
      },
    ],
    paths: {},
    components: {
      securitySchemes: {},
      schemas: {},
      parameters: {},
      responses: {},
      examples: {},
    },
    security: [
      {
        ApiKeyAuth: [],
      },
    ],
    tags: buildTagDefinitions(serviceRegistry),
  };

  // Filter OpenAPI paths by root (e.g., /bags/* or /all/*) when requested
  let filteredPaths = openApiSpec.paths || {};
  if (root && typeof root === "string") {
    const prefix = `/${root}/`;
    filteredPaths = Object.keys(filteredPaths)
      .filter((p) => p.startsWith(prefix) || p.startsWith("/meta/"))
      .sort()
      .reduce((acc, key) => {
        acc[key] = openApiSpec.paths[key];
        return acc;
      }, {});
    console.log(
      `🔎 Filtered OpenAPI paths by root '${root}': ${
        Object.keys(filteredPaths).length
      }`
    );
  }

  // Start with the filtered OpenAPI paths (preserving integration blocks, headers, etc.)
  const mergedPaths = JSON.parse(JSON.stringify(filteredPaths));

  // Merge parameters from registry into the operations
  mergePathsWithParameters(mergedPaths, serviceRegistry);

  // Merge components from OpenAPI, filtering Cognito out and keeping API Key
  if (openApiSpec.components) {
    if (openApiSpec.components.securitySchemes) {
      const filteredSchemes = {};
      Object.keys(openApiSpec.components.securitySchemes).forEach((key) => {
        if (key !== "CognitoAuth") {
          filteredSchemes[key] = openApiSpec.components.securitySchemes[key];
        }
      });
      swagger.components.securitySchemes = {
        ApiKeyAuth: {
          type: "apiKey",
          in: "header",
          name: "X-API-Key",
        },
        ...filteredSchemes,
      };
    } else {
      // Ensure ApiKeyAuth exists
      swagger.components.securitySchemes = {
        ApiKeyAuth: {
          type: "apiKey",
          in: "header",
          name: "X-API-Key",
        },
      };
    }

    if (openApiSpec.components.schemas) {
      Object.assign(swagger.components.schemas, openApiSpec.components.schemas);
    }
    if (openApiSpec.components.responses) {
      Object.assign(
        swagger.components.responses,
        openApiSpec.components.responses
      );
    }
    if (openApiSpec.components.parameters) {
      Object.assign(
        swagger.components.parameters,
        openApiSpec.components.parameters
      );
    }
    if (openApiSpec.components.examples) {
      Object.assign(
        swagger.components.examples,
        openApiSpec.components.examples
      );
    }
  }

  // Replace any remaining Cognito mentions in paths
  const pathsString = JSON.stringify(mergedPaths);
  const updatedPathsString = pathsString
    .replace(/CognitoAuth/g, "ApiKeyAuth")
    .replace(/ - Auth: cognito/g, " - Auth: api_key")
    .replace(/"cognito"/g, '"api_key"');
  swagger.paths = JSON.parse(updatedPathsString);

  // Extend components with our registry-based parameter/schema helpers
  Object.assign(
    swagger.components.parameters,
    buildParameterDefinitions(serviceRegistry)
  );
  Object.assign(
    swagger.components.schemas,
    buildSchemaDefinitions(serviceRegistry)
  );
  Object.assign(swagger.components.responses, buildResponseDefinitions());
  Object.assign(
    swagger.components.examples,
    buildExampleDefinitions(serviceRegistry)
  );

  return swagger;
}

/**
 * Merge OpenAPI paths with parameters from service registry
 */
function mergePathsWithParameters(openApiPaths, serviceRegistry) {
  const parameterDefinitions = buildParameterDefinitions(serviceRegistry);

  serviceRegistry.services.forEach((service) => {
    const tagName = stripRootSuffix(service.name);
    if (service.endpoints) {
      service.endpoints.forEach((endpoint) => {
        const fullPath = `${service.base_path}${endpoint.path}`;
        const method = (endpoint.method || "get").toLowerCase();

        if (openApiPaths[fullPath] && openApiPaths[fullPath][method]) {
          const operation = openApiPaths[fullPath][method];

          // Assemble parameters from endpoint definition using component refs
          const parameters = [];
          const epParams = endpoint.parameters || {};

          if (epParams.filters) {
            epParams.filters.forEach((filter) => {
              if (parameterDefinitions[filter]) {
                parameters.push({ $ref: `#/components/parameters/${filter}` });
              }
            });
          }
          if (epParams.group_by) {
            parameters.push({ $ref: "#/components/parameters/group_by" });
          }
          if (epParams.pagination) {
            epParams.pagination.forEach((p) => {
              if (parameterDefinitions[p]) {
                parameters.push({ $ref: `#/components/parameters/${p}` });
              }
            });
          }
          if (epParams.ordering) {
            epParams.ordering.forEach((p) => {
              if (parameterDefinitions[p]) {
                parameters.push({ $ref: `#/components/parameters/${p}` });
              }
            });
          }
          if (epParams.sorting) {
            epParams.sorting.forEach((p) => {
              if (parameterDefinitions[p]) {
                parameters.push({ $ref: `#/components/parameters/${p}` });
              }
            });
          }
          if (epParams.options) {
            epParams.options.forEach((p) => {
              if (parameterDefinitions[p]) {
                parameters.push({ $ref: `#/components/parameters/${p}` });
              }
            });
          }
          if (epParams.calc) {
            parameters.push({ $ref: "#/components/parameters/calc" });
          }

          // Attach or merge parameters with existing ones
          if (!operation.parameters) {
            operation.parameters = parameters;
          } else {
            const existing = new Set(
              operation.parameters.map((p) => p.$ref || p.name)
            );
            parameters.forEach((p) => {
              const key = p.$ref || p.name;
              if (!existing.has(key)) {
                operation.parameters.push(p);
                existing.add(key);
              }
            });
          }

          // Normalize security to API Key
          operation.security = [{ ApiKeyAuth: [] }];

          // Normalize description auth mention if present
          if (operation.description) {
            operation.description = operation.description.replace(
              / - Auth: cognito/g,
              " - Auth: api_key"
            );
          }

          // Assign normalized tag for grouping
          operation.tags = [tagName];
        }
      });
    }
  });
}

/**
 * Build comprehensive API description
 */
function buildApiDescription(serviceRegistry, root = null) {
  const heading =
    root === "bags"
      ? "# TRUSS Bags Market Data API"
      : "# TRUSS Market Data API";
  const intro =
    root === "bags"
      ? "A comprehensive analytics API for luxury handbags market data, enabling insights across brands, models, materials, hardware, and more."
      : "A comprehensive analytics API for garment market data, enabling powerful business insights across brands, types, colors, and more.";

  let description = `${heading}

${intro}

## Authentication

All endpoints require authentication:

- **API Key**: Use \`X-API-Key\` header for access

## Base URL

\`https://api.trussarchive.io/\`

## Services Overview

`;

  // Add service descriptions
  serviceRegistry.services.forEach((service) => {
    description += `### ${
      stripRootSuffix(service.name).charAt(0).toUpperCase() +
      stripRootSuffix(service.name).slice(1)
    } Service\n`;
    description += `${service.description}\n\n`;

    // Add endpoint summary
    if (service.endpoints && service.endpoints.length > 0) {
      description += `**Endpoints:**\n`;
      service.endpoints.forEach((endpoint) => {
        if (endpoint.path !== "/health") {
          description += `- \`${endpoint.method} ${service.base_path}${endpoint.path}\` - ${endpoint.description}\n`;
        }
      });
      description += "\n";
    }
  });

  description += `## Common Parameters

### Filters
- \`brands\` - Filter by specific brands (comma-separated)
- \`types\` - Filter by product types
- \`materials\` - Filter by materials
- \`colors\` - Filter by colors
- \`conditions\` - Filter by item conditions
- \`sizes\` - Filter by sizes
- \`vendors\` - Filter by vendors
- \`genders\` - Filter by target gender
- \`models\` - Filter by specific models
- \`monthly\` - Filter by specific months (YYYY-MM format)
- \`weekly\` - Filter by specific weeks (YYYY-WW format)

### Grouping
- \`group_by\` - Group results by entity (comma-separated)
  - Valid values: brand, type, material, color, condition, size, vendor, gender, model, monthly, weekly, decade, location, hardware

### Pagination
- \`limit\` - Number of results to return (default: 500, max: 1000)
- \`offset\` - Number of results to skip (default: 0)

### Options
- \`min_samples\` - Minimum number of samples required for a group to be included (default: 0)
- \`return_nulls\` - Whether to include null values in results (default: true)
- \`calc\` - Post-aggregation calculations (e.g., percentage_change)

### Sorting
- \`order\` - Sort order: ASC or DESC (default: DESC)

## Response Format

All endpoints return data in a consistent format:

\`\`\`json
{
  "data": [...],
  "metadata": {
    "service": "service-name",
    "endpoint": "endpoint-name",
    "generated_at": "2024-01-15T10:30:00Z",
    "request_id": "req_...",
    "execution_time_ms": 123
  },
  "pagination": {
    "limit": 50,
    "offset": 0,
    "total": 150,
    "has_more": true,
    "next_offset": 50
  },
  "component_type": "analysis_type"
}
\`\`\`

## Rate Limits

- **API Key**: 1000 requests per minute

## Error Handling

The API uses standard HTTP status codes:

- \`200\` - Success
- \`400\` - Bad Request (invalid parameters)
- \`401\` - Unauthorized (authentication required)
- \`403\` - Forbidden (insufficient permissions)
- \`404\` - Not Found
- \`429\` - Too Many Requests (rate limit exceeded)
- \`500\` - Internal Server Error

## Support

For API support, contact: api-support@truss.com

Generated on: ${new Date().toISOString()}
`;

  return description;
}

/**
 * Build schema definitions for common data structures
 */
function buildSchemaDefinitions(serviceRegistry) {
  return {
    Error: {
      type: "object",
      properties: {
        error: {
          type: "object",
          properties: {
            type: { type: "string" },
            message: { type: "string" },
            code: { type: "string" },
            timestamp: { type: "string", format: "date-time" },
            request_id: { type: "string" },
            service: { type: "string" },
          },
        },
        metadata: {
          type: "object",
          properties: {
            service: { type: "string" },
            version: { type: "string" },
            generated_at: { type: "string", format: "date-time" },
          },
        },
      },
    },
    Pagination: {
      type: "object",
      properties: {
        limit: { type: "integer", minimum: 1, maximum: 1000 },
        offset: { type: "integer", minimum: 0 },
        total: { type: "integer" },
        has_more: { type: "boolean" },
        next_offset: { type: "integer" },
      },
    },
    Metadata: {
      type: "object",
      properties: {
        service: { type: "string" },
        endpoint: { type: "string" },
        generated_at: { type: "string", format: "date-time" },
        request_id: { type: "string" },
        execution_time_ms: { type: "integer" },
        query_params: { type: "object" },
      },
    },
    AnalyticsResponse: {
      type: "object",
      properties: {
        data: {
          type: "array",
          items: {
            type: "object",
            properties: {
              brand: { type: "string" },
              type: { type: "string" },
              material: { type: "string" },
              color: { type: "string" },
              condition: { type: "string" },
              size: { type: "string" },
              vendor: { type: "string" },
              gender: { type: "string" },
              model: { type: "string" },
              monthly: { type: "string" },
              weekly: { type: "string" },
              value: { type: "number" },
              percentage_change: { type: "number" },
            },
          },
        },
        metadata: { $ref: "#/components/schemas/Metadata" },
        pagination: { $ref: "#/components/schemas/Pagination" },
        component_type: { type: "string" },
      },
    },
  };
}

/**
 * Build parameter definitions for common query parameters
 */
function buildParameterDefinitions(serviceRegistry) {
  const allParameters = {};

  // Helper function to create parameter definition
  function createParameter(name, description, schema, paramKey) {
    const example = getExampleValue(paramKey);
    const param = {
      name,
      in: "query",
      description,
      schema,
    };

    // Only add example if it exists
    if (example !== null) {
      param.example = example;
    }

    return param;
  }

  // Filter parameters
  allParameters.brands = createParameter(
    "brands",
    "Filter by specific brands (comma-separated)",
    { type: "string" },
    "brands"
  );

  allParameters.types = createParameter(
    "types",
    "Filter by product types (comma-separated)",
    { type: "string" },
    "types"
  );

  allParameters.materials = createParameter(
    "materials",
    "Filter by materials (comma-separated)",
    { type: "string" },
    "materials"
  );

  allParameters.colors = createParameter(
    "colors",
    "Filter by colors (comma-separated)",
    { type: "string" },
    "colors"
  );

  allParameters.conditions = createParameter(
    "conditions",
    "Filter by item conditions (comma-separated)",
    { type: "string" },
    "conditions"
  );

  allParameters.sizes = createParameter(
    "sizes",
    "Filter by sizes (comma-separated)",
    { type: "string" },
    "sizes"
  );

  allParameters.vendors = createParameter(
    "vendors",
    "Filter by vendors (comma-separated)",
    { type: "string" },
    "vendors"
  );

  allParameters.genders = createParameter(
    "genders",
    "Filter by target gender (comma-separated)",
    { type: "string" },
    "genders"
  );

  allParameters.models = createParameter(
    "models",
    "Filter by specific models (comma-separated)",
    { type: "string" },
    "models"
  );

  allParameters.decades = createParameter(
    "decades",
    "Filter by decades (comma-separated)",
    { type: "string" },
    "decades"
  );

  allParameters.locations = createParameter(
    "locations",
    "Filter by locations (comma-separated)",
    { type: "string" },
    "locations"
  );

  allParameters.hardwares = createParameter(
    "hardwares",
    "Filter by hardware materials (comma-separated)",
    { type: "string" },
    "hardwares"
  );

  allParameters.monthly = createParameter(
    "monthly",
    "Filter by specific months (YYYY-MM format, comma-separated)",
    { type: "string" },
    "monthly"
  );

  allParameters.weekly = createParameter(
    "weekly",
    "Filter by specific weeks (YYYY-WW format, comma-separated)",
    { type: "string" },
    "weekly"
  );

  allParameters.min_price = createParameter(
    "min_price",
    "Minimum price filter",
    { type: "number" },
    "min_price"
  );

  allParameters.max_price = createParameter(
    "max_price",
    "Maximum price filter",
    { type: "number" },
    "max_price"
  );

  allParameters.is_sold = createParameter(
    "is_sold",
    "Filter by sold status",
    { type: "boolean" },
    "is_sold"
  );

  allParameters.listed_after = createParameter(
    "listed_after",
    "Filter by listed after date (YYYY-MM-DD)",
    { type: "string", format: "date" },
    "listed_after"
  );

  allParameters.listed_before = createParameter(
    "listed_before",
    "Filter by listed before date (YYYY-MM-DD)",
    { type: "string", format: "date" },
    "listed_before"
  );

  allParameters.sold_after = createParameter(
    "sold_after",
    "Filter by sold after date (YYYY-MM-DD)",
    { type: "string", format: "date" },
    "sold_after"
  );

  // Grouping and calculation parameters
  allParameters.group_by = createParameter(
    "group_by",
    "Group results by entity (comma-separated)",
    { type: "string" },
    "group_by"
  );

  allParameters.calc = createParameter(
    "calc",
    "Post-aggregation calculations",
    {
      type: "string",
      enum: ["percentage_change"],
    },
    "calc"
  );

  // Pagination parameters
  allParameters.limit = createParameter(
    "limit",
    "Number of results to return (default: 500, max: 1000)",
    { type: "integer", minimum: 1, maximum: 1000, default: 500 },
    "limit"
  );

  allParameters.offset = createParameter(
    "offset",
    "Number of results to skip (default: 0)",
    { type: "integer", minimum: 0, default: 0 },
    "offset"
  );

  allParameters.page = createParameter(
    "page",
    "Page number (alternative to offset)",
    { type: "integer", minimum: 1, default: 1 },
    "page"
  );

  // Sorting parameters
  allParameters.sort = createParameter(
    "sort",
    "Sort field",
    { type: "string" },
    "sort"
  );

  allParameters.order = createParameter(
    "order",
    "Sort order (default: DESC)",
    {
      type: "string",
      enum: ["ASC", "DESC"],
      default: "DESC",
    },
    "order"
  );

  // Options parameters
  allParameters.min_samples = createParameter(
    "min_samples",
    "Minimum number of samples required for a group to be included (default: 0)",
    { type: "integer", minimum: 0, default: 0 },
    "min_samples"
  );

  allParameters.return_nulls = createParameter(
    "return_nulls",
    "Whether to include null values in results (default: true)",
    { type: "boolean", default: true },
    "return_nulls"
  );

  return allParameters;
}

/**
 * Build response definitions
 */
function buildResponseDefinitions() {
  return {
    200: {
      description: "Success",
      content: {
        "application/json": {
          schema: {
            type: "object",
            properties: {
              data: {
                type: "array",
                items: { type: "object" },
              },
              metadata: { $ref: "#/components/schemas/Metadata" },
              pagination: { $ref: "#/components/schemas/Pagination" },
              component_type: { type: "string" },
            },
          },
        },
      },
    },
    400: {
      description: "Bad Request - Invalid parameters",
      content: {
        "application/json": {
          schema: { $ref: "#/components/schemas/Error" },
        },
      },
    },
    401: {
      description: "Unauthorized - Authentication required",
      content: {
        "application/json": {
          schema: { $ref: "#/components/schemas/Error" },
        },
      },
    },
    403: {
      description: "Forbidden - Insufficient permissions",
      content: {
        "application/json": {
          schema: { $ref: "#/components/schemas/Error" },
        },
      },
    },
    404: {
      description: "Not Found",
      content: {
        "application/json": {
          schema: { $ref: "#/components/schemas/Error" },
        },
      },
    },
    429: {
      description: "Too Many Requests - Rate limit exceeded",
      content: {
        "application/json": {
          schema: { $ref: "#/components/schemas/Error" },
        },
      },
    },
    500: {
      description: "Internal Server Error",
      content: {
        "application/json": {
          schema: { $ref: "#/components/schemas/Error" },
        },
      },
    },
  };
}

/**
 * Build example definitions
 */
function buildExampleDefinitions(serviceRegistry) {
  const examples = {};

  // Add examples for each service
  serviceRegistry.services.forEach((service) => {
    if (service.endpoints) {
      service.endpoints.forEach((endpoint) => {
        if (endpoint.path !== "/health" && endpoint.parameters) {
          const exampleKey = `${stripRootSuffix(service.name)} ${
            endpoint.path
          } example`;

          // Build example query parameters using examples asset file
          const queryParams = {};

          if (endpoint.parameters.filters) {
            endpoint.parameters.filters.forEach((filter) => {
              const exampleValue = getExampleValue(filter);
              if (exampleValue !== null && exampleValue !== undefined) {
                queryParams[filter] = exampleValue;
              }
            });
          }

          if (endpoint.parameters.group_by) {
            queryParams.group_by = getExampleValue("group_by");
          }

          if (endpoint.parameters.pagination) {
            endpoint.parameters.pagination.forEach((param) => {
              const exampleValue = getExampleValue(param);
              if (exampleValue !== null && exampleValue !== undefined) {
                queryParams[param] = exampleValue;
              }
            });
          }

          if (endpoint.parameters.sorting) {
            endpoint.parameters.sorting.forEach((param) => {
              const exampleValue = getExampleValue(param);
              if (exampleValue !== null && exampleValue !== undefined) {
                queryParams[param] = exampleValue;
              }
            });
          }

          if (endpoint.parameters.options) {
            endpoint.parameters.options.forEach((param) => {
              const exampleValue = getExampleValue(param);
              if (exampleValue !== null && exampleValue !== undefined) {
                queryParams[param] = exampleValue;
              }
            });
          }

          if (endpoint.parameters.calc) {
            queryParams.calc = getExampleValue("calc");
          }

          examples[exampleKey] = {
            summary: `${stripRootSuffix(service.name)} ${
              endpoint.path
            } example`,
            description: `Example request for ${endpoint.description}`,
            value: {
              method: endpoint.method,
              url: `${service.base_path}${endpoint.path}`,
              query: queryParams,
            },
          };
        }
      });
    }
  });

  return examples;
}

/**
 * Get example value for a parameter from the examples asset file
 */
function getExampleValue(param) {
  try {
    const examplesAsset = require("../assets/examples.json");
    const paramExamples = examplesAsset[param];

    if (
      paramExamples &&
      paramExamples.examples &&
      paramExamples.examples.length > 0
    ) {
      // Return the first example value
      return Array.isArray(paramExamples.examples)
        ? paramExamples.examples[0]
        : paramExamples.examples;
    }
  } catch (error) {
    console.warn(
      `Could not load examples for parameter ${param}:`,
      error.message
    );
  }

  // No fallbacks - if not in assets file, return null
  return null;
}

/**
 * Build tag definitions for services
 */
function buildTagDefinitions(serviceRegistry) {
  return serviceRegistry.services.map((service) => ({
    name: stripRootSuffix(service.name),
    description: service.description,
    externalDocs: {
      description: "Service Documentation",
      url: `https://api.truss.com/docs/${stripRootSuffix(service.name)}`,
    },
  }));
}

// CLI interface
function main() {
  const args = process.argv.slice(2);
  let serviceRegistryPath = "api-gateway/service-registry.json";
  let openApiPath = "api-gateway/aggregated-openapi-prod.yaml";
  let outputPath = "swagger-template.json";
  let baseUrl =
    "https://lealvbo928.execute-api.eu-west-2.amazonaws.com/external-prod";
  let root = null;

  // Parse arguments
  for (let i = 0; i < args.length; i++) {
    const arg = args[i];
    if (arg === "--help") {
      console.log("Usage: node scripts/generate-swagger-template.js [options]");
      console.log("");
      console.log("Options:");
      console.log(
        "  --service-registry <path>  Service registry file (default: api-gateway/service-registry.json)"
      );
      console.log(
        "  --openapi <path>           OpenAPI YAML file (default: api-gateway/aggregated-openapi-prod.yaml)"
      );
      console.log(
        "  --output <path>            Output JSON file (default: swagger-template.json)"
      );
      console.log(
        "  --base-url <url>           Base URL for API (default: production URL)"
      );
      console.log(
        "  --root <name>              Partition root (e.g., bags, all) to filter paths and default registry"
      );
      console.log("  --help                     Show this help message");
      process.exit(0);
    } else if (arg.startsWith("--service-registry=")) {
      serviceRegistryPath = arg.split("=")[1];
    } else if (arg.startsWith("--openapi=")) {
      openApiPath = arg.split("=")[1];
    } else if (arg.startsWith("--output=")) {
      outputPath = arg.split("=")[1];
    } else if (arg.startsWith("--base-url=")) {
      baseUrl = arg.split("=")[1];
    } else if (arg.startsWith("--root=")) {
      root = arg.split("=")[1];
    }
  }

  // If root is specified and service registry path not explicitly overridden, choose per-root default
  if (root && serviceRegistryPath === "api-gateway/service-registry.json") {
    serviceRegistryPath = `api-gateway/service-registry-${root}.json`;
    console.log(
      `🔧 Using per-root service registry for '${root}': ${serviceRegistryPath}`
    );
  }

  try {
    const result = generateSwaggerTemplate({
      serviceRegistryPath,
      openApiPath,
      outputPath,
      baseUrl,
      root,
    });

    console.log("\n🎉 Swagger template generation complete!");
    console.log("═".repeat(50));
    console.log(`📁 Output file: ${result.outputPath}`);
    console.log(`📊 Services: ${result.servicesCount}`);
    console.log(`🔗 Paths: ${result.pathsCount}`);
    console.log(`🏷️  Tags: ${result.tagsCount}`);
    console.log("═".repeat(50));
  } catch (error) {
    console.error("❌ Swagger template generation failed:", error.message);
    process.exit(1);
  }
}

if (require.main === module) {
  main();
}

module.exports = { generateSwaggerTemplate };
