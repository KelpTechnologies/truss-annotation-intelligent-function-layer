// scripts/aggregate-openapi.js - UNIFIED API VERSION
// Aggregates all service OpenAPI specs into a single unified API Gateway spec

const fs = require("fs");
const path = require("path");
const yaml = require("js-yaml");
const glob = require("glob");

// Load layer configuration from single source of truth
const LAYER_CONFIG = (() => {
  const configPath = path.join(__dirname, "layer-config.json");
  if (!fs.existsSync(configPath)) {
    console.error("❌ layer-config.json not found! Create it first.");
    process.exit(1);
  }
  return JSON.parse(fs.readFileSync(configPath, "utf-8"));
})();

/**
 * Aggregates all service OpenAPI specs into a single unified API Gateway specification
 * NO LONGER generates separate internal/external specs
 */
function aggregateOpenAPISpecs(options = {}) {
  const {
    servicesDir = "services",
    outputFile = "api-gateway/aggregated-openapi.yaml",
    stage = "dev",
  } = options;

  console.log(`🔍 Discovering OpenAPI specs in ${servicesDir}/`);
  console.log(`🔎 Aggregation stage: '${stage}'`);

  // UNIFIED: Always look for openapi.yaml (no internal/external distinction)
  const openApiPattern = "openapi.yaml";
  console.log(`🔎 Using unified OpenAPI pattern: '${openApiPattern}'`);

  const openApiFiles = glob.sync(`${servicesDir}/**/${openApiPattern}`);

  if (openApiFiles.length === 0) {
    console.warn(
      `⚠️  No OpenAPI specs found in ${servicesDir}/ for pattern ${openApiPattern}`
    );
    return;
  }

  console.log(`📄 Found ${openApiFiles.length} OpenAPI specs:`);
  openApiFiles.forEach((file) => console.log(`   - ${file}`));

  // Base unified OpenAPI structure
  const aggregatedSpec = {
    openapi: "3.0.1",
    info: {
      title: LAYER_CONFIG.apiTitle,
      version: "1.0.0",
      description: `Unified API with hybrid authentication (API Key + Cognito JWT)\nLayer: ${LAYER_CONFIG.layerName}`,
    },
    servers: [
      {
        url: `https://{apiId}.execute-api.${LAYER_CONFIG.region}.amazonaws.com/{stage}`,
        variables: {
          apiId: { default: "your-api-id" },
          stage: { default: stage },
        },
      },
    ],
    paths: {},
    components: {
      securitySchemes: {
        // UNIFIED HybridAuth security scheme
        HybridAuth: {
          type: "apiKey",
          in: "header",
          name: "Authorization",
          description:
            "Hybrid authentication: Accepts both 'Bearer <JWT>' for Cognito and x-api-key header for API key auth",
          "x-amazon-apigateway-authtype": "custom",
          "x-amazon-apigateway-authorizer": {
            type: "request",
            authorizerUri: `arn:aws:apigateway:${LAYER_CONFIG.region}:lambda:path/2015-03-31/functions/\${stageVariables.hybridAuthorizerArn}/invocations`,
            authorizerCredentials: `arn:aws:iam::${LAYER_CONFIG.accountId}:role/truss-api-gateway-authorizer-\${stageVariables.authorizerStage}-role`,
            authorizerResultTtlInSeconds: 300,
            identitySource:
              "method.request.header.Authorization, method.request.header.x-api-key",
          },
        },
      },
      schemas: {},
      responses: {},
      parameters: {},
      examples: {},
    },
    security: [{ HybridAuth: [] }],
    tags: [],
  };

  const serviceConfigs = [];

  // Process each service
  for (const openApiFile of openApiFiles) {
    try {
      console.log(`🔄 Processing ${openApiFile}...`);

      const servicePath = path.dirname(openApiFile);
      const configPath = path.join(servicePath, "config.json");

      let serviceConfig = null;
      let serviceName = null;

      if (fs.existsSync(configPath)) {
        serviceConfig = JSON.parse(fs.readFileSync(configPath, "utf8"));

        const relativePath = path.relative(servicesDir, servicePath);
        const pathParts = relativePath.split(path.sep);
        serviceName = pathParts.join("-");

        const configCopy = JSON.parse(JSON.stringify(serviceConfig));
        configCopy.service.aggregatedName = serviceName;
        serviceConfigs.push(configCopy);
      }

      // Load OpenAPI spec
      const serviceSpec = yaml.load(fs.readFileSync(openApiFile, "utf8"));

      // Update stage variables in integrations
      if (serviceSpec.paths && serviceName) {
        Object.keys(serviceSpec.paths).forEach((pathKey) => {
          Object.keys(serviceSpec.paths[pathKey]).forEach((method) => {
            const operation = serviceSpec.paths[pathKey][method];

            // Ensure all protected endpoints use HybridAuth
            if (method !== "options" && pathKey !== "/health") {
              operation.security = [{ HybridAuth: [] }];
            }

            if (operation["x-amazon-apigateway-integration"]) {
              const integration = operation["x-amazon-apigateway-integration"];
              if (integration.uri && serviceConfig) {
                const originalServiceName = serviceConfig.service.name;
                const stageVarPattern = `\${stageVariables.${originalServiceName}FunctionArn}`;
                const newStageVarPattern = `\${stageVariables.${serviceName}FunctionArn}`;

                integration.uri = integration.uri.replace(
                  new RegExp(
                    stageVarPattern.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"),
                    "g"
                  ),
                  newStageVarPattern
                );
              }
            }
          });
        });
      }

      // Merge paths
      if (serviceSpec.paths) {
        Object.assign(aggregatedSpec.paths, serviceSpec.paths);
      }

      // Merge components (excluding security schemes - we use our unified HybridAuth)
      if (serviceSpec.components) {
        if (serviceSpec.components.schemas) {
          Object.assign(
            aggregatedSpec.components.schemas,
            serviceSpec.components.schemas
          );
        }
        if (serviceSpec.components.responses) {
          Object.assign(
            aggregatedSpec.components.responses,
            serviceSpec.components.responses
          );
        }
        if (serviceSpec.components.parameters) {
          Object.assign(
            aggregatedSpec.components.parameters,
            serviceSpec.components.parameters
          );
        }
      }

      // Add service tag
      if (serviceConfig) {
        const displayName = serviceName || serviceConfig.service.name;
        aggregatedSpec.tags.push({
          name: displayName,
          description: serviceConfig.service.description,
        });

        // Tag all paths for this service
        const basePath = serviceConfig.api.base_path;
        Object.keys(aggregatedSpec.paths).forEach((pathKey) => {
          if (pathKey.startsWith(basePath)) {
            Object.keys(aggregatedSpec.paths[pathKey]).forEach((method) => {
              if (
                method !== "options" &&
                aggregatedSpec.paths[pathKey][method].tags === undefined
              ) {
                aggregatedSpec.paths[pathKey][method].tags = [displayName];
              }
            });
          }
        });
      }

      console.log(`✅ Merged ${openApiFile}`);
    } catch (error) {
      console.error(`❌ Error processing ${openApiFile}:`, error.message);
    }
  }

  // Update description with service list
  aggregatedSpec.info.description += `\n\n## Services Included\n\n${serviceConfigs
    .map(
      (config) =>
        `- **${config.service.aggregatedName || config.service.name}**: ${config.service.description}`
    )
    .join("\n")}

## Authentication

This API supports hybrid authentication:
- **API Key**: Include \`x-api-key\` header with your API key
- **Cognito JWT**: Include \`Authorization: Bearer <token>\` header with your JWT

Both methods are validated by the same hybrid authorizer.`;

  // Sort paths alphabetically
  const sortedPaths = {};
  Object.keys(aggregatedSpec.paths)
    .sort()
    .forEach((key) => {
      sortedPaths[key] = aggregatedSpec.paths[key];
    });
  aggregatedSpec.paths = sortedPaths;

  // Ensure output directory exists
  const outputDir = path.dirname(outputFile);
  if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir, { recursive: true });
  }

  // Write aggregated spec
  const yamlContent = yaml.dump(aggregatedSpec, {
    lineWidth: -1,
    noRefs: true,
    quotingType: '"',
  });

  const finalContent = `# Unified API Gateway Specification with HybridAuth
# Generated by aggregate-openapi.js - DO NOT EDIT DIRECTLY
# Layer: ${LAYER_CONFIG.layerName}
# Generated: ${new Date().toISOString()}
# Services: ${serviceConfigs.map((c) => c.service.aggregatedName || c.service.name).join(", ")}

${yamlContent}`;

  fs.writeFileSync(outputFile, finalContent);

  console.log(`🎉 Unified OpenAPI spec written to: ${outputFile}`);
  console.log(`📊 Summary:`);
  console.log(`   - Services: ${serviceConfigs.length}`);
  console.log(`   - Paths: ${Object.keys(aggregatedSpec.paths).length}`);
  console.log(`   - Security: HybridAuth (unified)`);
  console.log(`   - Tags: ${aggregatedSpec.tags.length}`);

  return {
    outputFile,
    servicesCount: serviceConfigs.length,
    pathsCount: Object.keys(aggregatedSpec.paths).length,
    securitySchemes: ["HybridAuth"],
    services: serviceConfigs.map(
      (c) => c.service.aggregatedName || c.service.name
    ),
  };
}

// CLI interface
function main() {
  const args = process.argv.slice(2);
  let servicesDir = "services";
  let outputFile = "api-gateway/aggregated-openapi.yaml";
  let stage = "dev";

  for (let i = 0; i < args.length; i++) {
    let arg = args[i];
    if (arg.startsWith("--")) {
      let [flag, value] = arg.split("=");
      if (!value && i + 1 < args.length && !args[i + 1].startsWith("--")) {
        value = args[++i];
      }
      switch (flag) {
        case "--services-dir":
          if (value) servicesDir = value;
          break;
        case "--output":
          if (value) outputFile = value;
          break;
        case "--stage":
          if (value) stage = value;
          break;
        case "--help":
          console.log("Usage: node scripts/aggregate-openapi.js [options]");
          console.log("");
          console.log("Options:");
          console.log(
            "  --services-dir <dir>   Directory containing services (default: services)"
          );
          console.log(
            "  --output <file>        Output file path (default: api-gateway/aggregated-openapi.yaml)"
          );
          console.log(
            "  --stage <stage>        Deployment stage: dev, staging, or prod (default: dev)"
          );
          console.log("  --help                 Show this help message");
          process.exit(0);
      }
    }
  }

  console.log("Parsed CLI args:", { servicesDir, outputFile, stage });

  try {
    aggregateOpenAPISpecs({ servicesDir, outputFile, stage });
  } catch (error) {
    console.error("❌ Error aggregating OpenAPI specs:", error.message);
    process.exit(1);
  }
}

if (require.main === module) {
  main();
}

module.exports = { aggregateOpenAPISpecs, LAYER_CONFIG };
