const fs = require("fs");
const path = require("path");
const yaml = require("js-yaml");
const glob = require("glob");

/**
 * Aggregates all service OpenAPI specs into a single API Gateway specification
 */
function aggregateOpenAPISpecs(options = {}) {
  const {
    servicesDir = "services",
    outputFile = "api-gateway/aggregated-openapi.yaml",
    stage = "dev",
  } = options;

  console.log(`🔍 Discovering OpenAPI specs in ${servicesDir}/`);
  console.log(`🔎 Aggregation stage: '${stage}'`);

  // Determine which OpenAPI files to aggregate based on stage
  let openApiPattern = "openapi.yaml";
  if (stage.toLowerCase().includes("external")) {
    openApiPattern = "openapi.external.yaml";
  } else if (
    stage.toLowerCase().includes("prod") ||
    stage.toLowerCase().includes("dev")
  ) {
    openApiPattern = "openapi.internal.yaml";
  }
  console.log(`🔎 Using OpenAPI pattern: '${openApiPattern}'`);

  // Find all openapi files in services (including nested directories)
  const openApiFiles = glob.sync(`${servicesDir}/**/${openApiPattern}`);

  if (openApiFiles.length === 0) {
    console.warn(
      `⚠️  No OpenAPI specs found in ${servicesDir}/ for pattern ${openApiPattern}`
    );
    return;
  }

  console.log(`📄 Found ${openApiFiles.length} OpenAPI specs:`);
  openApiFiles.forEach((file) => console.log(`   - ${file}`));

  // Base OpenAPI structure
  const aggregatedSpec = {
    openapi: "3.0.1",
    info: {
      title: "Truss Data Service API",
      version: "1.0.0",
      description:
        "Comprehensive API for Truss Data Services with multi-authentication support",
    },
    servers: [
      {
        url: "https://{apiId}.execute-api.{region}.amazonaws.com/{stage}",
        variables: {
          apiId: { default: "your-api-id" },
          region: { default: "eu-west-2" },
          stage: { default: stage },
        },
      },
    ],
    paths: {},
    components: {
      securitySchemes: {},
      schemas: {},
      responses: {},
      parameters: {},
      examples: {},
    },
    security: [], // Will be populated based on services
    tags: [],
  };

  const serviceConfigs = [];
  const allSecurityModes = new Set();

  // If external, force API Key auth for all services
  const forceApiKeyOnly = stage.startsWith("external");

  // Process each service
  for (const openApiFile of openApiFiles) {
    try {
      console.log(`🔄 Processing ${openApiFile}...`);

      const servicePath = path.dirname(openApiFile);
      const configPath = path.join(servicePath, "config.json");

      // Load service config
      let serviceConfig = null;
      let serviceName = null;

      if (fs.existsSync(configPath)) {
        serviceConfig = JSON.parse(fs.readFileSync(configPath, "utf8"));

        // Calculate flattened service name for stage variables
        const relativePath = path.relative(servicesDir, servicePath);
        const pathParts = relativePath.split(path.sep);
        serviceName = pathParts.join("-"); // e.g., meta/component-types -> meta-component-types

        // Override config with calculated name for aggregation
        const configCopy = JSON.parse(JSON.stringify(serviceConfig));
        configCopy.service.aggregatedName = serviceName;

        // Force API Key auth for external stage
        if (forceApiKeyOnly) {
          configCopy.access = {
            internal: false,
            external: true,
            auth_config: {
              api_key: { cors_origin: "*" },
            },
          };
        }

        serviceConfigs.push(configCopy);

        // Track all security modes (after override)
        const authConfig = configCopy.access?.auth_config || {};
        if (configCopy.access?.internal && authConfig.cognito)
          allSecurityModes.add("cognito");
        if (configCopy.access?.external && authConfig.api_key)
          allSecurityModes.add("api_key");
        if (authConfig.public) allSecurityModes.add("public");
      }

      // Load OpenAPI spec
      const serviceSpec = yaml.load(fs.readFileSync(openApiFile, "utf8"));

      // Update stage variables in integrations to use flattened service name
      if (serviceSpec.paths && serviceName) {
        Object.keys(serviceSpec.paths).forEach((pathKey) => {
          Object.keys(serviceSpec.paths[pathKey]).forEach((method) => {
            const operation = serviceSpec.paths[pathKey][method];
            if (operation["x-amazon-apigateway-integration"]) {
              const integration = operation["x-amazon-apigateway-integration"];
              if (integration.uri) {
                // Replace service name references in stage variables
                const originalServiceName = serviceConfig.service.name;
                const stageVarPattern = `\\\${stageVariables.${originalServiceName}FunctionArn}`;
                const newStageVarPattern = `\\\${stageVariables.${serviceName}FunctionArn}`;

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

      // Merge components
      if (serviceSpec.components) {
        if (serviceSpec.components.securitySchemes) {
          Object.assign(
            aggregatedSpec.components.securitySchemes,
            serviceSpec.components.securitySchemes
          );
        }
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
        if (serviceSpec.components.examples) {
          Object.assign(
            aggregatedSpec.components.examples,
            serviceSpec.components.examples
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

  // Add global security schemes that are used across services
  console.log(
    `🔐 Detected security modes: ${Array.from(allSecurityModes).join(", ")}`
  );

  // Add info about services to description
  aggregatedSpec.info.description += `\n\n## Services Included\n\n${serviceConfigs
    .map((config) => {
      // Build security modes list for display
      const authConfig = config.access?.auth_config || {};
      const securityModes = [];
      if (config.access?.internal && authConfig.cognito)
        securityModes.push("cognito");
      if (config.access?.external && authConfig.api_key)
        securityModes.push("api_key");
      if (authConfig.public) securityModes.push("public");

      return `- **${config.service.aggregatedName || config.service.name}**: ${
        config.service.description
      } (${securityModes.join(", ")})`;
    })
    .join("\n")}`;

  // Sort paths alphabetically
  const sortedPaths = {};
  Object.keys(aggregatedSpec.paths)
    .sort()
    .forEach((key) => {
      sortedPaths[key] = aggregatedSpec.paths[key];
    });
  aggregatedSpec.paths = sortedPaths;

  // Ensure output directory exists
  const outputDir = outputFile.includes("api-gateway-external")
    ? "api-gateway-external"
    : "api-gateway";
  if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir, { recursive: true });
  }

  // Write aggregated spec
  const yamlContent = yaml.dump(aggregatedSpec, {
    lineWidth: -1,
    noRefs: true,
    quotingType: '"',
  });

  const finalContent = `# Generated by aggregate-openapi.js - DO NOT EDIT DIRECTLY
# This file aggregates all service OpenAPI specifications
# Generated on: ${new Date().toISOString()}
# Services included: ${serviceConfigs
    .map((c) => c.service.aggregatedName || c.service.name)
    .join(", ")}

${yamlContent}`;

  fs.writeFileSync(outputFile, finalContent);

  console.log(`🎉 Aggregated OpenAPI spec written to: ${outputFile}`);
  console.log(`📊 Summary:`);
  console.log(`   - Services: ${serviceConfigs.length}`);
  console.log(`   - Paths: ${Object.keys(aggregatedSpec.paths).length}`);
  console.log(
    `   - Security schemes: ${
      Object.keys(aggregatedSpec.components.securitySchemes).length
    }`
  );
  console.log(`   - Tags: ${aggregatedSpec.tags.length}`);

  return {
    outputFile,
    servicesCount: serviceConfigs.length,
    pathsCount: Object.keys(aggregatedSpec.paths).length,
    securitySchemes: Object.keys(aggregatedSpec.components.securitySchemes),
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

  // Robust argument parsing: support --flag value, --flag=value, any order
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
            "  --stage <stage>        Deployment stage (default: dev)"
          );
          console.log("  --help                 Show this help message");
          process.exit(0);
      }
    }
  }

  // Debug print for parsed CLI args
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

module.exports = { aggregateOpenAPISpecs };
