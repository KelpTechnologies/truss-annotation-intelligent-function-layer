#!/usr/bin/env node
// scripts/generate-service-registry.js - Simplified to only use config file data

const fs = require("fs");
const path = require("path");
const glob = require("glob");

function isServiceDirectory(dir) {
  // A service directory must have config.json AND (index.js OR index.py)
  return (
    fs.existsSync(path.join(dir, "config.json")) &&
    (fs.existsSync(path.join(dir, "index.js")) ||
      fs.existsSync(path.join(dir, "index.py")))
  );
}

function stripRootSuffix(name) {
  return typeof name === "string"
    ? name.replace(/-(bags|all|apparel|footwear)$/i, "")
    : name;
}

function deepClone(obj) {
  return JSON.parse(JSON.stringify(obj || {}));
}

function applyDiff(baseArray = [], add = [], remove = []) {
  const set = new Set(baseArray || []);
  (remove || []).forEach((r) => set.delete(r));
  (add || []).forEach((a) => set.add(a));
  return Array.from(set);
}

function applyPartitionModificationsToEndpoint(endpoint, routeCfg) {
  const ep = deepClone(endpoint);
  ep.parameters = ep.parameters ? deepClone(ep.parameters) : {};

  // Filters
  const modFilters = routeCfg?.modified_filters || null;
  if (modFilters) {
    const baseFilters = Array.isArray(ep.parameters.filters)
      ? ep.parameters.filters
      : [];
    ep.parameters.filters = applyDiff(
      baseFilters,
      modFilters.add || [],
      modFilters.remove || []
    );
  }

  // Group bys
  const modGroupBys = routeCfg?.modified_group_bys || null;
  if (modGroupBys) {
    const baseGroupBy = Array.isArray(ep.parameters.group_by)
      ? ep.parameters.group_by
      : [];
    ep.parameters.group_by = applyDiff(
      baseGroupBy,
      modGroupBys.add || [],
      modGroupBys.remove || []
    );
  }

  return ep;
}

function discoverServices() {
  const services = [];

  // Find all config.json files recursively in services/
  const configFiles = glob.sync("services/**/config.json");

  for (const configFile of configFiles) {
    const serviceDir = path.dirname(configFile);

    // Only include actual services (not just any directory with config.json)
    if (isServiceDirectory(serviceDir)) {
      try {
        const config = JSON.parse(fs.readFileSync(configFile, "utf8"));

        // Process endpoints with only the data from config
        const endpoints = (config.api.endpoints || []).map((endpoint) => {
          const endpointObj = {
            path: endpoint.path,
            method: endpoint.method,
            description: endpoint.description,
            parameters: endpoint.parameters || {},
          };
          return endpointObj;
        });

        const partitionRoutes = config.api?.partitions?.routes || {};
        const hasPartitions = Object.keys(partitionRoutes).length > 0;

        // Base service entry (only if no partitions configured)
        if (!hasPartitions) {
          services.push({
            name: config.service.name,
            description: config.service.description,
            base_path: config.api.base_path,
            endpoints,
          });
        }

        // Partition routes: add separate entries per partition with nested base path
        Object.entries(partitionRoutes).forEach(([routeKey, routeCfg]) => {
          if (routeCfg && typeof routeCfg.base_path === "string") {
            const partitionBase = `${routeCfg.base_path}${config.api.base_path}`;
            const partitionEndpoints = endpoints.map((ep) =>
              applyPartitionModificationsToEndpoint(ep, routeCfg)
            );
            services.push({
              name: `${config.service.name}-${routeKey}`,
              description: `${config.service.description} (${routeKey})`,
              base_path: partitionBase,
              endpoints: partitionEndpoints,
            });
          }
        });
        console.log(
          `✅ Discovered service: ${config.service.name} at ${serviceDir} (${endpoints.length} endpoints)`
        );
      } catch (error) {
        console.warn(
          `⚠️  Could not parse config for ${serviceDir}:`,
          error.message
        );
      }
    }
  }

  // Sort services by name for consistency
  services.sort((a, b) => a.name.localeCompare(b.name));

  return services;
}

/**
 * Build a registry object from a list of services
 */
function buildRegistry(services, options = {}) {
  const { stripSuffix = false } = options;

  const normalizedServices = stripSuffix
    ? services.map((s) => ({ ...s, name: stripRootSuffix(s.name) }))
    : services;

  const registry = {
    generated_at: new Date().toISOString(),
    generator_version: "3.1.0",
    total_services: normalizedServices.length,
    services: normalizedServices,
    summary: {
      total_endpoints: 0,
    },
  };

  // Generate summary statistics
  normalizedServices.forEach((service) => {
    registry.summary.total_endpoints += service.endpoints.length;
  });

  return registry;
}

/**
 * Main function to generate the service registry.
 * This function is now exportable.
 */
function generateServiceRegistry() {
  console.log("🔍 Discovering services...");

  const allDiscovered = discoverServices();

  // Combined registry (all services as discovered)
  const combinedRegistry = buildRegistry(allDiscovered);

  // Build per-root registries for route partitions (e.g., /bags, /all)
  const roots = new Set();
  allDiscovered.forEach((svc) => {
    const match = svc.base_path && svc.base_path.match(/^\/(\w+)\//);
    if (match) roots.add(match[1]);
  });

  // Also include canonical roots if present in configs
  ["bags", "all", "apparel", "footwear"].forEach((r) => roots.add(r));

  // Materialize per-root registries by filtering entries that start with that root
  const perRootRegistries = {};
  roots.forEach((root) => {
    const prefix = `/${root}/`;
    const filtered = allDiscovered.filter(
      (svc) =>
        typeof svc.base_path === "string" && svc.base_path.startsWith(prefix)
    );
    // For per-root outputs, strip the -bags/-all suffix from the name
    perRootRegistries[root] = buildRegistry(filtered, { stripSuffix: true });
  });

  return { combinedRegistry, perRootRegistries };
}

function writeRegistry(registry, outputPath) {
  // Ensure output directory exists
  const outputDir = path.dirname(outputPath);
  if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir, { recursive: true });
  }

  // Write the registry
  fs.writeFileSync(outputPath, JSON.stringify(registry, null, 2));

  console.log(`📝 Service registry written to: ${outputPath}`);
}

// Main execution function for when the script is run directly
function runMain() {
  try {
    const { combinedRegistry, perRootRegistries } = generateServiceRegistry();

    // Write combined registry to multiple locations for different deployment scenarios
    const combinedOutputs = [
      "services/meta/routes/service-registry.json", // For the routes service
      "api-gateway/service-registry.json", // For API Gateway generation
      "service-registry.json", // Root level for CI/CD
    ];

    combinedOutputs.forEach((outputPath) => {
      writeRegistry(combinedRegistry, outputPath);
    });

    // Write per-root registries only if they have services
    Object.entries(perRootRegistries).forEach(([root, registry]) => {
      if (registry.total_services > 0) {
        const suffix = `service-registry-${root}.json`;
        const outputs = [
          `services/meta/routes/${suffix}`,
          `api-gateway/${suffix}`,
          `${suffix}`,
        ];
        outputs.forEach((outputPath) => writeRegistry(registry, outputPath));
      }
    });

    console.log("✅ Service registry generation complete!");
    console.log(`📊 Combined services: ${combinedRegistry.total_services}`);
    Object.entries(perRootRegistries).forEach(([root, reg]) => {
      console.log(`📊 ${root} services: ${reg.total_services}`);
    });
  } catch (error) {
    console.error("❌ Service registry generation failed:", error);
    process.exit(1);
  }
}

// Export the main registry generation function
module.exports = { generateServiceRegistry, discoverServices, runMain };

// Run the main function only if the script is executed directly
if (require.main === module) {
  runMain();
}
