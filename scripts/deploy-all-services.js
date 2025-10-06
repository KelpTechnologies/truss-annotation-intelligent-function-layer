const fs = require("fs");
const path = require("path");
const { deployService } = require("./deploy-service");
const glob = require("glob");

/**
 * Deploy all services in parallel or sequential mode
 */
async function deployAllServices(options = {}) {
  const {
    servicesDir = "services",
    stage = "dev",
    parallel = false,
    maxConcurrency = 3,
    forceUpdate = false,
    skipPackaging = false,
    includeServices = null,
    excludeServices = [],
    continueOnError = false,
  } = options;

  console.log(`🚀 Deploying all services (${stage} stage)`);
  console.log(`   Mode: ${parallel ? "Parallel" : "Sequential"}`);
  if (parallel) {
    console.log(`   Max Concurrency: ${maxConcurrency}`);
  }

  // Copy utils to all services before deployment
  console.log(`📁 Copying utils to all services...`);
  try {
    const copyUtilsScript = path.join(__dirname, "copy-utils.js");
    require("child_process").execSync(`node "${copyUtilsScript}"`, {
      stdio: "inherit",
    });
    console.log(`✅ Utils copied successfully`);
  } catch (error) {
    console.error(`❌ Failed to copy utils: ${error.message}`);
    if (!continueOnError) {
      throw new Error(`Utils copying failed: ${error.message}`);
    }
  }

  // Discover services
  const services = discoverServices(
    servicesDir,
    includeServices,
    excludeServices
  );

  if (services.length === 0) {
    console.log("❌ No services found to deploy");
    return;
  }

  console.log(`📦 Found ${services.length} services to deploy:`);
  services.forEach((service) => {
    console.log(`   - ${service.name} (${service.path})`);
  });

  const results = {
    successful: [],
    failed: [],
    skipped: [],
  };

  const startTime = Date.now();

  if (parallel) {
    await deployServicesParallel(services, stage, options, results);
  } else {
    await deployServicesSequential(services, stage, options, results);
  }

  const endTime = Date.now();
  const duration = Math.round((endTime - startTime) / 1000);

  // Print summary
  printDeploymentSummary(results, duration, stage);

  // Return results for further processing
  return results;
}

function discoverServices(servicesDir, includeServices, excludeServices) {
  console.log(`🔍 Discovering services in ${servicesDir}/`);

  // Helper function to check if directory is a service
  function isServiceDirectory(dir) {
    const configPath = path.join(dir, "config.json");
    const hasHandler =
      fs.existsSync(path.join(dir, "index.js")) ||
      fs.existsSync(path.join(dir, "index.py"));
    return fs.existsSync(configPath) && hasHandler;
  }

  // Find all service directories recursively (up to 3 levels deep)
  const allServices = [];

  function findServicesRecursively(baseDir, currentDepth = 0, maxDepth = 3) {
    if (currentDepth > maxDepth) return;

    try {
      const entries = fs.readdirSync(baseDir, { withFileTypes: true });

      for (const entry of entries) {
        if (entry.isDirectory()) {
          const fullPath = path.join(baseDir, entry.name);

          // Check if this directory is a service
          if (isServiceDirectory(fullPath)) {
            // Calculate service name with parent folder prefix
            const relativePath = path.relative(servicesDir, fullPath);
            const pathParts = relativePath.split(path.sep);

            // Create flattened service name (e.g., meta/component-types -> meta-component-types)
            const serviceName = pathParts.join("-");

            try {
              const configPath = path.join(fullPath, "config.json");
              const config = JSON.parse(fs.readFileSync(configPath, "utf8"));

              // Override config service name with flattened name for deployment
              const deploymentServiceName = serviceName;

              allServices.push({
                name: deploymentServiceName, // Use flattened name for deployment
                path: fullPath,
                config: config,
                displayName: relativePath, // Keep the path-based name for display
                originalServiceName: config.service.name, // Keep original name from config
                relativePath: relativePath, // Keep relative path for reference
              });
            } catch (error) {
              console.warn(
                `   ⚠️  Invalid config.json in ${relativePath}: ${error.message}`
              );
            }
          } else if (currentDepth < maxDepth) {
            // If not a service, search deeper
            findServicesRecursively(fullPath, currentDepth + 1, maxDepth);
          }
        }
      }
    } catch (error) {
      console.warn(
        `   ⚠️  Could not read directory ${baseDir}: ${error.message}`
      );
    }
  }

  findServicesRecursively(servicesDir);

  // Apply filters
  let filteredServices = allServices;

  if (includeServices && includeServices.length > 0) {
    filteredServices = filteredServices.filter((service) => {
      // Match against multiple identifiers
      return includeServices.some(
        (includeName) =>
          service.name.includes(includeName) ||
          service.displayName.includes(includeName) ||
          service.originalServiceName === includeName ||
          service.relativePath.includes(includeName)
      );
    });
  }

  if (excludeServices.length > 0) {
    filteredServices = filteredServices.filter((service) => {
      return !excludeServices.some(
        (excludeName) =>
          service.name.includes(excludeName) ||
          service.displayName.includes(excludeName) ||
          service.originalServiceName === excludeName ||
          service.relativePath.includes(excludeName)
      );
    });
  }

  // Sort by name for consistent ordering
  return filteredServices.sort((a, b) => a.name.localeCompare(b.name));
}

async function deployServicesSequential(services, stage, options, results) {
  console.log("\n📋 Starting sequential deployment...");

  for (let i = 0; i < services.length; i++) {
    const service = services[i];
    console.log(`\n[${i + 1}/${services.length}] Deploying ${service.name}...`);

    try {
      const result = await deployService(service.path, stage, {
        forceUpdate: options.forceUpdate,
        skipPackaging: options.skipPackaging,
        // Pass the flattened service name for deployment
        serviceName: service.name,
      });

      results.successful.push({
        service: service.name,
        result: result,
      });

      console.log(`✅ ${service.name} deployed successfully`);
    } catch (error) {
      console.error(`❌ ${service.name} deployment failed: ${error.message}`);

      results.failed.push({
        service: service.name,
        error: error.message,
      });

      if (!options.continueOnError) {
        console.error(
          "❌ Stopping deployment due to error (use --continue-on-error to continue)"
        );
        break;
      }
    }
  }
}

async function deployServicesParallel(services, stage, options, results) {
  console.log("\n🔄 Starting parallel deployment...");

  const semaphore = new Semaphore(options.maxConcurrency);
  const promises = services.map(async (service, index) => {
    await semaphore.acquire();

    try {
      console.log(
        `[${index + 1}/${services.length}] Starting ${service.name}...`
      );

      const result = await deployService(service.path, stage, {
        forceUpdate: options.forceUpdate,
        skipPackaging: options.skipPackaging,
        // Pass the flattened service name for deployment
        serviceName: service.name,
      });

      results.successful.push({
        service: service.name,
        result: result,
      });

      console.log(`✅ ${service.name} completed`);
    } catch (error) {
      console.error(`❌ ${service.name} failed: ${error.message}`);

      results.failed.push({
        service: service.name,
        error: error.message,
      });

      if (!options.continueOnError) {
        throw error; // This will stop other deployments
      }
    } finally {
      semaphore.release();
    }
  });

  try {
    await Promise.all(promises);
  } catch (error) {
    if (!options.continueOnError) {
      console.error("❌ Parallel deployment stopped due to error");
    }
  }
}

class Semaphore {
  constructor(max) {
    this.max = max;
    this.current = 0;
    this.queue = [];
  }

  async acquire() {
    if (this.current < this.max) {
      this.current++;
      return;
    }

    return new Promise((resolve) => {
      this.queue.push(resolve);
    });
  }

  release() {
    this.current--;
    if (this.queue.length > 0) {
      this.current++;
      const resolve = this.queue.shift();
      resolve();
    }
  }
}

function printDeploymentSummary(results, duration, stage) {
  console.log("\n🎉 Deployment Complete!");
  console.log("═".repeat(50));
  console.log(`Stage: ${stage}`);
  console.log(`Duration: ${duration}s`);
  console.log(
    `Total Services: ${
      results.successful.length + results.failed.length + results.skipped.length
    }`
  );
  console.log(`✅ Successful: ${results.successful.length}`);
  console.log(`❌ Failed: ${results.failed.length}`);
  console.log(`⏭️  Skipped: ${results.skipped.length}`);

  if (results.successful.length > 0) {
    console.log("\n✅ Successfully Deployed:");
    results.successful.forEach(({ service, result }) => {
      console.log(`   - ${service} (${result.functionName})`);
    });
  }

  if (results.failed.length > 0) {
    console.log("\n❌ Failed Deployments:");
    results.failed.forEach(({ service, error }) => {
      console.log(`   - ${service}: ${error}`);
    });
  }

  if (results.skipped.length > 0) {
    console.log("\n⏭️  Skipped Services:");
    results.skipped.forEach(({ service, reason }) => {
      console.log(`   - ${service}: ${reason}`);
    });
  }

  console.log("═".repeat(50));
}

// CLI interface
async function main() {
  const args = process.argv.slice(2);

  const options = {
    servicesDir: "services",
    stage: "dev",
    parallel: false,
    maxConcurrency: 3,
    forceUpdate: false,
    skipPackaging: false,
    includeServices: null,
    excludeServices: [],
    continueOnError: false,
  };

  // Parse arguments
  for (let i = 0; i < args.length; i++) {
    const arg = args[i];

    if (arg === "--help") {
      console.log("Usage: node scripts/deploy-all-services.js [options]");
      console.log("");
      console.log("Options:");
      console.log(
        "  --stage <stage>           Deployment stage (default: dev)"
      );
      console.log("  --parallel                Deploy services in parallel");
      console.log(
        "  --max-concurrency <n>     Max parallel deployments (default: 3)"
      );
      console.log("  --force-update            Force Lambda code updates");
      console.log(
        "  --skip-packaging          Skip packaging for all services"
      );
      console.log(
        "  --include <services>      Only deploy specified services (comma-separated)"
      );
      console.log(
        "  --exclude <services>      Exclude specified services (comma-separated)"
      );
      console.log(
        "  --continue-on-error       Continue deployment if a service fails"
      );
      console.log(
        "  --services-dir <dir>      Services directory (default: services)"
      );
      console.log("");
      console.log("Examples:");
      console.log(
        "  node scripts/deploy-all-services.js --stage=prod --parallel"
      );
      console.log(
        "  node scripts/deploy-all-services.js --include=knowledge,listings"
      );
      console.log(
        "  node scripts/deploy-all-services.js --include=meta --continue-on-error"
      );
      console.log(
        "  node scripts/deploy-all-services.js --exclude=test-service --continue-on-error"
      );
      process.exit(0);
    } else if (arg.startsWith("--stage=")) {
      options.stage = arg.split("=")[1];
    } else if (arg === "--parallel") {
      options.parallel = true;
    } else if (arg.startsWith("--max-concurrency=")) {
      options.maxConcurrency = parseInt(arg.split("=")[1]);
    } else if (arg === "--force-update") {
      options.forceUpdate = true;
    } else if (arg === "--skip-packaging") {
      options.skipPackaging = true;
    } else if (arg.startsWith("--include=")) {
      options.includeServices = arg.split("=")[1].split(",");
    } else if (arg.startsWith("--exclude=")) {
      options.excludeServices = arg.split("=")[1].split(",");
    } else if (arg === "--continue-on-error") {
      options.continueOnError = true;
    } else if (arg.startsWith("--services-dir=")) {
      options.servicesDir = arg.split("=")[1];
    }
  }

  try {
    const results = await deployAllServices(options);

    // Exit with error code if any deployments failed
    if (results.failed.length > 0) {
      process.exit(1);
    }
  } catch (error) {
    console.error(`❌ Bulk deployment failed: ${error.message}`);
    process.exit(1);
  }
}

if (require.main === module) {
  main();
}

module.exports = { deployAllServices };
