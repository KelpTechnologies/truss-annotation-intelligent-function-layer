#!/usr/bin/env node

/**
 * Script to copy utils folder to all services
 * This ensures all services have access to the latest shared utilities
 */

const fs = require("fs");
const path = require("path");

/**
 * Copies a directory recursively
 * @param {string} src - Source directory
 * @param {string} dest - Destination directory
 */
function copyDirectory(src, dest) {
  // Create destination directory if it doesn't exist
  if (!fs.existsSync(dest)) {
    fs.mkdirSync(dest, { recursive: true });
  }

  // Read source directory
  const entries = fs.readdirSync(src, { withFileTypes: true });

  for (const entry of entries) {
    const srcPath = path.join(src, entry.name);
    const destPath = path.join(dest, entry.name);

    if (entry.isDirectory()) {
      // Recursively copy subdirectories
      copyDirectory(srcPath, destPath);
    } else {
      // Copy files
      fs.copyFileSync(srcPath, destPath);
      console.log(`Copied: ${srcPath} -> ${destPath}`);
    }
  }
}

/**
 * Removes a directory recursively
 * @param {string} dir - Directory to remove
 */
function removeDirectory(dir) {
  if (fs.existsSync(dir)) {
    fs.rmSync(dir, { recursive: true, force: true });
    console.log(`Removed: ${dir}`);
  }
}

/**
 * Detects the runtime of a service
 * @param {string} servicePath - Path to the service directory
 * @returns {string} - 'nodejs' or 'python'
 */
function detectServiceRuntime(servicePath) {
  const configPath = path.join(servicePath, "config.json");

  // Try to read runtime from config.json first
  if (fs.existsSync(configPath)) {
    try {
      const config = JSON.parse(fs.readFileSync(configPath, "utf8"));
      const runtime = config.deployment?.runtime || "";

      if (runtime.startsWith("python") || runtime.includes("python")) {
        return "python";
      }
      if (runtime.startsWith("nodejs") || runtime.includes("node")) {
        return "nodejs";
      }
    } catch (error) {
      // Fall through to file-based detection
    }
  }

  // Fallback: detect by handler file
  if (fs.existsSync(path.join(servicePath, "index.py"))) {
    return "python";
  }
  if (fs.existsSync(path.join(servicePath, "index.js"))) {
    return "nodejs";
  }

  // Default to nodejs if we can't determine
  return "nodejs";
}

/**
 * Main function to copy utils to all services
 */
function copyUtilsToServices() {
  const servicesDir = path.join(__dirname, "..", "services");
  const utilsSourceDir = path.join(servicesDir, "utils");

  // Check if utils source directory exists
  if (!fs.existsSync(utilsSourceDir)) {
    console.error("Error: utils directory not found in services folder");
    process.exit(1);
  }

  // Get all service directories (including nested ones)
  const serviceDirs = [];

  function findServiceDirs(dir, currentDepth = 0, maxDepth = 3) {
    if (currentDepth > maxDepth) return;

    try {
      const entries = fs.readdirSync(dir, { withFileTypes: true });

      for (const entry of entries) {
        if (entry.isDirectory()) {
          const fullPath = path.join(dir, entry.name);

          // Check if this directory is a service (has config.json and index.js)
          const configPath = path.join(fullPath, "config.json");
          const hasHandler =
            fs.existsSync(path.join(fullPath, "index.js")) ||
            fs.existsSync(path.join(fullPath, "index.py"));

          if (fs.existsSync(configPath) && hasHandler) {
            serviceDirs.push(fullPath);
          } else if (currentDepth < maxDepth) {
            // If not a service, search deeper
            findServiceDirs(fullPath, currentDepth + 1, maxDepth);
          }
        }
      }
    } catch (error) {
      console.warn(`Could not read directory ${dir}: ${error.message}`);
    }
  }

  findServiceDirs(servicesDir);

  console.log("Copying utils to services...");
  console.log(
    `Found ${serviceDirs.length} services: ${serviceDirs
      .map((dir) => path.relative(servicesDir, dir))
      .join(", ")}`
  );

  let successCount = 0;
  let skippedCount = 0;
  let errorCount = 0;

  for (const servicePath of serviceDirs) {
    const serviceName = path.relative(servicesDir, servicePath);
    const runtime = detectServiceRuntime(servicePath);

    // Only copy JavaScript utils to Node.js services
    if (runtime === "python") {
      console.log(
        `⏭️  Skipping ${serviceName} (Python service - no Python utils available)`
      );
      skippedCount++;
      continue;
    }

    const utilsDestDir = path.join(servicePath, "utils");

    try {
      // Remove existing utils directory if it exists
      removeDirectory(utilsDestDir);

      // Copy utils directory to service
      copyDirectory(utilsSourceDir, utilsDestDir);

      console.log(`✅ Successfully copied utils to ${serviceName}`);
      successCount++;
    } catch (error) {
      console.error(
        `❌ Failed to copy utils to ${serviceName}:`,
        error.message
      );
      errorCount++;
    }
  }

  console.log("\n📊 Copy Summary:");
  console.log(`✅ Successful: ${successCount}`);
  console.log(`⏭️  Skipped (Python): ${skippedCount}`);
  console.log(`❌ Failed: ${errorCount}`);
  console.log(`📁 Total services: ${serviceDirs.length}`);

  if (errorCount > 0) {
    console.log(
      "\n⚠️  Some services failed to receive utils. Please check the errors above."
    );
    process.exit(1);
  } else {
    console.log("\n🎉 All Node.js services successfully received utils!");
  }
}

/**
 * Function to copy utils to a specific service
 * @param {string} serviceName - Name of the service (can be nested like "meta/entities")
 */
function copyUtilsToService(serviceName) {
  const servicesDir = path.join(__dirname, "..", "services");
  const utilsSourceDir = path.join(servicesDir, "utils");
  const serviceDir = path.join(servicesDir, serviceName);
  const utilsDestDir = path.join(serviceDir, "utils");

  // Check if utils source directory exists
  if (!fs.existsSync(utilsSourceDir)) {
    console.error("Error: utils directory not found in services folder");
    process.exit(1);
  }

  // Check if service directory exists
  if (!fs.existsSync(serviceDir)) {
    console.error(`Error: Service directory '${serviceName}' not found`);
    process.exit(1);
  }

  // Check runtime and skip Python services
  const runtime = detectServiceRuntime(serviceDir);
  if (runtime === "python") {
    console.log(
      `⏭️  Skipping ${serviceName} (Python service - no Python utils available)`
    );
    process.exit(0);
  }

  try {
    // Remove existing utils directory if it exists
    removeDirectory(utilsDestDir);

    // Copy utils directory to service
    copyDirectory(utilsSourceDir, utilsDestDir);

    console.log(`✅ Successfully copied utils to ${serviceName}`);
  } catch (error) {
    console.error(`❌ Failed to copy utils to ${serviceName}:`, error.message);
    process.exit(1);
  }
}

// Handle command line arguments
const args = process.argv.slice(2);

if (args.length === 0) {
  // Copy to all services
  copyUtilsToServices();
} else if (args.length === 1) {
  // Copy to specific service
  copyUtilsToService(args[0]);
} else {
  console.log("Usage:");
  console.log(
    "  node copy-utils.js                    # Copy utils to all services"
  );
  console.log(
    "  node copy-utils.js <service-name>     # Copy utils to specific service"
  );
  process.exit(1);
}
