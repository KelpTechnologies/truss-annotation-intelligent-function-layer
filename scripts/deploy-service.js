const fs = require("fs");
const path = require("path");
const { execSync } = require("child_process");
const crypto = require("crypto");

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
    }
  }
}

/**
 * Deploy a single service Lambda function
 */
async function deployService(servicePath, stage = "dev", options = {}) {
  const {
    forceUpdate = false,
    forceCodeUpdate = true, // Default to true to ensure code changes are always deployed
    skipPackaging = false,
    verbose = false,
    serviceName = null, // Allow override of service name
    apiGatewayId = null, // NEW: Allow passing API Gateway ID
  } = options;

  // Calculate service name from path if not provided
  const defaultServiceName = calculateServiceName(servicePath);
  const actualServiceName = serviceName || defaultServiceName;

  console.log(`🚀 Deploying service: ${actualServiceName} (${stage})`);
  console.log(`   Path: ${servicePath}`);

  // Validate service structure
  if (!validateService(servicePath)) {
    throw new Error(`Invalid service structure: ${servicePath}`);
  }

  // Load service configuration
  const config = loadServiceConfig(servicePath);

  // Generate templates if needed
  await ensureTemplatesGenerated(servicePath, config, actualServiceName);

  // Package the service
  const packageInfo = await packageService(
    servicePath,
    config,
    stage,
    skipPackaging,
    actualServiceName
  );

  // Deploy CloudFormation stack
  const deploymentResult = await deployStack(
    servicePath,
    config,
    packageInfo,
    stage,
    forceUpdate,
    actualServiceName,
    apiGatewayId, // Pass API Gateway ID
    forceCodeUpdate // Pass force code update option
  );

  console.log(`✅ Service ${actualServiceName} deployed successfully`);
  return deploymentResult;
}

/**
 * Calculate service name from path (handles nested services)
 */
function calculateServiceName(servicePath) {
  // Get relative path from services directory
  const relativePath = path.relative("services", servicePath);

  // Split into parts and join with hyphens
  const pathParts = relativePath.split(path.sep);

  // Return flattened name (e.g., meta/component-types -> meta-component-types)
  return pathParts.join("-");
}

function validateService(servicePath) {
  const requiredFiles = ["config.json"];
  const handlerFiles = ["index.js", "index.py"];

  console.log(`   🔍 Validating service structure...`);

  // Check for config.json
  const configPath = path.join(servicePath, "config.json");
  if (!fs.existsSync(configPath)) {
    console.error(`   ❌ Missing required file: config.json`);
    return false;
  }

  // Check for at least one handler file
  const hasHandler = handlerFiles.some((file) =>
    fs.existsSync(path.join(servicePath, file))
  );

  if (!hasHandler) {
    console.error(
      `   ❌ Missing handler file: need either index.js or index.py`
    );
    return false;
  }

  // Check config.json syntax
  try {
    JSON.parse(fs.readFileSync(configPath, "utf8"));
  } catch (error) {
    console.error(`   ❌ Invalid JSON in config.json: ${error.message}`);
    return false;
  }

  console.log(`   ✅ Service structure is valid`);
  return true;
}

function loadServiceConfig(servicePath) {
  const configPath = path.join(servicePath, "config.json");
  const config = JSON.parse(fs.readFileSync(configPath, "utf8"));

  // Build security modes list for display
  const authConfig = config.access?.auth_config || {};
  const securityModes = [];
  if (config.access?.internal && authConfig.cognito)
    securityModes.push("cognito");
  if (config.access?.external && authConfig.api_key)
    securityModes.push("api_key");
  if (authConfig.public) securityModes.push("public");

  console.log(`   📋 Loaded config for ${config.service.name}`);
  console.log(`      Runtime: ${config.deployment.runtime}`);
  console.log(`      Memory: ${config.deployment.memory}MB`);
  console.log(`      Timeout: ${config.deployment.timeout}s`);
  console.log(`      Security: ${securityModes.join(", ")}`);
  console.log(
    `      Access: internal=${config.access?.internal || false}, external=${
      config.access?.external || false
    }`
  );

  return config;
}

async function ensureTemplatesGenerated(servicePath, config, serviceName) {
  const templatePath = path.join(servicePath, "template.yaml");
  const openApiPath = path.join(servicePath, "openapi.yaml");

  // Check if templates exist and are newer than config
  const configStat = fs.statSync(path.join(servicePath, "config.json"));
  const templateExists = fs.existsSync(templatePath);
  const openApiExists = fs.existsSync(openApiPath);

  let needsRegeneration = false;

  if (!templateExists || !openApiExists) {
    needsRegeneration = true;
    console.log(`   🔄 Missing templates, generating...`);
  } else {
    const templateStat = fs.statSync(templatePath);
    if (configStat.mtime > templateStat.mtime) {
      needsRegeneration = true;
      console.log(`   🔄 Config newer than templates, regenerating...`);
    }
  }

  if (needsRegeneration) {
    console.log(`   🛠️  Generating templates from config...`);
    try {
      const generateScript = path.join(
        __dirname,
        "generate-service-templates.js"
      );
      execSync(`node "${generateScript}" "${servicePath}"`, {
        stdio: "inherit",
      });
      console.log(`   ✅ Templates generated successfully`);
    } catch (error) {
      throw new Error(`Failed to generate templates: ${error.message}`);
    }
  } else {
    console.log(`   ✅ Templates are up to date`);
  }
}

async function packageService(
  servicePath,
  config,
  stage,
  skipPackaging,
  serviceName
) {
  if (skipPackaging) {
    console.log(`   ⏭️  Skipping packaging (using existing package)`);
    return null;
  }

  console.log(`   📦 Packaging service...`);

  const runtime = config.deployment.runtime;

  // Create temporary packaging directory
  const tempDir = path.join(servicePath, ".package-temp");
  if (fs.existsSync(tempDir)) {
    fs.rmSync(tempDir, { recursive: true });
  }
  fs.mkdirSync(tempDir);

  try {
    // Copy source files
    const sourceFiles = getSourceFiles(servicePath, runtime);
    for (const file of sourceFiles) {
      const srcPath = path.join(servicePath, file);
      const destPath = path.join(tempDir, file);

      if (fs.existsSync(srcPath)) {
        // Create directory if needed
        const destDir = path.dirname(destPath);
        if (!fs.existsSync(destDir)) {
          fs.mkdirSync(destDir, { recursive: true });
        }
        fs.copyFileSync(srcPath, destPath);
        console.log(`     📄 Copied ${file}`);
      }
    }

    // Python: include structured_logger.py, stage_urls.py, and all Python packages
    // Note: index.py is copied via getSourceFiles(), handler.py is legacy and not used
    if (runtime.startsWith("python")) {
      const structuredLoggerPath = path.join(
        servicePath,
        "structured_logger.py"
      );
      if (fs.existsSync(structuredLoggerPath)) {
        const destStructuredLogger = path.join(tempDir, "structured_logger.py");
        fs.copyFileSync(structuredLoggerPath, destStructuredLogger);
        console.log(`     📄 Copied structured_logger.py`);
      }
      const stageUrlsPath = path.join(servicePath, "stage_urls.py");
      if (fs.existsSync(stageUrlsPath)) {
        const destStageUrls = path.join(tempDir, "stage_urls.py");
        fs.copyFileSync(stageUrlsPath, destStageUrls);
        console.log(`     📄 Copied stage_urls.py`);
      }
      
      // Copy all Python package directories (core, agent_*, vector-classifiers)
      // Note: dsl/ is legacy and not used by index.py
      const pythonPackages = [
        "core",
        "agent_architecture",
        "agent_orchestration",
        "agent_utils",
        "vector-classifiers",
      ];
      
      for (const pkgName of pythonPackages) {
        const pkgDir = path.join(servicePath, pkgName);
        if (fs.existsSync(pkgDir)) {
          const destPkg = path.join(tempDir, pkgName);
          copyDirectory(pkgDir, destPkg);
          console.log(`     📁 Copied ${pkgName}/ package`);
        }
      }
    }

    // Copy shared utils folder (always needed for Node.js services)
    if (runtime.startsWith("nodejs")) {
      const sharedUtilsPath = path.join("services", "utils");
      if (fs.existsSync(sharedUtilsPath)) {
        const utilsDestPath = path.join(tempDir, "utils");
        copyDirectory(sharedUtilsPath, utilsDestPath);
        console.log(`     📁 Copied shared utils folder`);
      } else {
        console.warn(
          `     ⚠️  Shared utils folder not found at ${sharedUtilsPath}`
        );
      }
    }

    // Copy service-specific utils folder if it exists
    const serviceUtilsPath = path.join(servicePath, "utils");
    if (fs.existsSync(serviceUtilsPath)) {
      const utilsDestPath = path.join(tempDir, "utils");
      copyDirectory(serviceUtilsPath, utilsDestPath);
      console.log(`     📁 Copied service-specific utils folder`);
    }

    // Skip dependency installation since everything is in the toolkit layer
    console.log(
      `     📚 Skipping dependency installation (using toolkit layer)`
    );

    // Create deployment package
    const zipName = `${serviceName}-${stage}.zip`;
    const zipPath = path.resolve(servicePath, zipName);

    console.log(`     🗜️  Creating deployment package: ${zipName}`);
    execSync(`cd "${tempDir}" && zip -r "${zipPath}" .`, { stdio: "pipe" });

    // Upload to S3
    const s3Bucket = "truss-api-automated-deployments";
    const s3Key = `lambda-packages/${zipName}`;

    console.log(`     ☁️  Uploading to S3: s3://${s3Bucket}/${s3Key}`);
    execSync(`aws s3 cp "${zipPath}" s3://${s3Bucket}/${s3Key}`, {
      stdio: "inherit",
    });

    // Cleanup
    fs.rmSync(tempDir, { recursive: true });
    fs.unlinkSync(zipPath);

    console.log(`   ✅ Package created and uploaded`);

    return {
      s3Bucket,
      s3Key,
      zipName,
      uploaded: true, // Indicate that package was uploaded
    };
  } catch (error) {
    // Cleanup on error
    if (fs.existsSync(tempDir)) {
      fs.rmSync(tempDir, { recursive: true });
    }
    throw new Error(`Packaging failed: ${error.message}`);
  }
}

function getSourceFiles(servicePath, runtime) {
  const commonFiles = ["config.json"];

  if (runtime.startsWith("nodejs")) {
    const files = [...commonFiles, "index.js"];

    // For routes service, include the service registry
    if (servicePath.includes("routes")) {
      files.push("service-registry-all.json");
      files.push("service-registry-bags.json");
      files.push("service-registry-apparel.json");
      files.push("service-registry-footwear.json");
    }

    return files;
  } else if (runtime.startsWith("python")) {
    // Python files - index.py is the main handler, packages copied separately in packageService
    return [
      ...commonFiles,
      "index.py",
      "template.yaml",
      "openapi.yaml",
      "openapi.internal.yaml",
      "openapi.external.yaml",
    ];
  }

  return commonFiles;
}

async function deployStack(
  servicePath,
  config,
  packageInfo,
  stage,
  forceUpdate,
  serviceName,
  apiGatewayId,
  forceCodeUpdate
) {
  console.log(`   🏗️  Deploying CloudFormation stack...`);

  const stackName = `truss-aifl-${serviceName}-${stage}`;
  const functionName = `truss-aifl-${serviceName}-${stage}`;
  const templatePath = path.join(servicePath, "template.yaml");

  // Prepare parameters
  const parameters = [
    `StageName=${stage}`,
    `FunctionName=${functionName}`,
    `ServiceName=${serviceName}`,
  ];

  if (packageInfo) {
    parameters.push(`CodeS3Bucket=${packageInfo.s3Bucket}`);
    parameters.push(`CodeS3Key=${packageInfo.s3Key}`);
  }

  // Add database parameters if required
  if (config.database.required) {
    parameters.push("DatabaseUser=${DATABASE_USER}");
    parameters.push("DatabasePassword=${DATABASE_PASSWORD}");
  }

  // Add API Gateway ID parameter
  if (apiGatewayId) {
    parameters.push(`ApiGatewayId=${apiGatewayId}`);
    console.log(`     🔗 Using API Gateway ID: ${apiGatewayId}`);
  } else {
    console.log(
      `     ⚠️  No API Gateway ID provided; Lambda permissions may be incomplete`
    );
  }

  // Build CloudFormation command
  const cfnCommand = [
    "aws cloudformation deploy",
    `--stack-name ${stackName}`,
    `--template-file ${templatePath}`,
    `--parameter-overrides ${parameters.join(" ")}`,
    "--capabilities CAPABILITY_NAMED_IAM",
    "--no-fail-on-empty-changeset",
  ].join(" ");

  console.log(`     📋 Stack: ${stackName}`);
  console.log(`     📄 Template: ${templatePath}`);
  console.log(`     🏷️  Function: ${functionName}`);

  try {
    execSync(cfnCommand, { stdio: "inherit" });
    console.log(`   ✅ CloudFormation deployment successful`);

    // Always force Lambda code update if package was created and uploaded
    // This ensures code changes are deployed even when CloudFormation detects no changes
    if (packageInfo && (forceCodeUpdate || packageInfo.uploaded)) {
      console.log(
        `   🔄 Forcing Lambda code update to ensure latest code is deployed...`
      );
      await forceLambdaCodeUpdate(functionName, packageInfo);
    } else if (packageInfo && forceUpdate) {
      await forceLambdaCodeUpdate(functionName, packageInfo);
    }

    // Get stack outputs
    const outputs = getStackOutputs(stackName);

    return {
      stackName,
      functionName,
      functionArn: outputs.FunctionArn,
      outputs,
    };
  } catch (error) {
    throw new Error(`CloudFormation deployment failed: ${error.message}`);
  }
}

async function forceLambdaCodeUpdate(functionName, packageInfo) {
  console.log(`   🔄 Forcing Lambda code update for ${functionName}...`);

  const updateCommand = [
    "aws lambda update-function-code",
    `--function-name ${functionName}`,
    `--s3-bucket ${packageInfo.s3Bucket}`,
    `--s3-key ${packageInfo.s3Key}`,
    "--publish",
  ].join(" ");

  try {
    execSync(updateCommand, { stdio: "inherit" });
    console.log(`   ✅ Lambda code forcefully updated to latest version`);

    // Wait a moment for the update to propagate
    console.log(`   ⏳ Waiting for code update to propagate...`);
    await new Promise((resolve) => setTimeout(resolve, 2000));
  } catch (error) {
    console.error(`   ❌ Code update failed: ${error.message}`);
    console.log(
      `   💡 You may need to check AWS credentials or Lambda function permissions`
    );
    throw error;
  }
}

function getStackOutputs(stackName) {
  try {
    const command = `aws cloudformation describe-stacks --stack-name ${stackName} --query "Stacks[0].Outputs" --output json`;
    const result = execSync(command, { encoding: "utf8" });
    const outputs = JSON.parse(result);

    const outputMap = {};
    outputs.forEach((output) => {
      outputMap[output.OutputKey] = output.OutputValue;
    });

    return outputMap;
  } catch (error) {
    console.warn(`   ⚠️  Could not get stack outputs: ${error.message}`);
    return {};
  }
}

// CLI interface
async function main() {
  const args = process.argv.slice(2);

  if (args.length < 1) {
    console.log(
      "Usage: node scripts/deploy-service.js <service-path> [options]"
    );
    console.log("");
    console.log("Options:");
    console.log("  --stage <stage>        Deployment stage (default: dev)");
    console.log("  --force-update         Force Lambda code update (legacy)");
    console.log(
      "  --force-code-update    Force Lambda code update (default: true)"
    );
    console.log(
      "  --no-force-code-update Disable automatic code update forcing"
    );
    console.log("  --skip-packaging       Skip packaging step");
    console.log("  --verbose              Verbose output");
    console.log("  --service-name <n>  Override service name");
    console.log(
      "  --api-gateway-id <id>  API Gateway ID for Lambda permissions"
    );
    console.log("");
    console.log("Examples:");
    console.log("  node scripts/deploy-service.js services/knowledge");
    console.log(
      "  node scripts/deploy-service.js services/meta/component-types --service-name=meta-component-types"
    );
    console.log(
      "  node scripts/deploy-service.js services/analytics --stage=prod --force-update"
    );
    process.exit(1);
  }

  const servicePath = args[0];
  let stage = "dev";
  let forceUpdate = false;
  let forceCodeUpdate = true; // Default to true
  let skipPackaging = false;
  let verbose = false;
  let serviceName = null;
  let apiGatewayId = null;

  // Parse options
  for (let i = 1; i < args.length; i++) {
    const arg = args[i];
    if (arg.startsWith("--stage=")) {
      stage = arg.split("=")[1];
    } else if (arg === "--force-update") {
      forceUpdate = true;
    } else if (arg === "--force-code-update") {
      forceCodeUpdate = true;
    } else if (arg === "--no-force-code-update") {
      forceCodeUpdate = false;
    } else if (arg === "--skip-packaging") {
      skipPackaging = true;
    } else if (arg === "--verbose") {
      verbose = true;
    } else if (arg.startsWith("--service-name=")) {
      serviceName = arg.split("=")[1];
    } else if (arg.startsWith("--api-gateway-id=")) {
      apiGatewayId = arg.split("=")[1];
    }
  }

  try {
    const result = await deployService(servicePath, stage, {
      forceUpdate,
      forceCodeUpdate,
      skipPackaging,
      verbose,
      serviceName,
      apiGatewayId,
    });

    console.log("\n🎉 Deployment Summary:");
    console.log(`   Service: ${result.functionName}`);
    console.log(`   Stage: ${stage}`);
    console.log(`   Stack: ${result.stackName}`);
    console.log(`   Function: ${result.functionName}`);
    if (result.functionArn) {
      console.log(`   ARN: ${result.functionArn}`);
    }
  } catch (error) {
    console.error(`❌ Deployment failed: ${error.message}`);
    process.exit(1);
  }
}

if (require.main === module) {
  main();
}

module.exports = { deployService };
