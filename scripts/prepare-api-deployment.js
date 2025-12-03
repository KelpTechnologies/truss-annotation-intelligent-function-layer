// scripts/prepare-api-deployment.js - UNIFIED API VERSION
// Prepares a single unified API Gateway deployment with HybridAuth

const fs = require("fs");
const path = require("path");
const { execSync } = require("child_process");
const { aggregateOpenAPISpecs } = require("./aggregate-openapi");
const crypto = require("crypto");
const yaml = require("js-yaml");

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
 * Prepares unified API Gateway deployment
 * NO LONGER generates separate internal/external APIs
 */
async function prepareApiDeployment(stage = "dev") {
  console.log(`🚀 Preparing UNIFIED API Gateway deployment for stage: ${stage}`);
  console.log(`📦 Layer: ${LAYER_CONFIG.layerName}`);

  // UNIFIED: Single output directory
  const outputDir = "api-gateway";

  // Step 1: Aggregate OpenAPI specs
  const aggregatedSpecFile = `aggregated-openapi-${stage}.yaml`;
  const outputFile = path.join(outputDir, aggregatedSpecFile);

  console.log(`\n[1/5] Aggregating OpenAPI specs...`);
  const aggregationResult = aggregateOpenAPISpecs({
    servicesDir: "services",
    outputFile: outputFile,
    stage,
  });

  if (!aggregationResult) {
    console.error("❌ Failed to aggregate OpenAPI specs");
    process.exit(1);
  }

  console.log("✅ Step 1 completed");

  // Step 2: Get Lambda ARNs
  console.log("\n🔍 Step 2: Resolving Lambda function ARNs...");
  const lambdaArns = await resolveLambdaArns(aggregationResult.services, stage);
  console.log("✅ Step 2 completed");

  // Step 3: Get Hybrid Authorizer ARN
  console.log("\n🔐 Step 3: Resolving Hybrid Authorizer ARN...");
  const hybridAuthorizerArn = await resolveHybridAuthorizerArn(stage);
  console.log("✅ Step 3 completed");

  // Step 4: Create deployment-ready spec
  console.log("\n🔧 Step 4: Creating deployment-ready specification...");
  const deploymentSpec = await createDeploymentSpec(
    outputFile,
    lambdaArns,
    hybridAuthorizerArn,
    stage,
    outputDir
  );
  console.log("✅ Step 4 completed");

  // Step 5: Save and upload artifacts
  console.log("\n💾 Step 5: Saving deployment specifications...");
  const definitionS3Key = await saveDeploymentArtifacts(
    deploymentSpec,
    stage,
    outputDir
  );
  console.log("✅ Step 5 completed");

  // Step 6: Generate CloudFormation template
  console.log("\n📝 Step 6: Generating API Gateway template...");
  generateApiGatewayTemplate(
    stage,
    deploymentSpec,
    hybridAuthorizerArn,
    outputDir,
    definitionS3Key
  );
  console.log("✅ Step 6 completed");

  console.log("\n✅ Unified API Gateway deployment preparation complete!");

  return {
    stage,
    services: aggregationResult.services,
    pathsCount: aggregationResult.pathsCount,
    deploymentSpecFile: deploymentSpec.outputFile,
    hybridAuthorizerArn,
  };
}

async function resolveLambdaArns(serviceNames, stage) {
  const lambdaArns = {};

  console.log(`   Resolving Lambda ARNs for stage '${stage}'`);

  for (const serviceName of serviceNames) {
    try {
      const stackName = `${LAYER_CONFIG.stackPrefix}-${serviceName}-${stage}`;
      const exportName = `${stackName}-FunctionArn`;

      console.log(`   🔍 Looking for export: ${exportName}`);

      const command = `aws cloudformation list-exports --query "Exports[?Name=='${exportName}'].Value" --output text`;
      const arn = execSync(command, { encoding: "utf-8" }).trim();

      if (arn && arn !== "None" && !arn.includes("None")) {
        lambdaArns[`${serviceName}FunctionArn`] = arn;
        console.log(`   ✅ ${serviceName}: ${arn}`);
      } else {
        const fallbackArn = `arn:aws:lambda:${LAYER_CONFIG.region}:${LAYER_CONFIG.accountId}:function:${LAYER_CONFIG.stackPrefix}-${serviceName}-${stage}`;
        lambdaArns[`${serviceName}FunctionArn`] = fallbackArn;
        console.log(`   ⚠️  ${serviceName}: Using fallback ARN`);
      }
    } catch (error) {
      const fallbackArn = `arn:aws:lambda:${LAYER_CONFIG.region}:${LAYER_CONFIG.accountId}:function:${LAYER_CONFIG.stackPrefix}-${serviceName}-${stage}`;
      lambdaArns[`${serviceName}FunctionArn`] = fallbackArn;
      console.log(`   ⚠️  ${serviceName}: Using fallback ARN (error: ${error.message})`);
    }
  }

  return lambdaArns;
}

/**
 * Resolve the Hybrid Authorizer ARN from truss-api-platform deployment
 */
async function resolveHybridAuthorizerArn(stage) {
  try {
    const functionName = `truss-hybrid-authorizer-${stage}`;
    console.log(`   🔍 Looking for hybrid authorizer: ${functionName}`);

    // Try CloudFormation export first
    const exportName = `truss-hybrid-authorizer-${stage}-arn`;
    try {
      const command = `aws cloudformation list-exports --query "Exports[?Name=='${exportName}'].Value" --output text`;
      const arn = execSync(command, { encoding: "utf-8" }).trim();

      if (arn && arn !== "None" && !arn.includes("None")) {
        console.log(`   ✅ Hybrid authorizer ARN from export: ${arn}`);
        return arn;
      }
    } catch (exportError) {
      console.log(`   ⚠️  Export not found, trying direct lookup`);
    }

    // Try direct function lookup
    try {
      const command = `aws lambda get-function --function-name ${functionName} --query 'Configuration.FunctionArn' --output text`;
      const arn = execSync(command, { encoding: "utf-8" }).trim();

      if (arn && arn !== "None" && !arn.includes("None")) {
        console.log(`   ✅ Hybrid authorizer ARN from function: ${arn}`);
        return arn;
      }
    } catch (functionError) {
      console.log(`   ⚠️  Function not found`);
    }

    // Fallback
    const fallbackArn = `arn:aws:lambda:${LAYER_CONFIG.region}:${LAYER_CONFIG.accountId}:function:${functionName}`;
    console.log(`   ⚠️  Using fallback: ${fallbackArn}`);
    console.log(
      `   💡 Deploy hybrid authorizer from truss-api-platform if not done`
    );

    return fallbackArn;
  } catch (error) {
    const fallbackArn = `arn:aws:lambda:${LAYER_CONFIG.region}:${LAYER_CONFIG.accountId}:function:truss-hybrid-authorizer-${stage}`;
    console.log(`   ⚠️  Using fallback: ${fallbackArn}`);
    return fallbackArn;
  }
}

async function createDeploymentSpec(
  inputFile,
  lambdaArns,
  hybridAuthorizerArn,
  stage,
  outputDir
) {
  console.log(`   Processing ${inputFile}...`);

  let spec = yaml.load(fs.readFileSync(inputFile, "utf8"));
  let specString = JSON.stringify(spec);

  // Replace Lambda function ARNs
  Object.entries(lambdaArns).forEach(([variable, arn]) => {
    const placeholder = `\${stageVariables.${variable}}`;
    specString = specString.replace(
      new RegExp(placeholder.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"), "g"),
      arn
    );
    console.log(`   🔄 Replaced ${variable}`);
  });

  // Replace Hybrid Authorizer ARN
  const authPlaceholder = `\${stageVariables.hybridAuthorizerArn}`;
  specString = specString.replace(
    new RegExp(authPlaceholder.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"), "g"),
    hybridAuthorizerArn
  );
  console.log(`   🔄 Replaced hybridAuthorizerArn`);

  // Replace authorizer stage variable
  const stagePlaceholder = `\${stageVariables.authorizerStage}`;
  specString = specString.replace(
    new RegExp(stagePlaceholder.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"), "g"),
    stage
  );

  spec = JSON.parse(specString);

  // Update server URL
  spec.servers = [
    {
      url: `https://\${ApiId}.execute-api.${LAYER_CONFIG.region}.amazonaws.com/${stage}`,
      description: `${stage.charAt(0).toUpperCase() + stage.slice(1)} environment`,
    },
  ];

  // Add deployment metadata
  spec.info.version = `1.0.0-${stage}`;
  spec.info["x-deployment"] = {
    stage: stage,
    layer: LAYER_CONFIG.layerName,
    timestamp: new Date().toISOString(),
    hybridAuthorizerArn: hybridAuthorizerArn,
    services: Object.keys(lambdaArns).map((arn) =>
      arn.replace("FunctionArn", "")
    ),
  };

  const outputFile = `${outputDir}/deployment-ready-${stage}.yaml`;
  const yamlContent = yaml.dump(spec, { lineWidth: -1, noRefs: true });

  const finalContent = `# Deployment-ready Unified API specification
# Generated: ${new Date().toISOString()}
# Stage: ${stage}
# Layer: ${LAYER_CONFIG.layerName}
# Hybrid Authorizer: ${hybridAuthorizerArn}

${yamlContent}`;

  fs.writeFileSync(outputFile, finalContent);
  console.log(`   ✅ Created: ${outputFile}`);

  return {
    outputFile,
    spec,
    lambdaArns,
    hybridAuthorizerArn,
  };
}

function generateApiGatewayTemplate(
  stage,
  deploymentSpec,
  hybridAuthorizerArn,
  outputDir,
  definitionS3Key = null
) {
  const template = `AWSTemplateFormatVersion: "2010-09-09"
Transform: AWS::Serverless-2016-10-31
Description: "Unified ${LAYER_CONFIG.layerName} API Gateway with HybridAuth - ${stage}"

Parameters:
  StageName:
    Type: String
    Default: ${stage}
    Description: Deployment stage

Resources:
  UnifiedApi:
    Type: AWS::Serverless::Api
    Properties:
      Name: !Sub "${LAYER_CONFIG.stackPrefix}-api-\${StageName}"
      StageName: !Ref StageName
      DefinitionUri: s3://truss-api-automated-deployments/${
        definitionS3Key || `api-specs/deployment-ready-${stage}.yaml`
      }
      EndpointConfiguration:
        Type: REGIONAL
      TracingEnabled: true
      Variables:
        hybridAuthorizerArn: "${hybridAuthorizerArn}"
        authorizerStage: !Ref StageName
      MethodSettings:
        - ResourcePath: "/*"
          HttpMethod: "*"
          MetricsEnabled: true
          DataTraceEnabled: false
          LoggingLevel: INFO
          ThrottlingRateLimit: 1000
          ThrottlingBurstLimit: 2000

Outputs:
  ApiGatewayUrl:
    Description: Unified API Gateway URL
    Value: !Sub "https://\${UnifiedApi}.execute-api.\${AWS::Region}.amazonaws.com/\${StageName}"
    Export:
      Name: !Sub "${LAYER_CONFIG.stackPrefix}-api-\${StageName}-url"

  ApiGatewayId:
    Description: API Gateway ID
    Value: !Ref UnifiedApi
    Export:
      Name: !Sub "${LAYER_CONFIG.stackPrefix}-api-\${StageName}-id"

  HybridAuthorizerArn:
    Description: Hybrid Authorizer ARN
    Value: "${hybridAuthorizerArn}"
    Export:
      Name: !Sub "${LAYER_CONFIG.stackPrefix}-api-\${StageName}-authorizer-arn"
`;

  fs.writeFileSync(`${outputDir}/template-${stage}.yaml`, template);
  console.log(`   ✅ Generated: ${outputDir}/template-${stage}.yaml`);
}

async function saveDeploymentArtifacts(deploymentSpec, stage, outputDir) {
  console.log(`   💾 Saving artifacts for ${stage}...`);

  if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir, { recursive: true });
  }

  // Save debug JSON
  const debugFile = `${outputDir}/resolved-spec-${stage}.json`;
  fs.writeFileSync(debugFile, JSON.stringify(deploymentSpec.spec, null, 2));
  console.log(`   📄 Saved: ${debugFile}`);

  // Compute hash for versioned S3 key
  const fileBuffer = fs.readFileSync(deploymentSpec.outputFile);
  const hash = crypto
    .createHash("sha1")
    .update(fileBuffer)
    .digest("hex")
    .slice(0, 12);
  const s3Key = `api-specs/${LAYER_CONFIG.layerName}/${hash}/deployment-ready-${stage}.yaml`;

  // Upload to S3
  console.log(`   ☁️  Uploading to S3 (key: ${s3Key})...`);
  const s3Bucket = "truss-api-automated-deployments";

  try {
    execSync(
      `aws s3 cp "${deploymentSpec.outputFile}" "s3://${s3Bucket}/${s3Key}"`,
      { stdio: "inherit" }
    );
    console.log(`   ✅ Uploaded to s3://${s3Bucket}/${s3Key}`);
  } catch (uploadError) {
    console.error(`   ⚠️  S3 upload failed: ${uploadError.message}`);
    console.log(`   🔧 Manual upload: aws s3 cp ${deploymentSpec.outputFile} s3://${s3Bucket}/${s3Key}`);
  }

  return s3Key;
}

// CLI
async function main() {
  const args = process.argv.slice(2);
  const stageIndex = args.findIndex((arg) => arg.startsWith("--stage="));
  const stage =
    stageIndex !== -1 ? args[stageIndex].split("=")[1] : args[0] || "dev";

  // Validate stage (no more external- prefix needed)
  if (!["dev", "staging", "prod"].includes(stage)) {
    console.warn(`⚠️  Unusual stage: ${stage}. Expected: dev, staging, or prod`);
  }

  try {
    const result = await prepareApiDeployment(stage);
    console.log(`\n🎉 Success! Unified API Gateway ready for deployment.`);
    console.log(`\n📁 Generated files:`);
    console.log(`   - ${result.deploymentSpecFile}`);
    console.log(`   - api-gateway/deployment-ready-${stage}.yaml`);
    console.log(`   - api-gateway/resolved-spec-${stage}.json`);
    console.log(`   - api-gateway/template-${stage}.yaml`);
    console.log(`\n🔐 Hybrid Authorizer: ${result.hybridAuthorizerArn}`);
    console.log(`\n🔑 Authentication methods:`);
    console.log(`   - API Key: Include x-api-key header`);
    console.log(`   - JWT: Include Authorization: Bearer <token>`);
  } catch (error) {
    console.error("❌ Deployment preparation failed:", error.message);
    process.exit(1);
  }
}

if (require.main === module) {
  main();
}

module.exports = { prepareApiDeployment };
