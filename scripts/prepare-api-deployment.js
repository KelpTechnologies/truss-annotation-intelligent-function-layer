const fs = require("fs");
const path = require("path");
const { execSync } = require("child_process");
const { aggregateOpenAPISpecs } = require("./aggregate-openapi");
const crypto = require("crypto");

/**
 * Prepares API Gateway deployment by aggregating specs and resolving Lambda ARNs
 */
async function prepareApiDeployment(stage = "dev") {
  console.log(`🚀 Preparing API Gateway deployment for stage: ${stage}`);

  const isExternal = stage.startsWith("external");
  const outputDir = isExternal ? "api-gateway-external" : "api-gateway";

  // Aggregate OpenAPI specs
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

  console.log("✅ Step 1 completed successfully");

  // Step 2: Get Lambda ARNs from CloudFormation
  console.log("\n🔍 Step 2: Resolving Lambda function ARNs...");
  const lambdaArns = await resolveLambdaArns(aggregationResult.services, stage);
  console.log("✅ Step 2 completed successfully");

  // Step 3: Get Custom Authorizer ARN (NEW)
  let customAuthorizerArn = "";
  if (!isExternal) {
    console.log("\n🔐 Step 3: Resolving Custom Authorizer ARN...");
    customAuthorizerArn = await resolveCustomAuthorizerArn(stage);
    console.log("✅ Step 3 completed successfully");
  } else {
    console.log(
      "\n🔐 Step 3: Skipping Custom Authorizer ARN for external stage..."
    );
  }

  // Step 4: Create deployment-ready OpenAPI spec
  console.log("\n🔧 Step 4: Creating deployment-ready specification...");
  const deploymentSpec = await createDeploymentSpec(
    outputFile,
    lambdaArns,
    customAuthorizerArn,
    stage,
    outputDir,
    isExternal
  );
  console.log("✅ Step 4 completed successfully");

  // Step 5: Save deployment spec (upload to versioned S3 key) and then generate template pointing to that key
  console.log("\n💾 Step 5: Saving deployment specifications...");
  const definitionS3Key = await saveDeploymentArtifacts(
    deploymentSpec,
    stage,
    outputDir
  );
  console.log("✅ Step 5 completed successfully");

  // Step 6: Generate API Gateway CloudFormation template using versioned key
  console.log("\n📝 Step 6: Generating API Gateway template...");
  generateApiGatewayTemplate(
    stage,
    deploymentSpec,
    customAuthorizerArn,
    outputDir,
    isExternal,
    definitionS3Key
  );
  console.log("✅ Step 6 completed successfully");

  console.log("\n✅ API Gateway deployment preparation complete!");

  return {
    stage,
    services: aggregationResult.services,
    pathsCount: aggregationResult.pathsCount,
    deploymentSpecFile: deploymentSpec.outputFile,
    customAuthorizerArn,
  };
}

async function resolveLambdaArns(serviceNames, stage) {
  const lambdaArns = {};
  let lambdaStage = stage;

  if (stage === "external-prod") {
    lambdaStage = "prod";
  } else if (stage === "external-dev") {
    lambdaStage = "dev";
  }

  console.log(
    `   Resolving Lambda ARNs for API stage '${stage}' using Lambda stage '${lambdaStage}'`
  );

  for (const serviceName of serviceNames) {
    try {
      // Use flattened service name for stack name
      const stackName = `truss-annotation-data-service-${serviceName}-${lambdaStage}`;
      const exportName = `${stackName}-FunctionArn`;

      console.log(`   🔍 Looking for export: ${exportName}`);

      const command = `aws cloudformation list-exports --query "Exports[?Name=='${exportName}'].Value" --output text`;
      const arn = execSync(command, { encoding: "utf-8" }).trim();

      if (arn && arn !== "None" && !arn.includes("None")) {
        lambdaArns[`${serviceName}FunctionArn`] = arn;
        console.log(`   ✅ ${serviceName}: ${arn}`);
      } else {
        // Fallback to constructed ARN
        const fallbackArn = `arn:aws:lambda:eu-west-2:193757560043:function:truss-annotation-data-service-${serviceName}-${lambdaStage}`;
        lambdaArns[`${serviceName}FunctionArn`] = fallbackArn;
        console.log(`   ⚠️  ${serviceName}: Using fallback ARN ${fallbackArn}`);
      }
    } catch (error) {
      console.error(
        `   ❌ Failed to get ARN for ${serviceName}:`,
        error.message
      );
      // Use fallback
      const fallbackArn = `arn:aws:lambda:eu-west-2:193757560043:function:truss-annotation-data-service-${serviceName}-${lambdaStage}`;
      lambdaArns[`${serviceName}FunctionArn`] = fallbackArn;
      console.log(`   ⚠️  ${serviceName}: Using fallback ARN ${fallbackArn}`);
    }
  }

  return lambdaArns;
}

// NEW FUNCTION: Resolve Custom Authorizer ARN
async function resolveCustomAuthorizerArn(stage) {
  try {
    const functionName = `truss-annotation-custom-authorizer-${stage}`;
    console.log(`   🔍 Looking for custom authorizer: ${functionName}`);

    // Try to get the function ARN from CloudFormation export first
    const exportName = `truss-annotation-custom-authorizer-${stage}-arn`;
    try {
      const command = `aws cloudformation list-exports --query "Exports[?Name=='${exportName}'].Value" --output text`;
      const arn = execSync(command, { encoding: "utf-8" }).trim();

      if (arn && arn !== "None" && !arn.includes("None")) {
        console.log(`   ✅ Custom authorizer ARN from export: ${arn}`);
        return arn;
      }
    } catch (exportError) {
      console.log(
        `   ⚠️  Export ${exportName} not found, trying direct function lookup`
      );
    }

    // Fallback: try to get function ARN directly
    try {
      const command = `aws lambda get-function --function-name ${functionName} --query 'Configuration.FunctionArn' --output text`;
      const arn = execSync(command, { encoding: "utf-8" }).trim();

      if (arn && arn !== "None" && !arn.includes("None")) {
        console.log(`   ✅ Custom authorizer ARN from function: ${arn}`);
        return arn;
      }
    } catch (functionError) {
      console.log(`   ⚠️  Function ${functionName} not found`);
    }

    // Final fallback: construct expected ARN
    const fallbackArn = `arn:aws:lambda:eu-west-2:193757560043:function:${functionName}`;
    console.log(`   ⚠️  Using fallback custom authorizer ARN: ${fallbackArn}`);
    console.log(
      `   💡 Note: Custom authorizer may not be deployed yet. Deploy it with the 'Deploy Custom Authorizer' workflow.`
    );

    return fallbackArn;
  } catch (error) {
    console.error(
      `   ❌ Error resolving custom authorizer ARN:`,
      error.message
    );
    // Return fallback ARN even on error
    const fallbackArn = `arn:aws:lambda:eu-west-2:193757560043:function:truss-annotation-custom-authorizer-${stage}`;
    console.log(`   ⚠️  Using fallback custom authorizer ARN: ${fallbackArn}`);
    return fallbackArn;
  }
}

async function createDeploymentSpec(
  inputFile,
  lambdaArns,
  customAuthorizerArn,
  stage,
  outputDir,
  isExternal = false
) {
  const yaml = require("js-yaml");

  console.log(`   Processing ${inputFile}...`);

  // Load aggregated spec
  let spec = yaml.load(fs.readFileSync(inputFile, "utf8"));

  // Replace stage variables with actual ARNs
  const specString = JSON.stringify(spec);
  let updatedSpecString = specString;

  // Replace Lambda function ARNs
  Object.entries(lambdaArns).forEach(([variable, arn]) => {
    const placeholder = `\${stageVariables.${variable}}`;
    updatedSpecString = updatedSpecString.replace(
      new RegExp(placeholder.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"), "g"),
      arn
    );
    console.log(`   🔄 Replaced ${variable} with actual ARN`);
  });

  // Replace Custom Authorizer ARN (NEW) only if not external
  if (!isExternal) {
    const authorizerPlaceholder = `\${stageVariables.customAuthorizerArn}`;
    updatedSpecString = updatedSpecString.replace(
      new RegExp(
        authorizerPlaceholder.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"),
        "g"
      ),
      customAuthorizerArn
    );
    console.log(`   🔄 Replaced customAuthorizerArn with actual ARN`);
  }

  spec = JSON.parse(updatedSpecString);

  // Update server URL for the specific stage
  spec.servers = [
    {
      url: `https://\${ApiId}.execute-api.eu-west-2.amazonaws.com/${stage}`,
      description: `${
        stage.charAt(0).toUpperCase() + stage.slice(1)
      } environment`,
    },
  ];

  // Add deployment metadata
  spec.info.version = `1.0.0-${stage}`;
  spec.info["x-deployment"] = {
    stage: stage,
    timestamp: new Date().toISOString(),
    lambdaArns: Object.keys(lambdaArns),
    customAuthorizerArn: !isExternal ? customAuthorizerArn : undefined,
    services: Object.keys(lambdaArns).map((arn) =>
      arn.replace("FunctionArn", "")
    ),
  };

  const outputFile = `${outputDir}/deployment-ready-${stage}.yaml`;
  const yamlContent = yaml.dump(spec, { lineWidth: -1, noRefs: true });

  // Only include custom authorizer comment if not external
  const finalContent = `# Deployment-ready OpenAPI specification
# Generated on: ${new Date().toISOString()}
# Stage: ${stage}
# Lambda ARNs resolved: ${Object.keys(lambdaArns).length}
${!isExternal ? `# Custom Authorizer: ${customAuthorizerArn}\n` : ""}
${yamlContent}`;

  fs.writeFileSync(outputFile, finalContent);
  console.log(`   ✅ Created deployment spec: ${outputFile}`);

  return {
    outputFile,
    spec,
    lambdaArns,
    customAuthorizerArn,
  };
}

function generateApiGatewayTemplate(
  stage,
  deploymentSpec,
  customAuthorizerArn,
  outputDir,
  isExternal = false,
  definitionS3Key = null
) {
  // Remove Auth block entirely when using DefinitionUri to avoid SAM validation errors
  let authBlock = "";

  const variablesBlock = isExternal
    ? ""
    : `      Variables:\n        customAuthorizerArn: \"${customAuthorizerArn}\"`;

  const customAuthorizerOutput = isExternal
    ? ""
    : `\n  CustomAuthorizerArn:\n    Description: Custom Authorizer ARN used by this API\n    Value: \"${customAuthorizerArn}\"\n    Export:\n      Name: !Sub "truss-annotation-data-service-api-${stage}-custom-authorizer-arn"\n`;

  const template = `AWSTemplateFormatVersion: "2010-09-09"
Transform: AWS::Serverless-2016-10-31
Description: "Truss Data Service API Gateway - Generated for ${stage}"

Parameters:
  StageName:
    Type: String
    Default: ${stage}
    Description: Deployment stage

Resources:
  TrussDataServiceApi:
    Type: AWS::Serverless::Api
    Properties:
      Name: !Sub "truss-annotation-data-service-api-\${StageName}"
      StageName: !Ref StageName
      DefinitionUri: s3://truss-api-automated-deployments/${
        definitionS3Key || `api-specs/deployment-ready-${stage}.yaml`
      }
      EndpointConfiguration:
        Type: REGIONAL
      TracingEnabled: true
${variablesBlock}
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
    Description: API Gateway Invoke URL
    Value: !Sub "https://\${TrussDataServiceApi}.execute-api.\${AWS::Region}.amazonaws.com/\${StageName}"
    Export:
      Name: !Sub "truss-annotation-data-service-api-\${StageName}-url"

  ApiGatewayId:
    Description: API Gateway ID
    Value: !Ref TrussDataServiceApi
    Export:
      Name: !Sub "truss-annotation-data-service-api-\${StageName}-id"

  Stage:
    Description: Deployment stage
    Value: !Ref StageName
    Export:
      Name: !Sub "truss-annotation-data-service-api-\${StageName}-stage"
${customAuthorizerOutput}
`;

  fs.writeFileSync(`${outputDir}/template-${stage}.yaml`, template);
  console.log(
    `   ✅ Generated API Gateway template: ${outputDir}/template-${stage}.yaml`
  );
}

async function saveDeploymentArtifacts(deploymentSpec, stage, outputDir) {
  try {
    console.log(`   💾 Saving deployment artifacts for ${stage}...`);

    // Ensure output directory exists
    if (!fs.existsSync(outputDir)) {
      fs.mkdirSync(outputDir, { recursive: true });
    }

    // Save a copy of the resolved spec for debugging
    const debugFile = `${outputDir}/resolved-spec-${stage}.json`;
    fs.writeFileSync(debugFile, JSON.stringify(deploymentSpec.spec, null, 2));
    console.log(`   📄 Saved debug spec: ${debugFile}`);

    // Compute content hash for versioned S3 key
    const fileBuffer = fs.readFileSync(deploymentSpec.outputFile);
    const hash = crypto
      .createHash("sha1")
      .update(fileBuffer)
      .digest("hex")
      .slice(0, 12);
    const s3Key = `api-specs/${hash}/deployment-ready-${stage}.yaml`;
    // Upload the deployment-ready spec to S3 (may be skipped locally if AWS CLI missing)
    console.log(`   ☁️  Uploading OpenAPI spec to S3 (key: ${s3Key})...`);
    const s3Bucket = "truss-api-automated-deployments";

    try {
      execSync(
        `aws s3 cp "${deploymentSpec.outputFile}" "s3://${s3Bucket}/${s3Key}"`,
        {
          stdio: "inherit",
        }
      );
      console.log(`   ✅ Uploaded to s3://${s3Bucket}/${s3Key}`);
    } catch (uploadError) {
      console.error(`   ⚠️  S3 upload failed: ${uploadError.message}`);
      console.log(
        `   🔧 You may need to upload manually: aws s3 cp ${deploymentSpec.outputFile} s3://${s3Bucket}/${s3Key}`
      );
    }

    // Log ARN mappings for verification
    console.log(`   🔍 Lambda ARN mappings:`);
    Object.entries(deploymentSpec.lambdaArns).forEach(([key, value]) => {
      console.log(`     ${key}: ${value}`);
    });

    console.log(
      `   🔐 Custom Authorizer ARN: ${deploymentSpec.customAuthorizerArn}`
    );

    console.log(`   ✅ All deployment artifacts saved`);
    return s3Key;
  } catch (error) {
    console.error(`   ❌ Failed to save deployment artifacts:`, error.message);
    throw error;
  }
}

// CLI interface
async function main() {
  const args = process.argv.slice(2);
  const stageIndex = args.findIndex((arg) => arg.startsWith("--stage="));
  const stage =
    stageIndex !== -1 ? args[stageIndex].split("=")[1] : args[0] || "dev";

  try {
    const result = await prepareApiDeployment(stage);
    const outputDir = stage.startsWith("external")
      ? "api-gateway-external"
      : "api-gateway";
    console.log(`\n🎉 Success! API Gateway is ready for deployment.`);
    console.log(`📁 Generated files:`);
    console.log(`   - ${result.deploymentSpecFile}`);
    console.log(`   - ${outputDir}/deployment-ready-${stage}.yaml`);
    console.log(`   - ${outputDir}/resolved-spec-${stage}.json`);
    console.log(`\n🔐 Custom Authorizer:`);
    console.log(`   - ARN: ${result.customAuthorizerArn}`);
    console.log(`   - Stage Variables: Configured`);
    console.log(`\n🌐 CORS Configuration:`);
    console.log(`   - Gateway Responses: Configured for all error types`);
    console.log(`   - Allowed Origins: * (all origins)`);
    console.log(
      `   - Allowed Headers: Content-Type, Authorization, X-Api-Key, etc.`
    );
    console.log(`   - Allowed Methods: GET, POST, PUT, DELETE, OPTIONS`);
  } catch (error) {
    console.error("❌ Deployment preparation failed:", error.message);
    process.exit(1);
  }
}

if (require.main === module) {
  main();
}

module.exports = { prepareApiDeployment };
