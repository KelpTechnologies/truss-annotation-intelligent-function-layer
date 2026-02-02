// scripts/generate-service-templates.js - UNIFIED API VERSION
// Generates a single OpenAPI spec per service using HybridAuth

const fs = require("fs");
const path = require("path");
const yaml = require("js-yaml");
const registryModule = require("./generate-service-registry");

// Load layer configuration from single source of truth
const LAYER_CONFIG = (() => {
  const configPath = path.join(__dirname, "layer-config.json");
  if (!fs.existsSync(configPath)) {
    console.error("❌ layer-config.json not found! Create it first.");
    process.exit(1);
  }
  return JSON.parse(fs.readFileSync(configPath, "utf-8"));
})();

function generateServiceTemplates(servicePath, configFileArg, stageArg) {
  const configPath = path.join(servicePath, configFileArg || "config.json");

  if (!fs.existsSync(configPath)) {
    console.error(`Config file not found: ${configPath}`);
    process.exit(1);
  }

  const config = JSON.parse(fs.readFileSync(configPath, "utf8"));

  // Generate CloudFormation template
  generateCloudFormationTemplate(servicePath, config, "");

  // Generate SINGLE unified OpenAPI spec with HybridAuth
  generateUnifiedOpenAPISpec(servicePath, config);

  console.log(`✅ Generated templates for ${config.service.name} service`);

  // Update service registry
  try {
    console.log("🔄 Updating overall service-registry.json...");
    registryModule.runMain();
    console.log("✅ service-registry.json updated successfully.");
  } catch (error) {
    console.error("❌ Failed to update service-registry.json:", error.message);
    process.exit(1);
  }
}

function generateCloudFormationTemplate(
  servicePath,
  config,
  outputSuffix = "",
) {
  const requiresDatabase = config.database?.required || false;
  const requiresVPC = config.deployment?.vpc_config?.required || false;
  const requiresImageProcessing = config.image_processing !== undefined;
  const requiresDynamoDB =
    config.database?.connection_type === "dynamodb" && requiresDatabase;

  const runtime = config.deployment.runtime;
  // Use index.lambda_handler for Python (new agent architecture)
  // handler.py is legacy and should not be used
  const handler = runtime.startsWith("python")
    ? "index.lambda_handler"
    : "index.handler";

  let template = `AWSTemplateFormatVersion: "2010-09-09"
Transform: AWS::Serverless-2016-10-31
Description: "${config.service.description}"

Parameters:
  StageName:
    Type: String
    Default: dev
  FunctionName:
    Type: String
  ServiceName:
    Type: String
  CodeS3Bucket:
    Type: String
  CodeS3Key:
    Type: String`;

  if (requiresDatabase) {
    template += `
  DatabaseHost:
    Type: String
    Description: RDS Proxy endpoint
    Default: "${config.database.host}"
  DatabaseName:
    Type: String
    Default: "${config.database.name}"`;
  }

  if (requiresVPC) {
    template += `
  VpcId:
    Type: String
    Default: "${config.aws.vpc_id}"
  SubnetIds:
    Type: CommaDelimitedList
    Default: "${config.deployment.vpc_config.subnets.join(",")}"
  SecurityGroupIds:
    Type: CommaDelimitedList
    Default: "${config.deployment.vpc_config.security_groups.join(",")}"`;
  }

  template += `

Resources:
  ServiceRole:
    Type: AWS::IAM::Role
    Properties:
      RoleName: !Sub "truss-ai-\${ServiceName}-\${StageName}-role"
      AssumeRolePolicyDocument:
        Version: "2012-10-17"
        Statement:
          - Effect: Allow
            Principal:
              Service: lambda.amazonaws.com
            Action: sts:AssumeRole
      ManagedPolicyArns:
        - arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
      Policies:
        - PolicyName: !Sub "\${FunctionName}-Policy"
          PolicyDocument:
            Version: "2012-10-17"
            Statement:
              - Effect: Allow
                Action:
                  - secretsmanager:GetSecretValue
                Resource:
                  - "arn:aws:secretsmanager:${config.aws.region}:${config.aws.account_id}:secret:truss-platform-secrets*"`;

  if (requiresVPC) {
    template += `
              - Effect: Allow
                Action:
                  - ec2:CreateNetworkInterface
                  - ec2:DescribeNetworkInterfaces
                  - ec2:DeleteNetworkInterface
                Resource: "*"`;
  }

  if (
    requiresDynamoDB &&
    config.database.tables &&
    Array.isArray(config.database.tables)
  ) {
    // Build DynamoDB actions based on permissions array
    const hasWrite =
      config.database.permissions &&
      config.database.permissions.includes("write");
    const dynamoActions = [
      "dynamodb:GetItem",
      "dynamodb:Query",
      "dynamodb:Scan",
    ];
    if (hasWrite) {
      dynamoActions.push(
        "dynamodb:PutItem",
        "dynamodb:UpdateItem",
        "dynamodb:DeleteItem",
      );
    }

    template += `
              - Effect: Allow
                Action:
${dynamoActions.map((a) => `                  - ${a}`).join("\n")}
                Resource:`;
    config.database.tables.forEach((table) => {
      if (table.includes("${STAGE}")) {
        const tableName = table.replace("${STAGE}", "${StageName}");
        template += `
                  - !Sub "arn:aws:dynamodb:${config.aws.region}:${config.aws.account_id}:table/${tableName}"
                  - !Sub "arn:aws:dynamodb:${config.aws.region}:${config.aws.account_id}:table/${tableName}/index/*"`;
      } else {
        template += `
                  - "arn:aws:dynamodb:${config.aws.region}:${config.aws.account_id}:table/${table}"
                  - "arn:aws:dynamodb:${config.aws.region}:${config.aws.account_id}:table/${table}/index/*"`;
      }
    });
  }

  if (requiresImageProcessing) {
    template += `
              - Effect: Allow
                Action:
                  - s3:GetObject
                  - s3:PutObject
                  - s3:DeleteObject
                  - s3:ListBucket
                Resource:
                  - !Sub "arn:aws:s3:::truss-annotation-image-source-\${StageName}"
                  - !Sub "arn:aws:s3:::truss-annotation-image-source-\${StageName}/*"
                  - !Sub "arn:aws:s3:::truss-annotation-image-processed-\${StageName}"
                  - !Sub "arn:aws:s3:::truss-annotation-image-processed-\${StageName}/*"
              - Effect: Allow
                Action:
                  - dynamodb:GetItem
                  - dynamodb:PutItem
                  - dynamodb:UpdateItem
                  - dynamodb:DeleteItem
                  - dynamodb:Query
                  - dynamodb:Scan
                Resource:
                  - !Sub "arn:aws:dynamodb:\${AWS::Region}:\${AWS::AccountId}:table/truss-image-processing-\${StageName}"`;
  }

  template += `

  ServiceLambda:
    Type: AWS::Lambda::Function
    Properties:
      FunctionName: !Ref FunctionName
      Handler: ${handler}
      Runtime: ${runtime}
      Role: !GetAtt ServiceRole.Arn
      Code:
        S3Bucket: !Ref CodeS3Bucket
        S3Key: !Ref CodeS3Key
      Timeout: ${config.deployment.timeout}
      MemorySize: ${config.deployment.memory}`;

  const layers = (config.deployment.layers || [])
    .map((layer) => `        - "${layer}"`)
    .join("\n");
  if (layers && layers.trim().length > 0) {
    template += `
      Layers:
${layers}`;
  }

  if (requiresVPC) {
    template += `
      VpcConfig:
        SecurityGroupIds: !Ref SecurityGroupIds
        SubnetIds: !Ref SubnetIds`;
  }

  template += `
      Environment:
        Variables:
          STAGE: !Ref StageName
          SERVICE_NAME: "${config.service.name}"
          LAYER_NAME: "${LAYER_CONFIG.layerName}"
          TRUSS_SECRETS_ARN: "arn:aws:secretsmanager:${LAYER_CONFIG.region}:${LAYER_CONFIG.accountId}:secret:truss-platform-secrets-yVuz1R"`;

  if (config.deployment.extra_env) {
    Object.entries(config.deployment.extra_env).forEach(([k, v]) => {
      template += `
          ${k}: "${String(v)}"`;
    });
  }

  if (requiresDatabase) {
    template += `
          DB_HOST: !Ref DatabaseHost`;
  }

  if (requiresImageProcessing) {
    template += `
          SOURCE_BUCKET: !Sub "truss-annotation-image-source-\${StageName}"
          PROCESSED_BUCKET: !Sub "truss-annotation-image-processed-\${StageName}"
          PROCESSING_TABLE: !Sub "truss-image-processing-\${StageName}"`;
  }

  template += `

  ServiceLambdaPermission:
    Type: AWS::Lambda::Permission
    Properties:
      FunctionName: !Ref ServiceLambda
      Action: lambda:InvokeFunction
      Principal: apigateway.amazonaws.com
      SourceArn: !Sub "arn:aws:execute-api:\${AWS::Region}:\${AWS::AccountId}:*/*/*"

Outputs:
  FunctionArn:
    Description: "ARN of the ${config.service.name} Service Lambda"
    Value: !GetAtt ServiceLambda.Arn
    Export:
      Name: !Sub "\${AWS::StackName}-FunctionArn"

  FunctionName:
    Description: "Name of the ${config.service.name} Service Lambda"
    Value: !Ref ServiceLambda
    Export:
      Name: !Sub "\${AWS::StackName}-FunctionName"
`;

  fs.writeFileSync(
    path.join(servicePath, `template${outputSuffix}.yaml`),
    template,
  );
  console.log(`✅ Generated template${outputSuffix}.yaml`);
}

/**
 * Generate UNIFIED OpenAPI spec with HybridAuth security scheme
 * This replaces the separate internal/external spec generation
 */
function generateUnifiedOpenAPISpec(servicePath, config) {
  const corsOrigin = config.access?.auth_config?.cognito?.cors_origin || "*";

  const spec = {
    openapi: "3.0.1",
    info: {
      title: config.service.name,
      version: config.service.version,
      description: config.service.description,
    },
    components: {
      securitySchemes: {
        // Unified Hybrid Authorizer - supports both API Key (x-api-key) and Cognito JWT (Authorization)
        // Uses REQUEST type to receive all headers
        // No identitySource - let Lambda handle auth detection (supports either API Key OR JWT)
        // TTL set to 0 to ensure every request is evaluated (required for flexible auth)
        HybridAuth: {
          type: "apiKey",
          in: "header",
          name: "Authorization",
          "x-amazon-apigateway-authtype": "custom",
          "x-amazon-apigateway-authorizer": {
            type: "request",
            authorizerUri: `arn:aws:apigateway:${LAYER_CONFIG.region}:lambda:path/2015-03-31/functions/\${stageVariables.hybridAuthorizerArn}/invocations`,
            authorizerResultTtlInSeconds: 0,
          },
        },
      },
    },
    paths: {},
  };

  // Helper to add endpoint
  function addEndpointPath(basePath, endpoint) {
    const fullPath = `${basePath}${endpoint.path}`;
    const methodLower = endpoint.method.toLowerCase();

    // All endpoints use HybridAuth (except health checks)
    const isPublic =
      endpoint.path === "/health" || endpoint.security === "public";
    const security = isPublic ? [] : [{ HybridAuth: [] }];

    spec.paths[fullPath] = {
      [methodLower]: {
        summary: endpoint.description,
        description: `${endpoint.description}`,
        security: security,
        responses: {
          200: {
            description: "Success",
            headers: {
              "Access-Control-Allow-Origin": { schema: { type: "string" } },
              "Access-Control-Allow-Methods": { schema: { type: "string" } },
              "Access-Control-Allow-Headers": { schema: { type: "string" } },
            },
            content: {
              "application/json": {
                schema: { type: "object" },
              },
            },
          },
          400: { description: "Bad request" },
          401: { description: "Unauthorized" },
          403: { description: "Forbidden" },
          404: { description: "Not found" },
          500: { description: "Internal server error" },
        },
        "x-amazon-apigateway-integration": {
          uri: `arn:aws:apigateway:${LAYER_CONFIG.region}:lambda:path/2015-03-31/functions/\${stageVariables.${config.service.name}FunctionArn}/invocations`,
          passthroughBehavior: "when_no_match",
          httpMethod: "POST",
          type: "aws_proxy",
        },
      },
      options: {
        summary: "CORS support",
        security: [],
        responses: {
          200: {
            description: "CORS enabled",
            headers: {
              "Access-Control-Allow-Origin": { schema: { type: "string" } },
              "Access-Control-Allow-Methods": { schema: { type: "string" } },
              "Access-Control-Allow-Headers": { schema: { type: "string" } },
            },
          },
        },
        "x-amazon-apigateway-integration": {
          type: "mock",
          requestTemplates: { "application/json": '{"statusCode": 200}' },
          responses: {
            default: {
              statusCode: "200",
              responseParameters: {
                "method.response.header.Access-Control-Allow-Headers":
                  "'Content-Type,Authorization,X-Api-Key,X-Amz-Date,X-Amz-Security-Token'",
                "method.response.header.Access-Control-Allow-Methods": `'${endpoint.method},OPTIONS'`,
                "method.response.header.Access-Control-Allow-Origin": `'${corsOrigin}'`,
              },
            },
          },
        },
      },
    };
  }

  // Generate paths from endpoints
  (config.api.endpoints || []).forEach((endpoint) => {
    addEndpointPath(config.api.base_path, endpoint);
  });

  // Handle partitions if defined
  const partitionRoutes = config.api?.partitions?.routes || {};
  Object.values(partitionRoutes).forEach((routeCfg) => {
    if (routeCfg && typeof routeCfg.base_path === "string") {
      const partitionBase = `${routeCfg.base_path}${config.api.base_path}`;
      (config.api.endpoints || []).forEach((endpoint) => {
        addEndpointPath(partitionBase, endpoint);
      });
    }
  });

  const yamlContent = `# Generated OpenAPI spec with HybridAuth - DO NOT EDIT DIRECTLY
# Service: ${config.service.name}
# Layer: ${LAYER_CONFIG.layerName}
# Generated: ${new Date().toISOString()}
${yaml.dump(spec, { lineWidth: -1, noRefs: true, quotingType: '"' })}`;

  // Write SINGLE unified spec (no .internal or .external suffix)
  fs.writeFileSync(path.join(servicePath, `openapi.yaml`), yamlContent);
  console.log(`✅ Generated openapi.yaml with HybridAuth`);
}

// CLI interface
if (require.main === module) {
  const args = process.argv.slice(2);
  const servicePath = args[0];
  const configFileArg = args[1] || null;
  const stageArg = args[2] || null;
  if (!servicePath) {
    console.error(
      "Usage: node generate-service-templates.js <service-directory> [config-file] [stage]",
    );
    process.exit(1);
  }
  generateServiceTemplates(servicePath, configFileArg, stageArg);
}

module.exports = { generateServiceTemplates, LAYER_CONFIG };
