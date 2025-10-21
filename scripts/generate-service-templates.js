// scripts/generate-service-templates.js - Updated with comprehensive CORS handling
const fs = require("fs");
const path = require("path");
const yaml = require("js-yaml");
const registryModule = require("./generate-service-registry"); // Import registry module

function generateServiceTemplates(servicePath, configFileArg, stageArg) {
  const configPath = path.join(servicePath, configFileArg || "config.json");

  if (!fs.existsSync(configPath)) {
    console.error(`Config file not found: ${configPath}`);
    process.exit(1);
  }

  // Load the unified config
  const config = JSON.parse(fs.readFileSync(configPath, "utf8"));

  // Generate CloudFormation template
  generateCloudFormationTemplate(servicePath, config, "");

  // Generate OpenAPI specs based on access configuration
  if (config.access?.internal) {
    generateOpenAPISpec(servicePath, config, "internal", ".internal");
  }

  if (config.access?.external) {
    generateOpenAPISpec(servicePath, config, "external", ".external");
  }

  console.log(`✅ Generated templates for ${config.service.name} service`);

  // IMPORTANT: After generating individual service templates,
  // trigger the service registry generation to ensure it's up-to-date.
  // This will read all new/updated openapi.yaml files and compile the registry.
  try {
    console.log("🔄 Updating overall service-registry.json...");
    // Write combined and per-root registries to disk
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
  outputSuffix = ""
) {
  // Determine supported auth methods from access configuration
  const authConfig = config.access?.auth_config || {};
  const supportsApiKey = config.access?.external && authConfig.api_key;
  const supportsCognito = config.access?.internal && authConfig.cognito;
  const supportsServiceRole = authConfig.service_role;
  const requiresDatabase = config.database.required;
  const requiresVPC = config.deployment.vpc_config?.required || false;
  const requiresImageProcessing = config.image_processing !== undefined;

  // Build security modes list for environment variables
  const securityModes = [];
  if (config.access?.internal && authConfig.cognito)
    securityModes.push("cognito");
  if (config.access?.external && authConfig.api_key)
    securityModes.push("api_key");
  if (authConfig.public) securityModes.push("public");
  const defaultSecurity = securityModes[0] || "public";

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

  // Add database parameters only if needed
  if (requiresDatabase) {
    template += `
  DatabaseHost:
    Type: String
    Description: RDS Proxy endpoint
    Default: "${config.database.host}"
  DatabaseUser:
    Type: String
    NoEcho: true
    Description: Database username
  DatabasePassword:
    Type: String
    NoEcho: true
    Description: Database password
  DatabaseName:
    Type: String
    Default: "${config.database.name}"`;
  }

  // Add VPC parameters only if needed
  if (requiresVPC) {
    template += `
  VpcId:
    Type: String
    Description: VPC ID where RDS Proxy is located
    Default: "${config.aws.vpc_id}"
  SubnetIds:
    Type: CommaDelimitedList
    Description: Subnet IDs for Lambda
    Default: "${config.deployment.vpc_config.subnets.join(",")}"
  SecurityGroupIds:
    Type: CommaDelimitedList
    Description: Security Group IDs for Lambda
    Default: "${config.deployment.vpc_config.security_groups.join(",")}"`;
  }

  // Add API Gateway ID parameter (always include this)
  template += `
  ApiGatewayId:
    Type: String
    Description: API Gateway ID that will invoke this Lambda
    Default: "*"`;

  template += `

Resources:`;

  // Add API Key if supported
  if (supportsApiKey) {
    template += `
  ServiceApiKey:
    Type: AWS::ApiGateway::ApiKey
    Properties:
      Name: !Sub "\${FunctionName}-api-key"
      Description: "API Key for ${config.service.name} service"
      Enabled: true`;
  }

  // IAM Role - simplified approach
  template += `
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
            Action: sts:AssumeRole`;

  // Add service role principals if supported
  if (
    supportsServiceRole &&
    config.auth_config.service_role?.allowed_principals
  ) {
    config.auth_config.service_role.allowed_principals.forEach((principal) => {
      template += `
          - Effect: Allow
            Principal:
              AWS: "${principal}"
            Action: sts:AssumeRole
            Condition:
              StringEquals:
                "sts:ExternalId": !Sub "\${FunctionName}-service-access"`;
    });
  }

  template += `
      ManagedPolicyArns:
        - arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole`;

  // Add policies - always include BigQuery secrets access
  template += `
      Policies:
        - PolicyName: !Sub "\${FunctionName}-Policy"
          PolicyDocument:
            Version: "2012-10-17"
            Statement:`;

  // Always add secrets access for BigQuery and OpenAI
  template += `
              - Effect: Allow
                Action:
                  - secretsmanager:GetSecretValue
                Resource:
                  - "arn:aws:secretsmanager:${config.aws.region}:${config.aws.account_id}:secret:bigquery-service-account*"
                  - "arn:aws:secretsmanager:${config.aws.region}:${config.aws.account_id}:secret:openAI*"`;

  // Add database permissions if needed
  if (requiresDatabase) {
    template += `
              - Effect: Allow
                Action:
                  - rds-db:connect
                Resource:
                  - "arn:aws:rds-db:${config.aws.region}:${config.aws.account_id}:dbuser:prx-06ec245e2f74cefcd/*"`;
  }

  // Add VPC permissions if needed
  if (requiresVPC) {
    template += `
              - Effect: Allow
                Action:
                  - ec2:CreateNetworkInterface
                  - ec2:DescribeNetworkInterfaces
                  - ec2:DeleteNetworkInterface
                  - ec2:AttachNetworkInterface
                  - ec2:DetachNetworkInterface
                Resource: "*"`;
  }

  // Add S3 and DynamoDB permissions for image processing
  if (requiresImageProcessing) {
    template += `
              - Effect: Allow
                Action:
                  - s3:GetObject
                  - s3:PutObject
                  - s3:DeleteObject
                  - s3:ListBucket
                  - s3:GetObjectVersion
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

  // Lambda Function
  template += `

  ServiceLambda:
    Type: AWS::Lambda::Function
    Properties:
      FunctionName: !Ref FunctionName
      Handler: index.handler
      Runtime: ${config.deployment.runtime}
      Role: !GetAtt ServiceRole.Arn
      Code:
        S3Bucket: !Ref CodeS3Bucket
        S3Key: !Ref CodeS3Key
      Timeout: ${config.deployment.timeout}
      MemorySize: ${config.deployment.memory}
      Layers:
${config.deployment.layers.map((layer) => `        - "${layer}"`).join("\n")}`;

  // VPC Config only if required
  if (requiresVPC) {
    template += `
      VpcConfig:
        SecurityGroupIds: !Ref SecurityGroupIds
        SubnetIds: !Ref SubnetIds`;
  }

  // Environment Variables
  template += `
      Environment:
        Variables:
          STAGE: !Ref StageName
          SERVICE_NAME: "${config.service.name}"
          SUPPORTED_AUTH_MODES: "${securityModes.join(",")}"
          DEFAULT_AUTH_MODE: "${defaultSecurity}"
          BIGQUERY_SECRET_ARN: "arn:aws:secretsmanager:${config.aws.region}:${
    config.aws.account_id
  }:secret:bigquery-service-account-GipBFQ"
          OPENAI_SECRET_ARN: "arn:aws:secretsmanager:${config.aws.region}:${
    config.aws.account_id
  }:secret:openAI-FNAJfl"`;

  // Database environment variables only if needed
  if (requiresDatabase) {
    template += `
          RDS_PROXY_ENDPOINT: !Ref DatabaseHost
          DB_HOST: !Ref DatabaseHost
          DB_USER: !Ref DatabaseUser
          DB_PASSWORD: !Ref DatabasePassword
          DB_NAME: !Ref DatabaseName
          DB_CONNECTION_STRATEGY: "${config.database.connection_type}"`;
  }

  // Auth environment variables
  if (supportsApiKey) {
    template += `
          API_KEY_ID: !Ref ServiceApiKey`;
  }

  if (supportsServiceRole) {
    template += `
          SERVICE_ROLE_EXTERNAL_ID: !Sub "\${FunctionName}-service-access"`;
  }

  if (supportsCognito && authConfig.cognito) {
    // Handle both single ARN (backward compatibility) and multiple ARNs
    let cognitoArns = [];
    if (authConfig.cognito.user_pool_arns) {
      cognitoArns = authConfig.cognito.user_pool_arns;
    } else if (authConfig.cognito.user_pool_arn) {
      cognitoArns = [authConfig.cognito.user_pool_arn];
    }

    template += `
          COGNITO_USER_POOL_ARNS: "${cognitoArns.join(",")}"`;
  }

  // Image processing environment variables
  if (requiresImageProcessing) {
    template += `
          SOURCE_BUCKET: !Sub "truss-annotation-image-source-\${StageName}"
          PROCESSED_BUCKET: !Sub "truss-annotation-image-processed-\${StageName}"
          PROCESSING_TABLE: !Sub "truss-image-processing-\${StageName}"
          CLOUDFRONT_URL: !Sub "https://truss-annotation-image-processed-\${StageName}.s3.${config.aws.region}.amazonaws.com"`;
  }

  // Lambda Permissions - Allow any API Gateway to invoke
  template += `

  ServiceLambdaPermission:
    Type: AWS::Lambda::Permission
    Properties:
      FunctionName: !Ref ServiceLambda
      Action: lambda:InvokeFunction
      Principal: apigateway.amazonaws.com
      SourceArn: !Sub "arn:aws:execute-api:\${AWS::Region}:\${AWS::AccountId}:*/*/*"`;

  // Service role invoke permissions
  if (supportsServiceRole && authConfig.service_role?.allowed_principals) {
    authConfig.service_role.allowed_principals.forEach((principal, index) => {
      template += `

  ServiceInvokePermission${index + 1}:
    Type: AWS::Lambda::Permission
    Properties:
      FunctionName: !Ref ServiceLambda
      Action: lambda:InvokeFunction
      Principal: iam.amazonaws.com
      SourceArn: "${principal}"`;
    });
  }

  // Outputs
  template += `

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
      Name: !Sub "\${AWS::StackName}-FunctionName"`;

  if (supportsApiKey) {
    template += `

  ApiKeyId:
    Description: API Key ID for private access
    Value: !Ref ServiceApiKey
    Export:
      Name: !Sub "\${AWS::StackName}-ApiKeyId"`;
  }

  fs.writeFileSync(
    path.join(servicePath, `template${outputSuffix}.yaml`),
    template
  );
  console.log(`✅ Generated template${outputSuffix}.yaml`);
}

function generateOpenAPISpec(servicePath, config, stageArg, outputSuffix = "") {
  const stage = stageArg || process.env.STAGE || process.env.stage || "dev";
  const isExternal = stage.startsWith("external");

  // Determine available security modes based on access configuration
  const authConfig = config.access?.auth_config || {};
  const securityModes = [];

  if (isExternal && config.access?.external && authConfig.api_key) {
    securityModes.push("api_key");
  }

  if (!isExternal && config.access?.internal && authConfig.cognito) {
    securityModes.push("cognito");
  }

  if (authConfig.public) {
    securityModes.push("public");
  }

  const defaultSecurity = securityModes[0] || "public";

  const spec = {
    openapi: "3.0.1",
    info: {
      title: config.service.name,
      version: config.service.version,
      description: config.service.description,
    },
    components: {
      securitySchemes: {},
    },
    paths: {},
  };

  // For external, add API-level key source extension
  if (isExternal) {
    spec["x-amazon-apigateway-api-key-source"] = "HEADER";
  }

  // Only add security schemes that are actually supported
  if (!isExternal && securityModes.includes("cognito")) {
    // Use custom authorizer instead of direct Cognito integration
    spec.components.securitySchemes.CognitoAuth = {
      type: "apiKey",
      in: "header",
      name: "Authorization",
      "x-amazon-apigateway-authtype": "custom",
      "x-amazon-apigateway-authorizer": {
        type: "token",
        authorizerUri: `arn:aws:apigateway:${config.aws.region}:lambda:path/2015-03-31/functions/\${stageVariables.customAuthorizerArn}/invocations`,
        authorizerCredentials: `arn:aws:iam::${config.aws.account_id}:role/TrussApiGatewayAuthorizerRole-prod`,
        authorizerResultTtlInSeconds: 300,
        identitySource: "method.request.header.Authorization",
      },
    };
  }

  if (securityModes.includes("api_key")) {
    // For external, do NOT add x-amazon-apigateway-authtype
    if (isExternal) {
      spec.components.securitySchemes.ApiKeyAuth = {
        type: "apiKey",
        in: "header",
        name: "X-API-Key",
      };
    } else {
      spec.components.securitySchemes.ApiKeyAuth = {
        type: "apiKey",
        in: "header",
        name: "X-API-Key",
        "x-amazon-apigateway-authtype": "apiKey",
      };
    }
  }

  // Helper to add an endpoint operation into spec under a basePath
  function addEndpointPath(basePath, endpoint) {
    const fullPath = `${basePath}${endpoint.path}`;

    let security = [];
    if (isExternal) {
      // For external stages, public endpoints remain public, others require API Key
      if (securityModes.includes("public") || endpoint.security === "public") {
        security = [{}];
      } else {
        security = [{ ApiKeyAuth: [] }];
      }
    } else {
      // Determine which auth to use based on default security or first available
      if (defaultSecurity === "public") {
        security = [{}];
      } else if (
        defaultSecurity === "cognito" &&
        securityModes.includes("cognito")
      ) {
        security = [{ CognitoAuth: [] }];
      } else if (
        defaultSecurity === "api_key" &&
        securityModes.includes("api_key")
      ) {
        security = [{ ApiKeyAuth: [] }];
      } else {
        // Fallback to first available auth method
        if (securityModes.includes("cognito")) {
          security = [{ CognitoAuth: [] }];
        } else if (securityModes.includes("api_key")) {
          security = [{ ApiKeyAuth: [] }];
        } else {
          security = [{}]; // Public fallback
        }
      }
    }

    const methodLower = endpoint.method.toLowerCase();

    // Determine CORS origin based on auth config
    const corsOrigin =
      authConfig[defaultSecurity]?.cors_origin ||
      authConfig.cognito?.cors_origin ||
      "*";

    // Build the response schema without CORS headers in properties
    const responseSchema = {
      type: "object",
      properties: {
        component_type: { type: "string" },
        data: { type: "array" },
        metadata: {
          type: "object",
          properties: {
            auth_method: {
              type: "string",
              enum: securityModes,
            },
          },
        },
      },
    };

    spec.paths[fullPath] = {
      [methodLower]: {
        summary: endpoint.description,
        description: `${endpoint.description} - Auth: ${
          isExternal
            ? security.length === 0 ||
              (security.length === 1 && Object.keys(security[0]).length === 0)
              ? "public"
              : "api_key"
            : defaultSecurity
        }`,
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
                schema: responseSchema,
              },
            },
          },
          400: {
            description: "Bad request",
            headers: {
              "Access-Control-Allow-Origin": { schema: { type: "string" } },
              "Access-Control-Allow-Methods": { schema: { type: "string" } },
              "Access-Control-Allow-Headers": { schema: { type: "string" } },
            },
          },
          401: {
            description: "Unauthorized - authentication failed",
            headers: {
              "Access-Control-Allow-Origin": { schema: { type: "string" } },
              "Access-Control-Allow-Methods": { schema: { type: "string" } },
              "Access-Control-Allow-Headers": { schema: { type: "string" } },
            },
          },
          403: {
            description: "Forbidden - access denied",
            headers: {
              "Access-Control-Allow-Origin": { schema: { type: "string" } },
              "Access-Control-Allow-Methods": { schema: { type: "string" } },
              "Access-Control-Allow-Headers": { schema: { type: "string" } },
            },
          },
          404: {
            description: "Not found",
            headers: {
              "Access-Control-Allow-Origin": { schema: { type: "string" } },
              "Access-Control-Allow-Methods": { schema: { type: "string" } },
              "Access-Control-Allow-Headers": { schema: { type: "string" } },
            },
          },
          500: {
            description: "Internal server error",
            headers: {
              "Access-Control-Allow-Origin": { schema: { type: "string" } },
              "Access-Control-Allow-Methods": { schema: { type: "string" } },
              "Access-Control-Allow-Headers": { schema: { type: "string" } },
            },
          },
        },
        "x-amazon-apigateway-integration": {
          uri: `arn:aws:apigateway:${config.aws.region}:lambda:path/2015-03-31/functions/\${stageVariables.${config.service.name}FunctionArn}/invocations`,
          passthroughBehavior: "when_no_match",
          httpMethod: "POST",
          type: "aws_proxy",
          responses: {
            default: {
              statusCode: "200",
              responseParameters: {
                "method.response.header.Access-Control-Allow-Origin": `'${corsOrigin}'`,
                "method.response.header.Access-Control-Allow-Headers":
                  "'Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token'",
                "method.response.header.Access-Control-Allow-Methods": `'${endpoint.method},OPTIONS'`,
              },
            },
            "4\\d{2}": {
              statusCode: "400",
              responseParameters: {
                "method.response.header.Access-Control-Allow-Origin": `'${corsOrigin}'`,
                "method.response.header.Access-Control-Allow-Headers":
                  "'Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token'",
                "method.response.header.Access-Control-Allow-Methods": `'${endpoint.method},OPTIONS'`,
              },
            },
            "5\\d{2}": {
              statusCode: "500",
              responseParameters: {
                "method.response.header.Access-Control-Allow-Origin": `'${corsOrigin}'`,
                "method.response.header.Access-Control-Allow-Headers":
                  "'Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token'",
                "method.response.header.Access-Control-Allow-Methods": `'${endpoint.method},OPTIONS'`,
              },
            },
          },
        },
      },
      options: {
        summary: "CORS support",
        description: "CORS preflight - no authentication required",
        security: [], // No authentication for OPTIONS
        responses: {
          200: {
            description: "CORS enabled",
            headers: {
              "Access-Control-Allow-Origin": { schema: { type: "string" } },
              "Access-Control-Allow-Methods": { schema: { type: "string" } },
              "Access-Control-Allow-Headers": { schema: { type: "string" } },
              "Access-Control-Max-Age": { schema: { type: "string" } },
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
                  "'Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token'",
                "method.response.header.Access-Control-Allow-Methods": `'${endpoint.method},OPTIONS'`,
                "method.response.header.Access-Control-Allow-Origin": "'*'", // Always allow * for OPTIONS
                "method.response.header.Access-Control-Max-Age": "'86400'",
              },
            },
          },
        },
      },
    };
  }

  // Generate paths from config.api.endpoints for default base path
  const partitionRoutes = config.api?.partitions?.routes || {};
  const hasPartitions = Object.keys(partitionRoutes).length > 0;

  if (!hasPartitions) {
    (config.api.endpoints || []).forEach((endpoint) => {
      addEndpointPath(config.api.base_path, endpoint);
    });
  }

  // Also generate partition base paths (mounted directly at routeCfg.base_path), if defined
  Object.values(partitionRoutes).forEach((routeCfg) => {
    if (routeCfg && typeof routeCfg.base_path === "string") {
      const partitionBase = `${routeCfg.base_path}${config.api.base_path}`;
      (config.api.endpoints || []).forEach((endpoint) => {
        addEndpointPath(partitionBase, endpoint);
      });
    }
  });

  const yamlContent = `# Generated from config.json - DO NOT EDIT DIRECTLY
# Service: ${config.service.name}
# Generated: ${new Date().toISOString()}
# CORS Support: Enhanced with error response handling
${yaml.dump(spec, { lineWidth: -1, noRefs: true, quotingType: '"' })}`;

  fs.writeFileSync(
    path.join(servicePath, `openapi${outputSuffix}.yaml`),
    yamlContent
  );
  console.log(
    `✅ Generated openapi${outputSuffix}.yaml with enhanced CORS support`
  );
}

// CLI interface
if (require.main === module) {
  const args = process.argv.slice(2);
  const servicePath = args[0];
  const configFileArg = args[1] || null;
  const stageArg = args[2] || null;
  if (!servicePath) {
    console.error(
      "Usage: node generate-service-templates.js <service-directory> [config-file] [stage]"
    );
    process.exit(1);
  }
  generateServiceTemplates(servicePath, configFileArg, stageArg);
}
