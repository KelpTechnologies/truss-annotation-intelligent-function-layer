const fs = require("fs");
const path = require("path");

// Service templates
const SERVICE_TEMPLATES = {
  nodejs: {
    runtime: "nodejs18.x",
    handler: "index.handler",
    packageManager: "npm",
  },
  python: {
    runtime: "python3.9",
    handler: "index.handler",
    packageManager: "pip",
  },
};

function createNewService(parentFolder, serviceName, options = {}) {
  const {
    runtime = "nodejs",
    description = `${serviceName} service`,
    security_modes = ["cognito"], // Changed default from api_key to cognito
    database = false,
    vpc = false,
    memory = 512,
    timeout = 30,
    llm_agent = false, // New option for automated annotation services
  } = options;

  const servicePath = path.join(parentFolder, serviceName);

  // Check if service already exists
  if (fs.existsSync(servicePath)) {
    console.error(`❌ Service already exists: ${servicePath}`);
    process.exit(1);
  }

  console.log(`🚀 Creating new ${runtime} service: ${serviceName}`);

  // Create service directory
  fs.mkdirSync(servicePath, { recursive: true });

  // Generate all required files
  generateConfigJson(servicePath, serviceName, runtime, options);
  generateHandlerFile(servicePath, serviceName, runtime);
  generatePackageFile(servicePath, serviceName, runtime);
  generateReadme(servicePath, serviceName, runtime, options);

  console.log(`✅ Created ${runtime} service at: ${servicePath}`);
  console.log(`📁 Files created:`);
  console.log(`   - config.json`);
  console.log(`   - index.${runtime === "nodejs" ? "js" : "py"}`);
  console.log(
    `   - ${runtime === "nodejs" ? "package.json" : "requirements.txt"}`
  );
  console.log(`   - README.md`);
  console.log(`\n🔧 Next steps:`);
  console.log(`   1. Edit config.json to customize your service`);
  console.log(
    `   2. Implement your endpoints in index.${
      runtime === "nodejs" ? "js" : "py"
    }`
  );
  console.log(
    `   3. Run: node scripts/generate-service-templates.js ${servicePath}`
  );
  console.log(`   4. Deploy with your existing pipeline`);
}

function generateConfigJson(servicePath, serviceName, runtime, options) {
  const template = SERVICE_TEMPLATES[runtime];

  // Generate API endpoints based on service type
  let apiEndpoints;
  if (options.llm_agent) {
    apiEndpoints = [
      {
        path: "/extract-knowledge",
        method: "POST",
        description: "Extract knowledge from text using automated annotation",
        parameters: {
          body: {
            text: "string (required) - Text to extract knowledge from",
            extraction_type:
              "string (optional) - Type of extraction (entities, relationships, attributes)",
            domain:
              "string (optional) - Domain context (fashion, luxury, etc.)",
          },
        },
      },
      {
        path: "/annotate",
        method: "POST",
        description:
          "Annotate text with structured data using automated annotation",
        parameters: {
          body: {
            text: "string (required) - Text to annotate",
            annotation_schema:
              "object (optional) - Schema for annotation structure",
            confidence_threshold:
              "number (optional) - Minimum confidence for annotations",
          },
        },
      },
      {
        path: "/health",
        method: "GET",
        description:
          "Health check endpoint for the automated annotation service",
        parameters: {},
      },
    ];
  } else {
    apiEndpoints = [
      {
        path: "",
        method: "GET",
        description: `Get ${serviceName} data`,
      },
      {
        path: "/{id}",
        method: "GET",
        description: `Get specific ${serviceName} by ID`,
      },
    ];
  }

  const config = {
    service: {
      name: serviceName,
      description: options.description || `${serviceName} service`,
      security_modes: options.security_modes || ["cognito"], // Changed default
      default_security: options.security_modes?.[0] || "cognito", // Changed default
      version: "1.0.0",
    },
    deployment: {
      runtime: template.runtime,
      timeout: options.llm_agent ? 300 : options.timeout || 30, // Automated annotation services need longer timeout
      memory: options.llm_agent ? 1024 : options.memory || 512, // Automated annotation services need more memory
      layers:
        runtime === "nodejs"
          ? [
              "arn:aws:lambda:eu-west-2:193757560043:layer:truss-toolkit-nodejs-layer:194",
            ]
          : [],
      vpc_config: {
        required: options.vpc || false,
        security_groups: options.vpc ? ["sg-060c6f3dedc8ec1f8"] : [],
        subnets: options.vpc
          ? [
              "subnet-0f14f1c73238b61ac",
              "subnet-0fee2f00bb864b0f4",
              "subnet-00c097a58a3d0bea2",
            ]
          : [],
      },
    },
    database: {
      required: options.database || false,
      connection_type: options.database ? "proxy" : "none",
      permissions: options.database ? ["read"] : [],
      host: options.database
        ? "sql-connection.proxy-c4btkb6jssvc.eu-west-2.rds.amazonaws.com"
        : "",
      name: options.database ? "api_staging" : "",
    },
    api: {
      base_path: options.llm_agent
        ? `/automations/annotation`
        : `/api/${serviceName}`,
      cors_enabled: true,
      endpoints: apiEndpoints,
    },
    auth_config: generateAuthConfig(options.security_modes || ["cognito"]),
    aws: {
      region: "eu-west-2",
      account_id: "193757560043",
      vpc_id: "vpc-04e79f1770f4205de",
    },
  };

  // Add LLM configuration for automated annotation services
  if (options.llm_agent) {
    config.llm = {
      provider: "openai",
      model: "gpt-4",
      max_tokens: 2000,
      temperature: 0.1,
      api_key_secret: "openai-api-key",
    };
  }

  fs.writeFileSync(
    path.join(servicePath, "config.json"),
    JSON.stringify(config, null, 2)
  );
}

function generateAuthConfig(securityModes) {
  const authConfig = {};

  if (securityModes.includes("public")) {
    authConfig.public = {
      cors_origin: "*",
      rate_limit: { rate: 100, burst: 200 },
    };
  }

  if (securityModes.includes("cognito")) {
    authConfig.cognito = {
      user_pool_arn:
        "arn:aws:cognito-idp:eu-west-2:193757560043:userpool/eu-west-2_JyaPgaRFW",
      cors_origin: "*", // Changed from restricted to open for better flexibility
    };
  }

  if (securityModes.includes("api_key")) {
    authConfig.api_key = {
      cors_origin: "https://internal.truss.com",
      rate_limit: { rate: 1000, burst: 2000 },
    };
  }

  if (securityModes.includes("service_role")) {
    authConfig.service_role = {
      allowed_principals: [
        "arn:aws:iam::193757560043:role/truss-internal-llm-role",
        "arn:aws:iam::193757560043:role/truss-analysis-service-role",
      ],
      cors_origin: "*",
      rate_limit: { rate: 2000, burst: 4000 },
    };
  }

  return authConfig;
}

function generateHandlerFile(servicePath, serviceName, runtime) {
  if (runtime === "nodejs") {
    generateNodeJSHandler(servicePath, serviceName);
  } else if (runtime === "python") {
    generatePythonHandler(servicePath, serviceName);
  }
}

function generateNodeJSHandler(servicePath, serviceName) {
  const handlerCode = `// ${serviceName} service handler
const config = require('./config.json');

// Database connection (if required)
let pool = null;

function initDatabase() {
  if (!config.database.required) return null;
  
  const mysql = require('mysql');
  
  if (pool) return pool;
  
  pool = mysql.createPool({
    host: process.env.RDS_PROXY_ENDPOINT || process.env.DB_HOST,
    user: process.env.DB_USER,
    password: process.env.DB_PASSWORD,
    database: process.env.DB_NAME,
    connectionLimit: 10,
    acquireTimeout: 60000,
    timeout: 60000,
    ssl: false
  });
  
  return pool;
}

async function query(sql, args = []) {
  const connectionPool = initDatabase();
  if (!connectionPool) throw new Error('Database not configured');
  
  return new Promise((resolve, reject) => {
    connectionPool.getConnection((err, connection) => {
      if (err) return reject(err);
      
      connection.query(sql, args, (queryErr, results) => {
        connection.release();
        if (queryErr) return reject(queryErr);
        resolve(results);
      });
    });
  });
}

// Multi-auth security middleware
function determineAuthMethod(event) {
  const supportedModes = config.service.security_modes || ['cognito'];
  
  // Check for Cognito authentication
  if (supportedModes.includes('cognito') && event.requestContext?.authorizer?.claims) {
    return {
      method: 'cognito',
      user: event.requestContext.authorizer.claims,
      authorized: true
    };
  }

  // Check for API Key
  const apiKey = event.headers['X-API-Key'] || event.headers['x-api-key'];
  if (supportedModes.includes('api_key') && apiKey) {
    return {
      method: 'api_key',
      apiKey: apiKey,
      authorized: true
    };
  }

  // Check for Service Role (direct invocation)
  if (supportedModes.includes('service_role')) {
    if (event.requestContext?.identity?.userArn?.includes('role/') ||
        (!event.httpMethod && !event.requestContext)) {
      return {
        method: 'service_role',
        authorized: true
      };
    }
  }

  // Check if public access is allowed
  if (supportedModes.includes('public')) {
    return {
      method: 'public',
      authorized: true
    };
  }

  return {
    method: 'none',
    authorized: false,
    error: {
      statusCode: 401,
      message: \`Authentication required. Supported methods: \${supportedModes.join(', ')}\`
    }
  };
}

// Dynamic CORS headers based on auth method
function getCorsHeaders(authMethod = 'cognito') {
  const authConfig = config.auth_config?.[authMethod] || config.auth_config?.cognito || {};
  
  return {
    'Access-Control-Allow-Origin': authConfig.cors_origin || '*',
    'Access-Control-Allow-Headers': 'Content-Type,X-Amz-Date,Authorization,X-Api-Key',
    'Access-Control-Allow-Methods': 'GET,POST,PUT,DELETE,OPTIONS',
    'Content-Type': 'application/json'
  };
}

/**
 * Main Lambda handler
 */
exports.handler = async (event) => {
  console.log(\`\${config.service.name} service request:\`, {
    path: event.path,
    method: event.httpMethod,
    supportedAuth: config.service.security_modes
  });

  // Determine authentication method
  const authResult = determineAuthMethod(event);
  const headers = getCorsHeaders(authResult.method);

  console.log(\`Authentication method: \${authResult.method}\`);

  // Handle CORS preflight
  if (event.httpMethod === 'OPTIONS') {
    return { statusCode: 200, headers: getCorsHeaders('public'), body: '' };
  }

  // Check authorization
  if (!authResult.authorized) {
    return {
      statusCode: authResult.error.statusCode,
      headers,
      body: JSON.stringify({
        error: authResult.error.message,
        service: config.service.name
      })
    };
  }

  try {
    // Route to appropriate handler based on path
    const path = event.path?.replace(config.api.base_path, '').replace(/^\\/+/, '') || '';
    
    return await routeRequest(path, event, authResult);
  } catch (error) {
    console.error('Service error:', error);
    return {
      statusCode: 500,
      headers,
      body: JSON.stringify({
        error: 'Internal server error',
        message: error.message,
        service: config.service.name
      })
    };
  }
};

async function routeRequest(path, event, authResult) {
  const headers = getCorsHeaders(authResult.method);
  
  // Extract ID from path if present
  const id = path || null;
  
  switch (true) {
    case path === '':
      return await getItems(event, headers);
    case /^[^/]+$/.test(path):
      return await getItemById(id, event, headers);
    default:
      return {
        statusCode: 404,
        headers,
        body: JSON.stringify({
          error: 'Not Found',
          message: \`Endpoint \${path} not found\`,
          service: config.service.name
        })
      };
  }
}

/**
 * Get all items
 */
async function getItems(event, headers) {
  try {
    // TODO: Implement your logic here
    const data = {
      message: \`Welcome to \${config.service.name} service\`,
      endpoints: config.api.endpoints.map(e => \`\${e.method} \${config.api.base_path}\${e.path}\`),
      timestamp: new Date().toISOString()
    };

    return {
      statusCode: 200,
      headers,
      body: JSON.stringify({
        component_type: 'service_info',
        data: data,
        metadata: {
          service: config.service.name,
          endpoint: 'root',
          generated_at: new Date().toISOString()
        }
      })
    };
  } catch (error) {
    console.error('Error in getItems:', error);
    return {
      statusCode: 500,
      headers,
      body: JSON.stringify({
        error: 'Internal server error',
        message: error.message
      })
    };
  }
}

/**
 * Get item by ID
 */
async function getItemById(id, event, headers) {
  try {
    // TODO: Implement your logic here
    const data = {
      id: id,
      message: \`Item \${id} from \${config.service.name} service\`,
      timestamp: new Date().toISOString()
    };

    return {
      statusCode: 200,
      headers,
      body: JSON.stringify({
        component_type: 'item_detail',
        data: data,
        metadata: {
          service: config.service.name,
          endpoint: 'get_by_id',
          generated_at: new Date().toISOString()
        }
      })
    };
  } catch (error) {
    console.error('Error in getItemById:', error);
    return {
      statusCode: 500,
      headers,
      body: JSON.stringify({
        error: 'Internal server error',
        message: error.message
      })
    };
  }
}`;

  fs.writeFileSync(path.join(servicePath, "index.js"), handlerCode);
}

function generatePythonHandler(servicePath, serviceName) {
  const handlerCode = `"""${serviceName} service handler"""
import json
import os
import logging
from typing import Dict, Any, Optional

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Load config
with open('config.json', 'r') as f:
    CONFIG = json.load(f)

def determine_auth_method(event: Dict[str, Any]) -> Dict[str, Any]:
    """Determine authentication method from event"""
    supported_modes = CONFIG.get('service', {}).get('security_modes', ['cognito'])
    
    # Check for Cognito authentication
    if 'cognito' in supported_modes:
        claims = event.get('requestContext', {}).get('authorizer', {}).get('claims')
        if claims:
            return {
                'method': 'cognito',
                'user': claims,
                'authorized': True
            }
    
    # Check for API Key
    if 'api_key' in supported_modes:
        headers = event.get('headers', {})
        api_key = headers.get('X-API-Key') or headers.get('x-api-key')
        if api_key:
            return {
                'method': 'api_key',
                'apiKey': api_key,
                'authorized': True
            }
    
    # Check for Service Role (direct invocation)
    if 'service_role' in supported_modes:
        user_arn = event.get('requestContext', {}).get('identity', {}).get('userArn')
        if (user_arn and 'role/' in user_arn) or ('httpMethod' not in event):
            return {
                'method': 'service_role',
                'authorized': True
            }
    
    # Check if public access is allowed
    if 'public' in supported_modes:
        return {
            'method': 'public',
            'authorized': True
        }
    
    return {
        'method': 'none',
        'authorized': False,
        'error': {
            'statusCode': 401,
            'message': f"Authentication required. Supported methods: {', '.join(supported_modes)}"
        }
    }

def get_cors_headers(auth_method: str = 'cognito') -> Dict[str, str]:
    """Get CORS headers based on auth method"""
    auth_config = CONFIG.get('auth_config', {}).get(auth_method, {})
    cors_origin = auth_config.get('cors_origin', '*')
    
    return {
        'Access-Control-Allow-Origin': cors_origin,
        'Access-Control-Allow-Headers': 'Content-Type,X-Amz-Date,Authorization,X-Api-Key',
        'Access-Control-Allow-Methods': 'GET,POST,PUT,DELETE,OPTIONS',
        'Content-Type': 'application/json'
    }

def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """Main Lambda handler"""
    logger.info(f"{CONFIG['service']['name']} service request: {event.get('path')} {event.get('httpMethod')}")
    
    # Determine authentication method
    auth_result = determine_auth_method(event)
    headers = get_cors_headers(auth_result['method'])
    
    # Handle CORS preflight
    if event.get('httpMethod') == 'OPTIONS':
        return {
            'statusCode': 200,
            'headers': get_cors_headers('public'),
            'body': ''
        }
    
    # Check authorization
    if not auth_result['authorized']:
        return {
            'statusCode': auth_result['error']['statusCode'],
            'headers': headers,
            'body': json.dumps({
                'error': auth_result['error']['message'],
                'service': CONFIG['service']['name']
            })
        }
    
    try:
        # Route to appropriate handler based on path
        path = event.get('path', '').replace(CONFIG['api']['base_path'], '').lstrip('/')
        return route_request(path, event, auth_result)
        
    except Exception as error:
        logger.error(f"Service error: {str(error)}")
        return {
            'statusCode': 500,
            'headers': headers,
            'body': json.dumps({
                'error': 'Internal server error',
                'message': str(error),
                'service': CONFIG['service']['name']
            })
        }

def route_request(path: str, event: Dict[str, Any], auth_result: Dict[str, Any]) -> Dict[str, Any]:
    """Route request to appropriate handler"""
    headers = get_cors_headers(auth_result['method'])
    
    if path == '':
        return get_items(event, headers)
    elif '/' not in path:  # Single ID
        return get_item_by_id(path, event, headers)
    else:
        return {
            'statusCode': 404,
            'headers': headers,
            'body': json.dumps({
                'error': 'Not Found',
                'message': f"Endpoint {path} not found",
                'service': CONFIG['service']['name']
            })
        }

def get_items(event: Dict[str, Any], headers: Dict[str, str]) -> Dict[str, Any]:
    """Get all items"""
    try:
        # TODO: Implement your logic here
        data = {
            'message': f"Welcome to {CONFIG['service']['name']} service",
            'endpoints': [f"{e['method']} {CONFIG['api']['base_path']}{e['path']}" 
                         for e in CONFIG['api']['endpoints']],
            'timestamp': '2024-01-01T00:00:00Z'  # Use actual timestamp
        }
        
        return {
            'statusCode': 200,
            'headers': headers,
            'body': json.dumps({
                'component_type': 'service_info',
                'data': data,
                'metadata': {
                    'service': CONFIG['service']['name'],
                    'endpoint': 'root',
                    'generated_at': '2024-01-01T00:00:00Z'  # Use actual timestamp
                }
            })
        }
    except Exception as error:
        logger.error(f"Error in get_items: {str(error)}")
        return {
            'statusCode': 500,
            'headers': headers,
            'body': json.dumps({
                'error': 'Internal server error',
                'message': str(error)
            })
        }

def get_item_by_id(item_id: str, event: Dict[str, Any], headers: Dict[str, str]) -> Dict[str, Any]:
    """Get item by ID"""
    try:
        # TODO: Implement your logic here
        data = {
            'id': item_id,
            'message': f"Item {item_id} from {CONFIG['service']['name']} service",
            'timestamp': '2024-01-01T00:00:00Z'  # Use actual timestamp
        }
        
        return {
            'statusCode': 200,
            'headers': headers,
            'body': json.dumps({
                'component_type': 'item_detail',
                'data': data,
                'metadata': {
                    'service': CONFIG['service']['name'],
                    'endpoint': 'get_by_id',
                    'generated_at': '2024-01-01T00:00:00Z'  # Use actual timestamp
                }
            })
        }
    except Exception as error:
        logger.error(f"Error in get_item_by_id: {str(error)}")
        return {
            'statusCode': 500,
            'headers': headers,
            'body': json.dumps({
                'error': 'Internal server error',
                'message': str(error)
            })
        }

# For direct invocation compatibility
handler = lambda_handler`;

  fs.writeFileSync(path.join(servicePath, "index.py"), handlerCode);
}

function generatePackageFile(servicePath, serviceName, runtime) {
  if (runtime === "nodejs") {
    const packageJson = {
      name: serviceName,
      version: "1.0.0",
      description: `${serviceName} service`,
      main: "index.js",
      scripts: {
        test: 'echo "Error: no test specified" && exit 1',
      },
      dependencies: {
        mysql: "^2.18.1",
      },
      keywords: ["lambda", "aws", "truss", serviceName],
      author: "Truss Team",
      license: "ISC",
    };

    fs.writeFileSync(
      path.join(servicePath, "package.json"),
      JSON.stringify(packageJson, null, 2)
    );
  } else if (runtime === "python") {
    const requirements = `# Python dependencies for ${serviceName} service
# Add your dependencies here
# Example:
# requests==2.28.1
# boto3==1.26.0
`;

    fs.writeFileSync(path.join(servicePath, "requirements.txt"), requirements);
  }
}

function generateReadme(servicePath, serviceName, runtime, options) {
  const readme = `# ${serviceName} Service

${options.description || `${serviceName} service for the Truss Data API`}

## Configuration

This service is configured via \`config.json\`:

- **Runtime**: ${runtime}
- **Security**: ${(options.security_modes || ["cognito"]).join(", ")}
- **Database**: ${options.database ? "Required" : "Not required"}
- **VPC**: ${options.vpc ? "Required" : "Not required"}

## API Endpoints

Base path: \`/api/${serviceName}\`

- \`GET /api/${serviceName}\` - Get ${serviceName} data
- \`GET /api/${serviceName}/{id}\` - Get specific item by ID

## Development

### Setup

${
  runtime === "nodejs"
    ? `\`\`\`bash
npm install
\`\`\``
    : `\`\`\`bash
pip install -r requirements.txt
\`\`\``
}

### Local Testing

${
  runtime === "nodejs"
    ? `\`\`\`bash
node index.js
\`\`\``
    : `\`\`\`bash
python index.py
\`\`\``
}

### Generate Templates

\`\`\`bash
node scripts/generate-service-templates.js services/${serviceName}
\`\`\`

## Deployment

This service is deployed automatically via GitHub Actions when:
- Files are modified in \`services/${serviceName}/\`
- Pushed to \`main\` or \`dev\` branches

## Security

This service supports the following authentication methods:
${(options.security_modes || ["cognito"])
  .map((mode) => `- **${mode}**: ${getAuthDescription(mode)}`)
  .join("\n")}

## Implementation Notes

- Main handler: \`index.${runtime === "nodejs" ? "js" : "py"}\`
- Configuration: \`config.json\` (single source of truth)
- Templates: Auto-generated from config
- Database: ${
    options.database ? "Connected via RDS Proxy" : "No database connection"
  }
- VPC: ${options.vpc ? "Deployed in VPC" : "Public Lambda"}

## TODO

- [ ] Implement actual business logic in handler functions
- [ ] Add input validation
- [ ] Add error handling
- [ ] Add unit tests
- [ ] Add integration tests
- [ ] Update API documentation

## Architecture

\`\`\`
Client -> API Gateway -> Lambda -> ${
    options.database ? "RDS Proxy -> Database" : "Business Logic"
  }
\`\`\`
`;

  fs.writeFileSync(path.join(servicePath, "README.md"), readme);
}

function getAuthDescription(mode) {
  const descriptions = {
    public: "No authentication required",
    cognito: "Cognito User Pool JWT token required",
    api_key: "API key in X-API-Key header required",
    service_role: "Service role authentication for direct invocation",
  };
  return descriptions[mode] || "Unknown authentication method";
}

// CLI interface
function main() {
  const args = process.argv.slice(2);

  if (args.length < 2) {
    console.log(
      "Usage: node create-new-service.js <parent-folder> <service-name> [options]"
    );
    console.log("");
    console.log("Options:");
    console.log("  --runtime <nodejs|python>     Runtime (default: nodejs)");
    console.log('  --description "text"           Service description');
    console.log(
      "  --security <modes>             Comma-separated security modes: public,cognito,api_key,service_role (default: cognito)"
    );
    console.log("  --database                     Enable database connection");
    console.log("  --vpc                          Deploy in VPC");
    console.log("  --memory <number>              Memory in MB (default: 512)");
    console.log(
      "  --timeout <number>             Timeout in seconds (default: 30)"
    );
    console.log(
      "  --llm-agent                    Create automated annotation service with LangChain"
    );
    console.log("");
    console.log("Examples:");
    console.log("  node scripts/create-new-service.js services analytics");
    console.log(
      "  node scripts/create-new-service.js services user-profile --runtime python --security cognito --database"
    );
    console.log(
      "  node scripts/create-new-service.js services my-annotation --llm-agent --security cognito,api_key"
    );
    console.log(
      "  node scripts/create-new-service.js services internal-api --security api_key,service_role --vpc --database"
    );
    process.exit(1);
  }

  const parentFolder = args[0];
  const serviceName = args[1];

  // Parse options
  const options = {};
  for (let i = 2; i < args.length; i++) {
    const arg = args[i];

    if (arg === "--runtime" && i + 1 < args.length) {
      options.runtime = args[++i];
    } else if (arg === "--description" && i + 1 < args.length) {
      options.description = args[++i];
    } else if (arg === "--security" && i + 1 < args.length) {
      options.security_modes = args[++i].split(",");
    } else if (arg === "--database") {
      options.database = true;
    } else if (arg === "--vpc") {
      options.vpc = true;
    } else if (arg === "--memory" && i + 1 < args.length) {
      options.memory = parseInt(args[++i]);
    } else if (arg === "--timeout" && i + 1 < args.length) {
      options.timeout = parseInt(args[++i]);
    } else if (arg === "--llm-agent") {
      options.llm_agent = true;
    }
  }

  // Validate runtime
  if (options.runtime && !["nodejs", "python"].includes(options.runtime)) {
    console.error('❌ Invalid runtime. Must be "nodejs" or "python"');
    process.exit(1);
  }

  // Validate security modes
  if (options.security_modes) {
    const validModes = ["public", "cognito", "api_key", "service_role"];
    const invalidModes = options.security_modes.filter(
      (mode) => !validModes.includes(mode)
    );
    if (invalidModes.length > 0) {
      console.error(`❌ Invalid security modes: ${invalidModes.join(", ")}`);
      console.error(`Valid modes: ${validModes.join(", ")}`);
      process.exit(1);
    }
  }

  createNewService(parentFolder, serviceName, options);
}

// Run if called directly
if (require.main === module) {
  main();
}

module.exports = { createNewService };
