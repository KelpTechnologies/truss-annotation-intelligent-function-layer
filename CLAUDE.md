# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the **Truss Annotation Intelligent Function Layer** - a config-driven microservices platform for automated knowledge extraction, text annotation, and image processing. It uses LangChain with OpenAI GPT-4 for AI-powered classification of fashion/luxury goods data.

## Core Principle: Config-Driven Development

`config.json` is the **single source of truth** for each service. All CloudFormation templates, OpenAPI specs, and deployment configurations are auto-generated from this file.

## Common Commands

### Local Development (run these manually)

```bash
# Copy shared utilities to all services (run after updating services/utils/)
node scripts/copy-utils.js

# Copy utilities to a specific service
node scripts/copy-utils.js <service-name>

# Generate CloudFormation and OpenAPI templates from config.json
node scripts/generate-service-templates.js services/<service-name>

# Create a new service
node scripts/create-new-service.js <parent-folder> <service-name> [options]
# Options: --runtime <nodejs|python>, --security <modes>, --database, --vpc, --memory <MB>, --timeout <sec>
```

### CI/CD Only (DO NOT run locally - handled by GitHub Actions)

- `deploy-service.js`, `deploy-all-services.js` - Deployment
- `aggregate-openapi.js`, `prepare-api-deployment.js` - API Gateway preparation
- `generate-service-registry.js` - Registry generation

### Integration Tests

```bash
# Run tests locally
STAGING_API_URL=https://staging.example.com python .github/scripts/runner.py
```

## Architecture

### Services

| Service | Runtime | Purpose |
|---------|---------|---------|
| `automated-annotation` | Python 3.11 | AI-powered classification using LangChain agents. Handles model/property classification, keyword extraction, vector search via Pinecone |
| `image-service` | Node.js 18 | Image upload management with presigned URLs, S3 storage, CloudFront distribution |

### Key Directories

```
services/
├── automated-annotation/     # Main AI service
│   ├── config.json           # Service configuration (source of truth)
│   ├── index.py              # Lambda handler
│   ├── agent_architecture/   # Agent base classes, validation
│   ├── agent_orchestration/  # Orchestration handlers
│   ├── agent_utils/          # BigQuery, DSL, measurement utilities
│   ├── core/                 # API handlers, routes
│   └── vector-classifiers/   # Pinecone/BigQuery vector search
├── image-service/            # Image management
├── utils/                    # Shared utilities (copied to all Node.js services)
└── meta/routes/              # Auto-generated service registry

scripts/                      # Automation scripts (Node.js)
api-gateway/                  # Aggregated OpenAPI specs, service registry
.github/workflows/            # CI/CD pipelines
tests/                        # Integration tests (.txt specs + .py/.js implementations)
```

### External Services

- **Pinecone** - Vector database for similarity search
- **BigQuery** - Taxonomy and brand data
- **AWS**: Lambda, API Gateway, DynamoDB, S3, CloudFront, RDS Proxy, Secrets Manager, Cognito

## Workflow: Modifying a Service

1. Edit `config.json` (endpoints, auth, database settings)
2. Run `node scripts/generate-service-templates.js services/<service-name>`
3. If you updated `services/utils/`, run `node scripts/copy-utils.js`
4. Commit and push - GitHub Actions deploys automatically

## Workflow: Creating a New Service

```bash
node scripts/create-new-service.js services my-service --runtime nodejs --security api_key --database
node scripts/copy-utils.js my-service
# Edit config.json and implement logic in index.js
node scripts/generate-service-templates.js services/my-service
git add . && git commit -m "Add my-service" && git push
```

## config.json Structure

```json
{
  "service": { "name": "...", "description": "...", "version": "..." },
  "deployment": { "runtime": "nodejs18.x|python3.11", "timeout": 30, "memory": 512, "layers": [], "vpc_config": {} },
  "database": { "required": true, "connection_type": "mysql", "permissions": ["read", "write"] },
  "api": { "base_path": "/...", "cors_enabled": true, "endpoints": [{ "path": "/", "method": "GET", "parameters": {} }] },
  "access": { "internal": true, "external": true, "auth_config": { "cognito": {}, "api_key": {} } }
}
```

## Integration Testing

Tests use `.txt` files (natural language specs) paired with `.py`/`.js` implementations:

```
tests/
├── test_input_data/          # Test data
├── 001-feature.txt           # INPUT/OUTPUT/INTERNAL LOGIC sections
└── 001-feature.py            # Implementation matching the spec
```

## Authentication

All external endpoints require `x-api-key` header. Services support:
- Cognito User Pools
- API Key authentication
- Service-to-service authentication
