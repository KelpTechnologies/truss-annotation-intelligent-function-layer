Process Title: New Service Endpoint Deployment

Stakeholders:

Developer: Creates and configures the service.
Automated System (Scripts/GitHub Actions): Generates templates, updates registry, and performs deployments.
AWS Infrastructure (API Gateway, Lambda, RDS Proxy): Hosts and manages the deployed service.
Flow:

Initiation (Developer Action):

Activity: Service Creation
Tool: create-new-service.js script
Input: Service name, runtime, and optional configurations (description, security modes, database, VPC, memory, timeout).
Output: New service directory with initial config.json, index.js/index.py, package.json/requirements.txt, and README.md files.
Development & Configuration (Developer Action):

Activity: Service Customization
Tool: Text editor/IDE
Input: Generated service files.
Output: Updated config.json (endpoints, security, etc.) and implemented business logic in index.js/index.py.
Template & Registry Generation (Automated/Developer Triggered):

Activity: Template and Registry Update
Tool 1: generate-service-templates.js script
Tool 2: generate-service-registry.js script
Input: Updated config.json and service logic.
Output 1: template.yaml (CloudFormation/Serverless template) and openapi.yaml for the new service.
Output 2: Updated service-registry.json (central API Gateway configuration).
Deployment (Automated via CI/CD):

Activity: Infrastructure Deployment
Tool: GitHub Actions (deploy-services.yaml workflow).
Trigger: Push to main or dev branches, or changes in the service directory.
Actions: Deploys the service's Lambda function and integrates it with API Gateway based on generated templates and the updated service registry.
Verification (Automated/Manual):

Activity: Endpoint Testing
Tool: Monitoring tools, API testing clients.
Output: Confirmation of service accessibility and functionality.
Deployment Process README.md Content
Markdown

# Service Deployment Process

This document outlines the steps to deploy a new service endpoint within the Truss Annotation Data Service Layer. The process leverages automated scripts and GitHub Actions for efficient and consistent deployments.

## Overview

The deployment workflow involves:

1.  **Service Initialization:** Creating the basic service structure.
2.  **Customization:** Implementing service logic and configuring API endpoints and security.
3.  **Template Generation:** Auto-generating deployment templates and updating the central service registry.
4.  **Automated Deployment:** Utilizing GitHub Actions to deploy the service to AWS.

## Detailed Steps

### 1. Create the New Service

Use the `create-new-service.js` script to generate the foundational files for your service.

```bash
node scripts/create-new-service.js services/<YOUR_SERVICE_NAME> --runtime <nodejs|python> [options]
Example:

Bash

node scripts/create-new-service.js services/knowledge --runtime nodejs --security cognito,api_key --database
Key Options:

--runtime <nodejs|python>: Specify the Lambda runtime (defaults to nodejs).
--description "text": A brief description for your service.
--security <modes>: Comma-separated authentication methods (e.g., public, cognito, api_key, service_role). Defaults to cognito.
--database: Include boilerplate for RDS Proxy database connection.
--vpc: Deploy the Lambda function within the VPC.
--memory <number>: Allocate memory in MB (defaults to 512).
--timeout <number>: Set timeout in seconds (defaults to 30).
2. Customize Your Service
Navigate to your new service directory (services/<your-service-name>/).

config.json: Update this file to define your service's API endpoints, security configurations, database requirements, and other deployment settings. This file is the single source of truth for your service's configuration.
index.js (or index.py): Implement your service's core logic and API endpoint handlers here. The generated file includes placeholders for getItems and getItemById.
3. Generate Service Templates
After making changes to your config.json or adding new endpoints, you must regenerate the deployment templates and update the service registry.

Bash

node scripts/generate-service-templates.js services/<YOUR_SERVICE_NAME>
This script will:

Generate/update the openapi.yaml for your service.
Generate/update the template.yaml (AWS CloudFormation template) for your service's Lambda and API Gateway integration.
Update the central api-gateway/service-registry.json to include your new service's routing information.
4. Deploy the Service
Deployment is primarily handled by GitHub Actions. Pushing your changes to the main or dev branch will automatically trigger the deployment pipeline.

deploy-services.yaml: This workflow handles the deployment of individual services. It is configured to trigger on pushes to main and dev branches and path changes within the services/ directory.
Push your changes to your Git repository:

Bash

git add .
git commit -m "feat: Add new <YOUR_SERVICE_NAME> service"
git push origin <your-branch-name>
Once merged into dev or main, the GitHub Actions workflow will automatically deploy your service.

5. Verify Deployment
After the deployment pipeline completes, verify that your service is deployed correctly and accessible via the API Gateway.

Check the AWS API Gateway console for your new service's endpoints.
Test your API endpoints using tools like Postman, curl, or your application's front-end.
Security Considerations
The security_modes in config.json define the authentication methods supported by your service. Ensure you configure these appropriately for your service's requirements.
```
