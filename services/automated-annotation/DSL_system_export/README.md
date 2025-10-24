# Lean Classification Orchestrator - Cloud Export Package

This package contains a minimal LLM classification system designed for cloud deployment using the DSL API.

## Files Overview

### Core Application Files
- **`lean_orchestration.py`** - Main orchestration script that runs single property classifications
- **`llm_annotation_system/`** - Core classifier system (API-only mode)
  - `base_classifier.py` - LLM annotation agent
  - `__init__.py` - Package initialization
- **`dsl_api_client.py`** - Client for interacting with the Truss Annotation Data Service API
- **`config_loader.py`** - Configuration loader abstraction layer
- **`output_parser.py`** - Result parsing and CSV generation utilities

### Environment & Dependencies
- **`requirements.txt`** - Full dependencies (includes development tools)
- **`requirements_production.txt`** - Minimal production dependencies (Lambda-ready)
- **`env_sample.txt`** - Sample environment configuration

## Setup Instructions

### Local Development Setup

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   This includes `python-dotenv` for loading environment variables from `.env` files.

2. **Configure Environment:**
   Copy `env_sample.txt` to `.env` and fill in your actual values:
   ```
   DSL_API_BASE_URL=https://your-api-endpoint
   DSL_API_KEY=your_api_key
   ```

3. **Run Classification:**
   ```bash
   python lean_orchestration.py --dataset your_data.csv --property material --root-type-id 30 --input-mode auto
   ```

### Production/Cloud Setup

1. **Install Dependencies:**
   ```bash
   pip install -r requirements_production.txt
   ```
   This excludes development-only dependencies like `python-dotenv`.

2. **Configure Environment:**
   Set environment variables directly in your cloud platform (Lambda, Docker, etc.):
   - `DSL_API_BASE_URL=https://your-api-endpoint`
   - `DSL_API_KEY=your_api_key`

3. **Deploy and Run:**
   The application will automatically detect the production environment and use system environment variables.

## Why Two Requirements Files?

This package includes two requirements files to support different deployment scenarios:

### `requirements.txt` - Full Development Dependencies
- **Purpose**: Local development and testing
- **Includes**: `python-dotenv` for loading `.env` files
- **Use when**: Developing locally, running tests, or deploying to environments that support `.env` files
- **Size**: Slightly larger due to development tools

### `requirements_production.txt` - Minimal Production Dependencies
- **Purpose**: Cloud deployment (Lambda, Docker, etc.)
- **Excludes**: `python-dotenv` (not available/needed in serverless environments)
- **Use when**: Deploying to AWS Lambda, Docker containers, or any production environment
- **Benefits**:
  - Smaller deployment package
  - Faster cold starts (Lambda)
  - No unnecessary dependencies in production
  - Complies with serverless best practices

### Automatic Environment Detection

The code automatically adapts to your environment:

```python
# In lean_orchestration.py
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("Environment variables loaded from .env file (local development)")
except ImportError:
    print("Using system environment variables (Lambda/production)")
```

**Lambda Deployment**: Use `requirements_production.txt` and set environment variables in Lambda configuration.

**Local Development**: Use `requirements.txt` and create a `.env` file.

## Command Line Options

- `--dataset`: Path to your CSV dataset file
- `--property`: Property to classify (material, colour, condition, etc.)
- `--root-type-id`: Root type ID (30 for bags, etc.)
- `--input-mode`: auto, image-only, text-only, or multimodal
- `--limit`: Limit number of items to process
- `--output`: Custom output file path
- `--config`: Path to config file (optional, uses API by default)

## API-Only Operation

This package is designed to work exclusively with the DSL API:

- All configurations are loaded from the API
- All prompt templates are loaded from the API
- All context data is loaded from the API
- No local configuration files are required

## Output

Results are saved as CSV files with the following columns:
- garment_id, property, root_type_id, primary, alternatives
- confidence, reasoning, processing_time_seconds, success
- existing_value, image_url, has_text_metadata, input_mode_used

## AWS Lambda Deployment

This package is designed for easy deployment to AWS Lambda:

### Lambda Environment Setup:

1. **Environment Variables:**
   Set these in your Lambda function configuration (not in a .env file):
   ```
   DSL_API_BASE_URL=https://your-api-endpoint
   DSL_API_KEY=your_api_key
   ```

2. **Dependencies:**
   Use `requirements_production.txt` (see "Why Two Requirements Files?" section below):
   ```bash
   pip install -r requirements_production.txt
   ```
   This excludes `python-dotenv` which is not needed in Lambda.

3. **IAM Permissions:**
   Your Lambda function needs:
   - `vertexai.*` permissions for Google Cloud VertexAI
   - `secretsmanager:GetSecretValue` (if using Secrets Manager for API keys)
   - CloudWatch permissions for logging

4. **Lambda Layers/Deployment Package:**
   - Package all files from this directory
   - Ensure all dependencies are included
   - Set handler to your Lambda handler function

5. **Code Adaptation:**
   The code automatically detects Lambda environment and handles environment variables appropriately. See comments in `lean_orchestration.py` for Lambda-specific modifications.

### Lambda Handler Example:
```python
import json
from lean_orchestration import run_single_classification, load_test_data

def lambda_handler(event, context):
    # Extract parameters from event
    dataset_url = event.get('dataset_url')
    property_name = event.get('property')
    root_type_id = event.get('root_type_id', 30)
    input_mode = event.get('input_mode', 'image-only')
    limit = event.get('limit')

    # Download and process dataset
    # ... your dataset loading logic ...

    # Run classification
    results = run_single_classification(
        df=df,
        property_name=property_name,
        root_type_id=root_type_id,
        config=specific_config,
        config_loader=config_loader,
        input_mode=input_mode
    )

    return {
        'statusCode': 200,
        'body': json.dumps(results)
    }
```

## Dependencies

### Local Development:
Use `requirements.txt` which includes `python-dotenv` for `.env` file support.

### Production Deployment:
Use `requirements_production.txt` which excludes development-only dependencies:

- **langchain + vertexai integration** - Core LLM functionality
- **pydantic** - Data validation
- **pandas** - Data processing
- **requests** - API calls
- **Pillow** - Image processing
