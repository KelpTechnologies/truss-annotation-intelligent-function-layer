# Automated Annotation Service

A LangChain-powered service for automated knowledge extraction and text annotation using AI agents.

## Overview

This service provides intelligent text processing capabilities through LangChain integration, enabling:

- **Knowledge Extraction**: Extract entities, relationships, and attributes from text
- **Text Annotation**: Annotate text with structured data and confidence scores
- **Domain-Specific Processing**: Support for fashion, luxury, and other specialized domains

## API Endpoints

### Base URL

`/automations/annotation`

### Endpoints

#### 1. Extract Knowledge

**POST** `/extract-knowledge`

Extract structured knowledge from text using AI.

**Request Body:**

```json
{
  "text": "The Chanel handbag features black leather with gold hardware",
  "extraction_type": "entities",
  "domain": "fashion"
}
```

**Parameters:**

- `text` (required): Text to extract knowledge from
- `extraction_type` (optional): Type of extraction (`entities`, `relationships`, `attributes`)
- `domain` (optional): Domain context (`fashion`, `luxury`, `general`)

**Response:**

```json
{
  "success": true,
  "data": {
    "extraction_type": "entities",
    "data": {
      "entities": [
        {
          "name": "Chanel",
          "type": "brand",
          "confidence": 0.95
        },
        {
          "name": "handbag",
          "type": "product",
          "confidence": 0.9
        }
      ]
    },
    "timestamp": "2024-01-01T00:00:00.000Z"
  }
}
```

#### 2. Annotate Text

**POST** `/annotate`

Annotate text with structured data and confidence scores.

**Request Body:**

```json
{
  "text": "This Louis Vuitton bag is in excellent condition",
  "annotation_schema": {
    "brand": "string",
    "product_type": "string",
    "condition": "string"
  },
  "confidence_threshold": 0.8
}
```

**Parameters:**

- `text` (required): Text to annotate
- `annotation_schema` (optional): Schema for annotation structure
- `confidence_threshold` (optional): Minimum confidence for annotations (0.0-1.0)

**Response:**

```json
{
  "success": true,
  "data": {
    "annotations": {
      "brand": "Louis Vuitton",
      "product_type": "bag",
      "condition": "excellent"
    },
    "confidence_threshold": 0.8,
    "timestamp": "2024-01-01T00:00:00.000Z"
  }
}
```

#### 3. Health Check

**GET** `/health`

Check service health and LLM connectivity.

**Response:**

```json
{
  "success": true,
  "data": [
    {
      "status": "healthy",
      "service": "automated-annotation",
      "llm_provider": "openai",
      "model": "gpt-4",
      "timestamp": "2024-01-01T00:00:00.000Z"
    }
  ]
}
```

## Configuration

The service uses the following configuration:

```json
{
  "llm": {
    "provider": "openai",
    "model": "gpt-4",
    "max_tokens": 2000,
    "temperature": 0.1,
    "api_key_secret": "openai-api-key"
  }
}
```

## Dependencies

- **LangChain**: AI framework for LLM integration
- **@langchain/openai**: OpenAI integration for LangChain
- **@langchain/core**: Core LangChain functionality
- **AWS SDK**: For secrets management

## Authentication

The service supports multiple authentication methods:

- Cognito User Pools
- API Key authentication
- Service-to-service authentication

## Error Handling

The service provides detailed error responses:

```json
{
  "success": false,
  "error": {
    "message": "Validation error: text is required",
    "type": "ValidationError",
    "service": "automated-annotation",
    "timestamp": "2024-01-01T00:00:00.000Z"
  }
}
```

## Usage Examples

### Extract Fashion Entities

```bash
curl -X POST https://api.example.com/automations/annotation/extract-knowledge \
  -H "Content-Type: application/json" \
  -H "x-api-key: your-api-key" \
  -d '{
    "text": "The Hermès Birkin bag is made of crocodile leather",
    "extraction_type": "entities",
    "domain": "fashion"
  }'
```

### Annotate Product Description

```bash
curl -X POST https://api.example.com/automations/annotation/annotate \
  -H "Content-Type: application/json" \
  -H "x-api-key: your-api-key" \
  -d '{
    "text": "Vintage Rolex Submariner in mint condition",
    "confidence_threshold": 0.9
  }'
```

## Development

To create a new automated annotation service:

```bash
node scripts/create-new-service.js services my-annotation-service --llm-agent
```

This will generate a complete service structure with LangChain integration.
