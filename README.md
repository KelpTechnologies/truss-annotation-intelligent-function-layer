# Truss Annotation Intelligent Function Layer

An AI-powered service layer for automated product classification and image processing, using agent architecture with vector similarity search and LLM-based extraction.

---

## Services Overview

This repository contains two main services:

| Service | Base Path | Description |
|---------|-----------|-------------|
| **Automated Annotation** | `/automations/annotation` | Agent-based classification for models, materials, colours, and other properties |
| **Image Service** | `/images` | Image upload, processing, and management |

---

## Authentication

All endpoints require authentication:

```
x-api-key: <your-api-key>
```

**Production Base URL:**  
`https://lealvbo928.execute-api.eu-west-2.amazonaws.com/external-prod`

---

## Automated Annotation Service

AI-powered classification service using agent architecture with vector similarity (Pinecone) and LLM-based extraction (OpenAI GPT-4).

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/{category}/classify/model` | POST | Vector-based model classification using image embeddings |
| `/{category}/classify/{property}` | POST | LLM-based property classification (material, colour, condition, type) |
| `/csv-config` | POST | Generate CSV column mapping configuration |
| `/health` | GET | Service health check |

### Model Classification (Vector-Based)

Classifies product models using pre-computed image vectors and Pinecone similarity search.

```http
POST /automations/annotation/bags/classify/model
Content-Type: application/json
x-api-key: <your-api-key>

{
  "image": "processing-id-from-image-service",
  "brand": "louis-vuitton",
  "category": "bags"
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "model": "Speedy 25",
    "model_id": 12345,
    "root_model": "Speedy",
    "root_model_id": 100,
    "confidence": 0.92
  }
}
```

### Property Classification (LLM-Based)

Classifies properties like material, colour, condition using the agent architecture.

```http
POST /automations/annotation/bags/classify/material
Content-Type: application/json
x-api-key: <your-api-key>

{
  "image": "processing-id-from-image-service",
  "brand": "Chanel",
  "title": "Classic Flap Bag",
  "description": "Black caviar leather with gold hardware",
  "input_mode": "auto"
}
```

**Parameters:**
- `image` - Processing ID from image service (signed URL auto-resolved)
- `image_url` - Direct image URL (alternative to `image`)
- `text_input` - Pre-formatted text string
- `text_metadata` - JSON object with brand, title, description
- `input_mode` - `auto`, `image-only`, `text-only`, or `multimodal`

**Response:**
```json
{
  "success": true,
  "data": {
    "material": "Caviar Leather",
    "material_id": 456,
    "root_material": "Leather",
    "root_material_id": 10,
    "confidence": 0.95
  }
}
```

### CSV Config Generation

Analyzes CSV columns using LLM to generate column mapping configuration.

```http
POST /automations/annotation/csv-config
Content-Type: application/json
x-api-key: <your-api-key>

{
  "CSV_uuid": "unique-csv-identifier",
  "sample_rows": [
    {"Product Name": "Speedy 25", "Brand": "LV", "Image": "http://..."},
    {"Product Name": "Boy Bag", "Brand": "Chanel", "Image": "http://..."}
  ],
  "organisation_uuid": "org-123"
}
```

---

## Image Service

Handles image uploads, processing, and retrieval with presigned URLs.

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/upload-url` | POST/GET | Generate presigned URL for image upload |
| `/status/{processingId}` | GET | Get processing status for an image |
| `/processed/{uniqueId}` | GET | Get processed image URLs |
| `/process/{uniqueId}` | POST | Manually trigger image processing |
| `/list` | GET | List images with optional filtering |
| `/image/{uniqueId}` | DELETE | Delete an image and processed versions |
| `/health` | GET | Service health check |

### Upload Image

```http
POST /images/upload-url
Content-Type: application/json
x-api-key: <your-api-key>

{
  "filename": "product-image.jpg",
  "contentType": "image/jpeg",
  "expiresIn": 3600
}
```

### Check Processing Status

```http
GET /images/status/{processingId}
x-api-key: <your-api-key>
```

### Get Processed Images

```http
GET /images/processed/{uniqueId}
x-api-key: <your-api-key>
```

---

## Development Workflow

This repository uses a **config-driven approach** where `config.json` is the single source of truth for each service.

### Local Scripts

**Only run these scripts locally:**

1. **Copy Utilities** - Sync shared utils to all services
   ```bash
   node scripts/copy-utils.js                     # All services
   node scripts/copy-utils.js <service-name>      # Specific service
   ```

2. **Generate Templates** - Generate CloudFormation and OpenAPI from config
   ```bash
   node scripts/generate-service-templates.js services/<service-name>
   ```

### Deployment

Deployment is handled automatically by GitHub Actions when you push to the repository. Do not run deployment scripts locally.

### Workflow

1. Edit `services/<service-name>/config.json`
2. Run `node scripts/copy-utils.js` (if utils changed)
3. Run `node scripts/generate-service-templates.js services/<service-name>`
4. Commit and push - CI/CD handles deployment

---

## Project Structure

```
├── services/
│   ├── automated-annotation/    # Agent architecture and LLM orchestration
│   │   ├── agent_architecture/  # Base agent, validation, Pydantic models
│   │   ├── agent_orchestration/ # Classification orchestration handlers
│   │   ├── vector-classifiers/  # Pinecone vector similarity
│   │   ├── config.json          # Service configuration
│   │   └── index.py             # Lambda handler
│   ├── image-service/           # Image upload and processing
│   │   ├── config.json
│   │   └── index.js
│   └── utils/                   # Shared utilities
├── scripts/                     # Development and deployment scripts
├── api-gateway/                 # OpenAPI specs and API Gateway configs
└── .github/workflows/           # CI/CD pipelines
```

---

## Technology Stack

- **LangChain + OpenAI GPT-4** - Agent architecture for LLM interactions
- **Pinecone** - Vector similarity search for model classification
- **AWS Lambda** - Serverless compute
- **AWS DynamoDB** - Agent configs and image processing records
- **AWS S3** - Image storage
- **BigQuery** - Taxonomy and knowledge base lookups

---

## Health Monitoring

Check service health:

```http
GET /automations/annotation/health
GET /images/health
```

---

**For further support, contact your Truss representative.**
