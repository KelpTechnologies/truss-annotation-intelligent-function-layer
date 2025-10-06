# Truss Automated Annotation Service

A LangChain-powered AI service for automated knowledge extraction and text annotation, designed for intelligent processing of fashion and luxury goods data.

---

## Authentication

All endpoints require an API key:

```
x-api-key: <your-api-key>
```

**Production Base URL:**  
`https://lealvbo928.execute-api.eu-west-2.amazonaws.com/external-prod`

---

## Automated Annotation Service

The service provides AI-powered text processing capabilities using LangChain:

### Core Features

- **Knowledge Extraction**: Extract entities, relationships, and attributes from text
- **Text Annotation**: Annotate text with structured data and confidence scores
- **Domain-Specific Processing**: Support for fashion, luxury, and specialized domains
- **LangChain Integration**: Modern AI framework for reliable LLM interactions

### API Endpoints

| Endpoint                                    | Method | Description                        |
| ------------------------------------------- | ------ | ---------------------------------- |
| `/automations/annotation/extract-knowledge` | POST   | Extract knowledge from text        |
| `/automations/annotation/annotate`          | POST   | Annotate text with structured data |
| `/automations/annotation/health`            | GET    | Service health check               |

---

## Example Usage

**Extract knowledge from product description:**

```http
POST /automations/annotation/extract-knowledge
Content-Type: application/json
x-api-key: <your-api-key>

{
  "text": "The Chanel handbag features black leather with gold hardware",
  "extraction_type": "entities",
  "domain": "fashion"
}
```

**Annotate text with structured data:**

```http
POST /automations/annotation/annotate
Content-Type: application/json
x-api-key: <your-api-key>

{
  "text": "Vintage Rolex Submariner in mint condition",
  "confidence_threshold": 0.9
}
```

---

## Request & Response Structure

**Request Headers:**

```
x-api-key: <your-api-key>
Content-Type: application/json
```

**Typical Response:**

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
        }
      ]
    },
    "timestamp": "2024-01-15T10:30:00Z"
  },
  "metadata": {
    "service": "automated-annotation",
    "endpoint": "extract-knowledge",
    "timestamp": "2024-01-15T10:30:00Z",
    "component_type": "knowledge_extraction"
  }
}
```

---

## Best Practices

- Use appropriate `extraction_type` for your use case (`entities`, `relationships`, `attributes`)
- Set `confidence_threshold` to filter low-confidence annotations
- Use `domain` parameter for domain-specific processing (fashion, luxury, etc.)
- Always include your API key in the request header
- Test with the health endpoint before processing large volumes

---

## Health Monitoring

- **Check service health:**

  ```http
  GET /automations/annotation/health
  ```

  _Returns the health status of the automated annotation service, including LLM connectivity and service availability._

---

## Technology Stack

- **LangChain**: Modern AI framework for reliable LLM interactions
- **OpenAI GPT-4**: Advanced language model for text processing
- **AWS Lambda**: Serverless compute for scalable processing
- **AWS Secrets Manager**: Secure API key management

---

**For further support, contact your Truss Automated Annotation Service representative.**
