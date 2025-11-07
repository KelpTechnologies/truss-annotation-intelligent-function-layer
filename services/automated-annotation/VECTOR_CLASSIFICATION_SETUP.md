# Vector Classification Setup Guide

This document outlines the required AWS permissions, environment variables, and secrets needed for the `/classify-model` endpoint.

## AWS IAM Permissions

The Lambda function requires the following permissions:

### DynamoDB Permissions

- `dynamodb:GetItem` - Read individual items by ID
- `dynamodb:Query` - Query items (for GSI queries if needed)
- `dynamodb:Scan` - Scan table (if needed for namespace queries)

**Resource ARN:**

```
arn:aws:dynamodb:eu-west-2:193757560043:table/model_visual_classifier_nodes
arn:aws:dynamodb:eu-west-2:193757560043:table/model_visual_classifier_nodes/index/*
```

### Secrets Manager Permissions

- `secretsmanager:GetSecretValue` - Retrieve Pinecone API key

**Resource ARN:**

```
arn:aws:secretsmanager:eu-west-2:193757560043:secret:PineconeAPI-*
```

## Environment Variables

The following environment variables are configured in the CloudFormation template:

### Required Variables

| Variable                | Description                                               | Default/Example                                                                 |
| ----------------------- | --------------------------------------------------------- | ------------------------------------------------------------------------------- |
| `DYNAMODB_MODEL_TABLE`  | DynamoDB table name for model metadata                    | `model_visual_classifier_nodes`                                                 |
| `PINECONE_SECRET_ARN`   | ARN of Secrets Manager secret containing Pinecone API key | `arn:aws:secretsmanager:eu-west-2:193757560043:secret:PineconeAPI-qcy3De`       |
| `VECTORIZATION_API_URL` | URL of the image vectorization API                        | `https://image-vectorization-api-gpu-94434742359.us-central1.run.app/vectorize` |

### Optional Variables

| Variable                | Description                                 | Default             |
| ----------------------- | ------------------------------------------- | ------------------- |
| `VECTORIZATION_API_KEY` | API key for vectorization API (if required) | `""` (empty string) |

## AWS Secrets Manager Setup

### 1. Pinecone API Key Secret

The Pinecone API key is stored in AWS Secrets Manager.

**Secret Name:** `PineconeAPI-qcy3De`

**Secret ARN:** `arn:aws:secretsmanager:eu-west-2:193757560043:secret:PineconeAPI-qcy3De`

**Secret Format:**

The secret should contain the Pinecone API key. It can be stored as:

- Plain text: The API key string directly
- JSON: `{"api_key": "your-pinecone-api-key-here"}`

**Note:** The secret is already created and configured. The ARN is set in the template.yaml file.

## Pinecone Configuration

The `tds` package (which provides `pinecone_utils`) should be configured to read the Pinecone API key from environment variables or AWS Secrets Manager.

**Expected Pinecone Environment Variables:**

- `PINECONE_API_KEY` - Pinecone API key (retrieved from Secrets Manager)
- `PINECONE_ENVIRONMENT` - Pinecone environment/region (if required)
- `PINECONE_INDEX_NAME` - Default index name (defaults to `mfc-classifier-bags-models`)

**Note:** The `tds.pinecone_utils` package should handle reading from Secrets Manager. If it doesn't, you may need to add code in the handler to retrieve the secret and set `PINECONE_API_KEY` environment variable.

## DynamoDB Table Structure

**Table Name:** `model_visual_classifier_nodes`

**Primary Key:**

- `id` (String) - Image identifier

**Attributes:**

- `model` (String) - Specific model name
- `root_model` (String) - Parent/root model family
- `brand` (String) - Brand name
- `namespace` (String) - Brand name (lowercase, for indexing)
- `index` (String) - Pinecone index identifier
- `root_type` (String) - Product category
- `root_type_id` (Number) - Product category ID
- `initialised_timestamp` (String) - ISO 8601 timestamp

**Global Secondary Index (GSI):**

- Index Name: `namespace-index`
- Key: `namespace`

## Verification Checklist

- [ ] DynamoDB table `model_visual_classifier_nodes` exists
- [ ] DynamoDB GSI `namespace-index` exists
- [ ] IAM role has DynamoDB permissions (GetItem, Query, Scan)
- [x] Secrets Manager secret `PineconeAPI-qcy3De` exists
- [ ] IAM role has Secrets Manager permissions for Pinecone secret
- [ ] Environment variables set in CloudFormation template
- [x] `tds` package available via Lambda layer `truss-data-service-layer-nodejs:4`
- [ ] Vectorization API accessible from Lambda

## Testing

After deployment, test the endpoint:

```bash
curl -X POST https://your-api-gateway-url/automations/annotation/classify-model \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{
    "image_url": "https://example.com/image.jpg",
    "brand": "jacquemus",
    "k": 7
  }'
```

## Troubleshooting

### "Vector classifier pipeline not available"

- Check that `vector-classifiers/model_classifier_pipeline.py` exists
- Verify Python imports are working
- Check CloudWatch logs for import errors

### "No matches found in Pinecone"

- Verify Pinecone API key is correct
- Check Pinecone index name matches (`mfc-classifier-bags-models`)
- Verify namespace (brand name) exists in Pinecone
- Check that vectors are indexed in Pinecone

### "DynamoDB access denied"

- Verify IAM role has DynamoDB permissions
- Check table name matches `DYNAMODB_MODEL_TABLE` environment variable
- Verify table exists in the correct region (eu-west-2)

### "Pinecone authentication failed"

- Verify `PINECONE_SECRET_ARN` points to correct secret
- Check secret contains valid Pinecone API key
- Verify IAM role can access Secrets Manager
- Check if `tds.pinecone_utils` needs additional configuration
