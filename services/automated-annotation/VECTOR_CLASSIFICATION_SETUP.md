# Vector Classification Setup Guide

This document outlines the required AWS permissions, environment variables, and secrets needed for the `/classify-model` endpoint.

> The model classification flow now uses the pre-computed vectors stored by the image processing pipeline. The Lambda no longer uploads the image or calls the vectorization API during classification.

> Model metadata is now fetched from BigQuery (`model_knowledge_display` table) instead of DynamoDB.

> The Python Lambda requires the `pinecone`, `google-genai`, and `google-cloud-bigquery` dependencies (available in a Lambda layer or vendored into the package).
>
> **Note:** As of 2026-02, we migrated from `langchain-google-vertexai` (~230MB) to `google-genai` (~5MB) for significant layer size reduction.

## AWS IAM Permissions

The Lambda function requires the following permissions:

### DynamoDB Permissions (for image vectors only)

- `dynamodb:GetItem` - Read individual items by ID

**Resource ARN:**

```
arn:aws:dynamodb:eu-west-2:193757560043:table/truss-image-processing-<stage>
arn:aws:dynamodb:eu-west-2:193757560043:table/truss-image-processing-<stage>/index/*
```

> Replace `<stage>` with the deployment stage (e.g., `dev`, `prod`).

### Secrets Manager Permissions

- `secretsmanager:GetSecretValue` - Retrieve Pinecone API key and BigQuery credentials

**Resource ARN:**

```
arn:aws:secretsmanager:eu-west-2:193757560043:secret:truss-platform-secrets-*
arn:aws:secretsmanager:eu-west-2:193757560043:secret:PineconeAPI-*
```

## Environment Variables

The following environment variables are configured in the CloudFormation template:

### Required Variables

| Variable                 | Description                                               | Default/Example                                                           |
| ------------------------ | --------------------------------------------------------- | ------------------------------------------------------------------------- |
| `IMAGE_PROCESSING_TABLE` | DynamoDB table containing processed images + vectors      | `truss-image-processing-<stage>`                                          |
| `TRUSS_SECRETS_ARN`      | ARN of Secrets Manager secret containing platform secrets | `arn:aws:secretsmanager:eu-west-2:193757560043:secret:truss-platform-secrets` |

### Optional Variables

| Variable                | Description                                                                                | Default                                                                         |
| ----------------------- | ------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------- |
| `PINECONE_INDEX_NAME`   | Pinecone index name for model classification                                               | `mfc-classifier-bags-models-userdata`                                  |
| `VECTORIZATION_API_URL` | Legacy fallback for direct vectorization API calls (not used when vectors are precomputed) | `https://image-vectorization-api-gpu-94434742359.us-central1.run.app/vectorize` |
| `VECTORIZATION_API_KEY` | API key for vectorization API (if required for legacy usage)                               | `""` (empty string)                                                             |

## AWS Secrets Manager Setup

### 1. Platform Secrets (truss-platform-secrets)

The centralized platform secrets contain both Pinecone and BigQuery credentials.

**Secret ARN:** Set via `TRUSS_SECRETS_ARN` environment variable

**Secret Format:**

```json
{
  "pinecone": {
    "api_key": "your-pinecone-api-key"
  },
  "bigquery": {
    "project_id": "truss-data-science",
    "private_key_id": "...",
    "private_key": "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n",
    "client_email": "...",
    "client_id": "..."
  }
}
```

## Pinecone Configuration

**Expected Pinecone Environment Variables:**

- `PINECONE_API_KEY` - Pinecone API key (retrieved from Secrets Manager at runtime)
- `PINECONE_ENVIRONMENT` - Pinecone environment/region (if required)
- `PINECONE_INDEX_NAME` - Default index name (defaults to `mfc-classifier-bags-models-userdata`)

The pipeline automatically loads the Pinecone API key from the centralized secrets if not already set in the environment.

## BigQuery Configuration

Model metadata is fetched from BigQuery using the following tables:

### Classifier Table (per brand)

**Table Pattern:** `truss-data-science.model_classification.{brand}`

Contains the mapping between `listing_uuid` (Pinecone vector ID) and `model_id`.

### Model Knowledge Display Table

**Table:** `truss-data-science.api.model_knowledge_display`

Contains the model metadata:
- `id` - Model ID
- `model` - Model name
- `root_model` - Root model name
- `root_model_id` - Root model ID

### Query Used

```sql
SELECT 
    classifier_table.listing_uuid,
    mkd.id as model_id,
    mkd.model,
    mkd.root_model,
    mkd.root_model_id
FROM `truss-data-science.model_classification.{brand}` classifier_table
JOIN `truss-data-science.api.model_knowledge_display` mkd 
    ON mkd.id = classifier_table.model_id
WHERE classifier_table.listing_uuid IN ({uuid_list})
```

## Image Processing Table

Pre-computed image vectors are stored in the image processing DynamoDB table.

**Table Name Pattern:** `truss-image-processing-<stage>`

**Primary Key:**

- `processingId` (String)

**Relevant Attributes:**

- `imageVector` (List[Number]) – Pre-computed embedding used for classification
- `vectorDimension` (Number) – Dimension of the stored vector
- `processedImage` (Map) – Metadata about the processed image (S3 location, size, etc.)
- `vectorizationTimings` (Map) – Timing information recorded during vectorization (optional)
- `timestamp` (String) – When the record was last updated
- `stage` (String) – Deployment stage associated with the record

> The classification pipeline reads the vector directly from this table. No re-vectorization is performed at request time.

## Response Format

The classification endpoint returns the following fields:

```json
{
  "predicted_model_id": 123,
  "predicted_model": "Le Chiquito",
  "predicted_model_confidence": 85.7,
  "predicted_root_model": "Chiquito",
  "predicted_root_model_id": 45,
  "predicted_root_model_confidence": 71.4,
  "confidence": 71.4,
  "method": "threshold_voting",
  "message": "Model 'Le Chiquito' (id=123) has 6/7 votes"
}
```

## Verification Checklist

- [ ] DynamoDB table `truss-image-processing-<stage>` exists
- [ ] IAM role has DynamoDB permissions for image processing table
- [x] Secrets Manager secret with platform secrets exists
- [ ] IAM role has Secrets Manager permissions
- [ ] BigQuery service account has access to `model_classification` and `api.model_knowledge_display` tables
- [ ] `google-cloud-bigquery` package available via Lambda layer
- [ ] `pinecone` package available via Lambda layer
- [ ] `google-genai` package available via Lambda layer (replaced langchain-google-vertexai)
- [ ] Pinecone index `mfc-classifier-bags-models-userdata` accessible from Lambda

## Testing

After deployment, test the endpoint:

```bash
curl -X POST https://your-api-gateway-url/automations/annotation/bags/classify/model \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{
    "image_id": "img_1234567890",
    "brand": "jacquemus"
  }'
```

The service automatically queries 7 nearest neighbors for majority voting.

## Troubleshooting

### "Vector classifier pipeline not available"

- Check that `vector-classifiers/model_classifier_pipeline.py` exists
- Verify Python imports are working
- Check CloudWatch logs for import errors

### "Processing record does not contain an image vector"

- Confirm the image processing Lambda completed successfully for the given `processing_id`
- Verify the DynamoDB item contains the `imageVector` attribute
- Check `vectorizationError` in the processing record for failure details
- Re-run image processing if the vector is missing

### "No matches found in Pinecone"

- Verify Pinecone API key is correct
- Check Pinecone index name matches (`mfc-classifier-bags-models-userdata`)
- Verify namespace (brand name) exists in Pinecone
- Check that vectors are indexed in Pinecone

### "BigQuery model lookup failed"

- Verify BigQuery credentials are correctly configured in platform secrets
- Check that the classifier table exists for the brand (`model_classification.{brand}`)
- Verify the service account has permissions to query both tables
- Check CloudWatch logs for specific BigQuery error messages

### "Pinecone authentication failed"

- Verify `TRUSS_SECRETS_ARN` points to correct secret
- Check secret contains valid Pinecone API key under `pinecone.api_key`
- Verify IAM role can access Secrets Manager
