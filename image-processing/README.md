# Image Processing & Vectorization Pipeline

Automated image processing and vectorization system for the Truss Annotation platform. Handles image uploads, optimization, and AI-powered vector embedding generation.

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────────┐
│  Client Upload  │────▶│  S3 Source       │────▶│  Image Processing   │
│  (via presigned │     │  Bucket          │     │  Lambda (S3 trigger)│
│   URL)          │     │                  │     │                     │
└─────────────────┘     └──────────────────┘     └──────────┬──────────┘
                                                            │
                        ┌──────────────────┐                │
                        │  DynamoDB        │◀───────────────┤
                        │  (status/vector) │                │
                        └──────────────────┘                │
                                                            ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────────┐
│  Vectorization  │◀────│  S3 Processed    │◀────│  Sharp Image        │
│  API (GPU)      │     │  Bucket          │     │  Processing         │
│  (Google Cloud) │     │  (768x768 WebP)  │     │                     │
└─────────────────┘     └──────────────────┘     └─────────────────────┘
```

## Processing Flow

### 1. Upload Request

Client requests a presigned URL from the Image Service API:

```bash
POST /images/upload-url
{
  "filename": "product-image.jpg",
  "contentType": "image/jpeg"
}
```

Response:

```json
{
  "uploadUrl": "https://s3.eu-west-2.amazonaws.com/...",
  "processingId": "abc123-uuid",
  "key": "uploads/abc123-uuid.jpg",
  "expiresIn": 3600
}
```

### 2. S3 Upload & Lambda Trigger

- Client uploads image directly to S3 using the presigned URL
- S3 `ObjectCreated` event triggers the Image Processing Lambda
- Processing ID is extracted from the filename

### 3. Image Processing (Sharp)

The Lambda processes the image using Sharp:

| Step     | Description                                      |
| -------- | ------------------------------------------------ |
| Download | Fetch original image from source bucket          |
| Resize   | Crop to **768x768** pixels (cover fit, centered) |
| Convert  | Output as **WebP** format (quality 90, effort 6) |
| Upload   | Store processed image in destination bucket      |

**Supported Input Formats:** JPEG, PNG, WebP, AVIF, GIF

### 4. Vectorization

The processed image is sent to a GPU-powered vectorization API:

| Step    | Description                                        |
| ------- | -------------------------------------------------- |
| Normalize | Create unified RGB buffer (768x768, sRGB, no alpha) |
| Create JPEG | Generate JPEG from normalized buffer (single lossy conversion) |
| Send    | POST to vectorization API with multipart form-data |
| Receive | Embedding vector + dimension                       |
| Store   | Save vector to DynamoDB                            |

**Important:** Both WebP (for storage) and JPEG (for vectorization) are created from the same normalized RGB buffer, ensuring consistent embeddings regardless of source format.

**Vectorization API:** Google Cloud Run (GPU-enabled)

```
https://image-vectorization-api-gpu-*.us-central1.run.app/vectorize
```

**⚠️ CRITICAL: All production image processing MUST use this Lambda function.**

Do NOT use alternative processing paths (e.g., `temp.py` in annotation-data-service-layer) as they will produce inconsistent results. This Lambda ensures:
- Consistent color space normalization (sRGB)
- Identical resize/crop behavior (768x768, cover fit, centered)
- EXIF orientation handling
- Single lossy JPEG conversion for vectorization

### 5. Status Tracking

All processing states are tracked in DynamoDB:

```
uploaded → processing → completed/failed
```

## Data Schema

### DynamoDB Record

| Field                  | Type   | Description                                     |
| ---------------------- | ------ | ----------------------------------------------- |
| `processingId`         | String | Unique identifier (UUID)                        |
| `originalKey`          | String | S3 key of original upload                       |
| `status`               | String | `uploaded`, `processing`, `completed`, `failed` |
| `timestamp`            | String | ISO 8601 timestamp                              |
| `processedImage`       | Object | `{ key, url, size }`                            |
| `imageVector`          | Array  | Embedding vector (floats)                       |
| `vectorDimension`      | Number | Vector dimension (e.g., 768)                    |
| `vectorizationTimings` | Object | API performance metrics                         |
| `error`                | String | Error message (if failed)                       |
| `vectorizationError`   | Object | Vectorization error details                     |

## API Endpoints

### Image Service (`/images/*`)

| Endpoint                           | Method | Description                                  |
| ---------------------------------- | ------ | -------------------------------------------- |
| `/images/upload-url`               | POST   | Generate presigned upload URL                |
| `/images/upload-url`               | GET    | Generate presigned upload URL (query params) |
| `/images/status/{processingId}`    | GET    | Get processing status                        |
| `/images/processed/{processingId}` | GET    | Get processed image URLs                     |
| `/images/list`                     | GET    | List images with filtering                   |
| `/images/process/{uniqueId}`       | POST   | Manually trigger processing                  |
| `/images/image/{uniqueId}`         | DELETE | Delete image and processed versions          |
| `/images/health`                   | GET    | Service health check                         |

## Environment Variables

### Image Processing Lambda

| Variable                | Description                    | Example                                 |
| ----------------------- | ------------------------------ | --------------------------------------- |
| `SOURCE_BUCKET`         | S3 bucket for original uploads | `truss-annotation-image-source-prod`    |
| `PROCESSED_BUCKET`      | S3 bucket for processed images | `truss-annotation-image-processed-prod` |
| `PROCESSING_TABLE`      | DynamoDB table name            | `truss-image-processing-prod`           |
| `STAGE`                 | Deployment stage               | `prod`, `staging`, `dev`                |
| `VECTORIZATION_API_URL` | GPU vectorization API endpoint | `https://...run.app/vectorize`          |
| `VECTORIZATION_API_KEY` | API authentication key         | `sk-...`                                |

## Output Specifications

### Processed Image

| Property   | Value                        |
| ---------- | ---------------------------- |
| Dimensions | 768 x 768 pixels             |
| Format     | WebP                         |
| Quality    | 90                           |
| Fit        | Cover (center crop)          |
| Filename   | `{originalKey}_768x768.webp` |

### Vector Embedding

| Property  | Value                                   |
| --------- | --------------------------------------- |
| Dimension | Model-dependent (typically 768 or 1024) |
| Format    | Array of floats                         |
| Use Case  | Similarity search, visual search        |

## Error Handling

### Processing Errors

- AVIF format fallback if initial decode fails
- Graceful degradation if vectorization fails (image still processed)
- All errors logged with full stack traces
- DynamoDB status updated to `failed` with error details

### Vectorization Errors

Vectorization failures don't block image processing:

```javascript
// Vector result stored even on failure
{
  success: false,
  error: "Error message",
  statusCode: 500,
  details: { ... }
}
```

## Monitoring

### CloudWatch Logs

Comprehensive logging at each step:

- Record processing start/end
- S3 download/upload metrics
- Sharp processing timings
- Vectorization API request/response
- DynamoDB operations

### Timing Metrics

Each processed record includes timing breakdown:

```json
{
  "timings": {
    "download": 245,
    "process": 892,
    "upload": 156,
    "vectorization": 1234,
    "total": 2527
  }
}
```

## Deployment

Deployed via GitHub Actions:

```bash
# Deploy to staging
gh workflow run "Deploy Image Processing" --ref staging

# Deploy to production
gh workflow run "Deploy Image Processing" --ref main
```

## Local Development

```bash
# Install dependencies
npm install

# Run tests (if available)
npm test
```

## Dependencies

| Package     | Purpose                           |
| ----------- | --------------------------------- |
| `aws-sdk`   | AWS services (S3, DynamoDB)       |
| `sharp`     | Image processing                  |
| `axios`     | HTTP client for vectorization API |
| `form-data` | Multipart form handling           |

## Related Services

- **Image Service API:** `services/image-service/` - REST API for image management
- **Vectorization API:** External GPU service for embedding generation
- **S3 Buckets:** Source and processed image storage
- **DynamoDB:** Processing status and vector storage
