# Image Processing Infrastructure

This directory contains the image processing infrastructure for the Truss microservices platform, including S3 buckets, Lambda functions, and API endpoints for managing image uploads and processing.

## Architecture Overview

The image processing solution consists of two main components:

### 1. Image Processing Lambda (`image-processing/`)

- **Purpose**: Processes images uploaded to S3 source bucket
- **Trigger**: S3 object creation events
- **Functionality**:
  - Downloads images from source bucket
  - Processes images using Sharp (resize, format conversion)
  - Uploads processed images to destination bucket
  - Tracks processing status in DynamoDB

### 2. Image Service API (`services/image-service/`)

- **Purpose**: Provides API endpoints for image management
- **Functionality**:
  - Generate presigned URLs for uploads
  - Check processing status
  - List images and processed versions
  - Delete images and processed versions

## Infrastructure Components

### S3 Buckets

- **Source Bucket**: `truss-image-source-{stage}-{account_id}`
  - Stores original uploaded images
  - Triggers processing Lambda on upload
  - Private access only
- **Processed Bucket**: `truss-image-processed-{stage}-{account_id}`
  - Stores processed image variants
  - Public access via CloudFront
  - Contains multiple sizes (thumbnail, medium, large, webp)

### DynamoDB Table

- **Processing Status Table**: `truss-image-processing-{stage}`
  - Tracks processing status and metadata
  - TTL enabled for automatic cleanup
  - Global Secondary Index on timestamp

### CloudFront Distribution

- **Purpose**: CDN for processed images
- **Configuration**:
  - Caching disabled for real-time updates
  - Security headers enabled
  - HTTPS redirect

## API Endpoints

### Upload Management

- `POST /images/upload-url` - Generate presigned upload URL
- `GET /images/upload-url` - Generate presigned upload URL (query params)

### Processing Status

- `GET /images/status/{processingId}` - Get processing status
- `GET /images/list` - List images with filtering
- `POST /images/process/{uniqueId}` - Manually trigger processing

### Processed Images

- `GET /images/processed/{uniqueId}` - Get processed image URLs
- `DELETE /images/image/{uniqueId}` - Delete image and processed versions

### Health Check

- `GET /images/health` - Service health status

## Image Processing Workflow

1. **Upload Request**: Client requests presigned URL from image service
2. **Upload**: Client uploads image directly to S3 source bucket
3. **Processing Trigger**: S3 event triggers image processing Lambda
4. **Image Processing**: Lambda processes image into multiple sizes/formats
5. **Storage**: Processed images stored in processed bucket
6. **Status Tracking**: Processing status updated in DynamoDB
7. **Access**: Processed images accessible via CloudFront URLs

## Supported Image Formats

- **Input**: JPG, JPEG, PNG, WebP, GIF
- **Output**: WebP (optimized for web)
- **Sizes**:
  - Thumbnail: 300x300 (max)
  - Medium: 800x600 (max)
  - Large: 1920x1080 (max)
  - Original: WebP conversion

## Deployment

The image processing infrastructure is deployed via GitHub Actions workflow:

```bash
# Deploy to dev environment
gh workflow run "Deploy Image Processing Infrastructure" --ref dev

# Deploy to prod environment
gh workflow run "Deploy Image Processing Infrastructure" --ref main
```

### Manual Deployment Options

- `deploy_infrastructure`: Deploy S3 buckets and Lambda (default: true)
- `deploy_image_service`: Deploy image service API (default: true)

## Environment Variables

### Image Processing Lambda

- `SOURCE_BUCKET`: Source S3 bucket name
- `PROCESSED_BUCKET`: Processed S3 bucket name
- `PROCESSING_TABLE`: DynamoDB table name
- `STAGE`: Deployment stage (dev/prod)

### Image Service API

- `SOURCE_BUCKET`: Source S3 bucket name
- `PROCESSED_BUCKET`: Processed S3 bucket name
- `PROCESSING_TABLE`: DynamoDB table name
- `CLOUDFRONT_URL`: CloudFront distribution URL
- `STAGE`: Deployment stage (dev/prod)

## Security

- **Authentication**: Cognito JWT tokens and API keys
- **Authorization**: IAM roles with least privilege access
- **S3 Security**: Private buckets with CloudFront public access
- **CORS**: Configured for cross-origin requests

## Monitoring and Logging

- **CloudWatch Logs**: Lambda execution logs
- **DynamoDB**: Processing status and metadata
- **S3 Metrics**: Upload and processing metrics
- **CloudFront**: CDN performance metrics

## Cost Optimization

- **S3 Lifecycle**: Automatic cleanup of old versions
- **DynamoDB TTL**: Automatic record expiration
- **CloudFront**: Efficient content delivery
- **Lambda**: Pay-per-execution model

## Troubleshooting

### Common Issues

1. **Processing Not Triggered**

   - Check S3 event configuration
   - Verify Lambda permissions
   - Check CloudWatch logs

2. **Upload URL Expired**

   - Default expiration: 1 hour
   - Request new URL if needed

3. **Processing Failed**
   - Check image format support
   - Verify file size limits
   - Review Lambda logs

### Debugging Commands

```bash
# Check processing status
aws dynamodb get-item \
  --table-name truss-image-processing-dev \
  --key '{"processingId":{"S":"img_1234567890_abc123"}}'

# List S3 objects
aws s3 ls s3://truss-image-source-dev-123456789012/uploads/

# Check Lambda logs
aws logs describe-log-groups --log-group-name-prefix /aws/lambda/truss-image-processing-dev
```

## Future Enhancements

- **AI Processing**: Integration with AI services for image analysis
- **Batch Processing**: Support for bulk image processing
- **Advanced Formats**: Support for additional image formats
- **Metadata Extraction**: EXIF data processing
- **Image Optimization**: Advanced compression algorithms
- **Watermarking**: Automatic watermark application
- **Face Detection**: AI-powered face detection and blurring
