const AWS = require("aws-sdk");
const sharp = require("sharp");

const s3 = new AWS.S3();
const dynamodb = new AWS.DynamoDB.DocumentClient();

// Configuration
const SOURCE_BUCKET = process.env.SOURCE_BUCKET;
const PROCESSED_BUCKET = process.env.PROCESSED_BUCKET;
const PROCESSING_TABLE = process.env.PROCESSING_TABLE;
const STAGE = process.env.STAGE;

/**
 * Main Lambda handler for image processing
 * Triggered by S3 upload events to the source bucket
 */
exports.handler = async (event) => {
  console.log(
    "Image processing Lambda triggered:",
    JSON.stringify(event, null, 2)
  );

  try {
    // Process each S3 record
    const results = await Promise.all(
      event.Records.map((record) => processImageRecord(record))
    );

    console.log("Processing completed:", results);
    return {
      statusCode: 200,
      body: JSON.stringify({
        message: "Image processing completed",
        results: results,
      }),
    };
  } catch (error) {
    console.error("Error processing images:", error);
    throw error;
  }
};

/**
 * Process a single S3 record
 */
async function processImageRecord(record) {
  const bucket = record.s3.bucket.name;
  const key = decodeURIComponent(record.s3.object.key.replace(/\+/g, " "));

  console.log(`Processing image: ${bucket}/${key}`);

  try {
    // Generate processing ID
    const processingId = generateProcessingId();

    // Update processing status
    await updateProcessingStatus(processingId, key, "processing");

    // Download image from source bucket
    const imageData = await downloadImage(bucket, key);

    // Process the image
    const processedImages = await processImage(imageData, key);

    // Upload processed images to destination bucket
    const uploadResults = await uploadProcessedImages(processedImages, key);

    // Update processing status to completed
    await updateProcessingStatus(processingId, key, "completed", uploadResults);

    return {
      processingId,
      originalKey: key,
      status: "completed",
      processedImages: uploadResults,
    };
  } catch (error) {
    console.error(`Error processing ${key}:`, error);

    // Update processing status to failed
    const processingId = generateProcessingId();
    await updateProcessingStatus(
      processingId,
      key,
      "failed",
      null,
      error.message
    );

    throw error;
  }
}

/**
 * Download image from S3
 */
async function downloadImage(bucket, key) {
  const params = {
    Bucket: bucket,
    Key: key,
  };

  const result = await s3.getObject(params).promise();
  return result.Body;
}

/**
 * Process image using Sharp
 */
async function processImage(imageBuffer, originalKey) {
  const processedImages = {};

  try {
    // Get image metadata
    const metadata = await sharp(imageBuffer).metadata();
    console.log("Image metadata:", metadata);

    // Generate different sizes and formats
    const baseName = originalKey.replace(/\.[^/.]+$/, "");

    // Original size, WebP format
    processedImages.webp = await sharp(imageBuffer)
      .webp({ quality: 85 })
      .toBuffer();

    // Thumbnail (300x300)
    processedImages.thumbnail = await sharp(imageBuffer)
      .resize(300, 300, {
        fit: "inside",
        withoutEnlargement: true,
      })
      .webp({ quality: 80 })
      .toBuffer();

    // Medium size (800x600)
    processedImages.medium = await sharp(imageBuffer)
      .resize(800, 600, {
        fit: "inside",
        withoutEnlargement: true,
      })
      .webp({ quality: 85 })
      .toBuffer();

    // Large size (1920x1080)
    processedImages.large = await sharp(imageBuffer)
      .resize(1920, 1080, {
        fit: "inside",
        withoutEnlargement: true,
      })
      .webp({ quality: 90 })
      .toBuffer();

    return processedImages;
  } catch (error) {
    console.error("Error processing image:", error);
    throw new Error(`Image processing failed: ${error.message}`);
  }
}

/**
 * Upload processed images to S3
 */
async function uploadProcessedImages(processedImages, originalKey) {
  const baseName = originalKey.replace(/\.[^/.]+$/, "");
  const uploadResults = {};

  try {
    const uploadPromises = Object.entries(processedImages).map(
      async ([size, buffer]) => {
        const key = `${baseName}_${size}.webp`;

        const params = {
          Bucket: PROCESSED_BUCKET,
          Key: key,
          Body: buffer,
          ContentType: "image/webp",
          Metadata: {
            "original-key": originalKey,
            "processed-size": size,
            "processed-at": new Date().toISOString(),
          },
        };

        const result = await s3.upload(params).promise();
        return { size, key, url: result.Location };
      }
    );

    const results = await Promise.all(uploadPromises);

    // Store results in uploadResults object
    results.forEach(({ size, key, url }) => {
      uploadResults[size] = { key, url };
    });

    return uploadResults;
  } catch (error) {
    console.error("Error uploading processed images:", error);
    throw new Error(`Upload failed: ${error.message}`);
  }
}

/**
 * Update processing status in DynamoDB
 */
async function updateProcessingStatus(
  processingId,
  originalKey,
  status,
  processedImages = null,
  error = null
) {
  const params = {
    TableName: PROCESSING_TABLE,
    Item: {
      processingId,
      originalKey,
      status,
      timestamp: new Date().toISOString(),
      stage: STAGE,
    },
  };

  if (processedImages) {
    params.Item.processedImages = processedImages;
  }

  if (error) {
    params.Item.error = error;
  }

  try {
    await dynamodb.put(params).promise();
    console.log(`Updated processing status: ${processingId} - ${status}`);
  } catch (dbError) {
    console.error("Error updating processing status:", dbError);
    // Don't throw here as it's not critical for image processing
  }
}

/**
 * Generate a unique processing ID
 */
function generateProcessingId() {
  return uuidv4();
}
