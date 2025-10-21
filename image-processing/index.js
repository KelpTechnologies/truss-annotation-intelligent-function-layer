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
    // Extract processing ID from the S3 key (uploads/{processingId}.{ext})
    const keyParts = key.split("/");
    const filename = keyParts[keyParts.length - 1];
    const processingId = filename.split(".")[0]; // Remove file extension

    console.log(`Using processing ID from key: ${processingId}`);

    // Update processing status
    await updateProcessingStatus(processingId, key, "processing");

    // Download image from source bucket
    const imageData = await downloadImage(bucket, key);

    // Process the image
    const { processedImage, metadata, baseName } = await processImage(
      imageData,
      key
    );

    // Upload processed image to destination bucket
    const uploadResult = await uploadProcessedImage(processedImage, key);

    // Update processing status to completed
    await updateProcessingStatus(processingId, key, "completed", uploadResult);

    return {
      processingId,
      originalKey: key,
      status: "completed",
      processedImage: uploadResult,
    };
  } catch (error) {
    console.error(`Error processing ${key}:`, error);

    // Extract processing ID from the S3 key for error case too
    const keyParts = key.split("/");
    const filename = keyParts[keyParts.length - 1];
    const processingId = filename.split(".")[0]; // Remove file extension

    // Update processing status to failed
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
  try {
    // Get image metadata
    const metadata = await sharp(imageBuffer).metadata();
    console.log("Image metadata:", metadata);

    // Generate single optimized 768x768 WebP image for Gemini
    const baseName = originalKey.replace(/\.[^/.]+$/, "");

    const processedImage = await sharp(imageBuffer)
      .resize(768, 768, {
        fit: "cover", // Crop to fill exact dimensions
        position: "center", // Center the crop
      })
      .webp({
        quality: 90, // High quality for Gemini
        effort: 6, // Maximum compression effort
      })
      .toBuffer();

    return {
      processedImage,
      metadata,
      baseName,
    };
  } catch (error) {
    console.error("Error processing image:", error);
    throw new Error(`Image processing failed: ${error.message}`);
  }
}

/**
 * Upload processed images to S3
 */
async function uploadProcessedImage(processedImage, originalKey) {
  const baseName = originalKey.replace(/\.[^/.]+$/, "");
  const key = `${baseName}_768x768.webp`;

  try {
    const params = {
      Bucket: PROCESSED_BUCKET,
      Key: key,
      Body: processedImage,
      ContentType: "image/webp",
      Metadata: {
        "original-key": originalKey,
        "processed-size": "768x768",
        "processed-at": new Date().toISOString(),
      },
    };

    const result = await s3.upload(params).promise();

    return {
      key,
      url: result.Location,
      size: "768x768",
    };
  } catch (error) {
    console.error("Error uploading processed image:", error);
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
  processedImage = null,
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

  if (processedImage) {
    params.Item.processedImage = processedImage;
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
