const AWS = require("aws-sdk");
const sharp = require("sharp");
const axios = require("axios");
const FormData = require("form-data");
const crypto = require("crypto");

const s3 = new AWS.S3();
const dynamodb = new AWS.DynamoDB.DocumentClient();

// Configuration
const SOURCE_BUCKET = process.env.SOURCE_BUCKET;
const PROCESSED_BUCKET = process.env.PROCESSED_BUCKET;
const PROCESSING_TABLE = process.env.PROCESSING_TABLE;
const STAGE = process.env.STAGE;
const VECTORIZATION_API_URL =
  process.env.VECTORIZATION_API_URL ||
  "https://image-vectorization-api-gpu-94434742359.us-central1.run.app/vectorize";
const VECTORIZATION_API_KEY = process.env.VECTORIZATION_API_KEY || null;

// Log configuration on cold start (only log non-sensitive values)
console.log("Image Processing Lambda Configuration:", {
  STAGE,
  SOURCE_BUCKET,
  PROCESSED_BUCKET,
  PROCESSING_TABLE,
  VECTORIZATION_API_URL,
  VECTORIZATION_API_KEY_PRESENT: !!VECTORIZATION_API_KEY,
  NODE_ENV: process.env.NODE_ENV || "production",
  AWS_REGION: process.env.AWS_REGION || "not-set",
});

// Validate required configuration
if (!SOURCE_BUCKET) {
  console.error("ERROR: SOURCE_BUCKET environment variable is not set");
}
if (!PROCESSED_BUCKET) {
  console.error("ERROR: PROCESSED_BUCKET environment variable is not set");
}
if (!PROCESSING_TABLE) {
  console.error("ERROR: PROCESSING_TABLE environment variable is not set");
}

/**
 * Main Lambda handler for image processing
 * Triggered by S3 upload events to the source bucket
 */
exports.handler = async (event) => {
  const startTime = Date.now();
  console.log("=".repeat(80));
  console.log("Image processing Lambda triggered");
  console.log("Event summary:", {
    recordCount: event.Records?.length || 0,
    eventSource: event.Records?.[0]?.eventSource || "unknown",
    eventTime: event.Records?.[0]?.eventTime || "unknown",
  });
  console.log("Full event:", JSON.stringify(event, null, 2));
  console.log("=".repeat(80));

  // Validate event structure
  if (
    !event.Records ||
    !Array.isArray(event.Records) ||
    event.Records.length === 0
  ) {
    console.warn(
      "Invalid event: No Records found. Event:",
      JSON.stringify(event, null, 2)
    );
    return {
      statusCode: 400,
      body: JSON.stringify({
        message: "Invalid event: No S3 records found",
        event: event,
      }),
    };
  }

  // Filter and validate S3 records
  const s3Records = event.Records.filter((record) => {
    const isValid = record.eventSource === "aws:s3" && record.s3;
    if (!isValid) {
      console.warn("Skipping non-S3 record:", {
        eventSource: record.eventSource,
        eventName: record.eventName,
      });
    }
    return isValid;
  });

  if (s3Records.length === 0) {
    console.warn("No valid S3 records found in event");
    return {
      statusCode: 400,
      body: JSON.stringify({
        message: "No valid S3 records found",
        totalRecords: event.Records.length,
      }),
    };
  }

  try {
    // Process each S3 record
    const results = await Promise.all(
      s3Records.map((record, index) => {
        console.log(`Processing record ${index + 1}/${s3Records.length}`);
        return processImageRecord(record);
      })
    );

    const processingTime = Date.now() - startTime;
    console.log("=".repeat(80));
    console.log("Processing completed successfully");
    console.log("Summary:", {
      totalRecords: s3Records.length,
      successfulRecords: results.filter((r) => r.status === "completed").length,
      failedRecords: results.filter((r) => r.status !== "completed").length,
      totalProcessingTimeMs: processingTime,
      averageTimePerRecordMs: Math.round(processingTime / s3Records.length),
    });
    console.log("Results:", JSON.stringify(results, null, 2));
    console.log("=".repeat(80));

    return {
      statusCode: 200,
      body: JSON.stringify({
        message: "Image processing completed",
        results: results,
        summary: {
          totalRecords: s3Records.length,
          processingTimeMs: processingTime,
        },
      }),
    };
  } catch (error) {
    const processingTime = Date.now() - startTime;
    console.error("=".repeat(80));
    console.error("FATAL ERROR processing images");
    console.error("Error details:", {
      message: error.message,
      stack: error.stack,
      name: error.name,
      processingTimeMs: processingTime,
    });
    console.error("=".repeat(80));
    throw error;
  }
};

/**
 * Process a single S3 record
 */
async function processImageRecord(record) {
  const recordStartTime = Date.now();
  const bucket = record.s3.bucket.name;
  const key = decodeURIComponent(record.s3.object.key.replace(/\+/g, " "));

  console.log("-".repeat(80));
  console.log(`Processing image record: ${bucket}/${key}`);
  console.log("S3 Record details:", {
    bucket,
    key,
    eventName: record.eventName,
    eventTime: record.eventTime,
    eventSource: record.eventSource,
    s3ObjectSize: record.s3?.object?.size,
    s3ObjectETag: record.s3?.object?.eTag,
  });

  try {
    // Extract processing ID from the S3 key (uploads/{processingId}.{ext})
    const keyParts = key.split("/");
    const filename = keyParts[keyParts.length - 1];
    const processingId = filename.split(".")[0]; // Remove file extension

    console.log(
      `Extracted processing ID: ${processingId} from filename: ${filename}`
    );

    // Update processing status
    console.log(`Updating DynamoDB status to 'processing' for ${processingId}`);
    await updateProcessingStatus(processingId, key, "processing");

    // Download image from source bucket
    console.log(`Downloading image from S3: ${bucket}/${key}`);
    const downloadStartTime = Date.now();
    const imageData = await downloadImage(bucket, key);
    const downloadTime = Date.now() - downloadStartTime;
    console.log(
      `Image downloaded successfully (${imageData.length} bytes) in ${downloadTime}ms`
    );

    // Process the image
    console.log(`Starting image processing with Sharp`);
    const processStartTime = Date.now();
    const {
      processedImage,
      vectorizationBuffer,
      metadata,
      baseName,
      normalizedWidth,
      normalizedHeight,
    } = await processImage(imageData, key);
    const processTime = Date.now() - processStartTime;
    console.log(`Image processing completed in ${processTime}ms`);

    // Upload processed image to destination bucket
    console.log(`Uploading processed image to S3: ${PROCESSED_BUCKET}`);
    const uploadStartTime = Date.now();
    const uploadResult = await uploadProcessedImage(
      processedImage,
      key,
      normalizedWidth,
      normalizedHeight
    );
    const uploadTime = Date.now() - uploadStartTime;
    console.log(
      `Processed image uploaded successfully in ${uploadTime}ms:`,
      uploadResult
    );

    // Vectorize the image using dedicated JPEG buffer (single lossy conversion)
    console.log(`Starting image vectorization using dedicated JPEG buffer`);
    const vectorStartTime = Date.now();
    let vectorResult = null;
    try {
      vectorResult = await vectorizeImage(
        vectorizationBuffer,
        key,
        processingId
      );
      const vectorTime = Date.now() - vectorStartTime;
      if (vectorResult.success) {
        console.log(
          `Image vectorization successful for ${processingId} in ${vectorTime}ms`
        );
        console.log("Vectorization details:", {
          dimension: vectorResult.dimension,
          vectorLength: vectorResult.vector?.length,
          timings: vectorResult.timings,
        });
      } else {
        console.warn(
          `Image vectorization failed for ${processingId} in ${vectorTime}ms:`,
          {
            error: vectorResult.error,
            statusCode: vectorResult.statusCode,
          }
        );
      }
    } catch (vectorError) {
      const vectorTime = Date.now() - vectorStartTime;
      console.error(
        `Image vectorization exception for ${processingId} after ${vectorTime}ms:`,
        {
          error: vectorError.message,
          stack: vectorError.stack,
        }
      );
      // Don't fail the entire processing if vectorization fails
      // Store the error but continue with processing
      vectorResult = {
        success: false,
        error: vectorError.message,
        exception: true,
      };
    }

    // Update processing status to completed
    console.log(`Updating DynamoDB status to 'completed' for ${processingId}`);
    await updateProcessingStatus(
      processingId,
      key,
      "completed",
      uploadResult,
      null,
      vectorResult
    );

    const totalTime = Date.now() - recordStartTime;
    console.log(`Record processing completed in ${totalTime}ms`);
    console.log("-".repeat(80));

    return {
      processingId,
      originalKey: key,
      status: "completed",
      processedImage: uploadResult,
      vectorization: vectorResult,
      timings: {
        download: downloadTime,
        process: processTime,
        upload: uploadTime,
        vectorization: vectorResult ? Date.now() - vectorStartTime : null,
        total: totalTime,
      },
    };
  } catch (error) {
    const totalTime = Date.now() - recordStartTime;
    console.error(`Error processing ${key} after ${totalTime}ms:`, {
      message: error.message,
      stack: error.stack,
      name: error.name,
    });

    // Extract processing ID from the S3 key for error case too
    const keyParts = key.split("/");
    const filename = keyParts[keyParts.length - 1];
    const processingId = filename.split(".")[0]; // Remove file extension

    // Update processing status to failed
    console.log(`Updating DynamoDB status to 'failed' for ${processingId}`);
    await updateProcessingStatus(
      processingId,
      key,
      "failed",
      null,
      error.message
    );

    console.error("-".repeat(80));
    throw error;
  }
}

/**
 * Download image from S3
 */
async function downloadImage(bucket, key) {
  console.log(`S3 download request: bucket=${bucket}, key=${key}`);
  const params = {
    Bucket: bucket,
    Key: key,
  };

  try {
    const result = await s3.getObject(params).promise();
    console.log(`S3 download successful:`, {
      contentLength: result.ContentLength,
      contentType: result.ContentType,
      lastModified: result.LastModified,
      eTag: result.ETag,
    });
    return result.Body;
  } catch (error) {
    console.error(`S3 download failed:`, {
      bucket,
      key,
      error: error.message,
      code: error.code,
      statusCode: error.statusCode,
    });
    throw error;
  }
}

/**
 * Process image using Sharp - simplified approach
 * Resize to 768px width (maintains aspect ratio), create JPEG for vectorization and WebP for storage
 */
async function processImage(imageBuffer, originalKey) {
  try {
    const baseName = originalKey.replace(/\.[^/.]+$/, "");
    const sharpInstance = sharp(imageBuffer, { failOn: "none" });

    // Get metadata for dimensions
    const metadata = await sharpInstance.metadata();

    // Create a pipeline: auto-rotate (handles EXIF), resize to 768px width, convert formats
    // Sharp automatically handles EXIF orientation, color space conversion, and alpha removal
    const pipeline = sharpInstance
      .rotate() // Auto-rotate based on EXIF orientation
      .resize(768, null, {
        fit: "inside",
        withoutEnlargement: true,
      });

    // Create JPEG for vectorization and WebP for storage from the same pipeline
    const [vectorizationBuffer, processedImage] = await Promise.all([
      pipeline.clone().jpeg({ quality: 90, progressive: false }).toBuffer(),
      pipeline.clone().webp({ quality: 90 }).toBuffer(),
    ]);

    // Get final dimensions
    const finalMetadata = await sharp(vectorizationBuffer).metadata();
    const normalizedWidth = finalMetadata.width || 768;
    const normalizedHeight = finalMetadata.height || 768;

    console.log("Image processed:", {
      original: `${metadata.width}x${metadata.height}`,
      processed: `${normalizedWidth}x${normalizedHeight}`,
      jpegSize: vectorizationBuffer.length,
      webpSize: processedImage.length,
    });

    return {
      processedImage,
      vectorizationBuffer,
      metadata,
      baseName,
      normalizedWidth,
      normalizedHeight,
    };
  } catch (error) {
    console.error("Error processing image with Sharp:", {
      originalKey,
      error: error.message,
      stack: error.stack,
      name: error.name,
      code: error.code,
    });

    // Check if this is an AVIF-related error
    const keyParts = originalKey.split(".");
    const extension =
      keyParts.length > 1 ? keyParts[keyParts.length - 1].toLowerCase() : null;
    const isAVIF = extension === "avif";

    if (isAVIF) {
      console.error("AVIF processing error detected:", {
        errorMessage: error.message,
        errorCode: error.code,
        errorName: error.name,
      });

      // Check if error suggests AVIF format is not supported
      const unsupportedErrorMessages = [
        "unsupported",
        "unknown",
        "invalid",
        "unsupported image format",
        "Input buffer contains unsupported image format",
      ];

      const isUnsupportedError = unsupportedErrorMessages.some((msg) =>
        error.message.toLowerCase().includes(msg)
      );

      if (isUnsupportedError) {
        console.error(
          "AVIF format may not be supported by Sharp in this Lambda environment."
        );
        console.error(
          "Possible solutions:",
          "1. Ensure Sharp is built with AVIF support (requires libavif)",
          "2. Update Sharp to latest version with AVIF support",
          "3. Pre-convert AVIF images before uploading"
        );
      }

      // Try simplified fallback approach
      try {
        console.log("Attempting AVIF fallback");
        const fallbackInstance = sharp(imageBuffer, {
          failOn: "none",
          limitInputPixels: false,
        });

        const fallbackMetadata = await fallbackInstance.metadata();
        const pipeline = fallbackInstance
          .rotate()
          .resize(768, null, { fit: "inside", withoutEnlargement: true });

        const [fallbackVectorizationBuffer, fallbackImage] = await Promise.all([
          pipeline.clone().jpeg({ quality: 90, progressive: false }).toBuffer(),
          pipeline.clone().webp({ quality: 90 }).toBuffer(),
        ]);

        const finalMetadata = await sharp(
          fallbackVectorizationBuffer
        ).metadata();

        return {
          processedImage: fallbackImage,
          vectorizationBuffer: fallbackVectorizationBuffer,
          metadata: fallbackMetadata,
          baseName: originalKey.replace(/\.[^/.]+$/, ""),
          normalizedWidth: finalMetadata.width || 768,
          normalizedHeight: finalMetadata.height || 768,
        };
      } catch (fallbackError) {
        console.error("AVIF fallback conversion also failed:", {
          error: fallbackError.message,
          stack: fallbackError.stack,
          code: fallbackError.code,
        });
        throw new Error(
          `AVIF image processing failed: ${error.message}. Fallback also failed: ${fallbackError.message}. This may indicate that Sharp in this Lambda environment does not have AVIF support compiled.`
        );
      }
    }

    throw new Error(`Image processing failed: ${error.message}`);
  }
}

/**
 * Upload processed images to S3
 */
async function uploadProcessedImage(
  processedImage,
  originalKey,
  width = 768,
  height = 768
) {
  const baseName = originalKey.replace(/\.[^/.]+$/, "");
  const key = `${baseName}_${width}x${height}.webp`;
  const sizeString = `${width}x${height}`;

  console.log(`S3 upload request: bucket=${PROCESSED_BUCKET}, key=${key}`);
  console.log("Upload parameters:", {
    bucket: PROCESSED_BUCKET,
    key,
    contentLength: processedImage.length,
    contentType: "image/webp",
    dimensions: sizeString,
  });

  try {
    const params = {
      Bucket: PROCESSED_BUCKET,
      Key: key,
      Body: processedImage,
      ContentType: "image/webp",
      Metadata: {
        "original-key": originalKey,
        "processed-size": sizeString,
        "processed-at": new Date().toISOString(),
      },
    };

    const result = await s3.upload(params).promise();

    console.log(`S3 upload successful:`, {
      key,
      location: result.Location,
      bucket: result.Bucket,
      etag: result.ETag,
      size: sizeString,
    });

    return {
      key,
      url: result.Location,
      size: sizeString,
    };
  } catch (error) {
    console.error("Error uploading processed image to S3:", {
      bucket: PROCESSED_BUCKET,
      key,
      error: error.message,
      code: error.code,
      statusCode: error.statusCode,
      stack: error.stack,
    });
    throw new Error(`Upload failed: ${error.message}`);
  }
}

/**
 * Vectorize image using the vectorization API
 * Receives a pre-processed JPEG buffer (single lossy conversion from original).
 * This ensures consistent embeddings regardless of source format.
 * - Buffer is already JPEG format (no conversion needed)
 * - Properly formats multipart form-data with content-type
 * - Handles errors gracefully
 */
async function vectorizeImage(jpegBuffer, originalKey, processingId) {
  const vectorStartTime = Date.now();
  try {
    console.log(
      `Starting vectorization for: ${originalKey} (processingId: ${processingId})`
    );
    console.log("Vectorization API configuration:", {
      url: VECTORIZATION_API_URL,
      hasApiKey: !!VECTORIZATION_API_KEY,
    });

    // Validate API configuration
    if (!VECTORIZATION_API_URL) {
      throw new Error("VECTORIZATION_API_URL environment variable is not set");
    }

    // Get image metadata for logging
    console.log("Extracting image metadata for vectorization");
    const metadata = await sharp(jpegBuffer).metadata();
    console.log("JPEG buffer metadata for vectorization:", {
      format: metadata.format,
      width: metadata.width,
      height: metadata.height,
      channels: metadata.channels,
      size: jpegBuffer.length,
      hasAlpha: metadata.hasAlpha,
    });

    // Buffer is already JPEG format - use directly
    // Use JPEG content type and filename for the vectorization API
    const contentType = "image/jpeg";
    const baseName = originalKey.replace(/\.[^/.]+$/, "");
    const filename = `${baseName.split("/").pop() || "image"}.jpg`;

    console.log("Preparing multipart form-data:", {
      filename,
      contentType,
      bufferSize: jpegBuffer.length,
    });

    // Create multipart form-data with proper 3-tuple format: (filename, buffer, content-type)
    // This is CRITICAL to avoid the 500 error from the API
    const formData = new FormData();
    formData.append("file", jpegBuffer, {
      filename: filename,
      contentType: contentType,
    });

    // Prepare request headers
    const headers = formData.getHeaders();
    if (VECTORIZATION_API_KEY) {
      headers["x-api-key"] = VECTORIZATION_API_KEY;
      console.log("API key authentication added to request headers");
    } else {
      console.warn(
        "WARNING: VECTORIZATION_API_KEY not set - request will be sent without authentication"
      );
    }

    console.log("Making request to vectorization API:", {
      url: VECTORIZATION_API_URL,
      method: "POST",
      hasAuth: !!VECTORIZATION_API_KEY,
      timeout: 30000,
      contentType: headers["content-type"],
      contentLength: jpegBuffer.length,
    });

    const requestStartTime = Date.now();
    // Make request to vectorization API
    const response = await axios.post(VECTORIZATION_API_URL, formData, {
      headers: headers,
      timeout: 30000, // 30 second timeout
      maxContentLength: Infinity,
      maxBodyLength: Infinity,
    });

    const requestTime = Date.now() - requestStartTime;
    const totalTime = Date.now() - vectorStartTime;

    console.log(`Vectorization API request completed in ${requestTime}ms`, {
      status: response.status,
      statusText: response.statusText,
      dimension: response.data?.dimension,
      vectorLength:
        response.data?.vector?.length || response.data?.embedding?.length,
      hasTimings: !!response.data?.timings,
    });

    console.log(`Vectorization completed successfully in ${totalTime}ms`, {
      processingId,
      dimension: response.data.dimension,
      vectorLength: (response.data.vector || response.data.embedding)?.length,
      timings: response.data.timings,
    });

    return {
      vector: response.data.vector || response.data.embedding,
      dimension: response.data.dimension,
      timings: response.data.timings,
      success: true,
      requestTimeMs: requestTime,
      totalTimeMs: totalTime,
    };
  } catch (error) {
    const totalTime = Date.now() - vectorStartTime;
    console.error(`Vectorization failed after ${totalTime}ms:`, {
      processingId,
      originalKey,
      errorMessage: error.message,
      errorName: error.name,
      errorCode: error.code,
      statusCode: error.response?.status,
      statusText: error.response?.statusText,
      responseData: error.response?.data,
      requestUrl: error.config?.url,
      requestMethod: error.config?.method,
      hasAuth: !!error.config?.headers?.["x-api-key"],
    });

    // Log detailed error information for debugging
    if (error.response) {
      console.error("API response error details:", {
        status: error.response.status,
        statusText: error.response.statusText,
        headers: error.response.headers,
        data: error.response.data,
      });
    } else if (error.request) {
      console.error("Request made but no response received:", {
        url: error.config?.url,
        method: error.config?.method,
        timeout: error.config?.timeout,
      });
    } else {
      console.error("Error setting up request:", error.message);
    }

    return {
      success: false,
      error: error.message,
      statusCode: error.response?.status,
      statusText: error.response?.statusText,
      details: error.response?.data,
      requestTimeMs: totalTime,
    };
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
  error = null,
  vectorResult = null
) {
  console.log(`Updating DynamoDB status for ${processingId}:`, {
    status,
    hasProcessedImage: !!processedImage,
    hasError: !!error,
    hasVectorResult: !!vectorResult,
    vectorSuccess: vectorResult?.success,
  });

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
    console.log("Added processedImage to DynamoDB item");
  }

  if (error) {
    params.Item.error = error;
    console.log("Added error to DynamoDB item:", error);
  }

  // Store vector result if available
  if (vectorResult) {
    if (vectorResult.success && vectorResult.vector) {
      params.Item.imageVector = vectorResult.vector;
      params.Item.vectorDimension = vectorResult.dimension;
      if (vectorResult.timings) {
        params.Item.vectorizationTimings = vectorResult.timings;
      }
      console.log("Added vectorization result to DynamoDB:", {
        dimension: vectorResult.dimension,
        vectorLength: vectorResult.vector?.length,
        hasTimings: !!vectorResult.timings,
      });
    } else {
      // Store error information if vectorization failed
      params.Item.vectorizationError = {
        message: vectorResult.error || "Unknown error",
        statusCode: vectorResult.statusCode,
        statusText: vectorResult.statusText,
        details: vectorResult.details,
      };
      console.log(
        "Added vectorization error to DynamoDB:",
        params.Item.vectorizationError
      );
    }
  }

  try {
    console.log("DynamoDB put request:", {
      table: PROCESSING_TABLE,
      itemKeys: Object.keys(params.Item),
      itemSize: JSON.stringify(params.Item).length,
    });

    await dynamodb.put(params).promise();

    console.log(
      `DynamoDB status updated successfully: ${processingId} - ${status}`
    );
  } catch (dbError) {
    console.error("Error updating DynamoDB processing status:", {
      processingId,
      status,
      error: dbError.message,
      code: dbError.code,
      statusCode: dbError.statusCode,
      stack: dbError.stack,
    });
    // Don't throw here as it's not critical for image processing
  }
}

/**
 * Generate a unique processing ID
 */
function generateProcessingId() {
  return uuidv4();
}
