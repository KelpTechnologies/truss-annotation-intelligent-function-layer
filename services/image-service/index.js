const AWS = require("aws-sdk");
const { v4: uuidv4 } = require("uuid");

// Initialize AWS services
const s3 = new AWS.S3();
const dynamodb = new AWS.DynamoDB.DocumentClient();

// Configuration from environment variables
const SOURCE_BUCKET = process.env.SOURCE_BUCKET;
const PROCESSED_BUCKET = process.env.PROCESSED_BUCKET;
const PROCESSING_TABLE = process.env.PROCESSING_TABLE;
const CLOUDFRONT_URL = process.env.CLOUDFRONT_URL;
const STAGE = process.env.STAGE;

/**
 * Main Lambda handler for image service API endpoints
 */
exports.handler = async (event) => {
  console.log(
    "Image service Lambda triggered:",
    JSON.stringify(event, null, 2)
  );

  try {
    const { httpMethod, path, pathParameters, queryStringParameters, body } =
      event;
    const pathInfo = parsePath(path);

    console.log("Request details:", {
      httpMethod,
      path,
      pathInfo,
      queryStringParameters,
      body: body ? JSON.parse(body) : null
    });

    // Route requests based on HTTP method and path
    switch (httpMethod) {
      case "GET":
        return await handleGetRequest(pathInfo, queryStringParameters);
      case "POST":
        return await handlePostRequest(pathInfo, JSON.parse(body || "{}"));
      case "DELETE":
        return await handleDeleteRequest(pathInfo, pathParameters);
      default:
        return createResponse(405, { error: "Method not allowed" });
    }
  } catch (error) {
    console.error("Error in image service:", error);
    return createResponse(500, {
      error: "Internal server error",
      message: error.message,
    });
  }
};

/**
 * Parse the request path to determine the endpoint
 */
function parsePath(path) {
  const segments = path.split("/").filter((segment) => segment);

  if (segments.length >= 2 && segments[0] === "images") {
    return {
      endpoint: segments[1],
      id: segments[2] || null,
      subEndpoint: segments[3] || null,
    };
  }

  return { endpoint: "unknown" };
}

/**
 * Handle GET requests
 */
async function handleGetRequest(pathInfo, queryParams) {
  console.log("GET request handler:", { pathInfo, queryParams });
  
  switch (pathInfo.endpoint) {
    case "upload-url":
      return await generateUploadUrl(queryParams);
    case "status":
      return await getProcessingStatus(pathInfo.id);
    case "list":
      return await listImages(queryParams);
    case "processed":
      return await getProcessedImages(pathInfo.id);
    case "health":
      return createResponse(200, {
        status: "healthy",
        service: "image-service",
        stage: STAGE,
        timestamp: new Date().toISOString(),
      });
    default:
      console.log("Unknown endpoint:", pathInfo.endpoint);
      return createResponse(404, { error: "Endpoint not found" });
  }
}

/**
 * Handle POST requests
 */
async function handlePostRequest(pathInfo, body) {
  switch (pathInfo.endpoint) {
    case "upload-url":
      return await generateUploadUrl(body);
    case "process":
      return await triggerProcessing(pathInfo.id);
    default:
      return createResponse(404, { error: "Endpoint not found" });
  }
}

/**
 * Handle DELETE requests
 */
async function handleDeleteRequest(pathInfo, pathParams) {
  switch (pathInfo.endpoint) {
    case "image":
      return await deleteImage(pathInfo.id);
    default:
      return createResponse(404, { error: "Endpoint not found" });
  }
}

/**
 * Generate a presigned URL for uploading images
 */
async function generateUploadUrl(params) {
  try {
    const { filename, contentType, expiresIn = 3600 } = params;

    if (!filename) {
      return createResponse(400, { error: "Filename is required" });
    }

    // Generate unique key for the upload
    const fileExtension = filename.split(".").pop().toLowerCase();
    const uniqueId = uuidv4();
    const key = `uploads/${uniqueId}.${fileExtension}`;

    // Generate presigned URL for PUT operation
    const presignedUrl = s3.getSignedUrl("putObject", {
      Bucket: SOURCE_BUCKET,
      Key: key,
      ContentType: contentType || `image/${fileExtension}`,
      Expires: expiresIn,
      Metadata: {
        "original-filename": filename,
        "upload-timestamp": new Date().toISOString(),
        stage: STAGE,
      },
    });

    // Store upload record in DynamoDB
    await storeUploadRecord(uniqueId, key, filename, contentType);

    return createResponse(200, {
      uploadUrl: presignedUrl,
      key: key,
      uniqueId: uniqueId,
      expiresIn: expiresIn,
      processedImageUrl: `${CLOUDFRONT_URL}/${key.replace(
        "uploads/",
        "processed/"
      )}`,
    });
  } catch (error) {
    console.error("Error generating upload URL:", error);
    return createResponse(500, { error: "Failed to generate upload URL" });
  }
}

/**
 * Get processing status for an image
 */
async function getProcessingStatus(processingId) {
  try {
    if (!processingId) {
      return createResponse(400, { error: "Processing ID is required" });
    }

    const params = {
      TableName: PROCESSING_TABLE,
      Key: {
        processingId: processingId,
      },
    };

    const result = await dynamodb.get(params).promise();

    if (!result.Item) {
      return createResponse(404, { error: "Processing record not found" });
    }

    return createResponse(200, {
      processingId: result.Item.processingId,
      originalKey: result.Item.originalKey,
      status: result.Item.status,
      timestamp: result.Item.timestamp,
      processedImages: result.Item.processedImages,
      error: result.Item.error,
    });
  } catch (error) {
    console.error("Error getting processing status:", error);
    return createResponse(500, { error: "Failed to get processing status" });
  }
}

/**
 * List images with optional filtering
 */
async function listImages(queryParams) {
  try {
    const { status, limit = 50, lastKey } = queryParams;

    const params = {
      TableName: PROCESSING_TABLE,
      Limit: parseInt(limit),
      ScanIndexForward: false, // Most recent first
    };

    if (lastKey) {
      params.ExclusiveStartKey = { processingId: lastKey };
    }

    // If status filter is provided, we'll need to scan
    if (status) {
      params.FilterExpression = "#status = :status";
      params.ExpressionAttributeNames = { "#status": "status" };
      params.ExpressionAttributeValues = { ":status": status };
    }

    const result = await dynamodb.scan(params).promise();

    return createResponse(200, {
      images: result.Items,
      count: result.Count,
      lastEvaluatedKey: result.LastEvaluatedKey,
    });
  } catch (error) {
    console.error("Error listing images:", error);
    return createResponse(500, { error: "Failed to list images" });
  }
}

/**
 * Get processed images for a specific upload
 */
async function getProcessedImages(uniqueId) {
  try {
    if (!uniqueId) {
      return createResponse(400, { error: "Unique ID is required" });
    }

    // List objects in processed bucket with the unique ID prefix
    const params = {
      Bucket: PROCESSED_BUCKET,
      Prefix: `processed/${uniqueId}`,
      MaxKeys: 10,
    };

    const result = await s3.listObjectsV2(params).promise();

    const processedImages = result.Contents.map((obj) => ({
      key: obj.Key,
      url: `${CLOUDFRONT_URL}/${obj.Key}`,
      size: obj.Size,
      lastModified: obj.LastModified,
      sizeType: extractSizeType(obj.Key),
    }));

    return createResponse(200, {
      uniqueId: uniqueId,
      processedImages: processedImages,
      count: processedImages.length,
    });
  } catch (error) {
    console.error("Error getting processed images:", error);
    return createResponse(500, { error: "Failed to get processed images" });
  }
}

/**
 * Trigger manual processing of an image
 */
async function triggerProcessing(uniqueId) {
  try {
    if (!uniqueId) {
      return createResponse(400, { error: "Unique ID is required" });
    }

    // Find the original upload record
    const uploadRecord = await findUploadRecord(uniqueId);
    if (!uploadRecord) {
      return createResponse(404, { error: "Upload record not found" });
    }

    // Trigger processing by copying the object (this will trigger the S3 event)
    const copyParams = {
      Bucket: SOURCE_BUCKET,
      CopySource: `${SOURCE_BUCKET}/${uploadRecord.key}`,
      Key: uploadRecord.key,
      MetadataDirective: "REPLACE",
      Metadata: {
        ...uploadRecord.metadata,
        "manual-trigger": "true",
        "trigger-timestamp": new Date().toISOString(),
      },
    };

    await s3.copyObject(copyParams).promise();

    return createResponse(200, {
      message: "Processing triggered successfully",
      uniqueId: uniqueId,
      key: uploadRecord.key,
    });
  } catch (error) {
    console.error("Error triggering processing:", error);
    return createResponse(500, { error: "Failed to trigger processing" });
  }
}

/**
 * Delete an image and its processed versions
 */
async function deleteImage(uniqueId) {
  try {
    if (!uniqueId) {
      return createResponse(400, { error: "Unique ID is required" });
    }

    // Find the original upload record
    const uploadRecord = await findUploadRecord(uniqueId);
    if (!uploadRecord) {
      return createResponse(404, { error: "Upload record not found" });
    }

    // Delete from source bucket
    await s3
      .deleteObject({
        Bucket: SOURCE_BUCKET,
        Key: uploadRecord.key,
      })
      .promise();

    // Delete processed versions
    const processedParams = {
      Bucket: PROCESSED_BUCKET,
      Prefix: `processed/${uniqueId}`,
    };

    const processedObjects = await s3.listObjectsV2(processedParams).promise();

    if (processedObjects.Contents.length > 0) {
      const deleteParams = {
        Bucket: PROCESSED_BUCKET,
        Delete: {
          Objects: processedObjects.Contents.map((obj) => ({ Key: obj.Key })),
        },
      };

      await s3.deleteObjects(deleteParams).promise();
    }

    // Delete processing records
    await deleteProcessingRecords(uniqueId);

    return createResponse(200, {
      message: "Image deleted successfully",
      uniqueId: uniqueId,
      deletedProcessedImages: processedObjects.Contents.length,
    });
  } catch (error) {
    console.error("Error deleting image:", error);
    return createResponse(500, { error: "Failed to delete image" });
  }
}

/**
 * Store upload record in DynamoDB
 */
async function storeUploadRecord(uniqueId, key, filename, contentType) {
  const params = {
    TableName: PROCESSING_TABLE,
    Item: {
      processingId: `upload_${uniqueId}`,
      uniqueId: uniqueId,
      key: key,
      filename: filename,
      contentType: contentType,
      status: "uploaded",
      timestamp: new Date().toISOString(),
      stage: STAGE,
      ttl: Math.floor(Date.now() / 1000) + 30 * 24 * 60 * 60, // 30 days
    },
  };

  await dynamodb.put(params).promise();
}

/**
 * Find upload record by unique ID
 */
async function findUploadRecord(uniqueId) {
  const params = {
    TableName: PROCESSING_TABLE,
    FilterExpression: "uniqueId = :uniqueId",
    ExpressionAttributeValues: {
      ":uniqueId": uniqueId,
    },
  };

  const result = await dynamodb.scan(params).promise();
  return result.Items[0] || null;
}

/**
 * Delete processing records for a unique ID
 */
async function deleteProcessingRecords(uniqueId) {
  const params = {
    TableName: PROCESSING_TABLE,
    FilterExpression: "uniqueId = :uniqueId",
    ExpressionAttributeValues: {
      ":uniqueId": uniqueId,
    },
  };

  const result = await dynamodb.scan(params).promise();

  for (const item of result.Items) {
    await dynamodb
      .delete({
        TableName: PROCESSING_TABLE,
        Key: { processingId: item.processingId },
      })
      .promise();
  }
}

/**
 * Extract size type from processed image key
 */
function extractSizeType(key) {
  const match = key.match(/_(\w+)\.webp$/);
  return match ? match[1] : "unknown";
}

/**
 * Create standardized HTTP response
 */
function createResponse(statusCode, body) {
  return {
    statusCode: statusCode,
    headers: {
      "Content-Type": "application/json",
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Headers":
        "Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token",
      "Access-Control-Allow-Methods": "GET,POST,PUT,DELETE,OPTIONS",
    },
    body: JSON.stringify(body),
  };
}
