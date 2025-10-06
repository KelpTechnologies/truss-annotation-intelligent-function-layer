// Automated Annotation service handler
const config = require("./config.json");
const { LLMAgent } = require("./utils/llm-agent");
const {
  createSuccessResponse,
  createErrorResponse,
  createCorsResponse,
} = require("./utils/response");
const { validateRequest } = require("./utils/validation");

console.log("🤖 Loading Automated Annotation service...");
console.log("📁 Config loaded:", JSON.stringify(config, null, 2));

// Initialize Automated Annotation Agent
const annotationAgent = new LLMAgent(config.llm);

/**
 * Main Lambda handler for Automated Annotation service
 * @param {object} event - Lambda event
 * @param {context} context - Lambda context
 * @returns {object} Lambda response
 */
exports.handler = async (event, context) => {
  console.log("🚀 Automated Annotation service handler called");
  console.log("📅 Timestamp:", new Date().toISOString());
  console.log("🔍 Event:", JSON.stringify(event, null, 2));
  try {
    // Handle CORS preflight
    if (event.httpMethod === "OPTIONS") {
      return createCorsResponse();
    }

    // Extract path and body
    const path = event.path?.split("/").pop() || "";
    const body = event.body ? JSON.parse(event.body) : {};

    console.log("🔍 Path:", path);
    console.log("🔍 Body:", JSON.stringify(body, null, 2));

    // Route to appropriate handler
    let result;
    switch (path) {
      case "extract-knowledge":
        result = await extractKnowledge(body);
        break;
      case "annotate":
        result = await annotateText(body);
        break;
      case "health":
        result = await getHealth();
        break;
      default:
        return createErrorResponse(
          new Error("Endpoint not found"),
          config,
          404
        );
    }

    return result;
  } catch (error) {
    console.error("❌ Service error:", error);
    return createErrorResponse(error, config);
  }
};

/**
 * Extract knowledge from text using LLM agent
 */
async function extractKnowledge(body) {
  try {
    // Validate request
    const validation = validateRequest(body, {
      text: { type: "string", required: true },
      extraction_type: { type: "string", required: false, default: "entities" },
      domain: { type: "string", required: false, default: "general" },
    });

    if (!validation.valid) {
      throw new Error(`Validation error: ${validation.errors.join(", ")}`);
    }

    const { text, extraction_type, domain } = validation.data;

    // Extract knowledge using automated annotation agent
    const knowledge = await annotationAgent.extractKnowledge(text, {
      extractionType: extraction_type,
      domain: domain,
    });

    return createSuccessResponse(
      knowledge,
      config,
      "extract-knowledge",
      {},
      null,
      "knowledge_extraction"
    );
  } catch (error) {
    console.error("❌ Knowledge extraction error:", error);
    throw error;
  }
}

/**
 * Annotate text with structured data using LLM agent
 */
async function annotateText(body) {
  try {
    // Validate request
    const validation = validateRequest(body, {
      text: { type: "string", required: true },
      annotation_schema: { type: "object", required: false },
      confidence_threshold: { type: "number", required: false, default: 0.7 },
    });

    if (!validation.valid) {
      throw new Error(`Validation error: ${validation.errors.join(", ")}`);
    }

    const { text, annotation_schema, confidence_threshold } = validation.data;

    // Annotate text using automated annotation agent
    const annotations = await annotationAgent.annotateText(text, {
      schema: annotation_schema,
      confidenceThreshold: confidence_threshold,
    });

    return createSuccessResponse(
      annotations,
      config,
      "annotate",
      {},
      null,
      "text_annotation"
    );
  } catch (error) {
    console.error("❌ Text annotation error:", error);
    throw error;
  }
}

/**
 * Health check endpoint
 */
async function getHealth() {
  try {
    // Test automated annotation agent connection
    const testResult = await annotationAgent.healthCheck();

    const healthData = [
      {
        status: "healthy",
        service: "automated-annotation",
        llm_provider: config.llm.provider,
        model: config.llm.model,
        timestamp: new Date().toISOString(),
        test_result: testResult,
      },
    ];

    return createSuccessResponse(
      healthData,
      config,
      "health",
      {},
      null,
      "health_check"
    );
  } catch (error) {
    const healthData = [
      {
        status: "unhealthy",
        service: "automated-annotation",
        llm_provider: config.llm.provider,
        model: config.llm.model,
        error: error.message,
        timestamp: new Date().toISOString(),
      },
    ];

    return createSuccessResponse(
      healthData,
      config,
      "health",
      {},
      null,
      "health_check"
    );
  }
}
