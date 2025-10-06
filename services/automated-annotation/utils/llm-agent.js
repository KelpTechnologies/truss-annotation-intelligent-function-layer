const { ChatOpenAI } = require("@langchain/openai");
const { HumanMessage, SystemMessage } = require("@langchain/core/messages");
const AWS = require("aws-sdk");

/**
 * Automated Annotation Agent class for handling AI-powered text processing using LangChain
 */
class LLMAgent {
  constructor(config) {
    this.config = config;
    this.llm = null;
    this.secretsManager = new AWS.SecretsManager({
      region: config.region || "eu-west-2",
    });
  }

  /**
   * Initialize the LangChain ChatOpenAI client with API key from AWS Secrets Manager
   */
  async initialize() {
    if (this.llm) return;

    try {
      const apiKey = await this.getSecret(this.config.api_key_secret);
      this.llm = new ChatOpenAI({
        openAIApiKey: apiKey,
        modelName: this.config.model,
        temperature: this.config.temperature,
        maxTokens: this.config.max_tokens,
      });
      console.log("✅ LangChain ChatOpenAI client initialized successfully");
    } catch (error) {
      console.error("❌ Failed to initialize LangChain client:", error);
      throw error;
    }
  }

  /**
   * Get secret from AWS Secrets Manager
   */
  async getSecret(secretName) {
    try {
      const result = await this.secretsManager
        .getSecretValue({
          SecretId: secretName,
        })
        .promise();

      const secret = JSON.parse(result.SecretString);
      return secret.api_key || secret;
    } catch (error) {
      console.error(`❌ Failed to retrieve secret ${secretName}:`, error);
      throw error;
    }
  }

  /**
   * Extract knowledge from text using LangChain
   */
  async extractKnowledge(text, options = {}) {
    await this.initialize();

    const { extractionType = "entities", domain = "general" } = options;

    const prompt = this.buildExtractionPrompt(text, extractionType, domain);

    try {
      const messages = [
        new SystemMessage(this.getSystemPrompt(extractionType, domain)),
        new HumanMessage(prompt),
      ];

      const response = await this.llm.invoke(messages);
      const result = response.content;
      return this.parseExtractionResult(result, extractionType);
    } catch (error) {
      console.error("❌ Knowledge extraction failed:", error);
      throw error;
    }
  }

  /**
   * Annotate text with structured data using LangChain
   */
  async annotateText(text, options = {}) {
    await this.initialize();

    const { schema = null, confidenceThreshold = 0.7 } = options;

    const prompt = this.buildAnnotationPrompt(text, schema);

    try {
      const messages = [
        new SystemMessage(this.getAnnotationSystemPrompt(schema)),
        new HumanMessage(prompt),
      ];

      const response = await this.llm.invoke(messages);
      const result = response.content;
      return this.parseAnnotationResult(result, confidenceThreshold);
    } catch (error) {
      console.error("❌ Text annotation failed:", error);
      throw error;
    }
  }

  /**
   * Health check for the automated annotation agent
   */
  async healthCheck() {
    try {
      await this.initialize();

      // Simple test call using LangChain
      const messages = [
        new HumanMessage(
          "Hello, this is a health check. Please respond with 'OK'."
        ),
      ];

      const response = await this.llm.invoke(messages);
      const result = response.content;

      return {
        status: "healthy",
        response: result.trim(),
        model: this.config.model,
        provider: this.config.provider,
      };
    } catch (error) {
      return {
        status: "unhealthy",
        error: error.message,
        model: this.config.model,
        provider: this.config.provider,
      };
    }
  }

  /**
   * Build extraction prompt based on type and domain
   */
  buildExtractionPrompt(text, extractionType, domain) {
    const basePrompt = `Please analyze the following text and extract ${extractionType} information.`;

    const domainContext =
      domain !== "general"
        ? `Focus on ${domain}-related entities and concepts.`
        : "";

    const typeInstructions = this.getExtractionInstructions(extractionType);

    return `${basePrompt} ${domainContext}

${typeInstructions}

Text to analyze:
"${text}"

Please provide the extracted information in JSON format.`;
  }

  /**
   * Build annotation prompt
   */
  buildAnnotationPrompt(text, schema) {
    const schemaInstructions = schema
      ? `Use the following schema for annotation: ${JSON.stringify(
          schema,
          null,
          2
        )}`
      : "Use a general annotation schema for entities, relationships, and attributes.";

    return `Please annotate the following text with structured data.

${schemaInstructions}

Text to annotate:
"${text}"

Please provide annotations in JSON format with confidence scores.`;
  }

  /**
   * Get system prompt for extraction
   */
  getSystemPrompt(extractionType, domain) {
    return `You are an expert AI assistant specialized in ${extractionType} extraction from text. 
Your task is to accurately identify and extract relevant information based on the user's request.
Always respond with valid JSON format.
Focus on ${domain}-related content when specified.`;
  }

  /**
   * Get system prompt for annotation
   */
  getAnnotationSystemPrompt(schema) {
    return `You are an expert AI assistant specialized in text annotation and structured data extraction.
Your task is to annotate text with accurate, structured information including confidence scores.
Always respond with valid JSON format.
Be precise and only include annotations you are confident about.`;
  }

  /**
   * Get extraction instructions based on type
   */
  getExtractionInstructions(extractionType) {
    switch (extractionType) {
      case "entities":
        return "Extract all named entities (people, places, organizations, products, brands, etc.) with their types and any relevant attributes.";
      case "relationships":
        return "Extract relationships between entities, including the type of relationship and the entities involved.";
      case "attributes":
        return "Extract attributes and properties of entities mentioned in the text.";
      default:
        return "Extract all relevant information in a structured format.";
    }
  }

  /**
   * Parse extraction result from LLM response
   */
  parseExtractionResult(result, extractionType) {
    try {
      // Try to parse as JSON
      const parsed = JSON.parse(result);
      return {
        extraction_type: extractionType,
        data: parsed,
        raw_response: result,
        timestamp: new Date().toISOString(),
      };
    } catch (error) {
      // If JSON parsing fails, return as text
      return {
        extraction_type: extractionType,
        data: result,
        raw_response: result,
        timestamp: new Date().toISOString(),
        parse_warning: "Response was not valid JSON",
      };
    }
  }

  /**
   * Parse annotation result from LLM response
   */
  parseAnnotationResult(result, confidenceThreshold) {
    try {
      const parsed = JSON.parse(result);

      // Filter by confidence threshold if applicable
      if (parsed.annotations && Array.isArray(parsed.annotations)) {
        parsed.annotations = parsed.annotations.filter(
          (annotation) =>
            !annotation.confidence ||
            annotation.confidence >= confidenceThreshold
        );
      }

      return {
        annotations: parsed,
        confidence_threshold: confidenceThreshold,
        raw_response: result,
        timestamp: new Date().toISOString(),
      };
    } catch (error) {
      return {
        annotations: result,
        confidence_threshold: confidenceThreshold,
        raw_response: result,
        timestamp: new Date().toISOString(),
        parse_warning: "Response was not valid JSON",
      };
    }
  }
}

module.exports = { LLMAgent };
