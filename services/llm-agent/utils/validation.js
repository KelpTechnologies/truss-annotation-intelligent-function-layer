/**
 * Validation utility functions for LLM Agent service
 */

/**
 * Validate request data against schema
 */
function validateRequest(data, schema) {
  const errors = [];
  const validatedData = {};

  for (const [field, rules] of Object.entries(schema)) {
    const value = data[field];

    // Check if required field is missing
    if (
      rules.required &&
      (value === undefined || value === null || value === "")
    ) {
      errors.push(`${field} is required`);
      continue;
    }

    // Skip validation if field is not provided and not required
    if (value === undefined || value === null) {
      if (rules.default !== undefined) {
        validatedData[field] = rules.default;
      }
      continue;
    }

    // Type validation
    if (rules.type && !validateType(value, rules.type)) {
      errors.push(`${field} must be of type ${rules.type}`);
      continue;
    }

    // String length validation
    if (
      rules.type === "string" &&
      rules.minLength &&
      value.length < rules.minLength
    ) {
      errors.push(
        `${field} must be at least ${rules.minLength} characters long`
      );
      continue;
    }

    if (
      rules.type === "string" &&
      rules.maxLength &&
      value.length > rules.maxLength
    ) {
      errors.push(
        `${field} must be no more than ${rules.maxLength} characters long`
      );
      continue;
    }

    // Number range validation
    if (rules.type === "number") {
      const numValue = Number(value);
      if (isNaN(numValue)) {
        errors.push(`${field} must be a valid number`);
        continue;
      }

      if (rules.min !== undefined && numValue < rules.min) {
        errors.push(`${field} must be at least ${rules.min}`);
        continue;
      }

      if (rules.max !== undefined && numValue > rules.max) {
        errors.push(`${field} must be no more than ${rules.max}`);
        continue;
      }

      validatedData[field] = numValue;
    } else {
      validatedData[field] = value;
    }
  }

  return {
    valid: errors.length === 0,
    errors: errors,
    data: validatedData,
  };
}

/**
 * Validate data type
 */
function validateType(value, expectedType) {
  switch (expectedType) {
    case "string":
      return typeof value === "string";
    case "number":
      return typeof value === "number" || !isNaN(Number(value));
    case "boolean":
      return (
        typeof value === "boolean" || value === "true" || value === "false"
      );
    case "object":
      return (
        typeof value === "object" && value !== null && !Array.isArray(value)
      );
    case "array":
      return Array.isArray(value);
    default:
      return true;
  }
}

/**
 * Validate text input for LLM processing
 */
function validateTextInput(text, options = {}) {
  const { maxLength = 10000, minLength = 1, allowEmpty = false } = options;

  if (!text || typeof text !== "string") {
    return {
      valid: false,
      error: "Text must be a non-empty string",
    };
  }

  if (!allowEmpty && text.trim().length === 0) {
    return {
      valid: false,
      error: "Text cannot be empty",
    };
  }

  if (text.length < minLength) {
    return {
      valid: false,
      error: `Text must be at least ${minLength} characters long`,
    };
  }

  if (text.length > maxLength) {
    return {
      valid: false,
      error: `Text must be no more than ${maxLength} characters long`,
    };
  }

  return {
    valid: true,
    text: text.trim(),
  };
}

/**
 * Validate extraction type
 */
function validateExtractionType(extractionType) {
  const validTypes = ["entities", "relationships", "attributes", "all"];
  return validTypes.includes(extractionType);
}

/**
 * Validate domain context
 */
function validateDomain(domain) {
  const validDomains = [
    "general",
    "fashion",
    "luxury",
    "technology",
    "business",
    "academic",
  ];
  return validDomains.includes(domain);
}

/**
 * Validate confidence threshold
 */
function validateConfidenceThreshold(threshold) {
  const numThreshold = Number(threshold);
  return !isNaN(numThreshold) && numThreshold >= 0 && numThreshold <= 1;
}

module.exports = {
  validateRequest,
  validateType,
  validateTextInput,
  validateExtractionType,
  validateDomain,
  validateConfidenceThreshold,
};
