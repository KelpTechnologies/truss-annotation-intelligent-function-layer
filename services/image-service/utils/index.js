/**
 * Main utilities index file
 * Exports all shared utilities for easy importing in services
 */

// Export all utility functions
const database = require("./database");
const response = require("./response");
const validation = require("./validation");
const queryBuilder = require("./query-builder");
const dataParser = require("./data-parser");
const constants = require("./constants");
const logger = require("./logger");
const serviceTemplate = require("./service-template");
const structuredLogger = require("./structured-logger");

module.exports = {
  // Database utilities
  ...database,

  // Response utilities
  ...response,

  // Validation utilities
  ...validation,

  // Query builder utilities
  ...queryBuilder,

  // Data parser utilities
  ...dataParser,

  // Constants
  ...constants,

  // Logger utilities
  ...logger,

  // Service template utilities
  ...serviceTemplate,

  // Structured logger for metrics (use for REQUEST/RESPONSE/ERROR lifecycle events)
  createLogger: structuredLogger.createLogger,
  StructuredLogger: structuredLogger.StructuredLogger,
};
