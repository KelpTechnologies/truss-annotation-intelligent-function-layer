/**
 * Field Dictionary — Single source of truth for field name mappings across all services.
 *
 * Maps API parameter names to database column names (MySQL and BigQuery).
 * All other mapping objects (FIELD_MAPPINGS, GROUP_BY_FIELDS, BQ_COLUMN_MAPPINGS,
 * MULTI_VALUE_FILTERS) are derived from this dictionary.
 *
 * Database columns stay as-is (British spelling, legacy names). This layer
 * translates between the canonical API names and the actual column names.
 */

const FIELD_DICTIONARY = {
  brand: {
    apiName: "brand",
    dbColumn: "brand",
    bqColumn: "brand",
    displayName: "Brand",
    pluralParam: "brands",
    singularParam: "brand",
  },
  type: {
    apiName: "type",
    dbColumn: "type",
    bqColumn: "type",
    displayName: "Type",
    pluralParam: "types",
    singularParam: "type",
  },
  model: {
    apiName: "model",
    dbColumn: "model",
    bqColumn: "model",
    displayName: "Model",
    pluralParam: "models",
    singularParam: "model",
  },
  material: {
    apiName: "material",
    dbColumn: "material",
    bqColumn: "material",
    displayName: "Material",
    pluralParam: "materials",
    singularParam: "material",
  },
  color: {
    apiName: "color",
    dbColumn: "colour",
    bqColumn: "colour",
    displayName: "Color",
    pluralParam: "colors",
    singularParam: "color",
  },
  condition: {
    apiName: "condition",
    dbColumn: "`condition`",
    bqColumn: "condition",
    displayName: "Condition",
    pluralParam: "conditions",
    singularParam: "condition",
  },
  size: {
    apiName: "size",
    dbColumn: "size",
    bqColumn: "size",
    displayName: "Size",
    pluralParam: "sizes",
    singularParam: "size",
  },
  vendor: {
    apiName: "vendor",
    dbColumn: "vendor",
    bqColumn: "vendor",
    displayName: "Vendor",
    pluralParam: "vendors",
    singularParam: "vendor",
  },
  gender: {
    apiName: "gender",
    dbColumn: "gender",
    bqColumn: "gender",
    displayName: "Gender",
    pluralParam: "genders",
    singularParam: "gender",
  },
  hardware: {
    apiName: "hardware",
    dbColumn: "hardware",
    bqColumn: "hardware_material",
    displayName: "Hardware",
    pluralParam: "hardwares",
    singularParam: "hardware",
  },
  location: {
    apiName: "location",
    dbColumn: "sold_location",
    bqColumn: "sold_location",
    displayName: "Location",
    pluralParam: "locations",
    singularParam: "location",
  },
  decade: {
    apiName: "decade",
    dbColumn: "decade",
    bqColumn: null,
    displayName: "Decade",
    pluralParam: "decades",
    singularParam: "decade",
  },
  key_word: {
    apiName: "key_word",
    dbColumn: "listing_title",
    bqColumn: "listing_title",
    displayName: "Keywords",
    pluralParam: "key_words",
    singularParam: "key_word",
  },
  monthly: {
    apiName: "monthly",
    dbColumn: "listed_date",
    bqColumn: "listed_date",
    displayName: "Monthly",
    pluralParam: "monthly",
    singularParam: "monthly",
  },
  root_model: {
    apiName: "root_model",
    dbColumn: "root_model",
    bqColumn: "root_model",
    displayName: "Root Model",
    pluralParam: "root_models",
    singularParam: "root_model",
  },
  root_material: {
    apiName: "root_material",
    dbColumn: "root_material",
    bqColumn: "root_material",
    displayName: "Root Material",
    pluralParam: "root_materials",
    singularParam: "root_material",
  },
  root_type: {
    apiName: "root_type",
    dbColumn: "root_type",
    bqColumn: "root_type",
    displayName: "Root Type",
    pluralParam: "root_types",
    singularParam: "root_type",
  },
  root_hardware: {
    apiName: "root_hardware",
    dbColumn: "root_hardware",
    bqColumn: "root_hardware_material",
    displayName: "Root Hardware",
    pluralParam: "root_hardwares",
    singularParam: "root_hardware",
  },
};

// --- Derived lookup tables (computed once at module load) ---

// API name -> MySQL column name (used by query-builder.js)
const FIELD_MAPPINGS = Object.fromEntries(
  Object.entries(FIELD_DICTIONARY).map(([key, v]) => [key, v.dbColumn])
);

// API name -> MySQL column name (identical to FIELD_MAPPINGS, used for GROUP BY)
const GROUP_BY_FIELDS = Object.fromEntries(
  Object.entries(FIELD_DICTIONARY).map(([key, v]) => [key, v.dbColumn])
);

// API name -> BigQuery column name (used by bigquery-query-builder.js)
// Includes extra aliases for plural params and date/price fields
const BQ_COLUMN_MAPPINGS = Object.fromEntries(
  Object.entries(FIELD_DICTIONARY)
    .filter(([, v]) => v.bqColumn)
    .map(([key, v]) => [key, v.bqColumn])
);
// Add plural aliases used by BigQuery queries
BQ_COLUMN_MAPPINGS.genders = "gender";
BQ_COLUMN_MAPPINGS.models = "model";
// Add date/price fields not in the entity dictionary
BQ_COLUMN_MAPPINGS.sold_date = "sold_date";
BQ_COLUMN_MAPPINGS.listed_date = "listed_date";
BQ_COLUMN_MAPPINGS.sold_price = "sold_price";
BQ_COLUMN_MAPPINGS.listed_price = "listed_price";
BQ_COLUMN_MAPPINGS.is_sold = "is_sold";

// Plural param name -> MySQL column name (used for multi-value IN filters)
const MULTI_VALUE_FILTERS = {};
for (const v of Object.values(FIELD_DICTIONARY)) {
  if (v.pluralParam && v.dbColumn) {
    MULTI_VALUE_FILTERS[v.pluralParam] = v.dbColumn;
  }
}
// Also accept legacy "colours" plural
MULTI_VALUE_FILTERS.colours = "colour";

// Plural param -> singular param lookup
const PLURAL_TO_SINGULAR = Object.fromEntries(
  Object.entries(FIELD_DICTIONARY).map(([, v]) => [v.pluralParam, v.singularParam])
);

// --- Pricing algorithm mappings (BQ column-level) ---
// Maps any variant of an attribute name to the canonical BQ column name.
// Used by pricing-algorithm.js for missing attribute penalty and factor scoring.
const PRICING_ATTRIBUTE_MAPPING = {
  materials: "material",
  material: "material",
  colors: "colour",
  colours: "colour",
  color: "colour",
  colour: "colour",
  models: "model",
  model: "model",
  hardware_materials: "hardware_material",
  hardware_material: "hardware_material",
  hardware: "hardware_material",
  size: "size",
  sizes: "size",
  conditions: "condition",
  condition: "condition",
  root_models: "root_model",
  root_model: "root_model",
  root_materials: "root_material",
  root_material: "root_material",
  root_hardware_materials: "root_hardware_material",
  root_hardware_material: "root_hardware_material",
  root_sizes: "root_size",
  root_size: "root_size",
};

// BQ column -> BQ column (identity, used for detail logging in pricing)
const PRICING_DB_COLUMN_MAP = {
  material: "material",
  colour: "colour",
  model: "model",
  hardware_material: "hardware_material",
  size: "size",
  condition: "condition",
  root_model: "root_model",
  root_material: "root_material",
  root_hardware_material: "root_hardware_material",
  root_size: "root_size",
};

// Binary fields: BQ column -> all filter key variants that match it
const PRICING_BINARY_FIELDS = {
  material: ["materials", "material"],
  colour: ["colors", "colours", "color", "colour"],
  model: ["models", "model"],
  hardware_material: [
    "hardware_materials",
    "hardware_material",
    "hardware",
  ],
  root_material: ["root_materials", "root_material"],
  root_model: ["root_models", "root_model"],
  root_hardware_material: [
    "root_hardware_materials",
    "root_hardware_material",
    "root_hardware",
  ],
  size: ["size", "sizes"],
};

// Scale fields: BQ column -> all filter key variants that match it
const PRICING_SCALE_FIELDS = {
  condition: ["conditions", "condition"],
  root_size: ["root_sizes", "root_size"],
};

module.exports = {
  FIELD_DICTIONARY,
  FIELD_MAPPINGS,
  GROUP_BY_FIELDS,
  BQ_COLUMN_MAPPINGS,
  MULTI_VALUE_FILTERS,
  PLURAL_TO_SINGULAR,
  PRICING_ATTRIBUTE_MAPPING,
  PRICING_DB_COLUMN_MAP,
  PRICING_BINARY_FIELDS,
  PRICING_SCALE_FIELDS,
};
