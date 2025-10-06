/**
 * Parser function that removes null values and values below a specified percentage of the top value
 * @param {Array} data - Array of objects with a 'value' field
 * @param {number} thresholdPercent - Percentage threshold (default: 5)
 * @param {boolean} returnNulls - Whether to return null values (default: true)
 * @returns {Array} Filtered array
 */
function parseAndFilterData(data, thresholdPercent = 5, returnNulls = true) {
  // Remove nulls first if returnNulls is false
  const cleaned = returnNulls ? data : parseAndRemoveNulls(data);

  if (cleaned.length === 0) {
    return cleaned;
  }

  // Find the maximum value
  const maxValue = Math.max(...cleaned.map((item) => item.value));

  // Calculate threshold (default 5% of max value)
  const threshold = (maxValue * thresholdPercent) / 100;

  // Filter out values below the threshold
  const finalData = cleaned.filter((item) => item.value >= threshold);

  return finalData;
}

/**
 * Maps '_condition' field to 'condition' (no underscore) in each object of the array.
 * Removes the original '_condition' field.
 * @param {Array} data - Array of result objects
 * @returns {Array} Array with 'condition' field
 */
function mapDbConditionToApi(data) {
  if (!Array.isArray(data)) return data;
  return data.map((item) => {
    if (item && typeof item === "object" && "_condition" in item) {
      const newItem = { ...item };
      newItem.condition = newItem._condition;
      delete newItem._condition;
      return newItem;
    }
    return item;
  });
}

/**
 * Maps 'colour' field to 'color' (American spelling) in each object of the array.
 * Removes the original 'colour' field.
 * @param {Array} data - Array of result objects
 * @returns {Array} Array with 'color' field
 */
function mapDbColourToColor(data) {
  if (!Array.isArray(data)) return data;
  return data.map((item) => {
    if (item && typeof item === "object" && "colour" in item) {
      const newItem = { ...item };
      newItem.color = newItem.colour;
      delete newItem.colour;
      return newItem;
    }
    return item;
  });
}

/**
 * Maps 'primary_hardware_materials' field to 'hardware' in each object of the array.
 * Removes the original 'primary_hardware_materials' field.
 * @param {Array} data - Array of result objects
 * @returns {Array} Array with 'hardware' field
 */
function mapDbHardwareToApi(data) {
  if (!Array.isArray(data)) return data;
  return data.map((item) => {
    if (item && typeof item === "object" && "root_hardware_material" in item) {
      const newItem = { ...item };
      newItem.hardware = newItem.root_hardware_material;
      delete newItem.root_hardware_material;
      return newItem;
    }
    return item;
  });
}

/**
 * Maps 'sold_location' field to 'location' in each object of the array.
 * Removes the original 'sold_location' field.
 * @param {Array} data - Array of result objects
 * @returns {Array} Array with 'location' field
 */
function mapDbLocationToApi(data) {
  if (!Array.isArray(data)) return data;
  return data.map((item) => {
    if (item && typeof item === "object" && "sold_location" in item) {
      const newItem = { ...item };
      newItem.location = newItem.sold_location;
      delete newItem.sold_location;
      return newItem;
    }
    return item;
  });
}

/**
 * Maps 'material_parent' field to 'material' in each object of the array.
 * Removes the original 'material_parent' field.
 * @param {Array} data - Array of result objects
 * @returns {Array} Array with 'material' field
 */
function mapDbMaterialToApi(data) {
  console.log("🔧 mapDbMaterialToApi called with data length:", data.length);
  if (data.length > 0) {
    console.log(
      "🔧 Sample data before material mapping:",
      JSON.stringify(data[0], null, 2)
    );
  }

  if (!Array.isArray(data) || data.length === 0) return data;

  const mappedData = data.map((item) => {
    if (item && typeof item === "object" && "root_material" in item) {
      console.log(
        "🔧 Mapping root_material:",
        item.root_material,
        "to material"
      );
      const newItem = { ...item };
      newItem.material = newItem.root_material;
      delete newItem.root_material;
      return newItem;
    }
    return item;
  });

  console.log("🔧 Material mapping completed, data length:", mappedData.length);
  if (mappedData.length > 0) {
    console.log(
      "🔧 Sample data after material mapping:",
      JSON.stringify(mappedData[0], null, 2)
    );
  }

  return mappedData;
}

/**
 * Maps 'type' field to 'shape' in each object of the array.
 * Removes the original 'type' field.
 * @param {Array} data - Array of result objects
 * @returns {Array} Array with 'shape' field
 */
function mapDbTypeToShape(data) {
  console.log("🔧 mapDbTypeToShape called with data length:", data.length);
  if (data.length > 0) {
    console.log(
      "🔧 Sample data before type mapping:",
      JSON.stringify(data[0], null, 2)
    );
  }

  if (!Array.isArray(data) || data.length === 0) return data;

  const mappedData = data.map((item) => {
    if (item && typeof item === "object" && "type" in item) {
      console.log("🔧 Mapping type:", item.type, "to shape");
      const newItem = { ...item };
      newItem.shape = newItem.type;
      delete newItem.type;
      return newItem;
    }
    return item;
  });

  console.log("🔧 Type mapping completed, data length:", mappedData.length);
  if (mappedData.length > 0) {
    console.log(
      "🔧 Sample data after type mapping:",
      JSON.stringify(mappedData[0], null, 2)
    );
  }

  return mappedData;
}

/**
 * Maps monthly field from ISO timestamp to YYYY-MM format in each object of the array.
 * Handles both direct monthly field and nested monthly.value field.
 * @param {Array} data - Array of result objects
 * @returns {Array} Array with formatted monthly field
 */
function mapDbMonthlyToApi(data) {
  console.log("🔧 mapDbMonthlyToApi called with data length:", data.length);
  if (data.length > 0) {
    console.log(
      "🔧 Sample data before monthly mapping:",
      JSON.stringify(data[0], null, 2)
    );
  }

  if (!Array.isArray(data) || data.length === 0) return data;

  const mappedData = data.map((item) => {
    if (item && typeof item === "object") {
      const newItem = { ...item };

      // Handle nested monthly.value structure
      if (
        newItem.monthly &&
        typeof newItem.monthly === "object" &&
        newItem.monthly.value
      ) {
        console.log(
          "🔧 Mapping nested monthly.value:",
          newItem.monthly.value,
          "to YYYY-MM format"
        );
        const date = new Date(newItem.monthly.value);
        const year = date.getFullYear();
        const month = String(date.getMonth() + 1).padStart(2, "0");
        newItem.monthly = `${year}-${month}`;
      }
      // Handle direct monthly field (ISO timestamp)
      else if (
        newItem.monthly &&
        typeof newItem.monthly === "string" &&
        newItem.monthly.includes("T")
      ) {
        console.log(
          "🔧 Mapping direct monthly ISO timestamp:",
          newItem.monthly,
          "to YYYY-MM format"
        );
        const date = new Date(newItem.monthly);
        const year = date.getFullYear();
        const month = String(date.getMonth() + 1).padStart(2, "0");
        newItem.monthly = `${year}-${month}`;
      }
      // Handle monthly field that's already in YYYY-MM format
      else if (
        newItem.monthly &&
        typeof newItem.monthly === "string" &&
        /^\d{4}-\d{2}$/.test(newItem.monthly)
      ) {
        console.log(
          "🔧 Monthly field already in correct format:",
          newItem.monthly
        );
        // No change needed
      }

      return newItem;
    }
    return item;
  });

  console.log("🔧 Monthly mapping completed, data length:", mappedData.length);
  if (mappedData.length > 0) {
    console.log(
      "🔧 Sample data after monthly mapping:",
      JSON.stringify(mappedData[0], null, 2)
    );
  }

  return mappedData;
}

/**
 * Consolidated function that applies all entity field mappings.
 * Maps database field names to API field names for consistent response format.
 * @param {Array} data - Array of result objects
 * @returns {Array} Array with all mapped fields
 */
function mapAllEntities(data) {
  console.log("🔧 mapAllEntities called with data length:", data.length);
  if (data.length > 0) {
    console.log(
      "🔧 Sample data before mapping:",
      JSON.stringify(data[0], null, 2)
    );
  }

  let mappedData = data;

  // Apply all entity mappings in sequence
  console.log("🔧 Applying condition mapping...");
  mappedData = mapDbConditionToApi(mappedData);
  console.log("🔧 After condition mapping, data length:", mappedData.length);

  console.log("🔧 Applying colour mapping...");
  mappedData = mapDbColourToColor(mappedData);
  console.log("🔧 After colour mapping, data length:", mappedData.length);

  console.log("🔧 Applying hardware mapping...");
  mappedData = mapDbHardwareToApi(mappedData);
  console.log("🔧 After hardware mapping, data length:", mappedData.length);

  console.log("🔧 Applying location mapping...");
  mappedData = mapDbLocationToApi(mappedData);
  console.log("🔧 After location mapping, data length:", mappedData.length);

  console.log("🔧 Applying material mapping...");
  mappedData = mapDbMaterialToApi(mappedData);
  console.log("🔧 After material mapping, data length:", mappedData.length);

  console.log("🔧 Applying type mapping...");
  mappedData = mapDbTypeToShape(mappedData);
  console.log("🔧 After type mapping, data length:", mappedData.length);

  console.log("🔧 Applying monthly mapping...");
  mappedData = mapDbMonthlyToApi(mappedData);
  console.log("🔧 After monthly mapping, data length:", mappedData.length);

  if (mappedData.length > 0) {
    console.log(
      "🔧 Sample data after all mapping:",
      JSON.stringify(mappedData[0], null, 2)
    );
  }

  return mappedData;
}

/**
 * Parser function that removes null values only (no threshold filtering)
 * @param {Array} data - Array of objects with a 'value' field
 * @returns {Array} Filtered array
 */
function parseAndRemoveNulls(data) {
  if (!Array.isArray(data) || data.length === 0) {
    return data;
  }

  // Remove objects where all values are null or undefined, or where 'value' is null/undefined
  return data.filter((item) => {
    if (item && typeof item === "object") {
      // Remove if 'value' is null or undefined
      if (item.value === null || item.value === undefined) return false;
      return Object.values(item).some((v) => v !== null && v !== undefined);
    }
    return item !== null && item !== undefined;
  });
}

/**
 * Parser function that filters values below a specified percentage of the top value (assumes no nulls)
 * @param {Array} data - Array of objects with a 'value' field
 * @param {number} thresholdPercent - Percentage threshold (default: 5)
 * @returns {Array} Filtered array
 */
function filterByThreshold(data, thresholdPercent = 5) {
  if (!Array.isArray(data) || data.length === 0) {
    return data;
  }

  // Find the maximum value
  const maxValue = Math.max(...data.map((item) => item.value));

  // Calculate threshold
  const threshold = (maxValue * thresholdPercent) / 100;

  // Filter out values below the threshold
  const finalData = data.filter((item) => item.value >= threshold);

  return finalData;
}

/**
 * Applies post-aggregation calculations to the result set.
 * @param {Array} data - The aggregated data array (already grouped/sorted by group_by, monthly)
 * @param {string} calc - The calc type (e.g., 'percentage_change')
 * @param {string} groupBy - The group_by string (e.g., 'brand,monthly')
 * @param {object} validatedParams - Validated parameters (already normalized)
 * @param {object} originalParams - Original query parameters (before validation/expansion)
 * @returns {Array} The data array with calc fields added and working data points removed
 */
function postAggregationCalcs(
  data,
  calc,
  groupBy,
  validatedParams = {},
  originalParams = {}
) {
  console.log("🔧 postAggregationCalcs called");
  console.log("🔧 Calc type:", calc);
  console.log("🔧 Group by:", groupBy);
  console.log("🔧 Input data length:", data.length);
  console.log("🔧 Validated params:", JSON.stringify(validatedParams, null, 2));
  console.log("🔧 Original params:", JSON.stringify(originalParams, null, 2));

  if (!calc || !Array.isArray(data) || data.length === 0) {
    console.log("🔧 Early return - no calc, no data, or empty array");
    return data;
  }

  if (calc === "percentage_change") {
    console.log("🔧 Processing percentage_change calculation");

    const groupByFields = (groupBy || "").split(",").map((f) => f.trim());
    console.log("🔧 Group by fields:", groupByFields);

    const temporalField = groupByFields.find((f) =>
      VALID_TEMPORAL_GROUPINGS.includes(f)
    );
    console.log("🔧 Temporal field found:", temporalField);

    if (!temporalField) {
      throw new Error(
        "The 'percentage_change' calc requires group_by to include a valid temporal grouping (e.g., 'monthly' or 'weekly')."
      );
    }

    const groupFields = groupByFields.filter(
      (f) => f && !VALID_TEMPORAL_GROUPINGS.includes(f)
    );
    console.log("🔧 Entity group fields:", groupFields);

    const grouped = {};
    data.forEach((row) => {
      const key = groupFields.map((f) => row[f]).join("||");
      if (!grouped[key]) grouped[key] = [];
      grouped[key].push(row);
    });
    console.log("🔧 Grouped data keys:", Object.keys(grouped));
    console.log(
      "🔧 Grouped data structure:",
      Object.keys(grouped).map((key) => ({
        key,
        count: grouped[key].length,
        sample: grouped[key][0],
      }))
    );

    const result = [];
    const entityKeys = Object.keys(grouped).sort();
    console.log("🔧 Sorted entity keys:", entityKeys);

    entityKeys.forEach((key) => {
      const rows = grouped[key];
      console.log(
        `🔧 Processing entity group: ${key} with ${rows.length} rows`
      );

      // Sort by temporal field to ensure correct percentage calculation order
      rows.sort((a, b) => {
        if (a[temporalField] < b[temporalField]) return -1;
        if (a[temporalField] > b[temporalField]) return 1;
        return 0;
      });
      console.log(
        `🔧 Sorted temporal values for ${key}:`,
        rows.map((r) => r[temporalField])
      );

      for (let i = 0; i < rows.length; i++) {
        rows[i].pre_calc_value = rows[i].value;

        if (i === 0) {
          rows[i].value = null;
          console.log(`🔧 First row for ${key}: value set to null`);
        } else {
          const currentValue = rows[i].pre_calc_value;
          const previousValue = rows[i - 1].pre_calc_value;
          console.log(
            `🔧 Row ${i} for ${key}: current=${currentValue}, previous=${previousValue}`
          );

          if (
            currentValue !== null &&
            currentValue !== undefined &&
            previousValue !== null &&
            previousValue !== undefined &&
            previousValue !== 0
          ) {
            rows[i].value = (currentValue - previousValue) / previousValue;
            console.log(`🔧 Calculated percentage change: ${rows[i].value}`);
          } else {
            rows[i].value = null;
            console.log(
              `🔧 Row ${i} for ${key}: value set to null due to invalid data`
            );
          }
        }
      }
      result.push(...rows);
    });

    console.log(
      "🔧 Result after percentage calculation, length:",
      result.length
    );
    console.log("🔧 Sample result rows:", result.slice(0, 3));

    const finalResult = filterWorkingDataPoints(
      result,
      validatedParams,
      originalParams,
      temporalField
    );
    console.log(
      "🔧 After filterWorkingDataPoints, length:",
      finalResult.length
    );
    console.log("🔧 Sample filtered rows:", finalResult.slice(0, 3));

    // After calculating percentage changes, re-order by the new percentage change values
    // and apply limit/offset if specified
    const reorderedResult = reorderAndLimitCalcResults(
      finalResult,
      validatedParams,
      calc
    );
    console.log(
      "🔧 After reorderAndLimitCalcResults, length:",
      reorderedResult.length
    );
    console.log("🔧 Final result sample:", reorderedResult.slice(0, 3));

    return reorderedResult;
  }

  return data;
}

/**
 * Reorders and limits calculation results based on the calculated values
 * @param {Array} data - The data array after calculations
 * @param {object} validatedParams - Validated parameters (already normalized)
 * @param {string} calc - The calculation type (e.g., 'percentage_change')
 * @returns {Array} Reordered and limited data array
 */
function reorderAndLimitCalcResults(data, validatedParams, calc) {
  console.log("🔧 reorderAndLimitCalcResults called");
  console.log("🔧 Input data length:", data.length);
  console.log("🔧 Calc type:", calc);
  console.log("🔧 Validated params:", JSON.stringify(validatedParams, null, 2));

  if (!Array.isArray(data) || data.length === 0) {
    console.log("🔧 Early return - no data or empty array");
    return data;
  }

  // Filter out null values from calculations (e.g., first period in percentage_change)
  const validResults = data.filter((row) => {
    if (calc === "percentage_change") {
      const isValid = row.value !== null && row.value !== undefined;
      if (!isValid) {
        console.log(`🔧 Filtering out row with null/undefined value:`, row);
      }
      return isValid;
    }
    return true;
  });

  console.log(
    "🔧 After filtering null values, valid results length:",
    validResults.length
  );

  // Sort by the calculated value (descending by default for percentage_change)
  const sortOrder = validatedParams.order || "DESC";
  console.log("🔧 Sort order:", sortOrder);

  validResults.sort((a, b) => {
    const aValue = a.value || 0;
    const bValue = b.value || 0;

    if (sortOrder === "ASC") {
      return aValue - bValue;
    } else {
      return bValue - aValue;
    }
  });

  console.log("🔧 After sorting, sample results:", validResults.slice(0, 3));

  // Apply limit and offset
  const limit = validatedParams.limit || 500;
  const offset = validatedParams.offset || 0;

  console.log("🔧 Applying limit:", limit, "offset:", offset);
  console.log(
    "🔧 Total valid results before limit/offset:",
    validResults.length
  );

  const limitedResults = validResults.slice(offset, offset + limit);

  console.log("🔧 Final limited results length:", limitedResults.length);
  console.log("🔧 Final sample results:", limitedResults.slice(0, 3));

  return limitedResults;
}

/**
 * Limits results to unique entity combinations (excluding temporal fields)
 * @param {Array} data - The data array
 * @param {object} validatedParams - Validated parameters (already normalized)
 * @param {number} originalLimit - The original limit requested by the user
 * @param {number} temporalMultiplier - The temporal multiplier used in the query
 * @returns {Array} Limited data array with unique entity combinations
 */
function limitToUniqueEntityCombinations(
  data,
  validatedParams,
  originalLimit,
  temporalMultiplier = 1
) {
  if (!Array.isArray(data) || data.length === 0) return data;
  if (temporalMultiplier <= 1) return data; // No temporal grouping, return as is

  // Get the group by fields to identify entity fields (excluding temporal)
  const groupByFields = validatedParams.group_by
    ? validatedParams.group_by.split(",").map((f) => f.trim())
    : [];

  const temporalFields = ["monthly", "weekly"];
  const entityFields = groupByFields.filter(
    (field) => !temporalFields.includes(field)
  );

  if (entityFields.length === 0) {
    // No entity grouping, just temporal grouping - return limited results
    return data.slice(0, originalLimit);
  }

  // Group by entity combination (excluding temporal fields)
  const entityGroups = {};
  data.forEach((row) => {
    const entityKey = entityFields
      .map((field) => row[field] || "null")
      .join("||");
    if (!entityGroups[entityKey]) {
      entityGroups[entityKey] = [];
    }
    entityGroups[entityKey].push(row);
  });

  // Sort entity combinations by their best performing temporal period
  const sortOrder = validatedParams.order || "DESC";
  const sortedEntityKeys = Object.keys(entityGroups).sort((a, b) => {
    const aRows = entityGroups[a];
    const bRows = entityGroups[b];

    // Find the best value for each entity combination
    const aBestValue = Math.max(...aRows.map((row) => row.value || 0));
    const bBestValue = Math.max(...bRows.map((row) => row.value || 0));

    if (sortOrder === "ASC") {
      return aBestValue - bBestValue;
    } else {
      return bBestValue - aBestValue;
    }
  });

  // Take only the top entity combinations up to the original limit
  const limitedEntityKeys = sortedEntityKeys.slice(0, originalLimit);

  // Flatten the results back to individual rows
  const limitedResults = limitedEntityKeys.flatMap((key) => entityGroups[key]);

  return limitedResults;
}

/**
 * Filters out working data points that were added for percentage change calculations
 * @param {Array} data - The data array after calculations
 * @param {object} validatedParams - Validated parameters (already normalized)
 * @param {object} originalParams - Original query parameters (before validation/expansion)
 * @param {string} temporalField - The temporal field used (monthly/weekly)
 * @returns {Array} Filtered data array without working data points
 */
function filterWorkingDataPoints(
  data,
  validatedParams,
  originalParams,
  temporalField
) {
  console.log("🔧 filterWorkingDataPoints called");
  console.log("🔧 Input data length:", data.length);
  console.log("🔧 Temporal field:", temporalField);
  console.log("🔧 Validated params:", JSON.stringify(validatedParams, null, 2));
  console.log("🔧 Original params:", JSON.stringify(originalParams, null, 2));

  if (!validatedParams || !temporalField) {
    console.log("🔧 Early return - missing validatedParams or temporalField");
    return data;
  }

  // Get the originally requested temporal values (before expansion for calc)
  let originalTemporalValues = [];

  if (temporalField === "monthly") {
    // Check both singular and plural versions of the parameter
    if (originalParams.monthlys) {
      originalTemporalValues = originalParams.monthlys;
    } else if (originalParams.monthly) {
      // Convert single value to array if needed
      if (Array.isArray(originalParams.monthly)) {
        originalTemporalValues = originalParams.monthly;
      } else {
        // Handle comma-separated string like "2025-01,2025-02,2025-03"
        originalTemporalValues = originalParams.monthly
          .split(",")
          .map((m) => m.trim())
          .filter((m) => m.length > 0);
      }
    }
  } else if (temporalField === "weekly") {
    // Check both singular and plural versions of the parameter
    if (originalParams.weeklys) {
      originalTemporalValues = originalParams.weeklys;
    } else if (originalParams.weekly) {
      // Convert single value to array if needed
      if (Array.isArray(originalParams.weekly)) {
        originalTemporalValues = originalParams.weekly;
      } else {
        // Handle comma-separated string like "2025-W01,2025-W02,2025-W03"
        originalTemporalValues = originalParams.weekly
          .split(",")
          .map((w) => w.trim())
          .filter((w) => w.length > 0);
      }
    }
  }

  console.log("🔧 Original temporal values extracted:", originalTemporalValues);
  console.log(
    "🔧 Original temporal values type:",
    typeof originalTemporalValues
  );
  console.log(
    "🔧 Original temporal values length:",
    originalTemporalValues.length
  );
  if (originalTemporalValues.length > 0) {
    console.log("🔧 First temporal value:", originalTemporalValues[0]);
    console.log(
      "🔧 First temporal value type:",
      typeof originalTemporalValues[0]
    );
  }

  // If no original temporal filter, return all data
  if (originalTemporalValues.length === 0) {
    console.log("🔧 No original temporal filter, returning all data");
    return data;
  }

  console.log("🔧 Filtering data to keep only requested temporal periods");
  console.log("🔧 Requested periods:", originalTemporalValues);

  // Filter to keep only the originally requested periods
  const filteredData = data.filter((row) => {
    const rowTemporalValue = row[temporalField];
    const isIncluded = originalTemporalValues.includes(rowTemporalValue);
    if (!isIncluded) {
      console.log(
        `🔧 Filtering out row with ${temporalField}: ${rowTemporalValue}`
      );
    }
    return isIncluded;
  });

  console.log("🔧 After filtering, data length:", filteredData.length);
  console.log("🔧 Sample filtered data:", filteredData.slice(0, 3));

  return filteredData;
}

/**
 * Processes data with optional null filtering based on return_nulls parameter
 * @param {Array} data - Array of data objects
 * @param {boolean} returnNulls - Whether to return null values (default: true)
 * @returns {Array} Processed data array
 */
function processDataWithNullsOption(data, returnNulls = true) {
  if (returnNulls || !Array.isArray(data) || data.length === 0) {
    return data;
  }

  return data.filter((item) => {
    if (item && typeof item === "object") {
      return Object.values(item).some((v) => v !== null && v !== undefined);
    }
    return item !== null && item !== undefined;
  });
}

/**
 * Returns up to n random product links from the data array (assume each item has a 'product_link' field)
 * @param {Array} data - Array of data objects
 * @param {number} n - Number of samples (default 5)
 * @returns {Array} Array of product links
 */
function getSampleProductLinks(data, n = 5) {
  if (!Array.isArray(data) || data.length === 0) return [];
  // Shuffle and pick n
  const shuffled = data.slice().sort(() => 0.5 - Math.random());
  return shuffled
    .filter((item) => item.product_link)
    .slice(0, n)
    .map((item) => item.product_link);
}

/**
 * Returns the total listing count from the data array
 * @param {Array} data - Array of data objects
 * @returns {number} Total count
 */
function getTotalListingCount(data) {
  if (!Array.isArray(data)) return 0;
  return data.length;
}

/**
 * Returns the sum and count of a metric field in the data array
 * @param {Array} data - Array of data objects
 * @param {string} metricField - The field to sum and count (e.g., 'value', 'sold_price')
 * @returns {Object} { sum: number, count: number }
 */
function getSumAndCount(data, metricField = "value") {
  if (!Array.isArray(data) || data.length === 0) return { sum: 0, count: 0 };
  let sum = 0;
  let count = 0;
  for (const item of data) {
    if (item && item[metricField] !== null && item[metricField] !== undefined) {
      sum += Number(item[metricField]);
      count++;
    }
  }
  return { sum, count };
}

/**
 * Filter out internal calculation fields from API response
 * Removes fields that are used for internal calculations but shouldn't be exposed to clients
 * Uses a comprehensive whitelist of client-allowed columns
 * @param {Array} data - Array of result objects
 * @param {string} service - Service name (e.g., "analytics", "counts")
 * @param {string} endpoint - Endpoint name (e.g., "market-share", "gmv")
 * @param {Array} allowedMetrics - Optional array of metric fields to allow (e.g., ["value", "count"])
 * @returns {Array} Array with internal calculation fields removed
 */
function filterInternalCalculationFields(
  data,
  service = "",
  endpoint = "",
  allowedMetrics = null
) {
  console.log("🔧 filterInternalCalculationFields called");
  console.log("🔧 Service:", service, "Endpoint:", endpoint);
  console.log("🔧 Input data length:", data.length);

  if (!Array.isArray(data)) return data;

  // Comprehensive list of client-allowed columns (baseline)
  const clientAllowedColumns = new Set([
    // Entity fields (from meta/entities)
    "brand",
    "type",
    "material",
    "color",
    "condition",
    "size",
    "vendor",
    "gender",
    "model",
    "decade",
    "location",
    "hardware",

    // Temporal fields
    "monthly",
    "weekly",

    // Value fields (aggregation results)
    "value",
    "count",
    "percentage_change",

    // Confidence metrics fields (slow mode)
    "final_value",
    "n",
    "variance",
    "stddev",
    "sem",
    "ci95_lower",
    "ci95_upper",
    "min_val",
    "max_val",
    "range_val",
    "cv",
    "skewness",
    "kurtosis_excess",

    // Additional aggregation fields
    "avg_value",
    "sum_value",
    "entity_gmv",

    // Product fields (from productListing.json schema)
    "id",
    "garment_id",
    "display_size",
    "rrp",
    "listed_date",
    "currency",
    "sold_date",
    "likes",
    "purchase_location",
    "release_date",
    "model_child",
    "sold_price",
    "listed_price",
    "depth",
    "width",
    "height",
    "collab",
    "customer_segment",
    "shape",
    "neutral_bold",
    "natural_synthetic",
    "heritage_contemporary",
    "new_used",
    "continent",
    "vegan_nonvegan",
    "sell_through_days",
    "listed",
    "is_sold",

    // Product links (for drilldown endpoints)
    "product_link",

    // Pre-calc value (used for percentage_change calculations)
    "pre_calc_value",
  ]);

  // Identify metric fields subset we may want to constrain
  const metricFields = new Set([
    "value",
    "count",
    "final_value",
    "n",
    "variance",
    "stddev",
    "sem",
    "ci95_lower",
    "ci95_upper",
    "min_val",
    "max_val",
    "range_val",
    "cv",
    "skewness",
    "kurtosis_excess",
    "avg_value",
    "sum_value",
    "entity_gmv",
    "percentage_change",
  ]);

  // Build final allowed set: baseline non-metric fields + endpoint-allowed metrics (or all metrics if not provided)
  const baseAllowed = new Set(
    Array.from(clientAllowedColumns).filter((key) => !metricFields.has(key))
  );

  const allowedMetricFields = new Set(
    allowedMetrics && Array.isArray(allowedMetrics) && allowedMetrics.length > 0
      ? allowedMetrics
      : Array.from(metricFields)
  );

  // Merge into final allowed set
  const finalAllowed = new Set([...baseAllowed, ...allowedMetricFields]);

  console.log("🔧 Endpoint-allowed metrics:", Array.from(allowedMetricFields));
  console.log("🔧 Client allowed baseline count:", baseAllowed.size);

  // Remove any fields that are not in the final allowed list
  const filteredData = data.map((item) => {
    if (item && typeof item === "object") {
      const cleanedItem = {};
      const originalKeys = Object.keys(item);
      const cleanedKeys = [];

      Object.keys(item).forEach((key) => {
        if (finalAllowed.has(key)) {
          cleanedItem[key] = item[key];
          cleanedKeys.push(key);
        }
      });

      if (originalKeys.length !== cleanedKeys.length) {
        console.log(
          "🔧 Filtered item - Original keys:",
          originalKeys.length,
          "Cleaned keys:",
          cleanedKeys.length
        );
        console.log(
          "🔧 Removed keys:",
          originalKeys.filter((key) => !cleanedKeys.includes(key))
        );
      }

      return cleanedItem;
    }
    return item;
  });

  console.log(
    "🔧 Filtering completed, output data length:",
    filteredData.length
  );
  if (filteredData.length > 0) {
    console.log(
      "🔧 Sample filtered data structure:",
      Object.keys(filteredData[0])
    );
  }

  return filteredData;
}

/**
 * Generic confidence metrics processor that can be used by any service
 * @param {Array} data - Raw data from database
 * @param {string} queryMode - Query mode (fast or slow)
 * @param {object} config - Service configuration
 * @param {object} options - Processing options
 * @returns {Array} Processed data with appropriate metrics
 */
function processConfidenceMetrics(data, queryMode, config, options = {}) {
  if (queryMode === "basic") {
    // Basic mode: return basic metrics only
    return data.map((row) => {
      const processed = {
        ...row,
        value: row.value || row.final_value || row.avg_value || row.sum_value,
        count: row.count || row.n || 0,
      };

      // Add any additional basic mode fields if specified
      if (options.basicModeFields) {
        options.basicModeFields.forEach((field) => {
          if (row[field] !== undefined) {
            processed[field] = row[field];
          }
        });
      }

      return processed;
    });
  }

  // Basic diagnostics mode: return comprehensive metrics
  return data.map((row) => {
    const processed = {
      ...row,
      value: row.final_value || row.value || row.avg_value || row.sum_value,
      count: row.n || row.count || 0,
    };

    // Add confidence metrics if available
    const confidenceFields = [
      "variance",
      "stddev",
      "sem",
      "ci95_lower",
      "ci95_upper",
      "min_val",
      "max_val",
      "range_val",
      "cv",
      "skewness",
      "kurtosis_excess",
    ];

    confidenceFields.forEach((field) => {
      if (row[field] !== undefined) {
        processed[field] = row[field];
      }
    });

    // Add any additional basic diagnostics mode fields if specified
    if (options.basicDiagnosticsModeFields) {
      options.basicDiagnosticsModeFields.forEach((field) => {
        if (row[field] !== undefined) {
          processed[field] = row[field];
        }
      });
    }

    return processed;
  });
}

/**
 * Generic function to build confidence metrics SQL based on query mode and field type
 * @param {string} queryMode - Query mode (fast or slow)
 * @param {string} valueField - Field to calculate metrics on (e.g., 'sold_price', 'discount', etc.)
 * @param {string} aggregationType - Type of aggregation (avg, sum, count, etc.)
 * @param {object} config - Service configuration
 * @param {object} options - Additional options for specific calculations
 * @returns {string} SQL aggregation function
 */
function buildConfidenceMetricsSQL(
  queryMode,
  valueField,
  aggregationType = "avg",
  config = null,
  options = {}
) {
  if (queryMode === "basic") {
    // Basic mode: return basic metrics only
    switch (aggregationType) {
      case "avg":
        return `AVG(${valueField}) AS value, COUNT(*) AS count`;
      case "sum":
        return `SUM(${valueField}) AS value, COUNT(*) AS count`;
      case "count":
        return `COUNT(*) AS value, COUNT(*) AS count`;
      case "custom":
        return (
          options.customBasicSQL ||
          `AVG(${valueField}) AS value, COUNT(*) AS count`
        );
      default:
        return `AVG(${valueField}) AS value, COUNT(*) AS count`;
    }
  }

  // Basic diagnostics mode: comprehensive statistical analysis
  const zScore =
    config?.confidence_metrics?.modes?.basic_diagnostics?.z_score || 1.96;

  switch (aggregationType) {
    case "avg":
      return `
        AVG(${valueField}) AS final_value,
        COUNT(*) AS n,
        VAR_POP(${valueField}) AS variance,
        STDDEV_POP(${valueField}) AS stddev,
        STDDEV_POP(${valueField}) / SQRT(COUNT(*)) AS sem,
        AVG(${valueField}) - ${zScore} * STDDEV_POP(${valueField}) / SQRT(COUNT(*)) AS ci95_lower,
        AVG(${valueField}) + ${zScore} * STDDEV_POP(${valueField}) / SQRT(COUNT(*)) AS ci95_upper,
        MIN(${valueField}) AS min_val,
        MAX(${valueField}) AS max_val,
        (MAX(${valueField}) - MIN(${valueField})) AS range_val,
        CASE WHEN AVG(${valueField}) = 0 THEN NULL ELSE STDDEV_POP(${valueField}) / AVG(${valueField}) END AS cv,
        CASE WHEN STDDEV_POP(${valueField}) = 0 THEN NULL ELSE
          (AVG(POWER(${valueField},3)) - 3*AVG(${valueField})*AVG(POWER(${valueField},2)) + 2*POWER(AVG(${valueField}),3))
          / POWER(STDDEV_POP(${valueField}), 3)
        END AS skewness,
        CASE WHEN STDDEV_POP(${valueField}) = 0 THEN NULL ELSE
          (AVG(POWER(${valueField},4))
            - 4*AVG(${valueField})*AVG(POWER(${valueField},3))
            + 6*POWER(AVG(${valueField}),2)*AVG(POWER(${valueField},2))
            - 3*POWER(AVG(${valueField}),4))
          / POWER(STDDEV_POP(${valueField}), 4) - 3
        END AS kurtosis_excess
      `
        .replace(/\s+/g, " ")
        .trim();

    case "sum":
      return `
        SUM(${valueField}) AS final_value,
        COUNT(*) AS n,
        VAR_POP(${valueField}) AS variance,
        STDDEV_POP(${valueField}) AS stddev,
        STDDEV_POP(${valueField}) / SQRT(COUNT(*)) AS sem,
        SUM(${valueField}) - ${zScore} * STDDEV_POP(${valueField}) / SQRT(COUNT(*)) AS ci95_lower,
        SUM(${valueField}) + ${zScore} * STDDEV_POP(${valueField}) / SQRT(COUNT(*)) AS ci95_upper,
        MIN(${valueField}) AS min_val,
        MAX(${valueField}) AS max_val,
        (MAX(${valueField}) - MIN(${valueField})) AS range_val,
        CASE WHEN SUM(${valueField}) = 0 THEN NULL ELSE STDDEV_POP(${valueField}) / SUM(${valueField}) END AS cv,
        CASE WHEN STDDEV_POP(${valueField}) = 0 THEN NULL ELSE
          (AVG(POWER(${valueField},3)) - 3*AVG(${valueField})*AVG(POWER(${valueField},2)) + 2*POWER(AVG(${valueField}),3))
          / POWER(STDDEV_POP(${valueField}), 3)
        END AS skewness,
        CASE WHEN STDDEV_POP(${valueField}) = 0 THEN NULL ELSE
          (AVG(POWER(${valueField},4))
            - 4*AVG(${valueField})*AVG(POWER(${valueField},3))
            + 6*POWER(AVG(${valueField}),2)*AVG(POWER(${valueField},2))
            - 3*POWER(AVG(${valueField}),4))
          / POWER(STDDEV_POP(${valueField}), 4) - 3
        END AS kurtosis_excess
      `
        .replace(/\s+/g, " ")
        .trim();

    case "count":
      return `
        COUNT(*) AS final_value,
        COUNT(*) AS n,
        0 AS variance,
        0 AS stddev,
        0 AS sem,
        COUNT(*) AS ci95_lower,
        COUNT(*) AS ci95_upper,
        COUNT(*) AS min_val,
        COUNT(*) AS max_val,
        0 AS range_val,
        0 AS cv,
        0 AS skewness,
        0 AS kurtosis_excess
      `
        .replace(/\s+/g, " ")
        .trim();

    case "custom":
      return (
        options.customSlowSQL ||
        buildConfidenceMetricsSQL(queryMode, valueField, "avg", config)
      );

    default:
      return buildConfidenceMetricsSQL(queryMode, valueField, "avg", config);
  }
}

/**
 * Generic function to add confidence metrics metadata to response
 * @param {object} metadata - Response metadata
 * @param {string} queryMode - Query mode used
 * @param {object} config - Service configuration
 * @param {object} options - Additional options
 * @returns {object} Enhanced metadata with confidence information
 */
function addConfidenceMetadata(metadata, queryMode, config, options = {}) {
  if (queryMode === "basic") {
    return {
      ...metadata,
      confidence_mode: "basic",
      confidence_description: "Basic metrics only for basic queries",
      available_metrics: options.availableMetricsOverride?.basic ||
        config?.confidence_metrics?.modes?.basic?.metrics || ["value", "count"],
      ...options.basicModeMetadata,
    };
  }

  const basicDiagnosticsConfig =
    config?.confidence_metrics?.modes?.basic_diagnostics;
  const baseMetadata = {
    ...metadata,
    confidence_mode: "basic_diagnostics",
    confidence_description: "Comprehensive statistical analysis",
    confidence_level: basicDiagnosticsConfig?.confidence_level || 0.95,
    z_score: basicDiagnosticsConfig?.z_score || 1.96,
    available_metrics:
      options.availableMetricsOverride?.basic_diagnostics ||
      basicDiagnosticsConfig?.metrics ||
      [],
    statistical_interpretation: {
      cv: "Coefficient of Variation: ratio of standard deviation to mean (dimensionless measure of relative dispersion)",
      skewness:
        "Third standardized moment indicating distribution asymmetry (positive = right tail, negative = left tail)",
      kurtosis_excess:
        "Fourth standardized moment minus 3, measuring tail heaviness relative to normal distribution (positive = heavy tails, negative = light tails)",
      sem: "Standard Error of the Mean: standard deviation divided by square root of sample size",
      ci95: "95% Confidence Interval for the mean, indicating range where true population mean likely lies",
    },
    ...options.basicDiagnosticsModeMetadata,
  };

  return baseMetadata;
}

/**
 * Generic function to determine the correct ORDER BY field based on query mode
 * @param {string} queryMode - Query mode (fast or slow)
 * @param {string} fastField - Field to use for fast mode ordering
 * @param {string} slowField - Field to use for slow mode ordering
 * @returns {string} The correct field to use for ORDER BY
 */
function getOrderByField(
  queryMode,
  basicField = "value",
  basicDiagnosticsField = "final_value"
) {
  return queryMode === "basic_diagnostics" ? basicDiagnosticsField : basicField;
}

/**
 * Generic function to validate and process query_mode parameter
 * @param {object} queryParams - Query parameters
 * @param {object} config - Service configuration
 * @param {object} options - Validation options
 * @returns {string} Validated query mode
 */
function validateAndProcessQueryMode(queryParams, config, options = {}) {
  const queryMode =
    queryParams.query_mode ||
    config?.confidence_metrics?.default_mode ||
    "basic";

  // Validate query mode
  const validModes = ["basic", "basic_diagnostics"];
  if (!validModes.includes(queryMode)) {
    throw new Error(
      `Invalid query_mode parameter. Must be one of: ${validModes.join(", ")}`
    );
  }

  // Check if confidence metrics are enabled for basic diagnostics mode
  if (
    queryMode === "basic_diagnostics" &&
    !config?.confidence_metrics?.enabled
  ) {
    if (options.allowFallback !== false) {
      // Fallback to basic mode if confidence metrics not enabled
      return "basic";
    } else {
      throw new Error(
        "Basic diagnostics mode requires confidence metrics to be enabled in service configuration"
      );
    }
  }

  return queryMode;
}

const { VALID_TEMPORAL_GROUPINGS } = require("./validation");

module.exports = {
  parseAndFilterData,
  parseAndRemoveNulls,
  filterByThreshold,
  mapDbConditionToApi,
  mapDbColourToColor,
  mapDbHardwareToApi,
  mapDbLocationToApi,
  mapDbMaterialToApi,
  mapDbTypeToShape,
  mapDbMonthlyToApi,
  mapAllEntities,
  postAggregationCalcs,
  reorderAndLimitCalcResults,
  limitToUniqueEntityCombinations,
  filterWorkingDataPoints,
  processDataWithNullsOption,
  getSampleProductLinks,
  getTotalListingCount,
  getSumAndCount,
  filterInternalCalculationFields,
  processConfidenceMetrics,
  buildConfidenceMetricsSQL,
  addConfidenceMetadata,
  getOrderByField,
  validateAndProcessQueryMode,
};
