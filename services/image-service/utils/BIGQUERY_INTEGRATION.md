# BigQuery Integration for Truss Data Service Layer

This document outlines the BigQuery integration for the Truss Data Service Layer, including setup, configuration, permissions, and usage examples.

## Overview

The BigQuery integration allows services to query Google Cloud BigQuery instead of MySQL, providing better scalability and performance for large datasets. The integration maintains compatibility with the existing service architecture while adding BigQuery-specific features.

## Architecture

```
Client -> API Gateway -> Lambda (service) -> BigQuery Client -> Google Cloud BigQuery
```

## Setup Requirements

### 1. Google Cloud Project Setup

1. **Create a Google Cloud Project** (if not already exists)
2. **Enable BigQuery API** in the Google Cloud Console
3. **Create a BigQuery dataset** for your data
4. **Create tables** in the dataset with appropriate schemas

### 2. Service Account Setup

1. **Create a Service Account** in Google Cloud Console:

   - Go to IAM & Admin > Service Accounts
   - Click "Create Service Account"
   - Name: `truss-bigquery-service`
   - Description: "Service account for Truss Data Service Layer BigQuery access"

2. **Generate Service Account Key**:

   - Click on the created service account
   - Go to "Keys" tab
   - Click "Add Key" > "Create new key"
   - Choose "JSON" format
   - Download the JSON key file

3. **Store Credentials in AWS Secrets Manager**:
   ```bash
   aws secretsmanager create-secret \
     --name "bigquery-service-account" \
     --description "GCP Service Account credentials for BigQuery access" \
     --secret-string file://path/to/service-account-key.json
   ```

### 3. Required Permissions

The service account needs the following IAM roles:

#### BigQuery Permissions

- **BigQuery Data Viewer** (`roles/bigquery.dataViewer`)

  - `bigquery.datasets.get`
  - `bigquery.tables.get`
  - `bigquery.tables.list`
  - `bigquery.tables.getData`

- **BigQuery Job User** (`roles/bigquery.jobUser`)
  - `bigquery.jobs.create`
  - `bigquery.jobs.get`

#### Optional Permissions (for advanced features)

- **BigQuery Data Editor** (`roles/bigquery.dataEditor`) - if you need write access
- **BigQuery Admin** (`roles/bigquery.admin`) - for full administrative access

#### AWS Lambda Permissions

- **SecretsManagerReadWrite** (`SecretsManagerReadWrite`)
  - `secretsmanager:GetSecretValue`
  - `secretsmanager:DescribeSecret`

### 4. Environment Variables

Set the following environment variables in your Lambda function:

```bash
# Required
GCP_SECRET_NAME=bigquery-service-account
GCP_PROJECT_ID=your-gcp-project-id
AWS_REGION=eu-west-2

# Optional
BIGQUERY_DATASET=your_default_dataset
BIGQUERY_TIMEOUT=30000
BIGQUERY_USE_CACHE=true
```

## Configuration

### Service Configuration

Update your service's `config.json` to use BigQuery:

```json
{
  "database": {
    "required": true,
    "connection_type": "bigquery",
    "permissions": ["read"],
    "project_id": "truss-data-science",
    "dataset": "api",
    "table": "display_product_listings",
    "gcp_secret_name": "bigquery-service-account"
  }
}
```

### Table Configuration

For partitioned services, use root_type filtering instead of separate tables:

```json
{
  "api": {
    "partitions": {
      "routes": {
        "bags": {
          "table": "display_product_listings",
          "default_filters": { "root_type": "30" },
          "hosting_type": "BIGQUERY"
        },
        "apparel": {
          "table": "display_product_listings",
          "default_filters": { "root_type": "114" },
          "hosting_type": "BIGQUERY"
        },
        "footwear": {
          "table": "display_product_listings",
          "default_filters": { "root_type": "5" },
          "hosting_type": "BIGQUERY"
        }
      }
    }
  }
}
```

### Root Type Mappings

The following root_type IDs are used for filtering:

- `5`: Footwear
- `6`: Accessories
- `29`: Headwear
- `30`: Bags
- `45`: Eyewear
- `114`: Clothing
- `113`: All (no filtering)

## Usage Examples

### Basic Query

```javascript
const { query } = require("./utils");

// Simple aggregation query
const results = await query(
  "SELECT brand, SUM(sold_price) as total_gmv FROM `project.dataset.table` GROUP BY brand",
  [],
  config,
  { useCache: true }
);
```

### Parameterized Query

```javascript
// Using parameterized queries for security
const results = await query(
  "SELECT * FROM `project.dataset.table` WHERE brand = @brand AND sold_date >= @start_date",
  ["Gucci", "2024-01-01"],
  config
);
```

### Market Share Query

```javascript
const { buildMarketShareQuery } = require("./utils/bigquery-query-builder");

const { sql, args } = buildMarketShareQuery(
  { group_by: "brand", brands: "Gucci,Chanel" },
  "sold",
  "sold_price",
  "sold_date",
  config
);

const results = await query(sql, args, config);
```

### Text Search Query

```javascript
// Search for products containing specific keywords in listing titles
const results = await query(
  "SELECT * FROM `project.dataset.table` WHERE SEARCH(listing_title, @keyword)",
  ["Air max"],
  config
);

// Or using the query builder with key_words parameter
const { buildCompleteQuery } = require("./utils/bigquery-query-builder");
const { sql, args } = buildCompleteQuery(
  { key_words: "Air max,Nike" },
  "SUM(sold_price) AS value",
  "sold",
  "sold_price",
  "DESC",
  500,
  0,
  "sold_date",
  config
);
```

## BigQuery-Specific Features

### 1. Query Caching

BigQuery automatically caches query results for 24 hours. Enable/disable caching:

```javascript
const results = await query(sql, args, config, { useCache: false });
```

### 2. Query Timeout

Set custom timeout for long-running queries:

```javascript
const results = await query(sql, args, config, { timeout: 60000 }); // 60 seconds
```

### 3. Dry Run

Test queries without executing them:

```javascript
const results = await query(sql, args, config, { dryRun: true });
```

### 4. Result Limits

Limit the number of results returned:

```javascript
const results = await query(sql, args, config, { limit: 1000 });
```

## Data Schema Requirements

### Required Columns

Your BigQuery tables should include these columns for full compatibility:

```sql
CREATE TABLE `truss-data-science.api.display_product_listings` (
  -- Product identification
  brand STRING,
  model STRING,
  colour STRING,  -- Note: 'colour' not 'color'
  material STRING,
  hardware STRING,
  type STRING,    -- Used for both 'type' and 'shape'
  size STRING,

  -- Product attributes
  condition STRING,
  root_type INT64,  -- Integer ID, not string
  vendor STRING,
  gender STRING,
  decade STRING,

  -- Location
  sold_location STRING,  -- Note: 'sold_location' not 'sale_location_country'

  -- Pricing
  sold_price FLOAT64,
  listed_price FLOAT64,

  -- Dates
  sold_date DATE,
  listed_date DATE,

  -- Status
  is_sold BOOLEAN,

  -- Additional fields
  product_id STRING,
  listing_id STRING,
  listing_title STRING,  -- For text search functionality
)
PARTITION BY DATE(sold_date)
CLUSTER BY brand, model;
```

### Recommended Optimizations

1. **Partitioning**: Partition large tables by date (e.g., `sold_date`)
2. **Clustering**: Cluster by frequently queried columns (e.g., `brand`, `model`)
3. **Data Types**: Use appropriate BigQuery data types for better performance
4. **Nested Fields**: Consider using nested fields for complex data structures

## Error Handling

The BigQuery integration includes comprehensive error handling:

### Retryable Errors

- Rate limit exceeded
- Query timeout
- Service unavailable
- Internal errors

### Non-Retryable Errors

- Authentication errors
- Permission denied
- Invalid query syntax
- Table not found

### Error Response Format

```json
{
  "error": "BigQuery error message",
  "code": "ERROR_CODE",
  "service": "analytics",
  "timestamp": "2024-01-01T00:00:00.000Z",
  "query": "SELECT ...",
  "retryable": true
}
```

## Performance Considerations

### 1. Query Optimization

- Use `LIMIT` to reduce data transfer
- Filter early with `WHERE` clauses
- Use appropriate data types
- Leverage partitioning and clustering

### 2. Caching

- Enable query caching for repeated queries
- Use `useQueryCache: true` (default)

### 3. Cost Optimization

- Use `dryRun: true` to estimate costs
- Monitor query costs in BigQuery console
- Set appropriate limits on query results

### 4. Lambda Configuration

- Increase memory for complex queries (512MB+)
- Set appropriate timeout (60s+)
- Consider using provisioned concurrency for consistent performance

## Monitoring and Logging

### CloudWatch Logs

The integration logs:

- Query execution times
- Error details
- Retry attempts
- Cache hits/misses

### BigQuery Console

Monitor in BigQuery console:

- Query history
- Job details
- Performance metrics
- Cost analysis

### Custom Metrics

You can add custom CloudWatch metrics for:

- Query execution time
- Error rates
- Cache hit rates
- Data volume processed

## Migration from MySQL

### 1. Data Migration

Use BigQuery Data Transfer Service or custom scripts to migrate data from MySQL to BigQuery.

### 2. Query Migration

- Update SQL syntax for BigQuery compatibility
- Replace MySQL-specific functions with BigQuery equivalents
- Update date/time functions
- Adjust parameter binding syntax

### 3. Configuration Updates

- Change `connection_type` from `"proxy"` to `"bigquery"`
- Update table references
- Add BigQuery-specific configuration

### 4. Testing

- Test all endpoints with BigQuery
- Verify data consistency
- Performance testing
- Error handling validation

## Troubleshooting

### Common Issues

1. **Authentication Errors**

   - Verify service account key is correct
   - Check AWS Secrets Manager secret
   - Ensure IAM permissions are correct

2. **Permission Denied**

   - Verify BigQuery IAM roles
   - Check dataset/table permissions
   - Ensure service account has access

3. **Query Timeouts**

   - Increase Lambda timeout
   - Optimize query performance
   - Use query caching

4. **Memory Issues**
   - Increase Lambda memory allocation
   - Use `LIMIT` to reduce result size
   - Consider pagination

### Debug Mode

Enable debug logging:

```javascript
process.env.DEBUG = "bigquery:*";
```

### Health Checks

Use the health check endpoint to verify BigQuery connectivity:

```bash
curl https://your-api-gateway-url/analytics/health
```

## Security Best Practices

1. **Credential Management**

   - Store credentials in AWS Secrets Manager
   - Rotate service account keys regularly
   - Use least privilege principle

2. **Query Security**

   - Use parameterized queries
   - Validate input parameters
   - Implement query timeouts

3. **Data Access**

   - Use row-level security in BigQuery
   - Implement column-level permissions
   - Audit data access

4. **Network Security**
   - Use VPC endpoints if needed
   - Implement proper CORS policies
   - Use HTTPS for all communications

## Support and Maintenance

### Regular Tasks

- Monitor query performance
- Review and optimize costs
- Update service account keys
- Monitor error rates

### Updates

- Keep BigQuery client library updated
- Monitor BigQuery service updates
- Update documentation as needed

### Backup and Recovery

- BigQuery provides automatic backups
- Implement data retention policies
- Test disaster recovery procedures

## Additional Resources

- [BigQuery Documentation](https://cloud.google.com/bigquery/docs)
- [BigQuery SQL Reference](https://cloud.google.com/bigquery/docs/reference/standard-sql)
- [BigQuery Best Practices](https://cloud.google.com/bigquery/docs/best-practices)
- [AWS Secrets Manager Documentation](https://docs.aws.amazon.com/secretsmanager/)
