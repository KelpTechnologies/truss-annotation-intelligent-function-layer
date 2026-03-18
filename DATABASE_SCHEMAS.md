# Database Schemas

Canonical schema reference for BigQuery and Cloud SQL Postgres tables used by AIFL services.
Both databases share identical schemas. Source of truth: `rs-export-schema.sql` in this repo.

## Stage-to-Schema/Dataset Mapping

| Stage   | Postgres Schema | BigQuery Dataset                          |
|---------|-----------------|-------------------------------------------|
| dev     | `api_dev`       | `truss-data-science.api_Dev`              |
| staging | `api_staging`   | `truss-data-science.api_staging`          |
| prod    | `api`           | `truss-data-science.api`                  |
| legacy  | `api_legacy`    | `truss-data-science.api_legacy`           |

Postgres instance: `truss-data-science:europe-west2:truss-api-postgres`, database `truss-api`.

### Which tables does AIFL use?

| Table                        | Used by                                              |
|------------------------------|------------------------------------------------------|
| model_knowledge_display      | `bigquery_taxonomy_lookup.py`, `postgres_client.py`, `bigquery_utils.py` |
| material_knowledge_display   | `bigquery_taxonomy_lookup.py`, `postgres_client.py`  |
| brand_knowledge_display      | `bigquery_brand_tool.py`                             |
| model_size_knowledge_display | `bigquery_model_size_tool.py`                        |
| display_product_listings     | `bigquery-query-builder.js` (image-service)          |

---

## BigQuery

Project: `truss-data-science`

All tables below live under the dataset determined by stage (see mapping above).
Example fully-qualified name: `` `truss-data-science.api_staging.display_product_listings` ``

### display_product_listings

Main listings table, deduped on `listing_uuid`.

| Column                     | Type      | Notes                        |
|----------------------------|-----------|------------------------------|
| listing_uuid               | STRING    | Primary key                  |
| model                      | STRING    |                              |
| parent_model               | STRING    |                              |
| root_model                 | STRING    |                              |
| colour                     | STRING    |                              |
| material                   | STRING    |                              |
| parent_material            | STRING    |                              |
| root_material              | STRING    |                              |
| hardware_material          | STRING    |                              |
| parent_hardware_material   | STRING    |                              |
| root_hardware_material     | STRING    |                              |
| type                       | STRING    |                              |
| root_type                  | STRING    |                              |
| condition                  | STRING    |                              |
| gender                     | STRING    |                              |
| sold_location              | STRING    |                              |
| continent                  | STRING    |                              |
| sold_date                  | TIMESTAMP |                              |
| listed_date                | TIMESTAMP |                              |
| vendor                     | STRING    |                              |
| size                       | STRING    |                              |
| eu_size                    | FLOAT64   |                              |
| uk_size                    | FLOAT64   |                              |
| us_size                    | FLOAT64   |                              |
| jp_size                    | FLOAT64   |                              |
| sold_price                 | FLOAT64   |                              |
| listed_price               | FLOAT64   |                              |
| is_sold                    | BOOLEAN   |                              |
| product_link               | STRING    |                              |
| listing_title              | STRING    | Used for keyword/text search |
| primary_image_link         | STRING    |                              |
| backup_primary_image_link  | STRING    |                              |

### model_knowledge_display

| Column          | Type    |
|-----------------|---------|
| model_id        | INT64   |
| model           | STRING  |
| parent_model_id | INT64   |
| parent_model    | STRING  |
| root_model_id   | INT64   |
| root_model      | STRING  |
| type            | STRING  |
| type_id         | INT64   |
| brand           | STRING  |
| brand_id        | INT64   |
| is_phantom      | INT64   |

### material_knowledge_display

Also serves as the source for "hardware" entities — filter on `material_bag_category`:
- **Materials**: `material_bag_category IN ('Exterior', 'Both')`
- **Hardwares**: `material_bag_category IN ('Hardware', 'Both')`

| Column                | Type   |
|-----------------------|--------|
| material_id           | INT64  |
| material              | STRING |
| parent_material_id    | INT64  |
| parent_material       | STRING |
| root_material_id      | INT64  |
| root_material         | STRING |
| material_bag_category | STRING |

### colour_knowledge_display

| Column                   | Type   |
|--------------------------|--------|
| colour_id                | INT64  |
| colour                   | STRING |
| base_colour              | STRING |
| base_colour_display_name | STRING |

### type_knowledge_display

| Column        | Type   |
|---------------|--------|
| id            | INT64  |
| type          | STRING |
| root_type_id  | INT64  |
| root_type_name| STRING |

### brand_knowledge_display

| Column | Type   |
|--------|--------|
| id     | INT64  |
| brand  | STRING |

### listing_brand_links

| Column       | Type   |
|--------------|--------|
| listing_uuid | STRING |
| brand        | STRING |
| brand_id     | INT64  |

### model_size_knowledge_display

| Column   | Type    |
|----------|---------|
| id       | INT64   |
| model    | STRING  |
| model_id | INT64   |
| brand    | STRING  |
| brand_id | INT64   |
| size     | STRING  |
| height   | NUMERIC |
| width    | NUMERIC |
| depth    | NUMERIC |

### size_knowledge_display

| Column        | Type   |
|---------------|--------|
| size_id       | INT64  |
| root_type     | STRING |
| gender        | STRING |
| country       | STRING |
| root_size     | STRING |
| size          | STRING |
| parent_size   | STRING |
| parent_size_id| INT64  |

### condition_knowledge_display

| Column       | Type   |
|--------------|--------|
| condition_id | INT64  |
| condition    | STRING |

### vendor_knowledge_display

| Column    | Type   |
|-----------|--------|
| vendor_id | INT64  |
| vendor    | STRING |

### gender_knowledge_display

| Column    | Type   |
|-----------|--------|
| gender_id | INT64  |
| gender    | STRING |

### location_knowledge_display

| Column         | Type   |
|----------------|--------|
| location_id    | INT64  |
| country_code   | STRING |
| country_name   | STRING |
| continent_code | STRING |
| continent_name | STRING |

---

## Postgres (Cloud SQL)

Instance: `truss-data-science:europe-west2:truss-api-postgres`
Database: `truss-api`

All tables below live under the schema determined by stage (see mapping above).
Example fully-qualified name: `api_staging.display_product_listings`

### display_product_listings

Main listings table, deduped on `listing_uuid`.

| Column                     | Type             | Notes                        |
|----------------------------|------------------|------------------------------|
| listing_uuid               | text             | Primary key                  |
| model                      | text             |                              |
| parent_model               | text             |                              |
| root_model                 | text             |                              |
| colour                     | text             |                              |
| material                   | text             |                              |
| parent_material            | text             |                              |
| root_material              | text             |                              |
| hardware_material          | text             |                              |
| parent_hardware_material   | text             |                              |
| root_hardware_material     | text             |                              |
| type                       | text             |                              |
| root_type                  | text             |                              |
| condition                  | text             |                              |
| gender                     | text             |                              |
| sold_location              | text             |                              |
| continent                  | text             |                              |
| sold_date                  | timestamptz      |                              |
| listed_date                | timestamptz      |                              |
| vendor                     | text             |                              |
| size                       | text             |                              |
| eu_size                    | double precision |                              |
| uk_size                    | double precision |                              |
| us_size                    | double precision |                              |
| jp_size                    | double precision |                              |
| sold_price                 | double precision |                              |
| listed_price               | double precision |                              |
| is_sold                    | boolean          |                              |
| product_link               | text             |                              |
| listing_title              | text             | Used for keyword/text search |
| primary_image_link         | text             |                              |
| backup_primary_image_link  | text             |                              |

### model_knowledge_display

| Column          | Type     |
|-----------------|----------|
| model_id        | bigint   |
| model           | text     |
| parent_model_id | bigint   |
| parent_model    | text     |
| root_model_id   | bigint   |
| root_model      | text     |
| type            | text     |
| type_id         | integer  |
| brand           | text     |
| brand_id        | integer  |
| is_phantom      | smallint |

### material_knowledge_display

Also serves as the source for "hardware" entities — filter on `material_bag_category`:
- **Materials**: `material_bag_category IN ('Exterior', 'Both')`
- **Hardwares**: `material_bag_category IN ('Hardware', 'Both')`

| Column                | Type   |
|-----------------------|--------|
| material_id           | bigint |
| material              | text   |
| parent_material_id    | bigint |
| parent_material       | text   |
| root_material_id      | bigint |
| root_material         | text   |
| material_bag_category | text   |

### colour_knowledge_display

| Column                   | Type   |
|--------------------------|--------|
| colour_id                | bigint |
| colour                   | text   |
| base_colour              | text   |
| base_colour_display_name | text   |

### type_knowledge_display

| Column        | Type    |
|---------------|---------|
| id            | integer |
| type          | text    |
| root_type_id  | integer |
| root_type_name| text    |

### brand_knowledge_display

| Column | Type    |
|--------|---------|
| id     | integer |
| brand  | text    |

### listing_brand_links

| Column       | Type    |
|--------------|---------|
| listing_uuid | text    |
| brand        | text    |
| brand_id     | integer |

### model_size_knowledge_display

| Column   | Type    |
|----------|---------|
| id       | bigint  |
| model    | text    |
| model_id | integer |
| brand    | text    |
| brand_id | integer |
| size     | text    |
| height   | numeric |
| width    | numeric |
| depth    | numeric |

### size_knowledge_display

| Column        | Type    |
|---------------|---------|
| size_id       | bigint  |
| root_type     | text    |
| gender        | text    |
| country       | text    |
| root_size     | text    |
| size          | text    |
| parent_size   | text    |
| parent_size_id| integer |

### condition_knowledge_display

| Column       | Type   |
|--------------|--------|
| condition_id | bigint |
| condition    | text   |

### vendor_knowledge_display

| Column    | Type   |
|-----------|--------|
| vendor_id | bigint |
| vendor    | text   |

### gender_knowledge_display

| Column    | Type   |
|-----------|--------|
| gender_id | bigint |
| gender    | text   |

### location_knowledge_display

| Column         | Type   |
|----------------|--------|
| location_id    | bigint |
| country_code   | text   |
| country_name   | text   |
| continent_code | text   |
| continent_name | text   |
