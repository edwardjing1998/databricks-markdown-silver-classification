from pyspark.sql import SparkSession
from markdown_silver.config import parse_settings


def main() -> None:
    s = parse_settings()
    spark = SparkSession.builder.getOrCreate()
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {s.catalog}.{s.silver_schema}")
    spark.sql(f"""
      CREATE TABLE IF NOT EXISTS {s.sections_table} (
        section_id STRING NOT NULL, document_id STRING NOT NULL,
        book_id STRING, chapter_id STRING, page_id STRING, page_number BIGINT,
        section_number INT, heading_level INT, title STRING, content STRING,
        image_links ARRAY<STRING>, image_references ARRAY<STRUCT<figure_id: STRING,
          original_link: STRING, blob_path: STRING, file_name: STRING,
          content_type: STRING, file_size: BIGINT, content_hash: STRING,
          source_modified_at: TIMESTAMP>>,
        section_type STRING, subject STRING, topic STRING, difficulty STRING,
        language STRING, rule_confidence DOUBLE, classification_confidence DOUBLE,
        classification_method STRING, model_endpoint STRING,
        prompt_version STRING, source_content_hash STRING,
        classification_status STRING, classification_error STRING,
        created_at TIMESTAMP, updated_at TIMESTAMP,
        CONSTRAINT document_sections_pk PRIMARY KEY (section_id) NOT ENFORCED
      ) USING DELTA TBLPROPERTIES (delta.enableChangeDataFeed = true)
    """)
    print(f"Silver table is ready: {s.sections_table}")


if __name__ == "__main__":
    main()
