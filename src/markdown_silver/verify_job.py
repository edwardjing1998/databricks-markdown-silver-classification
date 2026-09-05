from pyspark.sql import SparkSession, functions as F
from markdown_silver.config import parse_settings


def main() -> None:
    s = parse_settings()
    spark = SparkSession.builder.getOrCreate()
    table = spark.table(s.sections_table)
    duplicates = table.groupBy("section_id").count().filter("count > 1").count()
    failures = table.filter(F.col("classification_status") == "FAILED").count()
    total = table.count()
    print(f"Silver verification: total={total}, duplicate_ids={duplicates}, classification_failures={failures}")
    if duplicates:
        raise RuntimeError("Silver verification failed: duplicate section IDs")


if __name__ == "__main__":
    main()
