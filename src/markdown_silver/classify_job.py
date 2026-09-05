from __future__ import annotations

import json

from delta.tables import DeltaTable
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from markdown_silver.config import parse_settings


def main() -> None:
    settings = parse_settings()
    spark = SparkSession.builder.getOrCreate()

    sections = spark.table(settings.sections_table)

    # Process:
    # 1. New PENDING records.
    # 2. Records previously left as RULE_ONLY.
    # 3. Previous AI failures.
    # 4. AI records produced using an older model or prompt.
    needs_refresh = (
        F.col("classification_status").isin(
            "PENDING",
            "RULE_ONLY",
            "FAILED",
        )
        |
        (
            (F.col("classification_method") == "AI")
            &
            (
                (
                    F.coalesce(
                        F.col("model_endpoint"),
                        F.lit(""),
                    )
                    != F.lit(settings.model_endpoint)
                )
                |
                (
                    F.coalesce(
                        F.col("prompt_version"),
                        F.lit(""),
                    )
                    != F.lit(settings.prompt_version)
                )
            )
        )
    )

    # Only ambiguous rule classifications are sent to AI.
    pending = sections.filter(
        needs_refresh
        & (
            F.col("rule_confidence")
            < F.lit(settings.confidence_threshold)
        )
    )

    # High-confidence rule results do not need AI.
    spark.sql(
        f"""
        UPDATE {settings.sections_table}
        SET
          classification_status = 'CLASSIFIED',
          updated_at = current_timestamp()
        WHERE classification_method = 'RULE'
          AND rule_confidence >= {settings.confidence_threshold}
          AND classification_status IN (
            'PENDING',
            'RULE_ONLY'
          )
        """
    )

    if pending.limit(1).count() == 0:
        print("No ambiguous sections require AI classification.")
        return

    if not settings.enable_ai:
        spark.sql(
            f"""
            UPDATE {settings.sections_table}
            SET
              classification_status = 'RULE_ONLY',
              updated_at = current_timestamp()
            WHERE classification_status IN (
              'PENDING',
              'FAILED'
            )
              AND rule_confidence < {settings.confidence_threshold}
            """
        )

        print(
            "AI classification is disabled. "
            "Ambiguous sections remain classified by rules only."
        )
        return

    allowed_types = (
        "TITLE, CHAPTER_HEADING, DEFINITION, THEOREM, "
        "EXAMPLE, EXERCISE, EXPLANATION, ANSWER, OTHER"
    )

    prompt = F.concat(
        F.lit(
            "Classify this school mathematics Markdown section. "
            "Return only data matching the required response schema. "
            "The section_type must be one of: "
            + allowed_types
            + ". "
            "The confidence value must be between 0 and 1."
            "\nTITLE: "
        ),
        F.coalesce(
            F.col("title"),
            F.lit(""),
        ),
        F.lit("\nCONTENT:\n"),
        F.substring(
            F.col("content"),
            1,
            12000,
        ),
    )

    # Databricks DDL structured output requires exactly one top-level field.
    # The six classification values are therefore nested under
    # the single top-level field named "classification".
    output_schema = (
        "STRUCT<classification:STRUCT<"
        "section_type:STRING,"
        "subject:STRING,"
        "topic:STRING,"
        "difficulty:STRING,"
        "language:STRING,"
        "confidence:DOUBLE"
        ">>"
    )

    queried = (
        pending
        .withColumn(
            "prompt",
            prompt,
        )
        .withColumn(
            "raw_result",
            F.expr(
                f"""
                ai_query(
                  '{settings.model_endpoint}',
                  prompt,
                  responseFormat => '{output_schema}',
                  failOnError => false
                )
                """
            ),
        )
    )

    # failOnError=false returns:
    #
    # raw_result.response.classification
    # raw_result.errorMessage
    classified = (
        queried
        .select(
            "section_id",
            "raw_result.response.classification.*",
            "raw_result.errorMessage",
        )
        .withColumn(
            "now",
            F.current_timestamp(),
        )
    )

    target = DeltaTable.forName(
        spark,
        settings.sections_table,
    )

    (
        target.alias("target")
        .merge(
            classified.alias("source"),
            "target.section_id = source.section_id",
        )
        .whenMatchedUpdate(
            set={
                "section_type": (
                    "coalesce("
                    "source.section_type, "
                    "target.section_type"
                    ")"
                ),
                "subject": "source.subject",
                "topic": "source.topic",
                "difficulty": "source.difficulty",
                "language": "source.language",
                "classification_confidence": (
                    "coalesce("
                    "source.confidence, "
                    "target.classification_confidence"
                    ")"
                ),
                "classification_method": "'AI'",
                "model_endpoint": json.dumps(
                    settings.model_endpoint
                ),
                "prompt_version": json.dumps(
                    settings.prompt_version
                ),
                "classification_status": (
                    "CASE "
                    "WHEN source.section_type IS NULL "
                    "THEN 'FAILED' "
                    "ELSE 'CLASSIFIED' "
                    "END"
                ),
                "classification_error": "source.errorMessage",
                "updated_at": "source.now",
            }
        )
        .execute()
    )

    total_ai = (
        spark.table(settings.sections_table)
        .filter(F.col("classification_method") == "AI")
        .count()
    )

    failed_ai = (
        spark.table(settings.sections_table)
        .filter(F.col("classification_status") == "FAILED")
        .count()
    )

    print(
        "AI classification completed: "
        f"AI rows={total_aiapha}, "
        f"failed rows={failed_ai}"
    )


if __name__ == "__main__":
    main()