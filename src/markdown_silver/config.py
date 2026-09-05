from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    catalog: str
    bronze_schema: str
    silver_schema: str
    model_endpoint: str
    enable_ai: bool
    confidence_threshold: float
    prompt_version: str

    @property
    def bronze_table(self) -> str:
        return f"{self.catalog}.{self.bronze_schema}.markdown_documents"

    @property
    def sections_table(self) -> str:
        return f"{self.catalog}.{self.silver_schema}.document_sections"


def parse_settings() -> Settings:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", default="education_rag")
    parser.add_argument("--bronze-schema", default="bronze")
    parser.add_argument("--silver-schema", default="silver")
    parser.add_argument("--model-endpoint", default="system.ai.gpt-oss-20b")
    parser.add_argument("--enable-ai", default="true")
    parser.add_argument("--confidence-threshold", type=float, default=0.85)
    parser.add_argument("--prompt-version", default="v1")
    args = parser.parse_args()
    return Settings(
        args.catalog, args.bronze_schema, args.silver_schema,
        args.model_endpoint, args.enable_ai.lower() == "true",
        args.confidence_threshold, args.prompt_version,
    )
