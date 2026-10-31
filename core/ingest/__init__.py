"""Document ingestion schemas, contracts, and pipelines."""

from core.ingest.pipeline import IngestionPipeline, create_default_ingestion_pipeline
from core.ingest.schema import IngestionResult, IngestionStatus

__all__ = [
    "IngestionPipeline",
    "IngestionResult",
    "IngestionStatus",
    "create_default_ingestion_pipeline",
]
