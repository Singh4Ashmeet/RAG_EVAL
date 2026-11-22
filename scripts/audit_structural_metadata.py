"""Audit existing structural metadata across all corpus documents and generate machine-readable inventory."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from core.ingest import create_default_ingestion_pipeline
from core.retrieval.structural.signals import (
    build_chunk_structural_metadata,
)
from eval.routing.generator import DEFAULT_DOCUMENT_SPECS


def generate_metadata_inventory() -> dict[str, Any]:
    pipeline = create_default_ingestion_pipeline(chunk_size=1000, chunk_overlap=200)

    inventory: dict[str, Any] = {
        "metadata_schema_fields": {
            "document_level": [
                "document_id",
                "filename",
                "mime_type",
                "content_sha256",
                "byte_size",
                "created_at",
                "processed_at",
                "source_uri",
                "custom_metadata",
                "page_count",
            ],
            "element_level": [
                "element_id",
                "element_type",
                "content",
                "order_index",
                "location (page_number, bbox, coordinates)",
                "table_data (headers, rows, caption)",
                "metadata (e.g. style, confidence)",
            ],
            "chunk_level": [
                "chunk_id",
                "document_id",
                "text",
                "chunk_index",
                "token_count",
                "source_element_ids",
                "locations",
                "metadata",
            ],
            "derived_structural_signals": [
                "document_title",
                "section_path",
                "heading_hierarchy",
                "table_headers",
                "table_caption",
                "element_types_present",
                "page_numbers",
                "is_table",
                "is_heading",
                "is_list",
                "neighbor_element_count",
            ],
        },
        "corpus_statistics": {
            "total_documents": len(DEFAULT_DOCUMENT_SPECS),
            "total_chunks": 0,
            "total_elements": 0,
            "element_type_counts": {},
            "chunks_with_heading_context": 0,
            "chunks_with_table_data": 0,
            "chunks_with_page_location": 0,
            "chunks_with_list_items": 0,
            "chunks_multi_element": 0,
        },
        "documents": [],
    }

    total_chunks = 0
    total_elements = 0
    elem_type_counts: dict[str, int] = {}
    chunks_with_heading = 0
    chunks_with_table = 0
    chunks_with_page = 0
    chunks_with_list = 0
    chunks_multi_elem = 0

    for spec in DEFAULT_DOCUMENT_SPECS:
        p = _project_root / spec.relative_path
        if not p.exists():
            continue
        res = pipeline.ingest(p)
        doc = res.document
        chunks = res.chunks

        if not doc or not chunks:
            continue

        doc_chunks = len(chunks)
        doc_elems = len(doc.elements)
        total_chunks += doc_chunks
        total_elements += doc_elems

        for elem in doc.elements:
            et = elem.element_type.value
            elem_type_counts[et] = elem_type_counts.get(et, 0) + 1

        doc_headings = []
        doc_tables = 0
        doc_pages = doc.page_count or (
            max(
                (
                    e.location.page_number
                    for e in doc.elements
                    if e.location and e.location.page_number
                ),
                default=None,
            )
        )

        for chunk in chunks:
            meta = build_chunk_structural_metadata(chunk, doc)
            if meta.section_path:
                chunks_with_heading += 1
            if meta.has_table:
                chunks_with_table += 1
                doc_tables += 1
            if meta.page_numbers:
                chunks_with_page += 1
            if meta.has_list:
                chunks_with_list += 1
            if len(chunk.source_element_ids) > 1:
                chunks_multi_elem += 1

            if meta.section_path and meta.section_path[-1] not in doc_headings:
                doc_headings.append(meta.section_path[-1])

        inventory["documents"].append(
            {
                "filename": spec.filename,
                "relative_path": spec.relative_path,
                "split": spec.split.value,
                "document_id": doc.metadata.document_id,
                "page_count": doc_pages,
                "element_count": doc_elems,
                "chunk_count": doc_chunks,
                "headings_sample": doc_headings[:5],
                "has_tables": doc_tables > 0,
            }
        )

    inventory["corpus_statistics"]["total_chunks"] = total_chunks
    inventory["corpus_statistics"]["total_elements"] = total_elements
    inventory["corpus_statistics"]["element_type_counts"] = elem_type_counts
    inventory["corpus_statistics"]["chunks_with_heading_context"] = chunks_with_heading
    inventory["corpus_statistics"]["chunks_with_table_data"] = chunks_with_table
    inventory["corpus_statistics"]["chunks_with_page_location"] = chunks_with_page
    inventory["corpus_statistics"]["chunks_with_list_items"] = chunks_with_list
    inventory["corpus_statistics"]["chunks_multi_element"] = chunks_multi_elem

    return inventory


def main() -> int:
    print("Auditing structural metadata across corpus...")
    inv = generate_metadata_inventory()
    out_path = _project_root / "docs" / "reports" / "phase17_metadata_inventory.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(inv, indent=2), encoding="utf-8")
    print(f"Saved metadata inventory to {out_path}")
    print(f"Total documents: {inv['corpus_statistics']['total_documents']}")
    print(f"Total chunks   : {inv['corpus_statistics']['total_chunks']}")
    print(f"Element types  : {inv['corpus_statistics']['element_type_counts']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
