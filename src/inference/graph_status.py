import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = (
    PROJECT_ROOT / "data" / "processed" / "graph_v3"
    / "graph_v3_manifest.json"
)


def resolve_project_path(value):
    path = Path(value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def get_graph_v3_status():
    """
    Read Graph V3 manifest and check artifact presence.

    Does not deserialize the graph tensor or scan the full evidence file.
    Validation values are reported as recorded in the manifest; they are
    not recomputed by this endpoint.
    """
    if not MANIFEST_PATH.is_file():
        return {
            "status": "missing",
            "ready": False,
            "message": "Graph V3 manifest was not found.",
            "prediction_available": False,
        }

    try:
        manifest = json.loads(
            MANIFEST_PATH.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "status": "invalid_manifest",
            "ready": False,
            "message": str(exc),
            "prediction_available": False,
        }

    graph_value = manifest.get(
        "output_graph",
        "data/processed/graph_v3/typed_graph_v3.pt",
    )
    source_value = manifest.get("source_evidence_extract")

    graph_path = resolve_project_path(graph_value)
    source_path = (
        resolve_project_path(source_value)
        if source_value else None
    )

    graph_exists = graph_path.is_file()
    source_exists = bool(source_path and source_path.is_file())

    validation = manifest.get("validation", {})
    recorded_validation_passed = (
        isinstance(validation, dict)
        and bool(validation)
        and all(value is True for value in validation.values())
    )

    ready = (
        graph_exists
        and source_exists
        and recorded_validation_passed
    )

    return {
        "status": "ready" if ready else "incomplete",
        "ready": ready,
        "graph_version": manifest.get("schema_version"),
        "graph_file": str(graph_path.relative_to(PROJECT_ROOT))
            if graph_path.is_relative_to(PROJECT_ROOT)
            else str(graph_path),
        "graph_file_exists": graph_exists,
        "graph_file_size_bytes": (
            graph_path.stat().st_size if graph_exists else None
        ),
        "manifest_file": str(MANIFEST_PATH.relative_to(PROJECT_ROOT)),
        "source_evidence_file": (
            str(source_path.relative_to(PROJECT_ROOT))
            if source_path and source_path.is_relative_to(PROJECT_ROOT)
            else str(source_path) if source_path else None
        ),
        "source_evidence_exists": source_exists,
        "source_evidence_size_bytes": (
            source_path.stat().st_size if source_exists else None
        ),
        "original_node_types_preserved": len(
            manifest.get("original_node_counts_preserved", {})
        ),
        "original_edge_types_preserved": len(
            manifest.get("original_edge_counts_preserved", {})
        ),
        "intergenic_node_type": manifest.get("new_node_type"),
        "intergenic_observations": manifest.get("new_node_count"),
        "new_relation": manifest.get("new_edge_type"),
        "new_relation_edges": manifest.get("new_edge_count"),
        "matched_samples": manifest.get("sample_nodes_matched"),
        "source_rows": manifest.get("source_rows"),
        "coordinate_reference": manifest.get("coordinate_reference"),
        "coordinate_convention": manifest.get("coordinate_convention"),
        "recorded_validation": validation,
        "recorded_validation_passed": recorded_validation_passed,
        "prediction_available": False,
        "note": (
            "This endpoint checks file presence and reports manifest "
            "metadata. It does not reload or independently revalidate "
            "the graph."
        ),
    }
