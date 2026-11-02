import json
import sqlite3
from pathlib import Path
from urllib.parse import quote

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATABASE_PATH = (
    PROJECT_ROOT / "data" / "processed" / "graph_v3"
    / "intergenic_evidence.sqlite"
)

INDEX_MANIFEST_PATH = (
    PROJECT_ROOT / "data" / "processed" / "graph_v3"
    / "intergenic_evidence_index_manifest.json"
)

MAX_PAGE_SIZE = 200


def _open_readonly_database():
    if not DATABASE_PATH.is_file():
        raise FileNotFoundError(
            "Intergenic evidence index has not been built."
        )

    if not INDEX_MANIFEST_PATH.is_file():
        raise FileNotFoundError(
            "Intergenic evidence index manifest is missing."
        )

    try:
        manifest = json.loads(
            INDEX_MANIFEST_PATH.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            "Intergenic evidence index manifest is invalid."
        ) from exc

    validation = manifest.get("validation", {})
    if (
        not isinstance(validation, dict)
        or not validation
        or not all(value is True for value in validation.values())
    ):
        raise RuntimeError(
            "Intergenic evidence index validation is incomplete."
        )

    database_uri = (
        "file:" + quote(str(DATABASE_PATH), safe="/") + "?mode=ro"
    )

    connection = sqlite3.connect(
        database_uri,
        uri=True,
        timeout=10,
    )
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only = ON")
    return connection


def get_sample_intergenic_evidence(
    sample_id: str,
    limit: int = 50,
    offset: int = 0,
):
    """
    Return one sample's intergenic observations in source-row order.

    The database is opened in read-only mode. Both pagination values
    are bounded, and all SQL values are parameterized.
    """
    sample_id = sample_id.strip()

    if not sample_id or len(sample_id) > 128:
        raise ValueError("Provide a valid sample barcode.")

    if not isinstance(limit, int) or not 1 <= limit <= MAX_PAGE_SIZE:
        raise ValueError(
            f"Page size must be between 1 and {MAX_PAGE_SIZE}."
        )

    if not isinstance(offset, int) or not 0 <= offset <= 10_000_000:
        raise ValueError("Offset must be between 0 and 10,000,000.")

    connection = _open_readonly_database()

    try:
        summary = connection.execute(
            """
            SELECT observation_count
            FROM sample_summary
            WHERE sample_id = ?
            """,
            (sample_id,),
        ).fetchone()

        if summary is None:
            raise ValueError(
                f"No intergenic observations found for sample "
                f"{sample_id!r}."
            )

        total_count = int(summary["observation_count"])

        rows = connection.execute(
            """
            SELECT
                source_maf_data_row,
                sample_id,
                chromosome,
                start_position,
                end_position,
                strand,
                variant_type,
                reference_allele,
                tumor_seq_allele1,
                tumor_seq_allele2,
                dbsnp_rs,
                genome_build,
                variant_classification
            FROM intergenic_observation
            WHERE sample_id = ?
            ORDER BY source_maf_data_row
            LIMIT ? OFFSET ?
            """,
            (sample_id, limit, offset),
        ).fetchall()

        items = [dict(row) for row in rows]
        next_offset = (
            offset + len(items)
            if offset + len(items) < total_count
            else None
        )

        return {
            "sample_id": sample_id,
            "total_count": total_count,
            "limit": limit,
            "offset": offset,
            "returned_count": len(items),
            "next_offset": next_offset,
            "coordinate_reference": "GRCh38",
            "coordinate_convention": "MAF positions, 1-based inclusive",
            "items": items,
        }

    finally:
        connection.close()
