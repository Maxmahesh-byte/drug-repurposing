import csv
import gzip
import hashlib
import json
import os
import sqlite3
from pathlib import Path

ROOT = Path.cwd()

SOURCE = ROOT / (
    "data/processed/graph_v2/intergenic_variants_maf.csv.gz"
)
GRAPH_MANIFEST = ROOT / (
    "data/processed/graph_v3/graph_v3_manifest.json"
)
OUT_DIR = ROOT / "data/processed/graph_v3"

DATABASE = OUT_DIR / "intergenic_evidence.sqlite"
TEMP_DATABASE = OUT_DIR / "intergenic_evidence.sqlite.building"
INDEX_MANIFEST = OUT_DIR / "intergenic_evidence_index_manifest.json"
TEMP_MANIFEST = OUT_DIR / "intergenic_evidence_index_manifest.json.tmp"

EXPECTED_ROWS = 4_522_696
EXPECTED_SAMPLES = 3_223
BATCH_SIZE = 20_000


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    for path in (SOURCE, GRAPH_MANIFEST):
        if not path.is_file():
            raise FileNotFoundError(path)

    for path in (DATABASE, INDEX_MANIFEST, TEMP_DATABASE):
        if path.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing file: {path}"
            )

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    graph_manifest = json.loads(
        GRAPH_MANIFEST.read_text(encoding="utf-8")
    )
    expected_hash = graph_manifest.get("source_evidence_sha256")

    if not expected_hash:
        raise ValueError(
            "Graph V3 manifest does not contain source SHA-256."
        )

    print("Checking source extract checksum...")
    source_hash_before = sha256_file(SOURCE)

    if source_hash_before != expected_hash:
        raise ValueError(
            "Source checksum differs from the Graph V3 manifest."
        )

    connection = None

    try:
        connection = sqlite3.connect(TEMP_DATABASE)
        connection.execute("PRAGMA journal_mode=DELETE")
        connection.execute("PRAGMA synchronous=NORMAL")
        connection.execute("PRAGMA temp_store=FILE")

        connection.execute("""
            CREATE TABLE intergenic_observation (
                source_maf_data_row INTEGER PRIMARY KEY,
                sample_id TEXT NOT NULL,
                chromosome TEXT NOT NULL,
                start_position INTEGER NOT NULL,
                end_position INTEGER NOT NULL,
                strand TEXT,
                variant_type TEXT NOT NULL,
                reference_allele TEXT NOT NULL,
                tumor_seq_allele1 TEXT NOT NULL,
                tumor_seq_allele2 TEXT NOT NULL,
                dbsnp_rs TEXT,
                genome_build TEXT NOT NULL,
                variant_classification TEXT NOT NULL
            )
        """)

        connection.execute("""
            CREATE TABLE sample_summary (
                sample_id TEXT PRIMARY KEY,
                observation_count INTEGER NOT NULL
            )
        """)

        required = {
            "source_maf_data_row",
            "Tumor_Sample_Barcode",
            "Chromosome",
            "Start_Position",
            "End_Position",
            "Strand",
            "Variant_Type",
            "Reference_Allele",
            "Tumor_Seq_Allele1",
            "Tumor_Seq_Allele2",
            "dbSNP_RS",
            "NCBI_Build",
            "Variant_Classification",
            "Hugo_Symbol",
        }

        insert_sql = """
            INSERT INTO intergenic_observation (
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
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        batch = []
        rows = 0
        previous_source_row = 0
        distinct_samples = set()

        print("Streaming source evidence into SQLite...")

        with gzip.open(
            SOURCE, "rt", encoding="utf-8", newline=""
        ) as f:
            reader = csv.DictReader(f)

            absent = sorted(required - set(reader.fieldnames or []))
            if absent:
                raise ValueError(
                    f"Source is missing columns: {absent}"
                )

            for row in reader:
                source_row = int(row["source_maf_data_row"])
                sample_id = row["Tumor_Sample_Barcode"]
                start = int(row["Start_Position"])
                end = int(row["End_Position"])

                if source_row <= previous_source_row:
                    raise ValueError(
                        f"Source row order problem at {source_row}"
                    )
                previous_source_row = source_row

                if not sample_id:
                    raise ValueError(
                        f"Missing sample ID at source row {source_row}"
                    )
                if start <= 0 or end < start:
                    raise ValueError(
                        f"Invalid coordinates at source row {source_row}"
                    )
                if (
                    row["NCBI_Build"] != "GRCh38"
                    or row["Variant_Classification"] != "IGR"
                    or row["Hugo_Symbol"] != "Unknown"
                ):
                    raise ValueError(
                        f"Unexpected annotation at source row {source_row}"
                    )

                distinct_samples.add(sample_id)

                batch.append((
                    source_row,
                    sample_id,
                    row["Chromosome"],
                    start,
                    end,
                    row["Strand"],
                    row["Variant_Type"],
                    row["Reference_Allele"],
                    row["Tumor_Seq_Allele1"],
                    row["Tumor_Seq_Allele2"],
                    row["dbSNP_RS"],
                    row["NCBI_Build"],
                    row["Variant_Classification"],
                ))

                rows += 1

                if len(batch) >= BATCH_SIZE:
                    connection.executemany(insert_sql, batch)
                    connection.commit()
                    batch.clear()

                    if rows % 200_000 == 0:
                        print(f"Indexed {rows:,} records...")

        if batch:
            connection.executemany(insert_sql, batch)
            connection.commit()

        if rows != EXPECTED_ROWS:
            raise ValueError(
                f"Expected {EXPECTED_ROWS:,} rows; got {rows:,}"
            )

        if len(distinct_samples) != EXPECTED_SAMPLES:
            raise ValueError(
                f"Expected {EXPECTED_SAMPLES:,} samples; "
                f"got {len(distinct_samples):,}"
            )

        print("Creating sample lookup index...")

        connection.execute("""
            CREATE INDEX idx_intergenic_sample_source
            ON intergenic_observation (
                sample_id,
                source_maf_data_row
            )
        """)

        connection.execute("""
            INSERT INTO sample_summary (
                sample_id,
                observation_count
            )
            SELECT sample_id, COUNT(*)
            FROM intergenic_observation
            GROUP BY sample_id
        """)

        connection.commit()

        db_count = connection.execute(
            "SELECT COUNT(*) FROM intergenic_observation"
        ).fetchone()[0]

        sample_count = connection.execute(
            "SELECT COUNT(*) FROM sample_summary"
        ).fetchone()[0]

        min_count, max_count = connection.execute(
            """
            SELECT MIN(observation_count), MAX(observation_count)
            FROM sample_summary
            """
        ).fetchone()

        integrity = connection.execute(
            "PRAGMA quick_check"
        ).fetchone()[0]

        if integrity != "ok":
            raise ValueError(f"SQLite integrity check failed: {integrity}")

        if db_count != EXPECTED_ROWS:
            raise ValueError("SQLite observation count mismatch.")

        if sample_count != EXPECTED_SAMPLES:
            raise ValueError("SQLite sample count mismatch.")

        connection.close()
        connection = None

        source_hash_after = sha256_file(SOURCE)
        if source_hash_after != source_hash_before:
            raise ValueError(
                "Source extract changed during indexing."
            )

        manifest = {
            "schema_version": "1.0.0",
            "database": str(DATABASE.relative_to(ROOT)),
            "source_file": str(SOURCE.relative_to(ROOT)),
            "source_sha256": source_hash_before,
            "observation_count": db_count,
            "sample_count": sample_count,
            "minimum_observations_per_sample": min_count,
            "maximum_observations_per_sample": max_count,
            "coordinate_reference": "GRCh38",
            "coordinate_convention": "MAF positions, 1-based inclusive",
            "tables": {
                "intergenic_observation": (
                    "Source-faithful observation fields excluding "
                    "matched-normal sample barcode."
                ),
                "sample_summary": (
                    "Indexed sample IDs and observation counts."
                ),
            },
            "validation": {
                "source_checksum_matches_graph_v3": True,
                "source_unchanged_during_indexing": True,
                "expected_observation_count": True,
                "expected_sample_count": True,
                "source_rows_strictly_increasing": True,
                "coordinates_valid": True,
                "annotations_grch38_igr_unknown": True,
                "sqlite_integrity_check": True,
            },
        }

        TEMP_MANIFEST.write_text(
            json.dumps(manifest, indent=2) + "\n",
            encoding="utf-8",
        )

        os.replace(TEMP_DATABASE, DATABASE)
        os.replace(TEMP_MANIFEST, INDEX_MANIFEST)

        print("\nEVIDENCE INDEX BUILD")
        print("=" * 48)
        print(f"Observations indexed: {db_count:,}")
        print(f"Samples indexed: {sample_count:,}")
        print(f"Minimum per sample: {min_count:,}")
        print(f"Maximum per sample: {max_count:,}")
        print(f"Database: {DATABASE}")
        print(f"Manifest: {INDEX_MANIFEST}")
        print(f"SQLite integrity: {integrity}")
        print("Graph V2 and Graph V3 were not modified.")

    except Exception:
        if connection is not None:
            connection.close()
        TEMP_DATABASE.unlink(missing_ok=True)
        Path(str(TEMP_DATABASE) + "-journal").unlink(missing_ok=True)
        TEMP_MANIFEST.unlink(missing_ok=True)
        raise


if __name__ == "__main__":
    main()
