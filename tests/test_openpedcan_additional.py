import json

import networkx as nx
import pandas as pd

from src.ingestion.openpedcan import build_phase1_dataset
from src.ingestion.openpedcan_additional import (
    CNV_FILENAME,
    FUSION_FILENAME,
    build_openpedcan_copy_number_dataset,
    build_openpedcan_fusion_dataset,
)
from src.preprocessing.build_graph import (
    EDGE_ATTRIBUTE_NAMES,
    _add_copy_number_summary,
    _add_fusion_edges,
    _edge_attribute_values,
)


def _build_clinical_release(output_dir, release_dir):
    release_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "Kids_First_Biospecimen_ID": ["BS-001"],
            "sample_id": ["S-001"],
            "cohort": ["PBTA"],
            "cancer_group": ["brain tumor"],
            "integrated_diagnosis": ["medulloblastoma"],
        }
    ).to_csv(release_dir / "histologies.tsv", sep="\t", index=False)
    build_phase1_dataset(output_dir=output_dir, raw_dir=release_dir)


def test_copy_number_ingestion_keeps_only_matched_non_neutral_calls(tmp_path):
    output_dir = tmp_path / "output"
    release_dir = tmp_path / "release" / "v15"
    _build_clinical_release(output_dir, release_dir)
    pd.DataFrame(
        {
            "biospecimen_id": ["BS-001", "BS-001", "BS-999"],
            "gene_symbol": ["MYCN", "TP53", "ALK"],
            "status": ["amplification", "neutral", "deletion"],
            "copy_number": ["8", "2", "0"],
            "ploidy": ["2", "2", "2"],
            "cytoband": ["2p24.3", "17p13.1", "2p23.2"],
        }
    ).to_csv(release_dir / CNV_FILENAME, sep="\t", index=False, compression="gzip")

    output_path = build_openpedcan_copy_number_dataset(output_dir=output_dir, raw_dir=release_dir)
    calls = pd.read_csv(output_path)
    manifest = json.loads((output_dir / "openpedcan_copy_number_manifest.json").read_text())

    assert calls[["sample_id", "gene_symbol", "status"]].values.tolist() == [["BS-001", "MYCN", "amplification"]]
    assert calls["cohort"].tolist() == ["PBTA"]
    assert manifest["unmatched_biospecimen_ids"] == 1


def test_fusion_ingestion_joins_biospecimen_and_clinical_metadata(tmp_path):
    output_dir = tmp_path / "output"
    release_dir = tmp_path / "release" / "v15"
    _build_clinical_release(output_dir, release_dir)
    pd.DataFrame(
        {
            "Sample": ["BS-001", "BS-999"],
            "FusionName": ["ETV6--NTRK3", "ALK--X"],
            "Gene1A": ["ETV6", "ALK"],
            "Gene1B": ["NTRK3", "X"],
            "Gene2A": ["ENSG00000139083", "ENSG1"],
            "Gene2B": ["ENSG00000186367", "ENSG2"],
            "Fusion_anno": ["Oncogene", "Oncogene"],
            "caller.count": ["2", "1"],
            "Fusion_Type": ["interchromosomal", "translocation"],
            "BreakpointLocation": ["exonic", "intergenic"],
        }
    ).to_csv(release_dir / FUSION_FILENAME, sep="\t", index=False)

    output_path = build_openpedcan_fusion_dataset(output_dir=output_dir, raw_dir=release_dir)
    fusions = pd.read_csv(output_path)
    manifest = json.loads((output_dir / "openpedcan_fusions_manifest.json").read_text())

    assert fusions["sample_id"].tolist() == ["BS-001"]
    assert fusions["fusion_name"].tolist() == ["ETV6--NTRK3"]
    assert fusions["diagnosis"].tolist() == ["medulloblastoma"]
    assert manifest["unmatched_biospecimen_ids"] == 1


def test_graph_keeps_copy_number_and_fusion_relations_distinct():
    graph = nx.Graph()
    calls = pd.DataFrame(
        {
            "sample_id": ["BS-001", "BS-002", "BS-004", "BS-005", "BS-003"],
            "gene_symbol": ["MYCN", "MYCN", "MYCN", "MYCN", "CDKN2A"],
            "status": [
                "amplification",
                "deep deletion",
                "gain",
                "loss",
                "unknown",
            ],
        }
    )
    _add_copy_number_summary(
        graph,
        calls,
        {
            "BS-001": "brain tumor",
            "BS-002": "brain tumor",
            "BS-003": "",
            "BS-004": "brain tumor",
            "BS-005": "brain tumor",
        },
    )

    fusions = pd.DataFrame(
        {
            "sample_id": ["BS-001", "BS-999"],
            "fusion_name": ["ETV6--NTRK3", "ALK--X"],
            "gene_a": ["ETV6", "ALK"],
            "gene_b": ["NTRK3", "X"],
            "breakpoint_location": ["exonic", "intergenic"],
        }
    )
    unmatched = _add_fusion_edges(graph, fusions, {"BS-001"})

    assert graph["brain tumor"]["MYCN"]["relation"] == "copy_number_in_tumor"
    assert graph["brain tumor"]["MYCN"]["copy_number_count"] == 4
    assert graph["brain tumor"]["MYCN"]["copy_number_gain_count"] == 2
    assert graph["brain tumor"]["MYCN"]["copy_number_loss_count"] == 2
    assert graph["brain tumor"]["MYCN"]["copy_number_status_counts"] == {
        "amplification": 1,
        "deep deletion": 1,
        "gain": 1,
        "loss": 1,
    }
    assert dict(
        zip(EDGE_ATTRIBUTE_NAMES, _edge_attribute_values(graph["brain tumor"]["MYCN"]))
    ) == {
        "variant_count": 0.0,
        "copy_number_count": 4.0,
        "fusion_count": 0.0,
        "ln_ic50": 0.0,
        "auc": 0.0,
        "copy_number_gain_count": 2.0,
        "copy_number_loss_count": 2.0,
        "copy_number_unclassified_count": 0.0,
    }
    assert graph["unclassified"]["CDKN2A"]["copy_number_count"] == 1
    assert graph["unclassified"]["CDKN2A"]["copy_number_unclassified_count"] == 1
    assert graph["unclassified"]["CDKN2A"]["copy_number_status_counts"] == {
        "unknown": 1,
    }
    assert unmatched == {"BS-999"}
    fusion_id = "BS-001|ETV6--NTRK3|exonic"
    assert graph["BS-001"][fusion_id]["relation"] == "has_fusion"
    assert graph[fusion_id]["ETV6"]["relation"] == "fusion_gene"