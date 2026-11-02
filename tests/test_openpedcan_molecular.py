import json

import pandas as pd
import pytest
import networkx as nx
import torch

from src.ingestion.openpedcan import build_phase1_dataset
from src.ingestion.openpedcan_molecular import (
    MAF_FILENAME,
    _resolve_maf_path,
    build_openpedcan_mutation_dataset,
)
from src.preprocessing.build_graph import _add_mutation_edges, _build_node_features


def test_build_openpedcan_mutation_dataset_joins_real_biospecimen_ids(tmp_path):
    output_dir = tmp_path / "output"
    release_dir = tmp_path / "release" / "testing"
    release_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "Kids_First_Biospecimen_ID": ["BS-001"],
            "sample_id": ["S-001"],
            "cohort": ["PBTA"],
            "cancer_group": ["brain tumor"],
        }
    ).to_csv(release_dir / "histologies.tsv", sep="\t", index=False)
    build_phase1_dataset(output_dir=output_dir, raw_dir=release_dir)

    maf_path = release_dir / "snv-consensus-plus-hotspots.maf.tsv.gz"
    pd.DataFrame(
        {
            "Tumor_Sample_Barcode": ["BS-001", "BS-999"],
            "Hugo_Symbol": ["TP53", "MYCN"],
            "Variant_Classification": ["Missense_Mutation", "Amplification"],
            "Variant_Type": ["SNP", "CNV"],
            "Chromosome": ["17", "2"],
            "Start_Position": ["7673803", "16000000"],
            "Reference_Allele": ["C", "N"],
            "Tumor_Seq_Allele2": ["T", "A"],
            "HGVSp_Short": ["p.R248Q", ""],
            "IMPACT": ["HIGH", "HIGH"],
        }
    ).to_csv(maf_path, sep="\t", index=False, compression="gzip")

    output_path = build_openpedcan_mutation_dataset(output_dir=output_dir, raw_dir=release_dir)
    mutations = pd.read_csv(output_path)
    manifest = json.loads((output_dir / "openpedcan_mutations_manifest.json").read_text())

    assert mutations["sample_id"].tolist() == ["BS-001"]
    assert mutations["gene_symbol"].tolist() == ["TP53"]
    assert mutations["cohort"].tolist() == ["PBTA"]
    assert mutations["tumor_type"].tolist() == ["brain tumor"]
    assert manifest["variant_rows_matched"] == 1
    assert manifest["unmatched_biospecimen_ids"] == 1
    assert manifest["release_scope"] == "testing_subset"


def test_mutation_ingestion_rejects_synthetic_clinical_dataset(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENPEDCAN_ALLOW_SYNTHETIC_FALLBACK", "1")
    build_phase1_dataset(output_dir=tmp_path, raw_dir=tmp_path / "empty")

    with pytest.raises(ValueError, match="requires real OpenPedCan clinical data"):
        build_openpedcan_mutation_dataset(output_dir=tmp_path, raw_dir=tmp_path / "empty")


def test_node_type_features_are_deterministic_and_non_random():
    graph = nx.Graph()
    graph.add_node("BS-001", type="sample")
    graph.add_node("TP53", type="gene")
    node_mapping = {node: index for index, node in enumerate(graph.nodes())}

    features = _build_node_features(graph, node_mapping)

    assert torch.equal(features, _build_node_features(graph, node_mapping))
    assert torch.equal(features.sum(dim=1), torch.ones(2))


def test_maf_resolver_prefers_full_release_over_testing_subset(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENPEDCAN_MAF_PATH", raising=False)
    testing_path = tmp_path / "testing" / MAF_FILENAME
    release_path = tmp_path / "v15" / MAF_FILENAME
    testing_path.parent.mkdir()
    release_path.parent.mkdir()
    testing_path.touch()
    release_path.touch()

    assert _resolve_maf_path(tmp_path) == release_path


def test_graph_aggregates_variant_rows_per_biospecimen_gene_pair():
    graph = nx.Graph()
    variants = pd.DataFrame(
        {
            "sample_id": ["BS-001", "BS-001", "BS-002"],
            "gene_symbol": ["TP53", "TP53", "MYCN"],
        }
    )

    unmatched = _add_mutation_edges(graph, variants, {"BS-001", "BS-002"})

    assert not unmatched
    assert graph["BS-001"]["TP53"]["variant_count"] == 2
    assert graph["BS-002"]["MYCN"]["variant_count"] == 1