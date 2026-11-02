import json

import networkx as nx
import pandas as pd

from src.ingestion.gdsc2 import build_gdsc2_response_dataset
from src.preprocessing.build_graph import _add_gdsc_response_edges


def test_build_gdsc2_response_dataset_preserves_official_measurements(tmp_path):
    workbook = tmp_path / "gdsc2.xlsx"
    pd.DataFrame(
        {
            "DATASET": ["GDSC2", "GDSC1", "GDSC2"],
            "NLME_RESULT_ID": [1, 2, 3],
            "NLME_CURVE_ID": [11, 12, 13],
            "COSMIC_ID": [101, 102, 103],
            "CELL_LINE_NAME": ["DAOY", "A549", "SJSA1"],
            "SANGER_MODEL_ID": ["SIDM001", "SIDM002", "SIDM003"],
            "TCGA_DESC": ["MB", "LUAD", "OS"],
            "DRUG_ID": [10, 11, 12],
            "DRUG_NAME": ["Drug A", "Drug B", "Drug C"],
            "PUTATIVE_TARGET": ["HDAC", "EGFR", "TOP2"],
            "PATHWAY_NAME": ["Chromatin", "RTK", "DNA damage"],
            "MIN_CONC": [0.01, 0.01, 0.02],
            "MAX_CONC": [10.0, 10.0, 20.0],
            "LN_IC50": [-2.3, -1.1, -0.7],
            "AUC": [0.42, 0.71, 0.83],
            "RMSE": [0.1, 0.2, 0.15],
            "Z_SCORE": [-1.2, 0.0, 1.1],
        }
    ).to_excel(workbook, index=False)

    output_path = build_gdsc2_response_dataset(output_dir=tmp_path / "out", raw_path=workbook)
    responses = pd.read_csv(output_path)
    manifest = json.loads((tmp_path / "out" / "gdsc2_manifest.json").read_text())

    assert responses["dataset"].tolist() == ["GDSC2", "GDSC2"]
    assert responses["sanger_model_id"].tolist() == ["SIDM001", "SIDM003"]
    assert responses["ln_ic50"].tolist() == [-2.3, -0.7]
    assert responses["reported_cancer_type"].tolist() == ["MB", "OS"]
    assert manifest["data_origin"] == "official_gdsc2_release"


def test_gdsc_response_graph_keeps_measured_ln_ic50_and_auc():
    responses = pd.DataFrame(
        {
            "sanger_model_id": ["SIDM001", "SIDM001"],
            "cell_line": ["DAOY", "DAOY"],
            "reported_cancer_type": ["MB", "MB"],
            "drug_id": ["10", "10"],
            "drug_name": ["Drug A", "Drug A"],
            "ln_ic50": ["-1.0", "-3.0"],
            "auc": ["0.4", "0.6"],
            "putative_target": ["HDAC", "HDAC"],
            "pathway": ["Chromatin", "Chromatin"],
        }
    )
    graph = nx.Graph()

    _add_gdsc_response_edges(graph, responses)

    model = "GDSC_MODEL:SIDM001"
    drug = "GDSC_DRUG:10"
    response = graph[drug][model]
    assert response["relation"] == "gdsc2_response"
    assert response["ln_ic50"] == -2.0
    assert response["auc"] == 0.5
    assert response["response_count"] == 2
    assert graph[model]["GDSC_CANCER:MB"]["relation"] == "has_reported_cancer_type"
    assert graph[drug]["GDSC_TARGET_ANNOTATION:HDAC"]["relation"] == "gdsc_putative_target_annotation"
    assert graph[drug]["GDSC_PATHWAY_ANNOTATION:Chromatin"]["relation"] == "gdsc_putative_pathway_annotation"