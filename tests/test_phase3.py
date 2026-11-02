import pandas as pd

from src.ingestion.gdsc_pediatric import build_phase3_dataset


def test_build_phase3_dataset_creates_response_layer(tmp_path):
    output_dir = tmp_path / "gdsc"
    workbook = tmp_path / "gdsc2.xlsx"
    pd.DataFrame(
        {
            "DATASET": ["GDSC2"],
            "NLME_RESULT_ID": [1],
            "NLME_CURVE_ID": [11],
            "COSMIC_ID": [101],
            "CELL_LINE_NAME": ["DAOY"],
            "SANGER_MODEL_ID": ["SIDM001"],
            "TCGA_DESC": ["MB"],
            "DRUG_ID": [10],
            "DRUG_NAME": ["Drug A"],
            "PUTATIVE_TARGET": ["HDAC"],
            "PATHWAY_NAME": ["Chromatin"],
            "MIN_CONC": [0.01],
            "MAX_CONC": [10.0],
            "LN_IC50": [-2.3],
            "AUC": [0.42],
            "RMSE": [0.1],
            "Z_SCORE": [-1.2],
        }
    ).to_excel(workbook, index=False)

    df = build_phase3_dataset(output_dir=output_dir, raw_path=workbook)

    assert isinstance(df, pd.DataFrame)
    assert df["sanger_model_id"].tolist() == ["SIDM001"]
    assert df["ln_ic50"].tolist() == [-2.3]
    assert (output_dir / "gdsc2_responses.csv").exists()
