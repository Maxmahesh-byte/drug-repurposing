import pandas as pd

from src.ingestion.openpedcan import normalize_columns


def test_normalize_columns():
    df = pd.DataFrame(
        {
            "Kids First Biospecimen ID": ["S1"],
            "Harmonized Diagnosis": ["Example"],
        }
    )
    out = normalize_columns(df)
    assert list(out.columns) == [
        "kids_first_biospecimen_id",
        "harmonized_diagnosis",
    ]
