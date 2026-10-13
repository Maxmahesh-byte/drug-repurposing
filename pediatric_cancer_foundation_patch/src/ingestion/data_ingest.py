from src.ingestion.openpedcan import download_openpedcan, standardize_histologies


def ingest():
    download_openpedcan()
    standardize_histologies()


if __name__ == "__main__":
    ingest()
