def predict_repurposing(drug_id: str, target_disease: str):
    return {
        "drug_id": drug_id,
        "target_disease": target_disease,
        "available": False,
        "repurposing_score": None,
        "confidence": "unavailable",
        "explanation": (
            "Candidate scoring is disabled until real drug-response labels and a validated "
            "drug-disease prediction task are available."
        ),
    }