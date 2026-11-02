from api.schemas import RepurposingResponse
from src.inference.predict import predict_repurposing


def test_prediction_is_unavailable_without_real_response_supervision():
    response = RepurposingResponse.model_validate(
        predict_repurposing("imatinib", "medulloblastoma")
    )

    assert response.available is False
    assert response.repurposing_score is None