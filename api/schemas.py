from pydantic import BaseModel

class RepurposingRequest(BaseModel):
    drug_id: str
    target_disease: str

class RepurposingResponse(BaseModel):
    drug_id: str
    target_disease: str
    available: bool
    repurposing_score: float | None
    confidence: str
    explanation: str