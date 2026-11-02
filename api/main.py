from fastapi import FastAPI, HTTPException, Query
from src.inference.predict import predict_repurposing
from api.schemas import RepurposingRequest, RepurposingResponse
from src.inference.graph_status import get_graph_v3_status
from src.inference.intergenic_evidence import get_sample_intergenic_evidence

app = FastAPI(title="Drug Repurposing GNN API")

@app.get("/")
def root():
    return {
        "message": "✅ MLOps Drug Repurposing Pipeline is running!",
        "docs": "/docs",
        "streamlit": "http://localhost:8501",
        "health": "/health"
    }

@app.post("/predict", response_model=RepurposingResponse)
def predict(request: RepurposingRequest):
    try:
        result = predict_repurposing(request.drug_id, request.target_disease)
        return result
    except Exception as e:
        return {"error": str(e)}

@app.get("/graph-v3/status")
def graph_v3_status():
    """Return read-only Graph V3 build and provenance status."""
    return get_graph_v3_status()



@app.get("/graph-v3/samples/{sample_id}/intergenic")
def graph_v3_sample_intergenic(
    sample_id: str,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0, le=10_000_000),
):
    """Return a bounded, paginated set of intergenic observations."""
    try:
        return get_sample_intergenic_evidence(
            sample_id=sample_id,
            limit=limit,
            offset=offset,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except (FileNotFoundError, RuntimeError) as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@app.get("/health")
def health():
    return {"status": "healthy", "message": "GNN pipeline is running"}