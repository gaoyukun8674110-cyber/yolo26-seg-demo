from pydantic import BaseModel


class PredictResponse(BaseModel):
    has_defect: bool
    confidence: float
    latency_ms: float
    overlay_url: str
