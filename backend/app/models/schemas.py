from pydantic import BaseModel, Field


class PredictResponse(BaseModel):
    label: str = Field(..., examples=["real", "fake"])
    prob_fake: str = Field(..., examples=["82.91%"])
    mode: str = Field(..., examples=["hybrid_av", "visual_only_fallback"])

class ErrorResponse(BaseModel):
    detail: str
