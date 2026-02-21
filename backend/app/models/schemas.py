from pydantic import BaseModel, Field


class PredictResponse(BaseModel):
    label: str = Field(..., examples=["real", "fake"])
    prob_fake: float = Field(..., ge=0.0, le=1.0, examples=[0.93])


class ErrorResponse(BaseModel):
    detail: str
