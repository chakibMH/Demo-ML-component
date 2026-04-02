from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    max_trials: int = Field(..., gt=0)
    time_step: int = Field(default=10, gt=0)
    epochs: int = Field(default=2000, gt=0)


class SearchResponse(BaseModel):
    best_hyperparameters: dict
    best_metrics: dict
    project_name: str
    elapsed_seconds: float
