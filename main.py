import warnings

from fastapi import FastAPI

from app.schemas import SearchRequest, SearchResponse
from app.search import run_bayesian_search

warnings.filterwarnings("ignore")

app = FastAPI(title="LSTM Bayesian HP Optimization")


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    return run_bayesian_search(req)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000)
