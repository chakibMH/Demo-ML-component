# Demo-ML-component

A FastAPI service that exposes an endpoint to run LSTM hyperparameter optimization using Keras Tuner's Bayesian Optimization strategy.

## Disclaimer

The original project is closed-source due to confidentiality, and the exact code, data, and model weights are proprietary. For this exercise, I shared an adapted version of the repository with modified variables, configurations, and anonymized data. As such, no specific open-source license applies, and compatibility with external licenses is not relevant in this context.

## Project Structure
├── main.py # FastAPI app, single POST /search route
├── requirements.txt
└── app/
├── schemas.py # Request / response models
├── search.py # LSTM model builder + Bayesian search logic
└── data.py # Synthetic data generator (replace with real data)


## Setup

Requires Python 3.10+.

```bash
pip install -r requirements.txt

For GPU support on Windows, conda is recommended:

conda create -n ml-demo python=3.11
conda activate ml-demo
conda install -c conda-forge tensorflow keras-tuner
pip install fastapi "uvicorn[standard]" scikit-learn

Running

python main.py

Server starts at http://127.0.0.1:8000.

API
POST /search
Triggers a Bayesian Optimization search over the LSTM architecture space.

Request body

Field	Type	Default	Description
max_trials	int	required	Number of BO trials to run
time_step	int	10	LSTM lookback window
epochs	int	2000	Max training epochs per trial
Example

curl -X POST http://127.0.0.1:8000/search \
  -H "Content-Type: application/json" \
  -d '{"max_trials": 5}'

Response
{
  "best_hyperparameters": { "num_layers": 2, "units_0": 64, ... },
  "best_metrics": { "loss": 0.002, "mean_absolute_error": 0.03, ... },
  "project_name": "perf_analyzis_Bayesian_LSTM_timestep_10",
  "elapsed_seconds": 142.7
}

Tuner artifacts are saved to ./output/, logs to ./output/logs/.

Note: the search is synchronous — the request blocks until all trials complete. Keep max_trials small for testing.