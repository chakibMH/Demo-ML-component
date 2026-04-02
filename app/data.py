import numpy as np


def generate_data(time_step: int, n_samples: int = 1000):
    """Synthetic sine-wave dataset shaped for LSTM. Replace with real data."""
    raw = np.sin(np.linspace(0, 50, n_samples + time_step))
    X, y = [], []
    for i in range(len(raw) - time_step):
        X.append(raw[i : i + time_step])
        y.append(raw[i + time_step])
    X = np.array(X).reshape(-1, time_step, 1).astype(np.float32)
    y = np.array(y).astype(np.float32)
    return X, y
