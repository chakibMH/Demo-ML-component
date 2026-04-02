import os
import time
from datetime import datetime
from functools import partial

import keras
import keras_tuner as kt
import tensorflow as tf

from app.data import generate_data
from app.schemas import SearchRequest, SearchResponse

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")
LOGS_DIR = os.path.join(OUTPUT_DIR, "logs")
os.makedirs(LOGS_DIR, exist_ok=True)


def build_model(hp, time_step: int):
    model = keras.Sequential()

    num_layers = hp.Int("num_layers", 1, 15)
    for i in range(num_layers):
        model.add(
            keras.layers.LSTM(
                units=hp.Choice(f"units_{i}", values=[16, 32, 64, 128, 256]),
                return_sequences=(i < num_layers - 1),
                input_shape=(time_step, 1) if i == 0 else None,
            )
        )
        model.add(
            keras.layers.Dropout(
                rate=hp.Float(f"dropout_{i}", min_value=0.0, max_value=0.8, step=0.2)
            )
        )

    model.add(keras.layers.Dense(1))

    model.compile(
        optimizer=keras.optimizers.Adam(
            hp.Choice("learning_rate", values=[1e-5, 1e-4, 1e-3, 1e-2])
        ),
        loss="mean_squared_error",
        metrics=[
            keras.metrics.MeanAbsoluteError(name="mean_absolute_error"),
            keras.metrics.MeanSquaredError(name="mean_squared_error"),
            tf.keras.metrics.RootMeanSquaredError(),
            tf.keras.metrics.R2Score(),
        ],
    )
    return model


def run_bayesian_search(req: SearchRequest) -> SearchResponse:
    start = time.time()

    X_train, y_train = generate_data(req.time_step)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    project_name = f"BO_LSTM_ts{req.time_step}_{timestamp}"

    tuner = kt.BayesianOptimization(
        hypermodel=partial(build_model, time_step=req.time_step),
        objective="val_mean_squared_error",
        max_trials=req.max_trials,
        directory=OUTPUT_DIR,
        project_name=project_name,
    )

    stop_early = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=250)
    csv_log_path = os.path.join(LOGS_DIR, f"{project_name}.csv")
    csv_logger = keras.callbacks.CSVLogger(csv_log_path, separator=",", append=True)

    tuner.search(
        X_train,
        y_train,
        epochs=req.epochs,
        validation_split=0.2,
        callbacks=[csv_logger, stop_early],
    )

    best_hp = tuner.get_best_hyperparameters(1)[0]
    best_model = tuner.get_best_models(1)[0]

    eval_results = best_model.evaluate(X_train, y_train, verbose=0)
    metric_names = ["loss"] + [m.name for m in best_model.metrics]
    best_metrics = dict(zip(metric_names, [float(v) for v in eval_results]))

    elapsed = time.time() - start

    return SearchResponse(
        best_hyperparameters=best_hp.values,
        best_metrics=best_metrics,
        project_name=project_name,
        elapsed_seconds=round(elapsed, 2),
    )
