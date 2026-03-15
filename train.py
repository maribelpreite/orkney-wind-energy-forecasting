########################################################################################################################
# IMPORTS
# You absolutely need these
import mlflow
import os


# You will probably need these
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import skops.io as sio

# This are for example purposes. You may discard them if you don't use them.
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit, cross_validate

### TODO -> HERE YOU CAN ADD ANY OTHER LIBRARIES YOU MAY NEED ###
import src.plots as plots
import src.preprocessing as preprocessing
import argparse
from xgboost import XGBRegressor
import src.custom_transformers as ct
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score 

########################################################################################################################
# allows both CV and training+evaluation

def main():
    args = parse_args()
    run_name = build_run_name(args)

    # Enable autologging for scikit-learn
    mlflow.sklearn.autolog()
    mlflow.set_tracking_uri(args.tracking_uri) # We set the MLFlow UI to display in our local host.
    mlflow.set_experiment(args.experiment)

    with mlflow.start_run(run_name=run_name):
        power_df, wind_df, X_train, X_test, y_train, y_test = load_data()
        pipeline = build_pipeline(args)

        if args.mode == "cv":
            scores = run_cv(pipeline, X_train, y_train, args.cv_splits)
            log_cv_results(scores, args.cv_splits)

        elif args.mode == "train":
            preds = train_and_evaluate(pipeline, X_train, y_train, X_test, y_test)
            log_plots(wind_df, power_df, y_test, preds)

            # if args.save_model:
            #    save_model(pipeline, args.model_out)


########################################################################################################################

# CUSTOMIZE RUN ON THE TERMINAL

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", type=str, required=True, choices=["cv", "train"])
    parser.add_argument("--model", type=str, required=True, choices=["linear", "xgboost"])

    parser.add_argument("--experiment", type=str, required=True)
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--tracking-uri", type=str, default="http://127.0.0.1:5000")

    parser.add_argument("--cv-splits", type=int, default=5)

    # xgboost hyperparams
    parser.add_argument("--n_estimators", type=int, default=300)
    parser.add_argument("--max_depth", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=0.1)

    return parser.parse_args()


def build_run_name(args):
    if args.model == "linear":
        base = "linear"
    elif args.model == "xgboost":
        base = f"xg_n={args.n_estimators}_depth={args.max_depth}_lr={args.learning_rate}"
    else:
        base = args.model

    if args.mode == "cv":
        base = f"{base}_cv"

    if args.tag:
        base = f"{base}_{args.tag}"

    return base

# LOAD DATA
def load_data():
    power_df = preprocessing.read_csv_with_time_index("data/power.csv")
    wind_df = preprocessing.read_csv_with_time_index("data/weather.csv")
    joined_df = preprocessing.align_data(power_df, wind_df)
    X_train, X_test, y_train, y_test = preprocessing.train_test_split(joined_df)
    return power_df, wind_df, X_train, X_test, y_train, y_test


# PIPELINE FUNCTIONS
def build_preprocesor(model_name):
    steps = [
        ("direction_encoder", ct.DirectionEncoder()),
        ("interpolator", ct.Interpolator()),
    ]

    if model_name == "linear":
        steps.append(("scaler", StandardScaler()))

    return Pipeline(steps)


def build_model(args):
    if args.model == "linear":
        model = LinearRegression()

    elif args.model == "xgboost":
        model = XGBRegressor(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
            random_state = 42
        )

    else:
        raise ValueError(f"Unsupported model: {args.model}")

    return model


def build_pipeline(args):
    preprocess = build_preprocesor(args.model)
    model = build_model(args)

    return Pipeline([
        ("preprocess", preprocess),
        ("regressor", model)
    ])    


# EXPERIMENTATION, TRAINING, EVALUATION FUNCTIONS
def run_cv(pipeline, X_train, y_train, n_splits):
    tscv = TimeSeriesSplit(n_splits=n_splits)

    scores = cross_validate( #per fold scores
        pipeline, #this function will call .fit() and .predict() automatically
        X_train,
        y_train,
        cv=tscv,
        scoring={ #which metrics to compute at each fold
            "rmse": "neg_root_mean_squared_error",
            "mae": "neg_mean_absolute_error",
            "r2": "r2"
        },
        return_train_score=False
    )

    return scores


def log_cv_results(scores, n_splits):
    rmse = -scores["test_rmse"]
    mae = -scores["test_mae"]
    r2 = scores["test_r2"]

    mlflow.log_param("cv_n_splits", n_splits)
    mlflow.log_param("cv_strategy", "TimeSeriesSplit")

    mlflow.log_metric("cv_rmse_mean", rmse.mean())
    mlflow.log_metric("cv_rmse_std", rmse.std())
    mlflow.log_metric("cv_mae_mean", mae.mean())
    mlflow.log_metric("cv_mae_std", mae.std())
    mlflow.log_metric("cv_r2_mean", r2.mean())
    mlflow.log_metric("cv_r2_std", r2.std())


def train_and_evaluate(pipeline, X_train, y_train, X_test, y_test):
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)

    rmse = root_mean_squared_error(y_test, preds) #took the square root to align the CV metrics and the mlflow automatic log metrics too 
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    # autologging probably does this already but it doesnt hurt
    mlflow.log_metric("test_rmse", rmse)
    mlflow.log_metric("test_mae", mae)
    mlflow.log_metric("test_r2", r2)

    return preds


# VISUALIZATIONS
def log_plots(wind_df, power_df, y_true, y_pred):
    os.makedirs("plots", exist_ok=True)

    plot_df = wind_df.join(power_df, how="inner")
    eda_fig = plots.create_eda_plots(plot_df)
    eda_path = "plots/eda_plots.png"
    eda_fig.savefig(eda_path)
    plt.close(eda_fig)
    mlflow.log_artifact(eda_path)

    pred_fig = plots.prediction_plot(y_pred, y_true)
    pred_path = "plots/predictions.png"
    pred_fig.savefig(pred_path)
    plt.close(pred_fig)
    mlflow.log_artifact(pred_path)


# --- Model section (same as before) ---
def load_and_predict_model(model_name, model_version, new_data):
    model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_version}")
    return model.predict(new_data)


##############################################################################################################3

if __name__ == "__main__":
    main()    