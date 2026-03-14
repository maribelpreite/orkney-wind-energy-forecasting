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
from sklearn.model_selection import TimeSeriesSplit

### TODO -> HERE YOU CAN ADD ANY OTHER LIBRARIES YOU MAY NEED ###
import src.plots as plots
import src.preprocessing as preprocessing
import argparse
from xgboost import XGBRegressor
import src.custom_transformers as ct

########################################################################################################################

def main():
    args = parse_args()
    run_ID = build_run_name(args)

    # Enable autologging for scikit-learn
    mlflow.sklearn.autolog()
    mlflow.set_tracking_uri(args.tracking_uri) # We set the MLFlow UI to display in our local host.
    mlflow.set_experiment(args.experiment)

    # Start a run
    with mlflow.start_run(run_name=run_ID):

        print("Loading data")

        # --- Load from CSVs ---
        power_df = preprocessing.read_csv_with_time_index("data/power.csv")
        wind_df = preprocessing.read_csv_with_time_index("data/weather.csv")

        print("Starting preprocessing")

        # --- Join datasets ---
        joined_dfs = preprocessing.align_data(power_df, wind_df) #there's gonna be a lot of NaNs due to upsampling in wind_df

        X_train = joined_dfs.drop(columns=["Total"])
        y_train = joined_dfs["Total"]

        # change split since this is time series data
        # X_train, X_test, y_train, y_test = train_test_split(X, y)

        # construct pipeline according to user input
        preprocess = build_preprocesor(args.model)
        model = build_model(args)

        pipeline = Pipeline([
            ("preprocess", preprocess),
            ("regressor", model)
        ])

        print("Starting training")


        # Train and evaluate model
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_train)


        # --- Create and save EDA plots ---
        plot_df = wind_df.join(power_df, how="inner")
        os.makedirs("plots", exist_ok=True)
        eda_fig = plots.create_eda_plots(plot_df)
        eda_fig.savefig("plots/eda_plots.png")
        mlflow.log_artifact("plots/eda_plots.png")
        plt.close(eda_fig)


        # Plot predictions
        pred_fig = plots.prediction_plot(predictions, y_train)
        plt.savefig(f"plots/predictions.png")
        plt.close()
        mlflow.log_artifact(f"plots/predictions.png")

        # No need to manually log metrics - autologging handles:
        # - Parameters
        # - Metrics (R², MSE, MAE)
        # - Model artifacts
        # - Model signature
        # - Feature importance (for supported models)


########################################################################################################################

# functions to customize each run

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, required=True, choices=["linear", "xgboost"])

    parser.add_argument("--experiment", type=str, required=True)
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--tracking-uri", type=str, default="http://127.0.0.1:5000")

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

    if args.tag:
        return f"{base}_{args.tag}"

    return base


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
            learning_rate=args.learning_rate
        )

    else:
        raise ValueError(f"Unsupported model: {args.model}")

    return model
    


# --- Model section (same as before) ---
def load_and_predict_model(model_name, model_version, new_data):
    model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_version}")
    return model.predict(new_data)


##############################################################################################################3

if __name__ == "__main__":
    main()    