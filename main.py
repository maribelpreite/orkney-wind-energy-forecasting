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

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment)


    if args.mode == "predict":
        print("Loading saved model...")
        model = sio.load(args.model_path)

        print("Loading new data...")
        new_data = preprocessing.read_csv_with_time_index(args.input_path)

        print("Generating predictions...")
        preds = model.predict(new_data)

        preds_df = pd.DataFrame(
            {"prediction": preds},
            index=new_data.index
        )

        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        preds_df.to_csv(args.output_path)
        print(f"Predictions saved to {args.output_path}")
        return

    with mlflow.start_run(run_name=run_name):
        print("Preparing data...")

        if args.mode == "cv":
            power_df, wind_df, X_train, X_test, y_train, y_test = load_data()
            pipeline = build_pipeline(args)

            print("Running CV...")
            scores = run_cv(pipeline, X_train, y_train, args.cv_splits)
            log_cv_results(scores, args.cv_splits)
            print("CV results logged.")

        elif args.mode == "eval":
            power_df, wind_df, X_train, X_test, y_train, y_test = load_data()
            pipeline = build_pipeline(args)

            mlflow.sklearn.autolog()

            print("Training and evaluating on holdout split...")
            preds = train_and_evaluate(pipeline, X_train, y_train, X_test, y_test)
            log_plots(wind_df, power_df, y_test, preds)
            print("Evaluation complete.")

        # elif args.mode == "train":
        #    power_df, wind_df, X, y = load_full_data()
        #    pipeline = build_pipeline(args)

        #    mlflow.sklearn.autolog()

        #    print("Training final model on all available data...")
        #    pipeline.fit(X, y)

        #    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
        #    sio.dump(pipeline, args.model_path)
        #    mlflow.log_artifact(args.model_path)

        #    print(f"Model saved to {args.model_path}")

        # else:
        #    raise ValueError(f"Unsupported mode: {args.mode}")

########################################################################################################################

# CUSTOMIZE RUN ON THE TERMINAL

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["cv", "eval", "train", "predict"]
    )

    parser.add_argument(
        "--model",
        type=str,
        choices=["linreg", "xgboost"],
        default=None
    )

    parser.add_argument("--experiment", type=str, default="default")
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--tracking-uri", type=str, default="http://127.0.0.1:5000")

    parser.add_argument("--cv-splits", type=int, default=5)

    # xgboost hyperparams
    parser.add_argument("--n_estimators", type=int, default=300)
    parser.add_argument("--max_depth", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=0.1)

    # save/load paths
    parser.add_argument("--model-path", type=str, default="artifacts/model.skops")
    parser.add_argument("--input-path", type=str, default=None)
    parser.add_argument("--output-path", type=str, default="artifacts/predictions.csv")

    args = parser.parse_args()

    if args.mode in ["cv", "eval", "train"] and args.model is None:
        parser.error("--model is required for cv, eval, and train modes.")

    if args.mode == "predict" and args.input_path is None:
        parser.error("--input-path is required for predict mode.")

    return args


def build_run_name(args):
    if args.mode == "predict":
        base = "predict"
    elif args.model == "linreg":
        base = f"{args.mode}_linreg"
    elif args.model == "xgboost":
        base = (
            f"{args.mode}_xg_"
            f"n={args.n_estimators}_"
            f"d={args.max_depth}_"
            f"lr={args.learning_rate}"
        )
    else:
        base = args.mode

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
def build_preprocesor(args):
    steps = [
        ("direction_encoder", ct.DirectionEncoder()),
        ("interpolator", ct.Interpolator()),
    ]

    if args.model == "linreg":
        steps.append(("scaler", StandardScaler()))

    return Pipeline(steps)


def build_model(args):
    if args.model == "linreg":
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
    preprocess = build_preprocesor(args)
    model = build_model(args)

    return Pipeline([
        ("preprocess", preprocess),
        ("regressor", model)
    ])    


# EXPERIMENTATION, TRAINING, EVALUATION FUNCTIONS
def run_cv(pipeline, X_train, y_train, n_splits):
    cv_splits = preprocessing.custom_time_series_split(
        X_train,
        n_splits=n_splits
    )

    scores = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=cv_splits,
        scoring={
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


# PREDICTION FUNCTIONS
def load_and_predict_model(model_name, model_version, new_data):
    model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_version}")
    return model.predict(new_data)


##############################################################################################################3

if __name__ == "__main__":
    main()    