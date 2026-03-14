import matplotlib.pyplot as plt
import numpy as np

def create_eda_plots(joined_dfs):
    """
    Create exploratory data analysis plots for wind power data
    """
    fig, ax = plt.subplots(1,3, figsize=(25,4))

    # Speed and Power for the last 7 days
    ax[0].plot(joined_dfs["Speed"].tail(int(7*60*24)), label="Speed", color="blue")
    ax[0].plot(joined_dfs["Total"].tail(int(7*60*24)), label="Power", color="tab:red")
    ax[0].set_title("Windspeed & Power Generation over last 7 days")
    ax[0].set_xlabel("Time")
    ax[0].tick_params(axis='x', labelrotation = 45)
    ax[0].set_ylabel("Windspeed [m/s], Power [MW]")
    ax[0].legend()

    # Speed vs Total (Power Curve nature)
    ax[1].scatter(joined_dfs["Speed"], joined_dfs["Total"])
    power_curve = joined_dfs.groupby("Speed").median(numeric_only=True)["Total"]
    ax[1].plot(power_curve.index, power_curve.values, "k:", label="Power Curve")
    ax[1].legend()
    ax[1].set_title("Windspeed vs Power")
    ax[1].set_ylabel("Power [MW]")
    ax[1].set_xlabel("Windspeed [m/s]")

    # Speed and Power per Wind Direction
    if "Direction" in joined_dfs.columns:
        wind_grouped_by_direction = joined_dfs.groupby("Direction").mean(numeric_only=True).reset_index()
        bar_width = 0.5
        x = np.arange(len(wind_grouped_by_direction.index))
        ax[2].bar(x, wind_grouped_by_direction.Total, width=0.5, label="Power", color="tab:red")
        ax[2].bar(x + bar_width, wind_grouped_by_direction.Speed, width=0.5, label="Speed", color="blue")
        ax[2].legend()
        ax[2].set_xticks(x)
        ax[2].set_xticklabels(wind_grouped_by_direction.Direction)
        ax[2].tick_params(axis='x', labelrotation = 45)
        ax[2].set_title("Speed and Power per Direction")
    else:
        ax[2].axis("off")

    plt.tight_layout()
    return fig



def prediction_plot(y_pred, y_true):
    fig = plt.figure(figsize=(15, 4))
    plt.plot(np.arange(len(y_pred)), y_pred, label="Predictions")
    plt.plot(np.arange(len(y_true)), y_true, label="Truth")
    plt.legend()
    return fig