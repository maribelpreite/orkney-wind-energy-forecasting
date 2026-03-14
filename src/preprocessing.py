import pandas as pd
import numpy as np

def preprocess_df(df, cols_to_drop):
    """
    General preprocessing function that drops custom columns, sets the time
    column as the index, converts it to datetime format, and sorts the data
    chronologically.
    """

    df = df.copy()

    # drop the unnecessary columns
    df = df.drop(columns=cols_to_drop)

    # set the time to be the index
    if df.index.name != "time":
        df["time"] = pd.to_datetime(df["time"])  # convert to datetime object
        df = df.set_index("time")

    # if time is index but not a datetime object
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    # sort chronologically
    df = df.sort_index()

    return df


def preprocess_wind_df(wind_df):

    # drop unwanted columns, sort chronologically. result: wind data in 3h intervals
    wind_df = preprocess_df(wind_df, cols_to_drop=["Lead_hours", "Source_time"])

    # transform categorical Direction column to numerical
    wind_df = dir_to_angle(wind_df)

    # generate minute wind data for the interval of the dataframe
    wind_df = wind_df.resample("1min").asfreq()

    # speed interpolation
    wind_df["Speed"] = wind_df["Speed"].interpolate(method="time")

    # direction interpolation
    wind_df["Direction_sin"] = wind_df["Direction_sin"].interpolate(method="time")
    wind_df["Direction_cos"] = wind_df["Direction_cos"].interpolate(method="time")

    return wind_df


def align_data(power_df, wind_df):
    power_df = preprocess_df(power_df, cols_to_drop=["ANM", "Non-ANM"])
    wind_df = preprocess_wind_df(wind_df)

    joined_df = wind_df.join(power_df, how="inner") #this will discard the values that go out of bounds for power data
    
    return joined_df


def dir_to_angle(wind_df):

    direction_map = {
    "N": 0.0,
    "NNE": 22.5,
    "NE": 45.0,
    "ENE": 67.5,
    "E": 90.0,
    "ESE": 112.5,
    "SE": 135.0,
    "SSE": 157.5,
    "S": 180.0,
    "SSW": 202.5,
    "SW": 225.0,
    "WSW": 247.5,
    "W": 270.0,
    "WNW": 292.5,
    "NW": 315.0,
    "NNW": 337.5
}
    
    angle = wind_df["Direction"].map(direction_map)
    wind_df["Direction_sin"] = np.sin(np.deg2rad(angle))
    wind_df["Direction_cos"] = np.cos(np.deg2rad(angle))
    wind_df = wind_df.drop(columns=["Direction"])

    return wind_df
