import pandas as pd
import numpy as np


def read_csv_with_time_index(path):
    """Helper to read CSVs with a datetime index."""
    df = pd.read_csv(path, parse_dates=["time"], index_col="time")
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    return df


def align_data(power_df, wind_df):
    #drop unnecessary columns from each dataset
    power_df = power_df.drop(columns=["ANM", "Non-ANM"])
    wind_df = wind_df.drop(columns=["Source_time", "Lead_hours"])

    # generate minute wind data for the interval of the dataframe and interpolate speed to handle missing values
    wind_df = wind_df.resample("1min").asfreq()

    joined_df = wind_df.join(power_df, how="inner") #this will discard the values that go out of bounds for power data
    
    return joined_df

