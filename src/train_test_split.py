import numpy as np

def train_test_split(df, target_col="Total", split_frac=0.8, window=500):
    X = df.drop(columns=[target_col])
    y = df[target_col]

    base_idx = int(len(df) * split_frac)

    start = base_idx
    end = min(len(df) - 1, base_idx + window)

    subset = X.iloc[start:end + 1]
    valid_mask = subset[["Speed", "Direction"]].notna().all(axis=1)

    valid_positions = np.where(valid_mask.to_numpy())[0]

    if len(valid_positions) == 0:
        # search backward
        start_back = max(1, base_idx - window)
        subset_back = X.iloc[start_back:base_idx]
        valid_mask_back = subset_back[["Speed", "Direction"]].notna().all(axis=1)
        valid_positions_back = np.where(valid_mask_back.to_numpy())[0]

        if len(valid_positions_back) == 0:
            raise ValueError(
                "Could not find a nearby split point where the first test row has "
                "non-null Speed and Direction."
            )

        split_idx = np.arange(start_back, base_idx)[valid_positions_back][-1]
    else:
        split_idx = np.arange(start, end + 1)[valid_positions][0]

    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]

    return X_train, X_test, y_train, y_test
