import os
import joblib
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def scale_dataframe(df: pd.DataFrame, scaler=None) -> tuple:
    """
    Scales the entire DataFrame.
    If scaler is provided, it will be used for transformation only.
    Otherwise, a new MinMaxScaler will be fitted.
    Returns:
        scaled_df: pd.DataFrame
        scaler: fitted or reused MinMaxScaler
    """

    if scaler is None:
        scaler = MinMaxScaler()
        scaled_array = scaler.fit_transform(df)
    else:
        scaled_array = scaler.transform(df)

    scaled_df = pd.DataFrame(scaled_array, columns=df.columns, index=df.index)
    return scaled_df, scaler


def save_scaler(scaler, path: str):
    """
    Saves a fitted scaler to the specified path using joblib.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(scaler, path)

def load_scaler(path: str):
    """
    Loads a saved scaler from the specified path.
    """
    return joblib.load(path)
