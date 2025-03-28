import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def scale_dataframe(df: pd.DataFrame) -> tuple:
    """
    Scales the entire DataFrame (except Date/Index).
    Returns:
        scaled_df: pd.DataFrame
        scaler: fitted MinMaxScaler (to inverse transform later)
    """
    scaler = MinMaxScaler()
    scaled_array = scaler.fit_transform(df)
    scaled_df = pd.DataFrame(scaled_array, columns=df.columns, index=df.index)
    return scaled_df, scaler
