import pandas as pd

def calculate_pivots(df: pd.DataFrame, freq: str = 'W') -> pd.DataFrame:
    """
    Calculate pivot points (P, S1-S3, R1-R3) based on the previous
    week or month using OHLC data.

    freq: 'W' for weekly pivots, 'M' for monthly pivots
    """
    df = df.copy()
    df.set_index("timestamp", inplace=True)

    # Resample to weekly or monthly to calculate OHLC of the previous period
    ohlc = df.resample(freq).agg({"high": "max", "low": "min", "close": "last"})

    # Pivot Point
    ohlc["P"] = (ohlc["high"] + ohlc["low"] + ohlc["close"]) / 3

    # Support and Resistance Levels
    ohlc["S1"] = (2 * ohlc["P"]) - ohlc["high"]
    ohlc["S2"] = ohlc["P"] - (ohlc["high"] - ohlc["low"])
    ohlc["S3"] = ohlc["low"] - 2 * (ohlc["high"] - ohlc["P"])

    ohlc["R1"] = (2 * ohlc["P"]) - ohlc["low"]
    ohlc["R2"] = ohlc["P"] + (ohlc["high"] - ohlc["low"])
    ohlc["R3"] = ohlc["high"] + 2 * (ohlc["P"] - ohlc["low"])

    # Forward fill pivot levels to each candle until next period
    pivots = ohlc[["P", "S1", "S2", "S3", "R1", "R2", "R3"]].reindex(df.index, method='ffill')

    # Join with original dataframe
    df = df.join(pivots)
    df.reset_index(inplace=True)
    return df


def calculate_support_resistance(df: pd.DataFrame, freq: str = 'W', tolerance: float = 0.001) -> pd.DataFrame:
    """
    Calculate pivot-based support and resistance for each candle:
    - Weekly or monthly pivots with S1-S3, R1-R3
    - Flags for near support/resistance based on tolerance

    Args:
        df: DataFrame with 'timestamp', 'high', 'low', 'close'
        freq: 'W' for weekly, 'M' for monthly
        tolerance: % of price to consider "near" a pivot level (default ~0.1%)

    Returns:
        df: DataFrame with pivot levels and support/resistance flags
    """
    df = calculate_pivots(df, freq=freq)

    # Calculate flags for being near support or resistance
    df['at_support'] = (
        (df['close'] - df[['S1', 'S2', 'S3']].min(axis=1)) / df['close'] < tolerance
    ).astype(int)

    df['at_resistance'] = (
        (df[['R1', 'R2', 'R3']].max(axis=1) - df['close']) / df['close'] < tolerance
    ).astype(int)

    return df

def merge_multi_timeframe_features(df_4h, df_1d, df_1w):
    """
    Merge 4H, 1D, and 1W features into a single 4H ML-ready dataset.

    Steps:
    1. Resample 1D and 1W data to 4H and forward-fill.
    2. Join higher timeframe features to 4H candles.
    3. Forward-fill missing values and drop early NaNs.
    """
    # Resample higher TF to 4H
    df_1d_res = df_1d.set_index('timestamp').resample('4H').ffill()
    df_1w_res = df_1w.set_index('timestamp').resample('4H').ffill()

    # Merge into 4H
    df_4h = df_4h.set_index('timestamp')
    df_merged = df_4h.join(df_1d_res, rsuffix='_1D')
    df_merged = df_merged.join(df_1w_res, rsuffix='_1W')

    # Reset timestamp
    df_merged.reset_index(inplace=True)

    # Forward-fill all NaNs (mainly first few rows)
    df_merged.ffill(inplace=True)

    # Drop any remaining NaNs (ensures ML-ready dataset)
    df_merged.dropna(inplace=True)

    return df_merged

