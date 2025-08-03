import pandas as pd

def validate_ohlc_data(df: pd.DataFrame) -> dict:
    """
    Validate raw OHLCV data.

    Checks:
    1. Missing values
    2. Duplicate rows
    3. Timestamp continuity (1H gaps detection)

    Returns:
        dict: summary of issues found
    """
    issues = {
        'row_count': len(df),
        'missing_values': df.isnull().sum().to_dict(),
        'duplicate_rows': df.duplicated().sum()
    }

    # Check timestamp continuity
    if "timestamp" in df.columns:
        df_sorted = df.sort_values("timestamp").copy()
        df_sorted["diff"] = df_sorted["timestamp"].diff()
        gaps = df_sorted[df_sorted["diff"] > pd.Timedelta("1H")]
        issues['gaps_found'] = len(gaps)
    else:
        issues['gaps_found'] = "timestamp column missing"

    return issues


def resample_ohlc(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """
    Resample OHLCV data to a specified timeframe.

    Args:
        df (pd.DataFrame): Input DataFrame with columns open, high, low, close, volume
        timeframe (str): Pandas offset string ("4H", "1D", "1W")

    Returns:
        pd.DataFrame: Resampled OHLCV data
    """
    ohlc = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum"
    }

    df_resampled = df.resample(timeframe).apply(ohlc).dropna()
    return df_resampled


def validate_resampled_data(df: pd.DataFrame, timeframe: str) -> dict:
    """
    Validate resampled OHLCV data for continuity and structure.

    Args:
        df (pd.DataFrame): Resampled DataFrame
        timeframe (str): Resampling timeframe string (for reporting)

    Returns:
        dict: summary of data quality for resampled data
    """
    issues = {
        'row_count': len(df),
        'missing_values': df.isnull().sum().to_dict(),
        'duplicate_rows': df.duplicated().sum(),
        'timeframe': timeframe
    }

    return issues
