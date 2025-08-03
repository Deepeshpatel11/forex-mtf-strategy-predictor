import pandas as pd

def calculate_trend(df: pd.DataFrame, period: int = 20, column: str = "close") -> pd.DataFrame:
    """
    Calculate simple trend signal based on EMA slope.
    
    Trend logic:
    - 1 = Uptrend (EMA rising)
    - -1 = Downtrend (EMA falling)
    - 0 = Flat/Neutral
    
    Args:
        df: DataFrame with at least 'timestamp' and OHLC columns
        period: EMA period to calculate trend
        column: Column to calculate EMA on (default 'close')
        
    Returns:
        df: DataFrame with added 'trend_signal' column
    """
    df = df.copy()
    
    # Calculate EMA
    df['ema'] = df[column].ewm(span=period, adjust=False).mean()
    
    # Determine slope for trend
    df['ema_diff'] = df['ema'].diff()
    
    # Generate trend signal
    df['trend_signal'] = df['ema_diff'].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    
    return df
