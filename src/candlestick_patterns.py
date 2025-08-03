import pandas as pd
import numpy as np

def detect_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect simple candlestick patterns:
    - Bullish: Hammer, Bullish Engulfing
    - Bearish: Shooting Star, Bearish Engulfing
    
    Returns:
        df with new column 'pattern_signal':
        1 = Bullish pattern
        -1 = Bearish pattern
        0 = No pattern
    """
    df = df.copy()
    
    # Default no pattern
    df['pattern_signal'] = 0
    
    # Calculate candle components
    df['body'] = df['close'] - df['open']
    df['upper_shadow'] = df['high'] - df[['close', 'open']].max(axis=1)
    df['lower_shadow'] = df[['close', 'open']].min(axis=1) - df['low']
    
    # Hammer: small body, long lower shadow
    hammer = (abs(df['body']) < (df['high'] - df['low']) * 0.3) & \
             (df['lower_shadow'] > 2 * abs(df['body']))
    
    # Shooting Star: small body, long upper shadow
    shooting_star = (abs(df['body']) < (df['high'] - df['low']) * 0.3) & \
                    (df['upper_shadow'] > 2 * abs(df['body']))
    
    # Bullish/Bearish Engulfing
    df['prev_open'] = df['open'].shift(1)
    df['prev_close'] = df['close'].shift(1)
    
    bullish_engulfing = (df['close'] > df['open']) & (df['prev_close'] < df['prev_open']) & \
                        (df['close'] > df['prev_open']) & (df['open'] < df['prev_close'])
    
    bearish_engulfing = (df['close'] < df['open']) & (df['prev_close'] > df['prev_open']) & \
                        (df['close'] < df['prev_open']) & (df['open'] > df['prev_close'])
    
    # Assign signals
    df.loc[hammer | bullish_engulfing, 'pattern_signal'] = 1
    df.loc[shooting_star | bearish_engulfing, 'pattern_signal'] = -1
    
    # Cleanup
    df.drop(['prev_open', 'prev_close'], axis=1, inplace=True)
    
    return df
