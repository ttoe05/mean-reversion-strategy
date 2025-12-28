"""
Sliding window clipping functionality for time series data.
Implements statistical clipping based on historical sliding windows.
"""

import pandas as pd
import numpy as np
from typing import Union, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def sliding_window_clip(
    data: pd.DataFrame,
    value_column: str,
    clipping_column: Optional[str] = None,
    window_size: int = 252,
    min_window_size: int = 50,
    random_seed: Optional[int] = None
) -> pd.Series:
    """
    Apply statistical clipping to time series data using historical sliding window.
    
    For extreme values (above historical max or below historical min), calculates 
    statistical thresholds based on standard deviations from historical mean and 
    applies random clipping within those bounds.
    
    Args:
        data: DataFrame with time series data (must have datetime index)
        value_column: Column name containing values to be clipped at time t
        clipping_column: Column for historical window calculation (if None, uses value_column)
        window_size: Size of historical sliding window (t-1 to t-m)
        min_window_size: Minimum window size required before clipping starts
        random_seed: Random seed for reproducibility
        
    Returns:
        Series with clipped values, same index as input data
        
    Raises:
        ValueError: If data is invalid or columns don't exist
        IndexError: If insufficient data for minimum window
    """
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("Data must have DatetimeIndex")
    
    if value_column not in data.columns:
        raise ValueError(f"Value column '{value_column}' not found in data")
    
    if clipping_column is None:
        clipping_column = value_column
    elif clipping_column not in data.columns:
        raise ValueError(f"Clipping column '{clipping_column}' not found in data")
    
    if len(data) < min_window_size + 1:
        raise IndexError(f"Insufficient data: need at least {min_window_size + 1} rows")
    
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Initialize result series
    result = data[value_column].copy()
    
    # Start clipping after minimum window size
    for t in range(min_window_size, len(data)):
        current_value = data[value_column].iloc[t]
        
        # Calculate historical window indices (t-1 to t-m)
        window_start = max(0, t - window_size)
        window_end = t  # Exclude current point
        
        # Get historical data for statistical analysis
        historical_data = data[clipping_column].iloc[window_start:window_end]
        
        if len(historical_data) == 0:
            continue
            
        # Calculate historical statistics
        hist_mean = historical_data.mean()
        hist_std = historical_data.std()
        hist_max = historical_data.max()
        hist_min = historical_data.min()
        
        # Skip if std is 0 (no variation in historical data)
        if hist_std == 0:
            continue
        
        clipped_value = current_value
        
        # Check for extreme high values
        if current_value > hist_max:
            # Calculate how many standard deviations the max is from mean
            max_std_from_mean = (hist_max - hist_mean) / hist_std
            
            # Bottom threshold: same std devs from mean as historical max
            bottom_threshold = hist_mean + max_std_from_mean * hist_std
            # Upper threshold: bottom + 0.5 std devs
            upper_threshold = bottom_threshold + 0.5 * hist_std
            
            # Generate random value between thresholds
            clipped_value = np.random.uniform(bottom_threshold, upper_threshold)
            
        # Check for extreme low values  
        elif current_value < hist_min:
            # Calculate how many standard deviations the min is from mean
            min_std_from_mean = (hist_min - hist_mean) / hist_std
            
            # Upper threshold: same std devs from mean as historical min
            upper_threshold = hist_mean + min_std_from_mean * hist_std
            # Bottom threshold: upper - 0.5 std devs
            bottom_threshold = upper_threshold - 0.5 * hist_std
            
            # Generate random value between thresholds
            clipped_value = np.random.uniform(bottom_threshold, upper_threshold)
        
        result.iloc[t] = clipped_value
    
    return result


def batch_sliding_window_clip(
    data: pd.DataFrame,
    columns: list[str],
    clipping_column: Optional[str] = None,
    window_size: int = 252,
    min_window_size: int = 50,
    random_seed: Optional[int] = None
) -> pd.DataFrame:
    """
    Apply sliding window clipping to multiple columns.
    
    Args:
        data: DataFrame with time series data
        columns: List of column names to clip
        clipping_column: Column for historical window calculation (if None, uses each column individually)
        window_size: Size of historical sliding window
        min_window_size: Minimum window size required
        random_seed: Random seed for reproducibility
        
    Returns:
        DataFrame with clipped values for specified columns
    """
    result_df = data.copy()
    
    for col in columns:
        logger.info(f"Applying sliding window clipping to column: {col}")
        result_df[col] = sliding_window_clip(
            data=data,
            value_column=col,
            clipping_column=clipping_column if clipping_column else col,
            window_size=window_size,
            min_window_size=min_window_size,
            random_seed=random_seed
        )
    
    return result_df


def get_clipping_statistics(
    data: pd.DataFrame,
    value_column: str,
    clipping_column: Optional[str] = None,
    window_size: int = 252,
    min_window_size: int = 50
) -> pd.DataFrame:
    """
    Get statistics about clipping thresholds for analysis.
    
    Args:
        data: DataFrame with time series data
        value_column: Column name containing values
        clipping_column: Column for historical analysis
        window_size: Size of historical sliding window
        min_window_size: Minimum window size
        
    Returns:
        DataFrame with clipping statistics for each time point
    """
    if clipping_column is None:
        clipping_column = value_column
        
    stats_list = []
    
    for t in range(min_window_size, len(data)):
        window_start = max(0, t - window_size)
        window_end = t
        
        historical_data = data[clipping_column].iloc[window_start:window_end]
        current_value = data[value_column].iloc[t]
        
        if len(historical_data) == 0:
            continue
            
        hist_mean = historical_data.mean()
        hist_std = historical_data.std()
        hist_max = historical_data.max()
        hist_min = historical_data.min()
        
        stats = {
            'date': data.index[t],
            'current_value': current_value,
            'hist_mean': hist_mean,
            'hist_std': hist_std,
            'hist_max': hist_max,
            'hist_min': hist_min,
            'is_extreme_high': current_value > hist_max,
            'is_extreme_low': current_value < hist_min,
            'window_size_actual': len(historical_data)
        }
        
        if hist_std > 0:
            if current_value > hist_max:
                max_std_from_mean = (hist_max - hist_mean) / hist_std
                stats['high_bottom_threshold'] = hist_mean + max_std_from_mean * hist_std
                stats['high_upper_threshold'] = stats['high_bottom_threshold'] + 0.5 * hist_std
            
            if current_value < hist_min:
                min_std_from_mean = (hist_min - hist_mean) / hist_std
                stats['low_upper_threshold'] = hist_mean + min_std_from_mean * hist_std
                stats['low_bottom_threshold'] = stats['low_upper_threshold'] - 0.5 * hist_std
        
        stats_list.append(stats)
    
    return pd.DataFrame(stats_list).set_index('date')