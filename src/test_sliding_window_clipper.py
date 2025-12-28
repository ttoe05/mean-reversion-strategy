"""
Test script for sliding window clipping functionality.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sliding_window_clipper import sliding_window_clip, batch_sliding_window_clip, get_clipping_statistics

def create_test_data(n_points: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Create test time series data with some extreme values."""
    np.random.seed(seed)
    
    # Create date range
    dates = pd.date_range(start='2020-01-01', periods=n_points, freq='D')
    
    # Generate base time series (random walk)
    base_values = np.cumsum(np.random.normal(0, 1, n_points))
    
    # Add some extreme outliers (only if we have enough points)
    if n_points > 100:
        extreme_indices = np.random.choice(range(100, n_points), size=min(20, n_points-100), replace=False)
        n_outliers = len(extreme_indices)
        for idx in extreme_indices[:n_outliers//2]:  # High outliers
            base_values[idx] = base_values[idx] + np.random.uniform(5, 10)
        for idx in extreme_indices[n_outliers//2:]:  # Low outliers  
            base_values[idx] = base_values[idx] - np.random.uniform(5, 10)
    else:
        # For small datasets, just add a few outliers
        if n_points > 20:
            outlier_indices = [n_points//3, 2*n_points//3]
            base_values[outlier_indices[0]] += 5
            base_values[outlier_indices[1]] -= 5
    
    # Create second series for testing different clipping column
    reference_values = base_values + np.random.normal(0, 0.5, n_points)
    
    df = pd.DataFrame({
        'value': base_values,
        'reference': reference_values
    }, index=dates)
    
    return df

def test_basic_clipping():
    """Test basic clipping functionality."""
    print("Testing basic sliding window clipping...")
    
    # Create test data
    data = create_test_data(n_points=500)
    
    # Apply clipping
    clipped_values = sliding_window_clip(
        data=data,
        value_column='value',
        window_size=50,
        min_window_size=20,
        random_seed=42
    )
    
    # Check results
    original_extremes = (data['value'] > data['value'].rolling(50).max().shift(1)) | \
                       (data['value'] < data['value'].rolling(50).min().shift(1))
    
    print(f"Original extreme values: {original_extremes.sum()}")
    print(f"Values changed: {(data['value'] != clipped_values).sum()}")
    
    # Verify no NaN values introduced
    assert not clipped_values.isna().any(), "Clipped values contain NaN"
    print("✓ Basic clipping test passed")

def test_different_clipping_column():
    """Test using different column for historical analysis."""
    print("Testing different clipping column...")
    
    data = create_test_data(n_points=300)
    
    # Use reference column for historical window
    clipped_values = sliding_window_clip(
        data=data,
        value_column='value',
        clipping_column='reference',
        window_size=30,
        min_window_size=15,
        random_seed=42
    )
    
    print(f"Values changed: {(data['value'] != clipped_values).sum()}")
    print("✓ Different clipping column test passed")

def test_batch_clipping():
    """Test batch processing of multiple columns."""
    print("Testing batch clipping...")
    
    data = create_test_data(n_points=200)
    data['value2'] = data['value'] * 1.2 + np.random.normal(0, 0.3, len(data))
    
    # Apply batch clipping
    clipped_df = batch_sliding_window_clip(
        data=data,
        columns=['value', 'value2'],
        window_size=40,
        min_window_size=20,
        random_seed=42
    )
    
    for col in ['value', 'value2']:
        changes = (data[col] != clipped_df[col]).sum()
        print(f"Column {col}: {changes} values changed")
    
    print("✓ Batch clipping test passed")

def test_statistics():
    """Test clipping statistics function."""
    print("Testing clipping statistics...")
    
    data = create_test_data(n_points=150)
    
    stats_df = get_clipping_statistics(
        data=data,
        value_column='value',
        window_size=30,
        min_window_size=20
    )
    
    print(f"Statistics calculated for {len(stats_df)} points")
    print(f"Extreme high values: {stats_df['is_extreme_high'].sum()}")
    print(f"Extreme low values: {stats_df['is_extreme_low'].sum()}")
    
    # Check required columns exist
    required_cols = ['current_value', 'hist_mean', 'hist_std', 'hist_max', 'hist_min']
    for col in required_cols:
        assert col in stats_df.columns, f"Missing column: {col}"
    
    print("✓ Statistics test passed")

def test_edge_cases():
    """Test edge cases and error handling."""
    print("Testing edge cases...")
    
    # Test insufficient data
    small_data = create_test_data(n_points=30)
    try:
        sliding_window_clip(small_data, 'value', window_size=100, min_window_size=50)
        assert False, "Should have raised IndexError"
    except IndexError:
        print("✓ Insufficient data error handled correctly")
    
    # Test invalid column
    data = create_test_data(n_points=100)
    try:
        sliding_window_clip(data, 'nonexistent_column')
        assert False, "Should have raised ValueError"
    except ValueError:
        print("✓ Invalid column error handled correctly")
    
    # Test non-datetime index
    data_no_datetime = data.reset_index()
    try:
        sliding_window_clip(data_no_datetime, 'value')
        assert False, "Should have raised ValueError"
    except ValueError:
        print("✓ Non-datetime index error handled correctly")

def create_visualization():
    """Create visualization of clipping results."""
    print("Creating visualization...")
    
    data = create_test_data(n_points=300, seed=123)
    
    # Apply clipping
    clipped_values = sliding_window_clip(
        data=data,
        value_column='value',
        window_size=50,
        min_window_size=25,
        random_seed=123
    )
    
    # Get statistics
    stats = get_clipping_statistics(
        data=data,
        value_column='value',
        window_size=50,
        min_window_size=25
    )
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot 1: Original vs Clipped values
    ax1.plot(data.index, data['value'], label='Original', alpha=0.7, linewidth=1)
    ax1.plot(data.index, clipped_values, label='Clipped', linewidth=1.5)
    
    # Highlight clipped points
    clipped_mask = data['value'] != clipped_values
    ax1.scatter(data.index[clipped_mask], data['value'][clipped_mask], 
               color='red', s=30, label='Extreme values', zorder=5)
    ax1.scatter(data.index[clipped_mask], clipped_values[clipped_mask], 
               color='green', s=30, label='Clipped values', zorder=5)
    
    ax1.set_title('Sliding Window Clipping Results')
    ax1.set_ylabel('Value')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Rolling statistics
    ax2.plot(stats.index, stats['hist_mean'], label='Historical Mean', alpha=0.8)
    ax2.fill_between(stats.index, 
                     stats['hist_mean'] - stats['hist_std'], 
                     stats['hist_mean'] + stats['hist_std'], 
                     alpha=0.2, label='±1 Std Dev')
    ax2.plot(stats.index, stats['hist_max'], '--', alpha=0.6, label='Historical Max')
    ax2.plot(stats.index, stats['hist_min'], '--', alpha=0.6, label='Historical Min')
    
    ax2.set_title('Historical Statistics (Sliding Window)')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Value')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/Users/mma0277/Documents/Development/mean_reversion_strategy/mean-reversion-strategy/src/clipping_visualization.png', 
                dpi=150, bbox_inches='tight')
    plt.close()
    
    print("✓ Visualization saved as 'clipping_visualization.png'")

if __name__ == "__main__":
    print("Running sliding window clipper tests...\n")
    
    try:
        test_basic_clipping()
        test_different_clipping_column()
        test_batch_clipping()
        test_statistics()
        test_edge_cases()
        create_visualization()
        
        print("\n🎉 All tests passed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        raise