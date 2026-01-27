# Mean Reversion Strategy: KRR Forecasting & GA Portfolio Optimization

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A time-series forecasting and portfolio optimization framework combining Kernel Ridge Regression (KRR) ensemble models with Genetic Algorithm-based portfolio management for automated trading strategies.

---

## Overview

This project provides a complete pipeline for financial asset price forecasting and portfolio optimization. It combines machine learning models for multi-step-ahead predictions with evolutionary algorithms for risk-aware portfolio rebalancing.

**Key Components:**
- **Multi-kernel ensemble forecasting** with automatic kernel selection
- **Autoregressive recursive prediction** for configurable forecast horizons
- **Walk-forward validation** with proper time series handling
- **Genetic algorithm portfolio optimization** with tournament selection
- **Multi-asset portfolio management** with automated rebalancing

**Target Audience:** Quantitative analysts, algorithmic traders, financial researchers, and ML practitioners working with time series forecasting.

---

## Table of Contents

- [Key Features](#key-features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Usage Guide](#usage-guide)
  - [Data Preparation](#data-preparation)
  - [Model Training](#model-training)
  - [Walk-Forward Validation](#walk-forward-validation)
  - [Portfolio Optimization](#portfolio-optimization)
- [API Reference](#api-reference)
- [Configuration Options](#configuration-options)
- [Examples](#examples)
- [Results and Output](#results-and-output)
- [Performance Considerations](#performance-considerations)
- [Troubleshooting](#troubleshooting)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

---

## Key Features

### 🔮 Advanced Forecasting Models

- **Multi-Kernel Ensemble**: Automatically selects from 5 kernel configurations
  - Radial Basis Function (RBF)
  - Rational Quadratic
  - Exponential Sine Squared (periodic patterns)
  - Composite kernels with DotProduct
  - White noise kernel for uncertainty modeling

- **Autoregressive Forecasting**: Recursive multi-step ahead predictions with configurable forecast horizons

- **Grid Search Optimization**: Automated hyperparameter tuning with time series cross-validation

### 🔄 Robust Validation Framework

- **Walk-Forward Validation**: Proper time series backtesting without look-ahead bias
- **Sliding & Growing Windows**: Flexible windowing strategies for stationary and non-stationary data
- **Automated Retraining**: Configurable model retraining intervals
- **Feature Scaling**: Optional StandardScaler integration

### 🧬 Genetic Algorithm Portfolio Optimization

- **Tournament-Style Selection**: Efficient evolutionary algorithm for weight optimization
- **Multiple Portfolio Types**:
  - Long-only portfolios with cash allocation
  - Long/short hedged portfolios
  - Buy-and-hold baseline comparison

- **Risk-Aware Optimization**:
  - Objective: Maximize P(future return > current return)
  - VaR-based tie-breaking for bear markets
  - Configurable risk thresholds

### 📊 Portfolio Management

- **Multi-Asset Support**: Manage portfolios with multiple assets
- **Automated Rebalancing**: Periodic portfolio adjustments based on predictions
- **Cost Basis Tracking**: LIFO/FIFO accounting methods
- **Comprehensive Metrics**: Total return, max drawdown, trade history

### 🚀 Production-Ready Features

- **Parallel Processing**: Multi-core support for model training
- **Efficient Storage**: Parquet format for data and results
- **Logging**: Comprehensive logging throughout the pipeline
- **Model Persistence**: Save and track model summaries and metadata

---

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository**:
```bash
git clone https://github.com/yourusername/mean-reversion-strategy.git
cd mean-reversion-strategy
```

2. **Create a virtual environment**:
```bash
python -m venv venv

# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

### Required Dependencies

```txt
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
scipy>=1.7.0
PyYAML>=5.4.0
tqdm>=4.62.0
pyarrow>=5.0.0  # For parquet support
```

---

## Quick Start

### Example 1: Simple Forecasting Pipeline

```python
from data_loader import DataLoader
from auto_kernel_ensemble import KernelAutoregressiveModel
from walk_forward import WalkForwardValidator

# Load data
data_loader = DataLoader(data_path="data/AAPL_backtesting.parquet")
features = [f'Lag {x}' for x in range(1, 11)]  # 10 lag features
data_loader.load_data(x=features, y=['30 Day Avg'], actuals=['Adj Close'])

# Initialize autoregressive model
model = KernelAutoregressiveModel(
    forecast_horizon=7,  # 7 days ahead
    n_lags=10,
    n_jobs=-1
)

# Set up walk-forward validator
validator = WalkForwardValidator(
    model=model,
    data_loader=data_loader,
    ticker="AAPL",
    lags=10,
    window_size=2000,
    window_type='sliding'
)

# Run validation
validator.run_walk_forward_validation(target_variable='30 Day Avg')

# Export results
validator.export_results("results/AAPL/")
```

### Example 2: Multi-Asset Portfolio Optimization

```python
from ga_optimizer import GeneticOptimizer, PortfolioManager
import pandas as pd

# Load prediction results for multiple assets
assets = ['AAPL', 'GOOGL', 'MSFT']
data_dict = {}
for asset in assets:
    df = pd.read_parquet(f'results/{asset}/{asset}.parquet')
    data_dict[asset] = df

# Initialize portfolio manager
portfolio = PortfolioManager(
    initial_capital=100000,
    cost_basis_method="LIFO"
)

# Run multi-asset backtest
metrics = portfolio.run_multi_asset_backtest(
    data_dict=data_dict,
    rebalance_frequency=5,      # Rebalance every 5 days
    population_size=50,          # GA population size
    num_samples=500,             # Monte Carlo samples
    portfolio_type="long"        # Long-only portfolio
)

# Display results
print(f"Total Return: {metrics['total_return']*100:.2f}%")
print(f"Max Drawdown: {metrics['max_drawdown']*100:.2f}%")
print(f"Number of Trades: {metrics['num_trades']}")
```

---

## Architecture

### System Overview

```
┌─────────────┐     ┌──────────────┐     ┌──────────────────┐     ┌──────────────────┐
│   Parquet   │────>│ Data Loader  │────>│  KRR Ensemble    │────>│   Walk-Forward   │
│   Dataset   │     │              │     │    Training      │     │    Validation    │
└─────────────┘     └──────────────┘     └──────────────────┘     └──────────────────┘
                                                                             │
                                                                             v
┌─────────────┐     ┌──────────────┐     ┌──────────────────┐     ┌──────────────────┐
│ Trade Exec  │<────│  Portfolio   │<────│   Genetic Algo   │<────│   Predictions    │
│             │     │  Rebalance   │     │  Optimization    │     │  + Uncertainty   │
└─────────────┘     └──────────────┘     └──────────────────┘     └──────────────────┘
```

### Core Components

#### 1. Data Management Layer
- **`data_loader.py`**: Time series data loading, windowing, and validation
  - Parquet file handling
  - Time window generation (sliding/growing)
  - Feature/target separation

#### 2. Forecasting Models Layer
- **`base_ensemble_model.py`**: Abstract base class defining model interface
- **`kernel_ridge_models.py`**: Multi-output KRR ensemble
  - Multiple kernel configurations
  - Grid search with TimeSeriesSplit
  - Prediction boundary constraints

- **`auto_kernel_ensemble.py`**: Single-output autoregressive KRR
  - Recursive multi-step forecasting
  - Horizon-specific uncertainty estimation

#### 3. Validation Framework Layer
- **`walk_forward.py`**: Backtesting engine
  - Proper time series validation
  - Automated model retraining
  - Feature scaling support
  - Results persistence

#### 4. Portfolio Optimization Layer
- **`ga_optimizer.py`**:
  - `GeneticOptimizer`: Tournament-based GA for weight optimization
  - `PortfolioManager`: Portfolio tracking, rebalancing, and performance metrics

#### 5. Orchestration Layer
- **`krr_experimentation.py`**: End-to-end pipeline coordination
- **`utils.py`**: Logging utilities

---

## Usage Guide

### Data Preparation

#### Expected Data Format

Your data should be in Parquet format with the following structure:

```python
# DataFrame structure
date (index)    Adj Close    Lag 1    Lag 2    ...    30 Day Avg
2020-01-01      150.25      148.50   147.80   ...     149.20
2020-01-02      151.30      150.25   148.50   ...     149.50
...
```

**Required columns:**
- DateTime index
- `Adj Close`: Actual closing prices (for validation)
- Feature columns: Lagged values or other features
- Target columns: Values to predict

#### Creating Lagged Features

```python
from krr_experimentation import load_data_transform

# Transform raw data into lagged features
df = load_data_transform(
    ticker='AAPL',
    lags=15,           # Number of lag features
    ewm=True,          # Use exponential weighted moving average
    persist=True       # Save transformed data
)
```

#### Loading Data

```python
from data_loader import DataLoader

data_loader = DataLoader(data_path="data/AAPL_backtesting.parquet")

# Specify features and targets
features = [f'Lag {x}' for x in range(1, 16)]
targets = ['30 Day Avg']
actuals = ['Adj Close']  # Ground truth for evaluation

data_loader.load_data(
    x=features,
    y=targets,
    actuals=actuals,
    adj_close=True
)
```

### Model Training

#### Kernel Ridge Ensemble (Multi-Output)

```python
from kernel_ridge_models import KernelRidgeEnsemble

# Initialize model
model = KernelRidgeEnsemble(
    training_metric='mse',  # Options: 'mse', 'rmse', 'r2_avg', 'r2_flat'
    random_state=42,
    n_jobs=-1  # Use all CPU cores
)

# Train on historical data
metrics = model.train_krr(
    x_train=X_train,
    y_train=y_train,
    alpha_params=[1e-2, 1e0, 1e3, 1e4]  # Regularization parameters
)

# Make predictions
predictions = model.predict_val(X_test)

# Generate prediction distributions
samples = model.predict_val_distribution(
    x=X_test,
    y=y_test,
    n_samples=1000
)
```

#### Kernel Autoregressive Model (Single-Output)

```python
from auto_kernel_ensemble import KernelAutoregressiveModel

# Initialize autoregressive model
model = KernelAutoregressiveModel(
    forecast_horizon=7,  # Predict 7 days ahead
    n_lags=15,          # Use 15 lag features
    training_metric='mse',
    n_jobs=-1
)

# Train model
metrics = model.train_historical(
    x=X_train,
    y=y_train,
    y_noise=y_train_noise  # Unsmoothed targets for residual calculation
)

# Make recursive predictions
predictions = model.predict_val(X_test)
```

### Walk-Forward Validation

Walk-forward validation ensures proper time series testing without look-ahead bias.

```python
from walk_forward import WalkForwardValidator

validator = WalkForwardValidator(
    model=model,
    data_loader=data_loader,
    ticker='AAPL',
    lags=15,
    window_size=2000,           # Training window size
    min_window_size=1000,       # Minimum window size
    step_size=1,                # Walk forward by 1 day
    model_retrain_interval=20,  # Retrain every 20 days
    use_scaling=True,           # Apply StandardScaler
    window_type='sliding',      # 'sliding' or 'growing'
    persist_samples=True,       # Save prediction distributions
    n_parallel_jobs=4
)

# Run validation
validator.run_walk_forward_validation(target_variable='30 Day Avg')

# Export results
validator.export_results('results/AAPL/')
```

#### Window Types

**Sliding Window**: Fixed-size window moves forward
```
Training:  [----2000 days----]
Testing:                      [1]
Next:       [----2000 days----]
                               [1]
```

**Growing Window**: Window grows from start
```
Training:  [----1000 days----]
Testing:                      [1]
Next:      [------1001 days-------]
                                  [1]
```

### Portfolio Optimization

#### Single-Asset Backtesting

```python
from ga_optimizer import PortfolioManager
import pandas as pd

# Load predictions with std
data = pd.read_parquet('results/AAPL/AAPL.parquet')

# Initialize portfolio
portfolio = PortfolioManager(
    initial_capital=100000,
    cost_basis_method="LIFO"  # Or "FIFO"
)

# Run backtest
metrics = portfolio.run_backtest(
    data=data,
    ticker='AAPL',
    rebalance_frequency=1,   # Daily rebalancing
    population_size=100,
    num_samples=1000
)
```

#### Multi-Asset Portfolio

```python
# Prepare data dictionary
assets = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']
data_dict = {
    asset: pd.read_parquet(f'results/{asset}/{asset}.parquet')
    for asset in assets
}

# Initialize portfolio
portfolio = PortfolioManager(initial_capital=100000)

# Run multi-asset backtest
metrics = portfolio.run_multi_asset_backtest(
    data_dict=data_dict,
    rebalance_frequency=5,
    population_size=50,
    num_samples=500,
    portfolio_type="long"  # 'long', 'short', or 'buy_and_hold'
)

# Results
print(f"Final Value: ${metrics['final_value']:,.2f}")
print(f"Total Return: {metrics['total_return']*100:.2f}%")
print(f"Max Drawdown: {metrics['max_drawdown']*100:.2f}%")
print(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 'N/A')}")
```

#### Portfolio Types

1. **Long-Only** (`portfolio_type="long"`):
   - Positive weights only
   - Weights sum between 0-1 (remainder is cash)
   - Suitable for traditional long portfolios

2. **Long/Short** (`portfolio_type="short"`):
   - Separate long and short positions
   - Hedged portfolio strategy
   - Weights for both long and short sides

3. **Buy-and-Hold** (`portfolio_type="buy_and_hold"`):
   - Equal weight allocation
   - No optimization performed
   - Baseline comparison strategy

---

## API Reference

### DataLoader

```python
class DataLoader(data_path: str)
```

**Key Methods:**

- `load_data(x, y, actuals=None, adj_close=True)`: Load parquet data with specified features and targets
- `get_time_windows(window_size=252, min_window_size=100, window_type='sliding')`: Generate validation windows
- `get_window_data(start_idx, end_idx, target_columns, feature_columns)`: Extract data for a time window
- `get_prediction_point(idx, feature_columns)`: Get features for single prediction
- `get_actual_value(idx, target_columns)`: Get ground truth for validation
- `get_date_for_index(idx)`: Get date for index position

### KernelRidgeEnsemble

```python
class KernelRidgeEnsemble(
    training_metric='mse',
    random_state=42,
    n_jobs=-1
)
```

**Key Methods:**

- `train_krr(x_train, y_train, alpha_params=None)`: Train with grid search CV
- `predict_val(x)`: Point predictions
- `predict_val_distribution(x, y, n_samples=1000)`: Generate prediction samples
- `get_model_summary()`: Model metadata and performance metrics
- `get_feature_importance_proxy(X)`: Feature importance estimates
- `apply_yield_constraints(predictions, is_inverse_transformed=False)`: Apply boundary constraints

**Training Metrics:**
- `'mse'`: Mean squared error (uniform average)
- `'mse_flat'`: Flattened MSE across all outputs
- `'rmse'`: Root mean squared error
- `'r2_avg'`: Average R² across outputs
- `'r2_flat'`: R² on flattened predictions

### KernelAutoregressiveModel

```python
class KernelAutoregressiveModel(
    forecast_horizon=1,
    n_lags=5,
    training_metric='mse',
    random_state=42,
    n_jobs=-1
)
```

**Key Methods:**

- `train_historical(x, y, y_noise)`: Train autoregressive model
- `predict_val(x)`: Recursive forecasting to horizon
- `predict_val_distribution(x, y, n_samples=1000)`: Distribution forecasting
- `get_model_summary()`: Model information and training scores
- `predict_autoregressive_batch(X, steps)`: Batch recursive predictions

### WalkForwardValidator

```python
class WalkForwardValidator(
    model,
    data_loader,
    ticker,
    lags=10,
    window_size=3000,
    min_window_size=2000,
    step_size=1,
    model_retrain_interval=20,
    use_scaling=False,
    window_type='sliding',
    persist_samples=True,
    n_parallel_jobs=5
)
```

**Key Methods:**

- `run_walk_forward_validation(target_variable)`: Execute validation loop
- `run_single_prediction(train_start_idx, train_end_idx, predict_idx, ...)`: Single prediction step
- `export_results(filepath)`: Save results and model summaries to parquet

### GeneticOptimizer

```python
class GeneticOptimizer(
    current_portfolio_value,
    current_cumulative_return=0.0,
    portfolio_type='long',
    var_threshold=0.05
)
```

**Key Methods:**

- `optimize(current_prices, predictions, population_size=100, num_samples=1000)`: Run GA optimization
- `generate_population(n_assets, population_size)`: Create random weight vectors
- `tournament_selection(population, sampled_returns)`: Tournament-style selection
- `compute_portfolio_returns(weights, sampled_returns)`: Calculate portfolio returns
- `calculate_var(returns, threshold=None)`: Value at Risk calculation

### PortfolioManager

```python
class PortfolioManager(
    initial_capital,
    cost_basis_method='LIFO'
)
```

**Key Methods:**

- `run_backtest(data, ticker, rebalance_frequency=1, ...)`: Single-asset backtest
- `run_multi_asset_backtest(data_dict, rebalance_frequency=5, portfolio_type='long', ...)`: Multi-asset backtest
- `get_portfolio_value(current_prices)`: Calculate current portfolio value
- `get_current_cumulative_return()`: Current cumulative return
- `update_positions(optimal_weights, current_prices, current_date)`: Rebalance portfolio
- `calculate_metrics()`: Compute performance metrics

---

## Configuration Options

### Model Configuration

```python
# Kernel Ridge Ensemble
model = KernelRidgeEnsemble(
    training_metric='mse',     # 'mse', 'mse_flat', 'rmse', 'r2_avg', 'r2_flat'
    random_state=42,
    n_jobs=-1                  # -1 for all cores
)

# Autoregressive Model
model = KernelAutoregressiveModel(
    forecast_horizon=7,        # Days to forecast ahead
    n_lags=15,                 # Number of lag features
    training_metric='mse',
    random_state=42,
    n_jobs=-1
)
```

### Validation Configuration

```python
validator = WalkForwardValidator(
    model=model,
    data_loader=data_loader,
    ticker='AAPL',
    lags=15,
    window_size=3000,          # Training window size
    min_window_size=2000,      # Minimum window
    step_size=1,               # Days to walk forward
    model_retrain_interval=20, # Retrain every N days
    use_scaling=True,          # Apply StandardScaler
    window_type='sliding',     # 'sliding' or 'growing'
    persist_samples=True,      # Save prediction distributions
    n_parallel_jobs=4
)
```

### Genetic Algorithm Configuration

```python
optimizer = GeneticOptimizer(
    current_portfolio_value=100000,
    current_cumulative_return=0.15,  # 15% current return
    portfolio_type='long',            # 'long', 'short', 'buy_and_hold'
    var_threshold=0.05                # 5% VaR threshold
)

optimal_weights, fitness = optimizer.optimize(
    current_prices=prices_dict,
    predictions=predictions_dict,
    population_size=100,              # Number of weight vectors
    num_samples=1000                  # Monte Carlo samples
)
```

---

## Examples

### Example 1: End-to-End Pipeline

```python
from krr_experimentation import run_end_to_end_pipeline

# Process multiple assets
portfolio_assets = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']

for ticker in portfolio_assets:
    print(f"Processing {ticker}...")
    run_end_to_end_pipeline(
        ticker=ticker,
        lags=15,
        ewm=True,
        persist=True
    )
```

### Example 2: Custom Validation with Scaling

```python
from data_loader import DataLoader
from kernel_ridge_models import KernelRidgeEnsemble
from walk_forward import WalkForwardValidator

# Load data
data_loader = DataLoader("data/asset.parquet")
features = [f'Lag {i}' for i in range(1, 11)]
data_loader.load_data(x=features, y=['price'], actuals=['actual_price'])

# Initialize model
model = KernelRidgeEnsemble(
    training_metric='r2_avg',
    n_jobs=-1
)

# Set up validator with scaling
validator = WalkForwardValidator(
    model=model,
    data_loader=data_loader,
    ticker='ASSET',
    lags=10,
    window_size=1500,
    use_scaling=True,           # Enable feature scaling
    window_type='growing'       # Growing window for non-stationary data
)

# Run validation
validator.run_walk_forward_validation(target_variable='price')
validator.export_results('results/ASSET/')
```

### Example 3: Long/Short Hedged Portfolio

```python
from ga_optimizer import PortfolioManager

# Load predictions
data_dict = {
    'AAPL': pd.read_parquet('results/AAPL/AAPL.parquet'),
    'GOOGL': pd.read_parquet('results/GOOGL/GOOGL.parquet'),
    'TSLA': pd.read_parquet('results/TSLA/TSLA.parquet')
}

# Initialize portfolio
portfolio = PortfolioManager(
    initial_capital=100000,
    cost_basis_method="LIFO"
)

# Run hedged portfolio backtest
metrics = portfolio.run_multi_asset_backtest(
    data_dict=data_dict,
    rebalance_frequency=5,
    population_size=64,
    num_samples=1000,
    portfolio_type="short"      # Long/short hedged strategy
)

print(f"\nHedged Portfolio Results:")
print(f"Total Return: {metrics['total_return']*100:.2f}%")
print(f"Max Drawdown: {metrics['max_drawdown']*100:.2f}%")
print(f"Final Value: ${metrics['final_value']:,.2f}")
```

### Example 4: Buy-and-Hold Comparison

```python
# Run optimized portfolio
portfolio_optimized = PortfolioManager(initial_capital=100000)
metrics_optimized = portfolio_optimized.run_multi_asset_backtest(
    data_dict=data_dict,
    portfolio_type="long"
)

# Run buy-and-hold baseline
portfolio_baseline = PortfolioManager(initial_capital=100000)
metrics_baseline = portfolio_baseline.run_multi_asset_backtest(
    data_dict=data_dict,
    portfolio_type="buy_and_hold"
)

# Compare results
print(f"Optimized Return: {metrics_optimized['total_return']*100:.2f}%")
print(f"Buy-and-Hold Return: {metrics_baseline['total_return']*100:.2f}%")
print(f"Excess Return: {(metrics_optimized['total_return'] - metrics_baseline['total_return'])*100:.2f}%")
```

---

## Results and Output

### Walk-Forward Validation Outputs

**1. Prediction Results** (`{ticker}.parquet`)
```python
# Columns:
- date: Prediction date
- actual_value: Ground truth values (list)
- prediction: Model predictions (list)
- std: Standard deviation of predictions (float)
- best_kernel: Selected kernel name
- best_alpha: Regularization parameter
- retrain: Boolean indicating if model was retrained
```

**2. Model Summaries** (`{ticker}_model_summary.parquet`)
```python
# Columns:
- date: Date of model training
- model_type: Model class name
- training_scores: Dictionary of training metrics
- best_params: Selected hyperparameters
- feature_importance: Feature importance scores
```

**3. Prediction Samples** (optional, `samples/{ticker}/{date}.parquet`)
- Distribution samples for uncertainty quantification
- 1000 samples per prediction by default

### Portfolio Backtest Outputs

**Performance Metrics Dictionary:**
```python
{
    'total_return': 0.234,           # 23.4% total return
    'max_drawdown': 0.087,           # 8.7% maximum drawdown
    'final_value': 123400.00,        # Final portfolio value
    'initial_capital': 100000.00,    # Starting capital
    'num_trades': 145,               # Number of trades executed
    'num_days': 500,                 # Trading days
    'tickers': ['AAPL', 'GOOGL'],   # Assets traded
    'portfolio_type': 'long'         # Strategy type
}
```

**Trade History** (`portfolio.trades_history`)
```python
# List of dictionaries:
{
    'date': datetime,
    'ticker': 'AAPL',
    'action': 'BUY',         # 'BUY' or 'SELL'
    'shares': 10.5,
    'price': 150.25,
    'amount': 1577.62
}
```

**Weight History** (`portfolio.weights_history`)
```python
# List of dictionaries:
{
    'date': datetime,
    'weights': {'AAPL': 0.35, 'GOOGL': 0.45, 'CASH': 0.20},
    'fitness_score': 0.67  # P(future return > current return)
}
```

---

## Performance Considerations

### Computational Efficiency

**Parallel Processing:**
```python
# Use all CPU cores for model training
model = KernelRidgeEnsemble(n_jobs=-1)

# Parallel jobs in walk-forward validation
validator = WalkForwardValidator(
    n_parallel_jobs=4,  # Adjust based on your CPU
    model=model,
    ...
)
```

**Model Retraining Trade-off:**
```python
# More frequent retraining = higher accuracy but slower
validator = WalkForwardValidator(
    model_retrain_interval=10,  # Retrain every 10 days
    ...
)

# Less frequent retraining = faster but may be less accurate
validator = WalkForwardValidator(
    model_retrain_interval=30,  # Retrain every 30 days
    ...
)
```

**GA Optimization Speed:**
```python
# Balance accuracy vs. speed
optimizer.optimize(
    population_size=50,   # Smaller = faster, less optimal
    num_samples=500       # Fewer samples = faster, more noise
)

# Higher quality (slower)
optimizer.optimize(
    population_size=128,  # Larger = slower, more optimal
    num_samples=2000      # More samples = slower, less noise
)
```

### Memory Management

**Window Size Impact:**
```python
# Large windows = more memory
validator = WalkForwardValidator(
    window_size=5000,     # High memory usage
    ...
)

# Sliding windows prevent unbounded growth
validator = WalkForwardValidator(
    window_type='sliding',  # Fixed memory footprint
    ...
)

# Growing windows increase memory over time
validator = WalkForwardValidator(
    window_type='growing',  # Increasing memory usage
    ...
)
```

**Sample Persistence:**
```python
# Save samples to disk instead of memory
validator = WalkForwardValidator(
    persist_samples=True,   # Save to disk
    ...
)
```

### Optimization Tips

1. **For stationary data**: Use `window_type='sliding'`
2. **For non-stationary data**: Use `window_type='growing'`
3. **High volatility**: Reduce `model_retrain_interval`
4. **Low volatility**: Increase `model_retrain_interval`
5. **Different feature scales**: Use `use_scaling=True`
6. **Production systems**: Start with conservative GA settings and tune based on performance

### Benchmark Performance

Typical performance on a modern workstation (8-core CPU, 32GB RAM):

| Task | Window Size | Time (approx) |
|------|-------------|---------------|
| Model training | 2000 samples | 30-60 seconds |
| Single prediction | - | 0.1-0.5 seconds |
| Walk-forward (1000 steps) | 2000 samples | 1-2 hours |
| GA optimization (50 pop) | - | 5-10 seconds |
| Multi-asset backtest (3 assets, 500 days) | - | 30-60 minutes |

---

## Troubleshooting

### Common Issues and Solutions

#### Issue: Negative Yield Predictions

**Problem**: Model predicts negative values for yields/prices

**Solution**:
```python
# Apply yield constraints
predictions = model.apply_yield_constraints(
    predictions=raw_predictions,
    is_inverse_transformed=True
)
```

#### Issue: Memory Errors with Large Windows

**Problem**: `MemoryError` during walk-forward validation

**Solutions**:
1. Reduce window size:
```python
validator = WalkForwardValidator(
    window_size=1500,  # Reduced from 3000
    ...
)
```

2. Use sliding windows instead of growing:
```python
validator = WalkForwardValidator(
    window_type='sliding',  # Fixed memory
    ...
)
```

3. Enable sample persistence:
```python
validator = WalkForwardValidator(
    persist_samples=True,  # Save to disk
    ...
)
```

#### Issue: Slow Training Performance

**Problem**: Model training takes too long

**Solutions**:
1. Reduce kernel configurations:
```python
# Edit kernel_ridge_models.py or auto_kernel_ensemble.py
# Comment out some kernels in create_kernel_configuration()
```

2. Reduce CV splits:
```python
# In train_krr method, n_splits defaults to 3-5
# Modify TimeSeriesSplit parameters in the model file
```

3. Increase retrain interval:
```python
validator = WalkForwardValidator(
    model_retrain_interval=30,  # Less frequent
    ...
)
```

#### Issue: NaN or Null Values

**Problem**: `ValueError: Feature DataFrame contains null values`

**Solution**: Data is automatically cleaned with `dropna()` in `load_data()`. Ensure your source data has sufficient non-null values:
```python
# Check data before loading
df = pd.read_parquet("data/asset.parquet")
print(f"Null values: {df.isnull().sum()}")
print(f"Shape before dropna: {df.shape}")
df_clean = df.dropna()
print(f"Shape after dropna: {df_clean.shape}")
```

#### Issue: Poor Prediction Performance

**Problem**: Model predictions are inaccurate

**Diagnostic Steps**:
1. Check training metrics:
```python
metrics = model.train_historical(x, y, y_noise)
print(f"Training R²: {metrics['r2']}")
print(f"Training RMSE: {metrics['rmse']}")
```

2. Try different training metrics:
```python
model = KernelRidgeEnsemble(
    training_metric='r2_avg',  # Instead of 'mse'
    ...
)
```

3. Enable feature scaling:
```python
validator = WalkForwardValidator(
    use_scaling=True,  # Normalize features
    ...
)
```

4. Adjust forecast horizon:
```python
# Shorter horizons are generally more accurate
model = KernelAutoregressiveModel(
    forecast_horizon=3,  # Instead of 7
    ...
)
```

#### Issue: GA Optimization Returns Zero Weights

**Problem**: Portfolio optimizer allocates all capital to cash

**Explanation**: This occurs in bear markets when P(future return > current return) ≈ 0 for all assets

**Expected Behavior**: The algorithm uses VaR as a tie-breaker to minimize losses

**Solutions**:
1. Review market conditions in your data
2. Consider using `portfolio_type='short'` for hedged positions
3. Adjust `var_threshold` for risk tolerance

---

## Project Structure

```
mean-reversion-strategy/
│
├── src/                                # Source code
│   ├── base_ensemble_model.py         # Abstract base class for models
│   ├── kernel_ridge_models.py         # Multi-output KRR ensemble
│   ├── auto_kernel_ensemble.py        # Autoregressive KRR model
│   ├── data_loader.py                 # Data loading and windowing
│   ├── walk_forward.py                # Validation framework
│   ├── ga_optimizer.py                # Genetic algorithm & portfolio manager
│   ├── krr_experimentation.py         # End-to-end pipeline orchestration
│   └── utils.py                       # Logging utilities
│
├── data/                               # Data directory
│   ├── sample/                        # Processed sample data
│   │   └── {ticker}_backtesting.parquet
│   └── {ticker}.parquet               # Raw data files
│
├── results/                            # Output directory
│   ├── {ticker}/                      # Per-asset results
│   │   ├── {ticker}.parquet          # Predictions
│   │   └── {ticker}_model_summary.parquet
│   └── samples/                       # Prediction distributions
│       └── {ticker}/
│           └── {date}.parquet
│
├── notebooks/                          # Jupyter notebooks (if any)
│   ├── Full_Experiment1.ipynb
│   ├── full_experiement_2.ipynb
│   └── ga_optimization_report.ipynb
│
├── logs/                               # Log files
│   └── *.log
│
├── claude/                             # Planning and documentation
│   └── readme_plan.md
│
├── requirements.txt                    # Python dependencies
├── README.md                          # This file
├── LICENSE                            # License file
└── .gitignore                         # Git ignore rules
```

---

## Contributing

Contributions are welcome! Please follow these guidelines:

### Code Style

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guide
- Use [Black](https://github.com/psf/black) for code formatting
- Add type hints using the `typing` module
- Write docstrings for all public methods

### Docstring Format

```python
def function_name(param1: Type1, param2: Type2) -> ReturnType:
    """
    Brief description of function.

    Args:
        param1: Description of param1
        param2: Description of param2

    Returns:
        Description of return value

    Raises:
        ValueError: When invalid input is provided
    """
```

### Pull Request Process

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Make your changes with clear commit messages
4. Add tests for new functionality
5. Ensure all tests pass
6. Update documentation as needed
7. Submit a pull request with a clear description

### Testing

```bash
# Run unit tests (if available)
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_data_loader.py
```

### Reporting Issues

When reporting bugs, please include:
- Python version
- Operating system
- Minimal code example to reproduce
- Error messages and stack traces
- Expected vs. actual behavior

---

## Future Enhancements

Potential features for future development:

### Model Enhancements
- [ ] Additional kernel types (Matern, Polynomial)
- [ ] LSTM and Transformer models as alternatives
- [ ] Ensemble model stacking
- [ ] Online learning / incremental training

### Portfolio Optimization
- [ ] Multi-objective optimization (return + Sharpe + drawdown)
- [ ] Transaction cost modeling
- [ ] Tax-aware rebalancing
- [ ] Dynamic risk budgeting

### Risk Management
- [ ] Conditional Value at Risk (CVaR)
- [ ] Sortino ratio optimization
- [ ] Stress testing framework
- [ ] Correlation-aware position sizing

### Infrastructure
- [ ] Real-time data integration
- [ ] Live trading API connectors
- [ ] Web dashboard for monitoring
- [ ] Model versioning and tracking
- [ ] Hyperparameter tuning with Optuna

### Performance
- [ ] GPU acceleration for model training
- [ ] Distributed computing support
- [ ] Caching strategies
- [ ] Model compression

---

## References and Citations

### Kernel Ridge Regression
- Kernel Ridge Regression: [scikit-learn documentation](https://scikit-learn.org/stable/modules/kernel_ridge.html)
- Gaussian Process kernels: [scikit-learn GP kernels](https://scikit-learn.org/stable/modules/gaussian_process.html#kernels-for-gaussian-processes)

### Genetic Algorithms
- Genetic Algorithms in Portfolio Optimization: Holland, J.H. (1992). "Adaptation in Natural and Artificial Systems"
- Tournament Selection: Goldberg, D.E., & Deb, K. (1991). "A Comparative Analysis of Selection Schemes Used in Genetic Algorithms"

### Time Series Validation
- Walk-Forward Analysis: Pardo, R. (2008). "The Evaluation and Optimization of Trading Strategies"
- Time Series Cross-Validation: Bergmeir, C., & Benítez, J.M. (2012). "On the use of cross-validation for time series predictor evaluation"

### Risk Management
- Value at Risk: Jorion, P. (2006). "Value at Risk: The New Benchmark for Managing Financial Risk"

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2025

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## Authors and Acknowledgments

**Primary Author**: Based on work by Terrill Toe

**Acknowledgments**:
- Genetic Algorithm portfolio optimization implementation inspired by evolutionary computation research
- Kernel Ridge Regression ensemble methodology based on scikit-learn framework
- Walk-forward validation framework following industry best practices for time series backtesting

---

## Contact

For questions, suggestions, or bug reports:

- **GitHub Issues**: [Create an issue](https://github.com/yourusername/mean-reversion-strategy/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/mean-reversion-strategy/discussions)
- **Email**: your.email@example.com

---

## Changelog

### Version 1.0.0 (Initial Release)
- Multi-kernel ensemble forecasting with KRR
- Autoregressive recursive prediction model
- Walk-forward validation framework
- Genetic algorithm portfolio optimization
- Multi-asset portfolio management
- Comprehensive documentation and examples

---

**⭐ If you find this project useful, please consider giving it a star on GitHub!**

