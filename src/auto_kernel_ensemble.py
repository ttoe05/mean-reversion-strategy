"""
Kernel Autoregressive Model for bond yield forecasting.
Implements single-output recursive forecasting with configurable forecast horizons.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.gaussian_process.kernels import (
    WhiteKernel, ConstantKernel, RBF, RationalQuadratic, ExpSineSquared, DotProduct
)
from scipy.stats import multivariate_normal
import warnings

from base_ensemble_model import BaseEnsembleModel

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')


class KernelAutoregressiveModel(BaseEnsembleModel):
    """
    Kernel Autoregressive Model using recursive prediction for multi-step forecasting.
    
    Transforms multi-output prediction into single-output autoregressive forecasting
    where each prediction uses previous predictions as lag features.
    """
    
    def __init__(self, 
                 forecast_horizon: int = 1,
                 n_lags: int = 5,
                 training_metric: str = 'mse',
                 random_state: int = 42,
                 n_jobs: int = -1):
        """
        Initialize Kernel Autoregressive Model.
        
        Args:
            forecast_horizon: Number of steps to forecast ahead (r)
            n_lags: Number of lag features to use (p)
            training_metric: Metric for grid search optimization
            random_state: Random state for reproducibility
            n_jobs: Number of parallel jobs
        """
        super().__init__(random_state=random_state, n_jobs=n_jobs)
        
        self.forecast_horizon = forecast_horizon
        self.n_lags = n_lags
        self.training_metric = training_metric
        
        # Model components
        self.best_model = None
        self.best_params = None
        self.training_scores = {}
        self.horizon_std = None
        self.best_kernel_name = None
        
        # Kernel configurations
        self.kernel_configs = None
        self.best_alpha_name = None
        
        # Grid search parameters
        self.param_grid = {
            'alpha': [0.001, 0.01, 0.1, 1.0, 10.0],
            'gamma': [0.001, 0.01, 0.1, 1.0, 10.0]
        }

        self.create_kernel_configuration()

    def create_kernel_configuration(self) -> List:
        """
        Create list of available kernels based on analysis/kernel_regression.ipynb.

        Returns:
            List of kernel configurations
        """
        kernel_test = (ConstantKernel(1.0, (1e-3, 1e3)) *
                       # DotProduct() *
                       RBF(length_scale=1.5, length_scale_bounds=(1e-7, 1e7)) +
                       WhiteKernel(noise_level=1e-5))

        kernel_test2 = (ConstantKernel(1.0, (1e-3, 1e3)) *
                        # DotProduct() *
                        RationalQuadratic(length_scale=1.0, alpha=0.1,
                                          length_scale_bounds=(1e-5, 1e5),
                                          alpha_bounds=(1e-5, 1e5)) +
                        WhiteKernel(noise_level=1e-5))

        kernel_test3 = (ConstantKernel(1.0, (1e-3, 1e3)) *
                        DotProduct() +
                        RationalQuadratic(length_scale=1.0, alpha=0.1,
                                          length_scale_bounds=(1e-5, 1e5),
                                          alpha_bounds=(1e-5, 1e5)) +
                        WhiteKernel(noise_level=1e-5))

        kernel_test4 = (ConstantKernel(1.0, (1e-3, 1e3)) *
                        DotProduct() +
                        ExpSineSquared(length_scale=2, periodicity=20.0,
                                       length_scale_bounds=(0.01, 10),
                                       periodicity_bounds=(1e-2, 1e2)) +
                        WhiteKernel(noise_level=1e-5))

        kernel_test5 = (ConstantKernel(1.0, (1e-3, 1e3)) *
                        # DotProduct() *
                        ExpSineSquared(length_scale=2, periodicity=20.0,
                                       length_scale_bounds=(0.01, 10),
                                       periodicity_bounds=(1e-2, 1e2)) +
                        WhiteKernel(noise_level=1e-5))
        kernels = [
            {'kernel': kernel_test},
            {'kernel': kernel_test2},
            {'kernel': kernel_test3},
            {'kernel': kernel_test4},
            {'kernel': kernel_test5},
        ]

        self.kernel_configs = kernels
        logger.info(f"Created {len(kernels)} kernel configurations")
        return kernels
    
    def train_historical(self, x: pd.DataFrame, y: pd.DataFrame) -> Dict[str, Any]:
        """
        Train the autoregressive model on historical data.
        
        Args:
            x: Feature DataFrame with lag columns
            y: Target DataFrame (single column expected)
            
        Returns:
            Dictionary with training metrics and model information
        """
        self._validate_inputs(x, y)
        self._store_training_boundaries(y)
        
        # Convert to single target if multi-target
        if y.shape[1] > 1:
            logger.warning(f"Multiple targets provided, using first column: {y.columns[0]}")
            y_single = y.iloc[:, 0]
        else:
            y_single = y.iloc[:, 0]
        
        # Store training data for horizon validation
        self.X_train = x.copy()
        self.y_train = y_single.copy()
        
        # Setup cross-validation
        tscv = TimeSeriesSplit(n_splits=3)
        
        best_score = float('inf')
        best_model = None
        best_params = None
        
        # Try different kernel configurations
        for kernel_config in self.kernel_configs:
            logger.debug(f"Training with kernel: {kernel_config}")
            
            try:
                # Create kernel ridge model
                if isinstance(kernel_config, tuple):
                    # For composite kernels, use RBF as base
                    model = KernelRidge(kernel='rbf')
                else:
                    model = KernelRidge(**kernel_config)
                
                # Grid search with time series CV
                grid_search = GridSearchCV(
                    model,
                    self.param_grid,
                    cv=tscv,
                    scoring='neg_mean_squared_error',
                    n_jobs=self.n_jobs,
                    verbose=0
                )
                
                # Fit grid search
                grid_search.fit(x, y_single)
                
                # Check if this is the best model
                if -grid_search.best_score_ < best_score:
                    best_score = -grid_search.best_score_
                    best_model = grid_search.best_estimator_
                    best_params = {
                        'kernel_config': kernel_config,
                        **grid_search.best_params_
                    }
                
            except Exception as e:
                logger.warning(f"Failed to train kernel {kernel_config}: {str(e)}")
                continue
        
        if best_model is None:
            raise ValueError("All kernel configurations failed during training")
        
        self.best_model = best_model
        self.best_kernel_name = best_params['kernel_config']['kernel'].__class__.__name__
        self.best_params = best_params
        
        # Calculate horizon-specific residuals for distribution modeling
        self.horizon_std = self._calculate_horizon_residuals(x, y_single, self.forecast_horizon)
        
        # Calculate training metrics
        train_pred = self.best_model.predict(x)
        self.training_scores = {
            'mse': mean_squared_error(y_single, train_pred),
            'rmse': np.sqrt(mean_squared_error(y_single, train_pred)),
            'mae': mean_absolute_error(y_single, train_pred),
            'r2': r2_score(y_single, train_pred),
            'horizon_std': self.horizon_std,
            'best_score': best_score
        }
        
        self.is_trained = True
        
        # logger.info(f"Training completed. Best kernel: {best_params['kernel_config']}, "
        #            f"RMSE: {self.training_scores['rmse']:.4f}")
        
        return self.training_scores
    
    def predict_val(self, x: pd.DataFrame) -> np.ndarray:
        """
        Make point predictions using recursive forecasting.
        
        Args:
            x: Feature DataFrame with lag columns
            
        Returns:
            Array of predictions at forecast horizon
        """
        self._validate_trained()
        self._validate_inputs(x)
        
        # Get predictions for each row
        predictions = []
        for idx in range(len(x)):
            initial_lags = x.iloc[idx].values
            pred = self._forecast_recursive(initial_lags, self.forecast_horizon)
            predictions.append(pred[-1])  # Take final prediction at horizon
        
        predictions = np.array(predictions)
        
        # Apply prediction boundaries
        if self.training_columns:
            predictions = self._apply_prediction_boundaries(
                predictions.reshape(-1, 1), 
                self.training_columns
            ).flatten()
        
        return predictions
    
    def predict_val_distribution(self, x: pd.DataFrame, y: pd.DataFrame, n_samples: int = 1000) -> pd.DataFrame:
        """
        Generate prediction distribution samples using recursive forecasting.
        
        Args:
            x: Feature DataFrame with lag columns
            y: Target DataFrame (for column names)
            n_samples: Number of samples to generate
            
        Returns:
            DataFrame with prediction samples
        """
        self._validate_trained()
        self._validate_inputs(x)
        
        # Get point predictions
        point_predictions = self.predict_val(x)
        
        # Generate samples using normal distribution
        samples = []
        for i, pred in enumerate(point_predictions):
            # Sample from normal distribution with horizon-specific std
            sample_values = np.random.normal(
                loc=pred, 
                scale=self.horizon_std, 
                size=n_samples
            )
            
            # Apply yield constraints
            # sample_values = self.apply_yield_constraints(sample_values)
            samples.append(sample_values)
        
        # Convert to DataFrame
        samples_array = np.array(samples)  # Shape: (n_samples, n_predictions)
        
        # Create column names based on target structure
        if y.shape[1] == 1:
            columns = [f"{y.columns[0]}_sample_{i}" for i in range(len(x))]
        else:
            columns = [f"prediction_sample_{i}" for i in range(len(x))]
        
        return pd.DataFrame(samples_array.T, columns=columns)
    
    def get_model_summary(self) -> Dict[str, Any]:
        """
        Get summary of trained model performance and parameters.
        
        Returns:
            Dictionary with model summary information
        """
        if not self.is_trained:
            return {"status": "not_trained"}
        
        return {
            "model_type": "KernelAutoregressiveModel",
            "forecast_horizon": self.forecast_horizon,
            "n_lags": self.n_lags,
            "best_params": self.best_params,
            "training_scores": self.training_scores,
            "is_trained": self.is_trained
        }
    
    def get_feature_importance_proxy(self, X: pd.DataFrame) -> pd.Series:
        """
        Get feature importance proxy using kernel ridge coefficients.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Series with feature importance scores
        """
        if not self.is_trained:
            logger.warning("Model not trained, returning uniform importance")
            return pd.Series(
                np.ones(len(X.columns)) / len(X.columns),
                index=X.columns,
                name="importance"
            )
        
        # For kernel methods, use dual coefficients as proxy
        if hasattr(self.best_model, 'dual_coef_'):
            importance = np.abs(self.best_model.dual_coef_).mean()
            importance_scores = np.full(len(X.columns), importance)
        else:
            # Fallback to uniform importance
            importance_scores = np.ones(len(X.columns)) / len(X.columns)
        
        return pd.Series(
            importance_scores,
            index=X.columns,
            name="importance"
        )
    
    def _forecast_recursive(self, initial_lags: np.ndarray, steps: int) -> np.ndarray:
        """
        Recursively forecast r steps ahead.
        
        Args:
            initial_lags: Array of shape (p,) with initial lag values [t-1, t-2, ..., t-p]
            steps: Number of steps to forecast (r)
        
        Returns:
            Array of shape (steps,) with predictions [t+1, t+2, ..., t+r]
        """
        predictions = []
        current_lags = initial_lags.copy()
        
        for step in range(steps):
            # Make prediction using current lag features
            pred = self.best_model.predict(current_lags.reshape(1, -1))[0]
            predictions.append(pred)
            
            # Update lag vector for next prediction
            current_lags = self._create_future_features(current_lags, pred)
        
        return np.array(predictions)
    
    def _create_future_features(self, current_lags: np.ndarray, new_prediction: float) -> np.ndarray:
        """
        Create feature vector for next prediction using previous forecast.
        
        Args:
            current_lags: Current lag features [t-1, t-2, ..., t-p]
            new_prediction: Latest prediction to add as t lag
        
        Returns:
            Updated feature vector [new_prediction, t-1, ..., t-(p-1)]
        """
        # Shift lags and add new prediction as most recent lag
        new_lags = np.zeros_like(current_lags)
        new_lags[0] = new_prediction
        new_lags[1:] = current_lags[:-1]
        
        return new_lags
    
    def _calculate_horizon_residuals(self, X_train: pd.DataFrame, y_train: pd.Series, horizon: int) -> float:
        """
        Calculate residuals at forecast horizon r for distribution modeling.
        
        Args:
            X_train: Training features
            y_train: Training targets
            horizon: Forecast horizon
            
        Returns:
            Standard deviation of horizon residuals
        """
        if len(y_train) < horizon + 1:
            logger.warning("Insufficient data for horizon validation, using training std")
            return y_train.std()
        
        # Create horizon-shifted targets for validation
        y_horizon = y_train.shift(-horizon).dropna()
        X_horizon = X_train.iloc[:len(y_horizon)]
        
        if len(y_horizon) == 0:
            logger.warning("No valid horizon data, using training std")
            return y_train.std()
        
        # Make recursive predictions for horizon validation
        horizon_predictions = []
        for idx in range(len(X_horizon)):
            initial_lags = X_horizon.iloc[idx].values
            pred_sequence = self._forecast_recursive(initial_lags, horizon)
            horizon_predictions.append(pred_sequence[-1])
        
        # Calculate residuals
        residuals = np.abs(np.array(horizon_predictions) - y_horizon.values)
        horizon_std = np.mean(residuals)  # Use mean absolute residual as std
        
        logger.debug(f"Calculated horizon std: {horizon_std:.4f} from {len(residuals)} samples")
        
        return horizon_std
    
    def predict_autoregressive_batch(self, X: pd.DataFrame, steps: int) -> pd.DataFrame:
        """
        Apply recursive forecasting to each row in walk-forward validation.
        
        Args:
            X: DataFrame where each row contains p lag features
            steps: Forecast horizon (r)
        
        Returns:
            DataFrame with predictions for each row at horizon r
        """
        self._validate_trained()
        self._validate_inputs(X)
        
        predictions = []
        for idx in range(len(X)):
            initial_lags = X.iloc[idx].values
            pred_sequence = self._forecast_recursive(initial_lags, steps)
            predictions.append(pred_sequence)
        
        # Convert to DataFrame
        pred_array = np.array(predictions)  # Shape: (n_samples, steps)
        columns = [f"step_{i+1}" for i in range(steps)]
        
        return pd.DataFrame(pred_array, columns=columns, index=X.index)

if __name__ == "__main__":
    sample_data = pd.read_parquet('data/sample_data_for_testing_functions.parquet')
    auto_kernel = KernelAutoregressiveModel(forecast_horizon=2, n_jobs=2)
    # split the sample data
    x_cols = [f'Lag {x + 1}' for x in range(5)]
    y_col = '30 Day Avg'

    # train the autoregressive model on the first 400 events
    auto_kernel.train_historical(x=sample_data[x_cols].iloc[:400], y=sample_data[y_col].iloc[:400].to_frame())
    print(f"model trained")
    test = sample_data.iloc[400:430]
    predictions = []
    for idx in test[x_cols].index:
        x_vals = test[x_cols].loc[[idx]]
        single_prediction = auto_kernel.predict_val(x_vals)
        predictions.append(single_prediction[0])

    len(predictions)