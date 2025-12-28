"""
Walk-forward validation framework for bond yield forecasting.
Implements sliding window validation with proper time series handling.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
import logging
from datetime import datetime
import gc
from data_loader import DataLoader
# from feature_manager import FeatureManager
from gp_models import GaussianProcessEnsemble
# from bayesian_ridge_models import BayesianRidgeEnsemble
from kernel_ridge_models import KernelRidgeEnsemble
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WalkForwardValidator:
    """Walk-forward validation framework for time series forecasting."""
    
    def __init__(self,
                 model: GaussianProcessEnsemble | KernelRidgeEnsemble,
                 data_loader: DataLoader,
                 ticker: str,
                 lags: int = 10,
                 # forecast_horizon: int = 1,
                 # feature_manager: FeatureManager,
                 # time_prediction: str,
                 persist_samples: bool = True,
                 window_size: int = 3000,
                 min_window_size: int = 2000,
                 step_size: int = 1,
                 model_retrain_interval: int = 20,
                 n_parallel_jobs: int = 5,
                 use_scaling: bool = False,
                 window_type: str = 'sliding'):
        """
        Initialize walk-forward validator.
        
        Args:
            model: Model instance (GP, Bayesian Ridge, or KRR)
            data_loader: Configured data loader
            # feature_manager: Feature manager instance
            # time_prediction: Time prediction horizon
            ticker: Asset ticker name
            lags: Number of lagged features to use
            persist_samples: Whether to persist prediction samples
            window_size: Size of training window (default: 3000 trading days)
            min_window_size: Minimum window size for training
            step_size: Step size for walking forward (default: 1 day)
            model_retrain_interval: Interval for model retraining
            n_parallel_jobs: Number of parallel jobs for training
            use_scaling: Whether to apply standard scaling to features and targets
            window_type: Type of window ('sliding' or 'growing')
        """
        self.data_loader = data_loader
        # self.time_prediction = time_prediction
        # self.feature_manager = feature_manager
        # self.forecast_horizon = forecast_horizon
        self.ticker = ticker
        self.lags = lags
        self.window_size = window_size
        self.min_window_size = min_window_size
        self.step_size = step_size
        self.n_parallel_jobs = n_parallel_jobs
        self.model_retrain_interval = model_retrain_interval
        self.initial_run = True
        self.model_retrain_counter = 0
        self.model = model
        self.persist_samples = persist_samples
        self.feature_importance = None
        self.use_scaling = use_scaling
        self.window_type = window_type
        
        # Initialize scalers for features and targets
        if self.use_scaling:
            self.scaler_x = StandardScaler()
            self.scaler_y = StandardScaler()
            self.scaler_y_noise = StandardScaler()
            self.scalers_fitted = False
        else:
            self.scaler_x = None
            self.scaler_y = None
            self.scaler_y_noise = None
            self.scalers_fitted = False

        # set up sample directory if true
        if self.persist_samples:
            # create the samples directory
            self.sample_dir = Path(f'results/samples/{self.ticker}')
            Path(self.sample_dir).mkdir(parents=True, exist_ok=True)
        else:
            self.sample_dir = None
        
        # Results storage
        self.results_list = []
        # Model tracking
        self.model_summaries: List[Dict] = []
        
        logger.info(f"Initialized walk-forward validator: window_size={window_size}, "
                   f"min_window_size={min_window_size}, step_size={step_size}")

    
    def run_single_prediction(self, 
                            train_start_idx: int,
                            train_end_idx: int, 
                            predict_idx: int,
                            target_columns: list[str],
                            noise_columns: list[str],
                            features: List[str]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Run a single prediction step in walk-forward validation.
        
        Args:
            train_start_idx: Start index of training window
            train_end_idx: End index of training window  
            predict_idx: Index to make prediction for
            noise_columns: the name of the noise column, this is for calculating the residuals for the distribtion output
            target_columns: str: Target variable names
            bond_index: Bond index name
            features: List of feature names
            
        Returns:
            Dictionary with prediction results
        """
        # Get training data
        x_train, y_train = self.data_loader.get_window_data(
            start_idx=train_start_idx,
            end_idx=train_end_idx,
            target_columns=target_columns,
            feature_columns=features
        )

        _, y_train_noise = self.data_loader.get_window_data(
            start_idx=train_start_idx,
            end_idx=train_end_idx,
            target_columns=noise_columns,
            feature_columns=features
        )
        
        # Validate training data before scaling
        if x_train.empty:
            raise ValueError(f"X_train is empty for window {train_start_idx}-{train_end_idx}")
        if y_train.empty:
            raise ValueError(f"Y_train is empty for window {train_start_idx}-{train_end_idx}")
        if y_train_noise.empty:
            raise ValueError(f"Y_train_noise is empty for window {train_start_idx}-{train_end_idx}")
        
        logger.debug(f"Training data shapes: X={x_train.shape}, Y={y_train.shape}, Y_noise={y_train_noise.shape}")
        
        # Apply scaling if enabled
        if self.use_scaling:
            if self.model_retrain_counter == self.model_retrain_interval or self.initial_run:
                # Fit scalers on new training data during retrain
                x_train_scaled = pd.DataFrame(
                    self.scaler_x.fit_transform(x_train),
                    columns=x_train.columns,
                    index=x_train.index
                )
                y_train_scaled = pd.DataFrame(
                    self.scaler_y.fit_transform(y_train),
                    columns=y_train.columns,
                    index=y_train.index
                )

                y_train_noise_scaled = pd.DataFrame(
                    self.scaler_y_noise.fit_transform(y_train_noise),
                    columns=y_train_noise.columns,
                    index=y_train_noise.index
                )
                self.scalers_fitted = True
            else:
                # Transform using existing scalers
                x_train_scaled = pd.DataFrame(
                    self.scaler_x.transform(x_train),
                    columns=x_train.columns,
                    index=x_train.index
                )
                y_train_scaled = pd.DataFrame(
                    self.scaler_y.transform(y_train),
                    columns=y_train.columns,
                    index=y_train.index
                )
                y_train_noise_scaled = pd.DataFrame(
                    self.scaler_y_noise.transform(y_train_noise),
                    columns=y_train_noise.columns,
                    index=y_train_noise.index
                )
        else:
            x_train_scaled = x_train
            y_train_scaled = y_train
            y_train_noise_scaled = y_train_noise
        # Check if a retrain is needed
        if self.model_retrain_counter == self.model_retrain_interval or self.initial_run:
            retrain = True
            logger.debug(f"Running retrain for window {x_train.index.min()}-{x_train.index.max()}")
            # train the model with scaled or original data
            self.model.train_historical(x=x_train_scaled, y=y_train_scaled, y_noise=y_train_noise_scaled)
            self.feature_importance = self.model.get_feature_importance_proxy(X=x_train_scaled)
            # Reset counter
            self.model_retrain_counter = 0
            if self.initial_run:
                logger.debug("Initial run completed, model retraining interval reached")
                self.initial_run = False
        else:
            retrain = False

        # Get prediction features
        x_predict = self.data_loader.get_prediction_point(idx=predict_idx, feature_columns=features)
        
        # Scale prediction features if scaling is enabled
        if self.use_scaling and self.scalers_fitted:
            x_predict_scaled = pd.DataFrame(
                self.scaler_x.transform(x_predict),
                columns=x_predict.columns,
                index=x_predict.index
            )
        else:
            x_predict_scaled = x_predict

        # Get actual value
        actual_value = self.data_loader.get_actual_value(idx=predict_idx, target_columns=target_columns)

        # Get prediction date
        predict_date = self.data_loader.get_date_for_index(predict_idx)
        # Make prediction with scaled features
        prediction_val_raw = self.model.predict_val(x=x_predict_scaled)
        
        # Apply inverse scaling to predictions if scaling is enabled
        if self.use_scaling and self.scalers_fitted:
            # Ensure prediction_val_raw is numpy array and has correct shape
            if isinstance(prediction_val_raw, pd.DataFrame):
                prediction_val_raw = prediction_val_raw.values
            if len(prediction_val_raw.shape) == 1:
                prediction_val_raw = prediction_val_raw.reshape(1, -1)
            prediction_val = self.scaler_y.inverse_transform(prediction_val_raw).flatten()
            
            # Apply prediction boundaries after inverse transformation
            prediction_val = self.model._apply_prediction_boundaries(prediction_val.reshape(1, -1), target_columns).flatten()
        else:
            prediction_val = prediction_val_raw
        
        # Make prediction samples
        prediction_samples_raw = self.model.predict_val_distribution(x=x_predict_scaled, y=y_train_scaled, n_samples=1000)
        
        # Apply inverse scaling to samples if scaling is enabled
        if self.use_scaling and self.scalers_fitted:
            # Reshape samples for inverse transform
            # samples_shape = prediction_samples_raw.shape
            # if len(samples_shape) == 3:  # (1, n_outputs, n_samples)  # (n_samples, n_outputs)
            prediction_samples_transformed = self.scaler_y.inverse_transform(prediction_samples_raw)
            prediction_samples = pd.DataFrame(prediction_samples_transformed, columns=prediction_samples_raw.columns) # Back to original shape
            
            # Apply prediction boundaries to samples after inverse transformation
            constrained_samples = self.model._apply_prediction_boundaries(prediction_samples.values, target_columns)
            prediction_samples = pd.DataFrame(constrained_samples, columns=prediction_samples.columns)
        else:
            prediction_samples = prediction_samples_raw


        # Persist samples if needed
        if self.persist_samples:
            # Convert the 1, 4, 1000 array to a dataframe and save as parquet

            samples_df = prediction_samples

            samples_df.to_parquet(self.sample_dir / f"{predict_date}.parquet")
        if isinstance(self.model, GaussianProcessEnsemble):
            result = {
                'date': predict_date,
                'actual_value': list(actual_value.to_numpy()),
                'prediction': list(prediction_val),
                # 'std': list(self.model.horizon_std),
                # 'prediction_std': list(prediction_std),
                'best_kernel': self.model.best_kernel_name,
                'retrain': retrain,
            }
        # elif isinstance(self.model, BayesianRidgeEnsemble):
        #     result = {
        #         'date': predict_date,
        #         'actual_value': list(actual_value.to_numpy()),
        #         'prediction': list(prediction_val),
        #         # 'prediction_std': list(prediction_std),
        #         'best_alpha': self.model.best_alpha_name,
        #         'retrain': retrain,
        #     }
        else:  # KernelRidgeEnsemble
            result = {
                'date': predict_date,
                'actual_value': list(actual_value.to_numpy()),
                'prediction': list(prediction_val),
                'std': float(self.model.horizon_std),
                'best_kernel': self.model.best_kernel_name,
                'best_alpha': self.model.best_alpha_name,
                'retrain': retrain,
            }
        model_summary = self.model.get_model_summary()
        model_summary['date'] = predict_date
        # get the feature importance
        model_summary['feature_importance'] = self.feature_importance.to_dict()
        # Increment retrain counter
        self.model_retrain_counter += 1


        logger.debug(f"Prediction completed for {predict_date}: "
                    f"actual={actual_value}, pred={prediction_val}")

        return result, model_summary

    
    def run_walk_forward_validation(self, target_variable:str) -> None:
        """
        Run complete walk-forward validation.
        
        Args:
            target_variable: the name of the target variable
            
        Returns:
            Dictionary with complete validation results
        """
        logger.info(f"Starting walk-forward validation for {self.ticker}")
        
        # Get features
        features = [f'Lag {x + 1}' for x in range(self.lags)]
        target_columns = [target_variable]
        
        # Get time windows
        windows = self.data_loader.get_time_windows(
            window_size=self.window_size, 
            min_window_size=self.min_window_size,
            window_type=self.window_type
        )

        # testing comment the following out when running the full validation
        # check if the window is greater than 2500, if so select the last 2500 records
        if len(windows) > 2500:
            logging.info(f"The window size is greater than 2500, only running the last 2500 records")
            windows = windows[-2500:]
        # windows = windows[-30:]

        logger.info(f"Running {len(windows)} predictions with {len(features)} features")
        # Run predictions
        for i, (train_start_idx, train_end_idx) in tqdm(enumerate(windows), desc='Running Walk Forward Validation', total=len(windows)):
            # set the prediction index
            predict_idx = train_end_idx

            # Safety check for prediction index
            if predict_idx >= len(self.data_loader.data):
                logger.warning(f"Prediction index {predict_idx} out of bounds, stopping")
                break
            
            # Run single prediction
            result, model_summary = self.run_single_prediction(
                train_start_idx=train_start_idx,
                train_end_idx=train_end_idx,
                predict_idx=predict_idx,
                target_columns=target_columns,
                noise_columns=['Adj Close'],
                features=features
            )
            # Store result
            self.results_list.append(result)
            self.model_summaries.append(model_summary)

    
    def export_results(self, filepath: str) -> None:
        """
        Export validation results to file.
        
        Args:
            filepath: Path to save results
        """

        results_df = pd.DataFrame(self.results_list)
        model_summary_df = pd.DataFrame(self.model_summaries)
        # create file path if not exists
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        try:
            results_df.to_parquet(f'{filepath}/{self.ticker}.parquet', index=False)
        except Exception as e:
            logger.error(f"Error exporting results: {str(e)}")
        try:
            model_summary_df.to_parquet(f'{filepath}/{self.ticker}_model_summary.parquet', index=False)
            logger.info(f"Results exported to {filepath}")
        except Exception as e:
            logger.error(f"Error exporting model summary: {str(e)}")
