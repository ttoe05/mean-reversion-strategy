"""
Comprehensive metrics calculation for bond yield forecasting evaluation.
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any, Union
from scipy import stats
from seaborn import color_palette
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
from pathlib import Path
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from percentile_calc import get_list_sample_files
# from src.KRR_simulation_test import actuals_variables

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ForecastingMetrics:
    """Comprehensive forecasting metrics calculator."""
    
    def __init__(self, ticker: str, actuals_file: str) -> None:
        """Initialize metrics calculator."""
        self.ticker = ticker
        self.actuals_df: pd.DataFrame = None
        self.forecasts_df: pd.DataFrame = None
        self.actuals_file = actuals_file
        self.dependent_varaibles = [
            '30 Day Avg',
        ]
        self.actuals_variables = [
            '30 Day Avg future value',
        ]
        self._get_actuals_df()
        logger.info(self.actuals_df.head())
        self._get_forecast_df()

    def _get_actuals_df(self) -> None:
        """Get the file path for actuals based on time prediction."""
        # generate case-specific file path
        # read in the actuals data
        df = pd.read_parquet(self.actuals_file)
        # df = df[df.index.to_timestamp() >= pd.to_datetime('2010-01-01')]
        self.actuals_df = df

    def _get_forecast_record(self, file: str) -> dict[str, Any]:
        """
        Get the forecast record from the given file.
        Args:
            file:
        Returns: Dict with forecast record
        """
        # Read the data from the file
        df_tmp = pd.read_parquet(file)
        record = {}
        # get the date from the file name and convert to datetime
        date_str = file.split('/')[-1].replace('.parquet', '')
        record['date'] = pd.to_datetime(date_str)
        for col in self.dependent_varaibles:
            col = f'{col}_sample_0'
            record[f"{col}_mean"] = df_tmp[col].mean()
            record[f"{col}_std"] = df_tmp[col].std()
            record[f"{col}_min"] = df_tmp[col].min()
            record[f"{col}_max"] = df_tmp[col].max()
            try:
                record[f"{col}_97_5"] = df_tmp[df_tmp[f"{col}_percentile"] >= 0.95][col].values[0]
            except IndexError:
                record[f"{col}_97_5"] = df_tmp[col].max()
            try:
                record[f"{col}_2_5"] = df_tmp[df_tmp[f"{col}_percentile"] <= 0.05][col].values[0]
            except IndexError:
                record[f"{col}_2_5"] = df_tmp[col].min()


        return record

    def _get_forecast_df(self)-> None:
        """
        Get the forecast Dataframe with the 95% confidence bands
        Returns: None
        """
        # get the list of sample files
        sample_files = get_list_sample_files(ticker=self.ticker, percentile=True)
        max_workers = 8
        # user mutlithreading to read in all the forecast files in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            records = list(tqdm(executor.map(self._get_forecast_record, sample_files),
                                total=len(sample_files)))

        self.forecasts_df = pd.DataFrame(records)
        # combine with actuals
        sample_df_tmp = self.actuals_df.copy()
        sample_df_tmp['date'] = sample_df_tmp.index
        self.forecasts_df = self.forecasts_df.merge(sample_df_tmp, left_on='date', right_on='date', how='left')


    def _create_confidence_plot_df(self, df: pd.DataFrame, col: str, confidence_bands: bool, 
                                   smoothing: bool = False, smoothing_window: int = 7) -> pd.DataFrame:
        """
        Create a DataFrame for confidence plot for a given column.
        Args:
            df: DataFrame with actuals and forecasts
            col: Column name to create plot for
            smoothing: Whether to apply exponential moving average smoothing
            smoothing_window: Window size for exponential moving average (default: 7)
        """
        # remove appendix from col name if exists
        col_original = col
        col = f'{col}_sample_0'
        if confidence_bands:
            full_cols = ['date', col_original, f"{col}_mean", f"{col}_2_5", f"{col}_97_5"]
            value_vars = [col_original, f"{col}_mean", f"{col}_2_5", f"{col}_97_5"]
        else:
            full_cols = ['date', col_original, f"{col}_mean"]
            value_vars = [col_original, f"{col}_mean"]
        
        df_tmp = df[full_cols].copy().sort_values('date')
        
        # Apply exponential moving average smoothing if requested
        if smoothing:
            # Calculate alpha for exponential moving average
            alpha = 2.0 / (smoothing_window + 1.0)
            
            # Apply smoothing to prediction mean and confidence intervals (but not actuals)
            if f"{col}_mean" in df_tmp.columns:
                df_tmp[f"{col}_mean"] = df_tmp[f"{col}_mean"].ewm(alpha=alpha, adjust=False).mean()
            if confidence_bands:
                if f"{col}_2_5" in df_tmp.columns:
                    df_tmp[f"{col}_2_5"] = df_tmp[f"{col}_2_5"].ewm(alpha=alpha, adjust=False).mean()
                if f"{col}_97_5" in df_tmp.columns:
                    df_tmp[f"{col}_97_5"] = df_tmp[f"{col}_97_5"].ewm(alpha=alpha, adjust=False).mean()
            
            logger.debug(f"Applied EMA smoothing with window={smoothing_window} (alpha={alpha:.3f}) to {col}")
        
        return df_tmp.melt(id_vars=['date'],
                           value_vars=value_vars,
                           var_name='type',
                           value_name='value')

    def plot_time_confidence_intervals(self, train: bool, confidence_bands: bool, save_fig: bool,
                                       smoothing: bool = False, smoothing_window: int = 7) -> None:
        """
        Generate DataFrame for plotting confidence intervals for a given column.
        Args:
            train: Whether to plot training data only
            confidence_bands: Whether to include confidence bands
            save_fig: Whether to save the figure
            smoothing: Whether to apply exponential moving average smoothing to predictions (default: False)
            smoothing_window: Window size for exponential moving average (default: 7)
        """
        # create a 4 by 3 subplot of the confidence intervals for each bond index
        
        smooth_suffix = "_smoothed" if smoothing else ""
        
        for i, col in enumerate(self.dependent_varaibles):
            plot_df = self._create_confidence_plot_df(self.forecasts_df, col, confidence_bands, 
                                                    smoothing, smoothing_window)
            # if train:
            #     plot_df = plot_df[(plot_df['date'] < pd.to_datetime('2017-12-31')) & (plot_df['date'] > pd.to_datetime('2005-01-01'))]
            col_orginal = col
            col = f'{col}_sample_0'
            color_palette = {
                col_orginal: 'blue',
                f"{col}_mean": 'orange',
                f"{col}_2_5": 'red',
                f"{col}_97_5": 'red'
            }
            title_suffix = f" (EMA-{smoothing_window})" if smoothing else ""
            sns.lineplot(plot_df, x='date', y='value', hue='type',
                         palette=color_palette).set_title(f'Confidence Intervals for {col}{title_suffix} - {self.ticker}')
            # axes[i].set_xlabel('Date')
            # axes[i].set_ylabel('Yield (%)')
            # axes[i].legend()
            
        plt.tight_layout()
        
        if save_fig:
            # create plots directory if not exists
            Path(f'plots/{self.ticker}/').mkdir(parents=True, exist_ok=True)
            filename = f'confidence_intervals_{"train" if train else "full"}{smooth_suffix}.png'
            plt.savefig(f'plots/{self.ticker}/{filename}')
            logger.info(f"Saved plot to plots/{self.ticker}/{filename}")
            
        plt.show()

    # def plot_actuals(self, smoothing: bool = False, smoothing_window: int = 7, save_fig: bool = False):
    #     """
    #     Plot actual bond yields over time.
    #     Args:
    #         smoothing: Whether to apply exponential moving average smoothing (default: False)
    #         smoothing_window: Window size for exponential moving average (default: 7)
    #         save_fig: Whether to save the figure (default: False)
    #     """
    #     fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(20, 17))
    #     axes = axes.flatten()
    #
    #     actuals_df_plot = self.actuals_df.copy()
    #
    #     # Apply smoothing if requested
    #     if smoothing:
    #         alpha = 2.0 / (smoothing_window + 1.0)
    #         for col in self.dependent_varaibles:
    #             col_rename = col.replace('_future_val', '')
    #             if col_rename in actuals_df_plot.columns:
    #                 actuals_df_plot[col_rename] = actuals_df_plot[col_rename].ewm(alpha=alpha, adjust=False).mean()
    #         logger.debug(f"Applied EMA smoothing with window={smoothing_window} to actual yields")
    #
    #     smooth_suffix = "_smoothed" if smoothing else ""
        
        # for i, col in enumerate(self.dependent_varaibles):
        #     col_rename = col.replace('_future_val', '')
        #     sns.lineplot(data=actuals_df_plot, x=actuals_df_plot.index.to_timestamp(), y=col_rename, ax=axes[i])
        #
        #     title_suffix = f" (EMA-{smoothing_window})" if smoothing else ""
        #     axes[i].set_title(f'Actual Yields for {col_rename}{title_suffix} - {self.time_prediction}')
        #     axes[i].set_xlabel('Date')
        #     axes[i].set_ylabel('Yield (%)')
        #
        # plt.tight_layout()
        #
        # if save_fig:
        #     Path(f'plots/{self.time_prediction}/').mkdir(parents=True, exist_ok=True)
        #     filename = f'actual_yields{smooth_suffix}.png'
        #     plt.savefig(f'plots/{self.time_prediction}/{filename}')
        #     logger.info(f"Saved plot to plots/{self.time_prediction}/{filename}")
        #
        # plt.show()

    def metric_calculator(self, actual_value: float, std: float, mean: float, min: float, max: float) -> tuple[str, str]:
        """
        Output a string value if the value is within 1 std of the mean 2 standard deviations or surpasse
        Returns:

        """
        # calcualte normalized value of the prediction
        try:
            normalized_actual = (mean - actual_value) / std
        except ZeroDivisionError:
            normalized_actual = 1000
        if abs(normalized_actual) <= 1:
            return "Within 1 std", "Within sample"
        elif abs(normalized_actual) > 1 and abs(normalized_actual) <= 2:
            return "Within 2 std", "Within sample"
        elif normalized_actual > max or  normalized_actual < min:
            return "Surpasses sample boundary", "out of sample"
        elif normalized_actual == 1000:
            return "Std 0", "out of sample"
        else:
            return "Greater than 2 std", "Within sample"

    def plot_sample_metrics(self) -> None:
        """
        Create a stacked bar chart of sample metrics.
        Returns:

        """
        color_palette = {
                "Within 1 std": "#002F6C",
                "Within 2 std": "#398FFF",
                "Surpasses sample boundary": "#EB2254",
            }
        for i, col in enumerate(self.dependent_varaibles):
            col_orginal = col
            col = f'{col}_sample_0'
            logging.info(self.forecasts_df.head())
            
            # Create crosstab for stacked bar chart
            ct = pd.crosstab(self.forecasts_df[f"{col}_sample_range"], 
                           self.forecasts_df[f"{col}_sample_metric"])
            
            # Create stacked bar chart
            ax = ct.plot(kind='bar', stacked=True, color=[color_palette.get(x, 'gray') for x in ct.columns])
            ax.set_title(f'Sample Metrics for {col_orginal} - {self.ticker}')
            ax.set_xlabel('Sample Range')
            ax.set_ylabel('Count')
            ax.legend(title='Sample Metric')
            
        plt.tight_layout()
        plt.show()


    def calculate_sample_metrics(self):
        """
        Calculate sample metrics by creating a column for each dependent variable
        that labels predictions where the absolute value of the actual value fell within 1 std of the mean 2 standard deviations or surpasse
        the max value or the min value.
        Returns:

        """
        for col, actual_col in zip(self.dependent_varaibles, self.actuals_variables):
            col_orginal = actual_col
            col = f'{col}_sample_0'
            self.forecasts_df[[f"{col}_sample_metric", f"{col}_sample_range"]] = self.forecasts_df.apply(lambda x: self.metric_calculator(x[f"{col_orginal}"], x[f"{col}_std"], x[f"{col}_mean"], x[f"{col}_min"], x[f"{col}_max"]),
                                                                                                         axis=1,
                                                                                                         result_type='expand'
                                                                                                         )



if __name__ == "__main__":
    forecasts = ForecastingMetrics(ticker='RGLD',
                                   actuals_file="data/rgld_train.parquet")

    print(forecasts.forecasts_df.head())
    print(forecasts.actuals_df.head())
    # print current working directory
    print(os.getcwd())
    # Original plots without smoothing
    forecasts.plot_time_confidence_intervals(train=True, confidence_bands=True, smoothing=False, smoothing_window=31, save_fig=True)
    # forecasts.plot_time_confidence_intervals(train=False, confidence_bands=False, smoothing=True, smoothing_window=31, save_fig=True)
    # forecasts.plot_time_confidence_intervals(train=False, confidence_bands=True,smoothing=True, smoothing_window=31, save_fig=True)
    # forecasts.plot_time_confidence_intervals(train=True, confidence_bands=False, save_fig=True)
    
    # New plots with exponential moving average smoothing (window=7)
    forecasts.calculate_sample_metrics()
    forecasts.plot_sample_metrics()



