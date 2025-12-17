import sys

import pandas as pd
import numpy as np
import os
from scipy.optimize import curve_fit
#import threadpool executor for parallel processing
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from itertools import repeat
from pathlib import Path

# DEPENDENT_COLUMN_NAMES = [
#     'DGS1MO_future_val', 'DGS3MO_future_val', 'DGS6MO_future_val',
#     'DGS1_future_val', 'DGS2_future_val', 'DGS3_future_val', 'DGS5_future_val',
#     'DGS7_future_val', 'DGS10_future_val', 'DGS20_future_val', 'DGS30_future_val'
# ]
def get_list_sample_files(ticker: str, percentile: bool = False) -> list[str]:
    """
    Get a list of all sample files for a given time prediction.
    Args:
        time_prediction: str
            The time prediction identifier (e.g., 'one-day-ahead', 'seven-day-ahead', etc.)
        percentile: bool
            Whether to read from percentile subdirectory
        monte_carlo: bool
            Whether to read from monte_carlo samples instead of dgs_yields
    Returns:
        list[str]: List of file paths to sample files.
    """
    # check the time_prediction passed in is valid


    if percentile:
        sample_dir = f"results/samples/{ticker}/percentile/"
    else:
        sample_dir = f"results/samples/{ticker}"
    
    if not os.path.exists(sample_dir):
        raise FileNotFoundError(f"Sample directory not found: {sample_dir}")
    
    sample_files = [os.path.join(sample_dir, f) for f in os.listdir(sample_dir) if f.endswith('.parquet')]
    return sample_files

def get_percentiles(sample_file: str, ticker: str) -> None:
    """
    Calculate percentiles for the dgs yields
    Args:
        sample_file: str
            Path to the sample file
        ticker: str
            ticker symbol

    Returns:
        None
    """
    sample_df = pd.read_parquet(sample_file)
    for col in sample_df.columns:
            sample_df[f"{col}_percentile"] = sample_df[col].rank(pct=True)
    
    # Create output directory based on source type
    output_dir = f"results/samples/{ticker}/percentile/"
    
    # Create the percentile subdirectory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # persist the results
    sample_df.to_parquet(f"{output_dir}{sample_file.split('/')[-1]}")


def calc_all_percentiles(sample_files: list[str], ticker: str, max_workers: int=4) -> None:
    """
    Concurrently calculate percentiles for all sample files.
    Args:
        sample_files: list[str]
            List of sample file paths
        time_prediction: str
            The time prediction identifier
        monte_carlo: bool
            Whether processing monte_carlo samples
        max_workers: int
            Maximum number of workers for concurrent processing
    Returns: None
    """
    # create the percentile subdirectory if it doesn't exist
    percentile_dir = Path(f'results/samples/{ticker}/percentile/')
    
    percentile_dir.mkdir(parents=True, exist_ok=True)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        list(tqdm(executor.map(get_percentiles, sample_files, repeat(ticker)), total=len(sample_files)))


def run_pipeline_transformations(ticker: str, max_workers: int=4) -> None:
    """
    Run the complete transformation pipeline for a given time prediction.
    Args:
        ticker: str
            ticker symbol
        max_workers: int
            Maximum number of workers for concurrent processing.
    Returns: None
    """
    # run the pipeline transformations
    # get the list of sample files
    sample_files = get_list_sample_files(ticker=ticker)

    print(f"Transforming {len(sample_files)} sample files for ticker: {ticker}")
    
    calc_all_percentiles(sample_files=sample_files, ticker=ticker, max_workers=max_workers)
    print(f"Pipeline transformations completed for samples, ticker: {ticker}")


if __name__ == "__main__":
    # pass the time prediction and monte_carlo flag as arguments commiting
    try:
        ticker = sys.argv[1]
    except IndexError:
        ticker = None
    
    if ticker is None:
        # run for all-time predictions
        tickers = [
            'OSK',
            ]
        for ticker in tickers:
            try:
                run_pipeline_transformations(ticker=ticker, max_workers=8)
            except FileNotFoundError as e:
                print(f"Skipping {ticker}: {e}")
                continue
    else:
        run_pipeline_transformations(ticker=ticker, max_workers=8)

    print(f"All transformations completed.")