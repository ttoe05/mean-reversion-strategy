import pandas as pd
import numpy as np
from auto_kernel_ensemble import KernelAutoregressiveModel
from data_loader import DataLoader
from walk_forward import WalkForwardValidator
from pathlib import Path


def load_data_transform(ticker: str, lags: int, ewm: bool = True, persist: bool = True) -> pd.DataFrame:
    """
    Load the data and transform it for KRR auto regression. Create the lagged features. and the moving average target
    variable
    :param ticker: str
        ticker symbol
    :param lags: int
        the number of lags to use as feature variables
    :param ewm: bool
        whether to use exponential weighted moving average or simple moving average
    :param persist: bool
        persist the data to disk
    :return:
    df: pd.DataFrame
    """
    # load the data from the general source
    df_ticker = pd.read_parquet(f"data/{ticker}.parquet")
    # create the target variable
    if ewm:
        df_ticker["30 Day Avg"] = df_ticker["Adj Close"].ewm(span=30, adjust=False).mean()
    else:
        df_ticker["30 Day Avg"] = df_ticker["Adj Close"].rolling(30).mean()
    # create the future value variable
    df_ticker['30 Day Avg future value'] = df_ticker['30 Day Avg'].shift(-6)
    # create the feature variables
    for lag in range(1, lags + 1):
        df_ticker[f"Lag {lag}"] = df_ticker["30 Day Avg"].shift(lag)
    # drop any null
    df_final = df_ticker.dropna()
    if persist:
        directory = Path(f"data/sample/")
        directory.mkdir(parents=True, exist_ok=True)
        df_final.to_parquet(f"data/sample/{ticker}_backtesting.parquet")
    return df_final

def run_krr_pipeline(ticker: str, lags: int):
    """
    Run the end-to-end pipeline for KRR auto regression.
    :param ticker:
    :param lags:
    :param ewm:
    :param persist:
    :return:
    """
    data_load = DataLoader(data_path=f"data/sample/{ticker}_backtesting.parquet")
    auto_kernel = KernelAutoregressiveModel(forecast_horizon=7, n_jobs=2)

    independent_variables = [f'Lag {x}' for x in range(1, lags + 1)]
    data_load.load_data(x=independent_variables, y=['30 Day Avg'], actuals=['30 Day Avg future value'])
    walks = WalkForwardValidator(
        model=auto_kernel,
        data_loader=data_load,
        ticker=ticker,
        lags=lags,
        persist_samples=False,
        # forecast_horizon=5,
        # n_jobs=2,
        window_size=2000,
        min_window_size=2000,
        n_parallel_jobs=2,
        use_scaling=True,
        window_type='sliding'
    )
    walks.run_walk_forward_validation(target_variable='30 Day Avg')
    results_df = pd.DataFrame(walks.results_list)
    # create directory if it has not been created
    directory = Path(f"data/results/{ticker}/auto_kernel")
    directory.mkdir(parents=True, exist_ok=True)
    walks.export_results(filepath=f"data/results/{ticker}/auto_kernel/")
    results_df.to_parquet(f"data/results/{ticker}/auto_kernel/{ticker}_results.parquet")



def run_end_to_end_pipeline(ticker: str, lags: int, ewm: bool = True, persist: bool = True):
    df = load_data_transform(ticker=ticker, lags=lags, ewm=ewm, persist=persist)
    run_krr_pipeline(ticker=ticker, lags=lags)


if __name__ == "__main__":
    # list of assets currently in portfolio
    portfolio_assets = [
        'ACM',
        'OSK',
        'PATH',
        'RGLD',
        'AAAU',
        'CIBR',
        'IBIT',
        'ITA',
        'SGDM',
        'USO',
        'USXF',
        'XHE'
    ]
    for ticker in portfolio_assets:
        print(f"Evaluating ticker:\t{ticker} ...")
        run_end_to_end_pipeline(ticker=ticker, lags=15)
