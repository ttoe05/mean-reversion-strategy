"""
Author: Terrill Toe
Title: Genetic Algorithm Portfolio Optimizer

Description:
A class for running a genetic algorithm on a portfolio of assets. The optimizer takes in samples of future yields for
each asset to optimize the weights of the portfolio. The objective function of the algorithm is to minimize a risk metric
while maximizing returns.

Objective function: Maximize the probability that future cumulative returns exceed current cumulative returns

Genetic Algorithm:
    - Generate a population of z random weight vectors
    - Tournament-style selection where best weights advance
    - Selection criteria: max(P(future_cumulative_return > current_cumulative_return))
    - Tie-breaker: For long portfolios with zero probability, use VaR; otherwise random
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import logging


class GeneticOptimizer:
    """
    Runs a tournament-style genetic algorithm to optimize portfolio weights given forecasted
    yield distributions for each asset.
    """

    def __init__(
        self,
        current_portfolio_value: float,
        current_cumulative_return: float = 0.0,
        portfolio_type: str = "long",
        var_threshold: float = 0.05
    ):
        """
        Initialize the Genetic Optimizer.

        Parameters:
        -----------
        current_portfolio_value : float
            Current total value of the portfolio
        current_cumulative_return : float
            Current cumulative return of the portfolio (default 0.0)
        portfolio_type : str
            Portfolio type: "long", "short", or "buy_and_hold" (default: "long")
            - "long": Long-only positions with GA optimization
            - "short": Long/short hedged positions with GA optimization
            - "buy_and_hold": Equal weight allocation (no optimization)
        var_threshold : float
            Value at Risk threshold for tie-breaking (default: 0.05 for 5%)
        """
        PORTFOLIO_TYPE_VALS = ["long", "short", "buy_and_hold"]
        if portfolio_type not in PORTFOLIO_TYPE_VALS:
            raise ValueError(f"Invalid portfolio type: {portfolio_type}. Must be one of {PORTFOLIO_TYPE_VALS}")

        self.portfolio_type = portfolio_type
        self.current_portfolio_value = current_portfolio_value
        self.current_cumulative_return = current_cumulative_return
        self.var_threshold = var_threshold

    def generate_population(self, n_assets: int, population_size: int) -> np.ndarray:
        """
        Generate random weight vectors for the population.

        Parameters:
        -----------
        n_assets : int
            Number of assets in the portfolio
        population_size : int
            Number of individuals in the population

        Returns:
        --------
        np.ndarray
            Array of shape (population_size, n_assets) with weight vectors
            Weights sum between 0 and 1, allowing for cash holdings
        """
        if self.portfolio_type == "long":
            # For long portfolio: generate base weights that sum to 1
            base_weights = np.random.dirichlet(np.ones(n_assets), size=population_size)

            # Scale by random allocation factor between 0 and 1
            # This allows weights to sum between 0 and 1 (remaining is cash)
            allocation_factors = np.random.uniform(0, 1, size=population_size)
            weights = base_weights * allocation_factors[:, np.newaxis]
        else:
            # For hedged portfolio: each asset has long and short positions
            # Shape will be (population_size, n_assets * 2)
            base_weights = np.random.dirichlet(np.ones(n_assets * 2), size=population_size)

            # Scale by random allocation factor between 0 and 1
            allocation_factors = np.random.uniform(0, 1, size=population_size)
            weights = base_weights * allocation_factors[:, np.newaxis]

        return weights


    def _generate_buy_and_hold_weights(
        self,
        tickers: List[str]
    ) -> Tuple[Dict[str, float], float]:
        """
        Generate equal weights for buy-and-hold strategy.

        For a buy-and-hold (passive) strategy, all assets receive equal allocation.
        This provides a simple baseline for comparison with optimized strategies.

        Parameters:
        -----------
        tickers : list
            List of asset tickers to allocate

        Returns:
        --------
        tuple
            (equal_weights_dict, fitness_score)
            equal_weights_dict: Dictionary mapping each ticker to 1/n_assets
            fitness_score: 1.0 (marker indicating no optimization performed)

        Example:
        --------
        >>> tickers = ["AAPL", "GOOGL", "MSFT"]
        >>> weights, fitness = optimizer._generate_buy_and_hold_weights(tickers)
        >>> print(weights)
        {"AAPL": 0.333, "GOOGL": 0.333, "MSFT": 0.333}
        >>> print(fitness)
        1.0

        Notes:
        ------
        - Weights always sum to exactly 1.0 (fully invested)
        - No cash allocation for buy-and-hold
        - Fitness score of 1.0 is a marker, not a probability
        - No tournament selection or optimization performed
        """
        n_assets = len(tickers)
        equal_weight = 1.0 / n_assets

        # Create equal weight dictionary
        weights_dict = {ticker: equal_weight for ticker in tickers}

        # Fitness score of 1.0 indicates "no optimization needed"
        fitness_score = 1.0

        return weights_dict, fitness_score


    def return_calculator(
        self,
        current_price: float,
        predicted_price: float,
        predicted_std: float,
        num_samples: int,
        long: bool = True
    ) -> np.ndarray:
        """
        Calculate sampled returns based on model predictions.

        Parameters:
        -----------
        current_price : float
            Current asset price
        predicted_price : float
            Model's predicted price
        predicted_std : float
            Standard deviation of the prediction
        num_samples : int
            Number of Monte Carlo samples to generate
        long : bool
            If True, calculate long position returns; if False, short position returns

        Returns:
        --------
        np.ndarray
            Array of sampled log returns
        """
        # Sample from normal distribution of predicted prices
        sampled_prices = np.random.normal(predicted_price, predicted_std, num_samples)

        # Ensure prices are positive
        sampled_prices = np.maximum(sampled_prices, 1e-6)

        if long:
            # Long position: log(predicted / current)
            log_returns = np.log(sampled_prices / current_price)
        else:
            # Short position: log(current / predicted) - inverted
            log_returns = np.log(current_price / sampled_prices)

        return log_returns

    def compute_portfolio_returns(
        self,
        weights: np.ndarray,
        sampled_returns: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """
        Compute portfolio returns given weights and individual asset returns.

        Parameters:
        -----------
        weights : np.ndarray
            Portfolio weights for each asset (or 2x for hedged: long + short)
        sampled_returns : dict
            Dictionary mapping asset tickers to their sampled returns
            For hedged: keys include "_long" and "_short" suffixes

        Returns:
        --------
        np.ndarray
            Array of portfolio cumulative returns
        """
        # Get asset tickers in consistent order
        tickers = sorted(sampled_returns.keys())

        # Stack individual asset returns
        returns_matrix = np.column_stack([sampled_returns[ticker] for ticker in tickers])

        # Calculate weighted portfolio log returns
        portfolio_log_returns = np.dot(returns_matrix, weights)

        # Convert to cumulative returns: exp(sum(log_returns)) - 1
        portfolio_cumulative_returns = np.exp(portfolio_log_returns) - 1

        # Add to current cumulative return
        final_cumulative_returns = (1 + self.current_cumulative_return) * (1 + portfolio_cumulative_returns) - 1

        return final_cumulative_returns

    def calculate_var(self, returns: np.ndarray, threshold: float = None) -> float:
        """
        Calculate Value at Risk (VaR) for a given return distribution.

        Parameters:
        -----------
        returns : np.ndarray
            Array of returns
        threshold : float
            Percentile threshold (default uses self.var_threshold)

        Returns:
        --------
        float
            VaR value (negative number representing potential loss)
        """
        if threshold is None:
            threshold = self.var_threshold

        return np.percentile(returns, threshold * 100)

    def tournament_selection(
        self,
        population: np.ndarray,
        sampled_returns: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, float]:
        """
        Perform tournament selection to find optimal weights.

        Parameters:
        -----------
        population : np.ndarray
            Population of weight vectors
        sampled_returns : dict
            Dictionary of sampled returns for each asset

        Returns:
        --------
        tuple
            (optimal_weights, fitness_score)
        """
        current_population = population.copy()

        while len(current_population) > 1:
            next_generation = []

            # Randomly shuffle population for pairing
            np.random.shuffle(current_population)

            # Pair individuals and run tournaments
            for i in range(0, len(current_population), 2):
                if i + 1 < len(current_population):
                    individual1 = current_population[i]
                    individual2 = current_population[i + 1]

                    # Compute fitness for both individuals
                    returns1 = self.compute_portfolio_returns(individual1, sampled_returns)
                    returns2 = self.compute_portfolio_returns(individual2, sampled_returns)

                    # Calculate probability of beating current cumulative return
                    prob1 = np.mean(returns1 > self.current_cumulative_return)
                    prob2 = np.mean(returns2 > self.current_cumulative_return)

                    # Select winner based on probability
                    if prob1 > prob2:
                        winner = individual1
                    elif prob2 > prob1:
                        winner = individual2
                    else:
                        # Tie-breaker logic
                        if self.portfolio_type == "long" and prob1 == 0.0 and prob2 == 0.0:
                            # Bear market: use VaR as tie-breaker (prefer lower VaR)
                            var1 = self.calculate_var(returns1)
                            var2 = self.calculate_var(returns2)
                            winner = individual1 if var1 > var2 else individual2  # Higher VaR is less loss
                        else:
                            # Random tie-breaker
                            winner = individual1 if np.random.random() < 0.5 else individual2

                    next_generation.append(winner)
                else:
                    # Odd number: last individual advances automatically
                    next_generation.append(current_population[i])

            current_population = np.array(next_generation)

        # Calculate final fitness score
        optimal_weights = current_population[0]
        final_returns = self.compute_portfolio_returns(optimal_weights, sampled_returns)
        fitness_score = np.mean(final_returns > self.current_cumulative_return)

        return optimal_weights, fitness_score

    def optimize(
        self,
        current_prices: Dict[str, float],
        predictions: Dict[str, Dict[str, float]],
        population_size: int = 100,
        num_samples: int = 1000
    ) -> Tuple[Dict[str, float], float]:
        """
        Run the genetic algorithm to optimize portfolio weights.

        For portfolio_type="buy_and_hold", returns equal weights without running
        the genetic algorithm, providing a computational shortcut for baseline comparison.

        Parameters:
        -----------
        current_prices : dict
            Dictionary mapping asset tickers to current prices
        predictions : dict
            Dictionary mapping asset tickers to prediction dicts with 'predicted' and 'std'
            (Not used for buy_and_hold strategy)
        population_size : int
            Size of the population (default: 100)
            (Ignored for buy_and_hold strategy)
        num_samples : int
            Number of Monte Carlo samples (default: 1000)
            (Ignored for buy_and_hold strategy)

        Returns:
        --------
        tuple
            (optimal_weights_dict, fitness_score)
            For buy_and_hold: equal weights for all assets, fitness = 1.0
            For long/short: GA-optimized weights, fitness = P(future > current return)
        """
        tickers = sorted(current_prices.keys())
        n_assets = len(tickers)

        # Early return for buy-and-hold strategy (no optimization needed)
        if self.portfolio_type == "buy_and_hold":
            return self._generate_buy_and_hold_weights(tickers)

        # Generate population
        population = self.generate_population(n_assets, population_size)

        # Sample returns for each asset
        sampled_returns = {}

        if self.portfolio_type == "long":
            # Long-only: one set of returns per asset
            for ticker in tickers:
                sampled_returns[ticker] = self.return_calculator(
                    current_price=current_prices[ticker],
                    predicted_price=predictions[ticker]['predicted'],
                    predicted_std=predictions[ticker]['std'],
                    num_samples=num_samples,
                    long=True
                )
        else:
            # Hedged: separate long and short returns for each asset
            for ticker in tickers:
                # Long returns
                sampled_returns[f"{ticker}_long"] = self.return_calculator(
                    current_price=current_prices[ticker],
                    predicted_price=predictions[ticker]['predicted'],
                    predicted_std=predictions[ticker]['std'],
                    num_samples=num_samples,
                    long=True
                )
                # Short returns (inverted)
                sampled_returns[f"{ticker}_short"] = self.return_calculator(
                    current_price=current_prices[ticker],
                    predicted_price=predictions[ticker]['predicted'],
                    predicted_std=predictions[ticker]['std'],
                    num_samples=num_samples,
                    long=False
                )

        # Run tournament selection
        optimal_weights_array, fitness_score = self.tournament_selection(population, sampled_returns)

        # Convert to dictionary
        if self.portfolio_type == "long":
            optimal_weights_dict = {ticker: weight for ticker, weight in zip(tickers, optimal_weights_array)}
        else:
            # Hedged: map to ticker_long and ticker_short
            weight_keys = []
            for ticker in tickers:
                weight_keys.append(f"{ticker}_long")
                weight_keys.append(f"{ticker}_short")
            optimal_weights_dict = {key: weight for key, weight in zip(weight_keys, optimal_weights_array)}

        return optimal_weights_dict, fitness_score


class PortfolioManager:
    """
    Manages portfolio state over time, tracking positions, cost basis, portfolio value,
    and uninvested cash.
    """

    def __init__(
        self,
        initial_capital: float,
        cost_basis_method: str = "LIFO"
    ):
        """
        Initialize the Portfolio Manager.

        Parameters:
        -----------
        initial_capital : float
            Starting capital for the portfolio
        cost_basis_method : str
            Method for cost basis tracking: "LIFO" or "FIFO" (default: "LIFO")
        """
        if cost_basis_method not in ["LIFO", "FIFO"]:
            raise ValueError(f"Invalid cost basis method: {cost_basis_method}. Must be 'LIFO' or 'FIFO'")

        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.cost_basis_method = cost_basis_method

        # Positions: {ticker: [{"shares": float, "cost_basis": float, "purchase_date": datetime}, ...]}
        self.positions: Dict[str, List[Dict]] = {}

        # History tracking
        self.portfolio_value_history: List[Dict] = []
        self.weights_history: List[Dict] = []
        self.trades_history: List[Dict] = []

        # Performance tracking
        self.current_date = None

    def get_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """
        Calculate current portfolio value.

        Parameters:
        -----------
        current_prices : dict
            Dictionary mapping tickers to current prices

        Returns:
        --------
        float
            Total portfolio value (cash + positions)
        """
        positions_value = 0.0

        for ticker, blocks in self.positions.items():
            if ticker in current_prices:
                total_shares = sum(block["shares"] for block in blocks)
                positions_value += total_shares * current_prices[ticker]

        return self.cash + positions_value

    def get_current_cumulative_return(self) -> float:
        """
        Calculate current cumulative return.

        Returns:
        --------
        float
            Cumulative return since inception
        """
        if not self.portfolio_value_history:
            return 0.0

        current_value = self.portfolio_value_history[-1]["portfolio_value"]
        return (current_value - self.initial_capital) / self.initial_capital

    def update_positions(
        self,
        optimal_weights: Dict[str, float],
        current_prices: Dict[str, float],
        current_date: datetime
    ):
        """
        Update positions based on optimal weights.

        Parameters:
        -----------
        optimal_weights : dict
            Target weights for each asset (may have _long/_short suffixes for hedged)
        current_prices : dict
            Current prices for each asset (base tickers only)
        current_date : datetime
            Current date for tracking
        """
        self.current_date = current_date
        portfolio_value = self.get_portfolio_value(current_prices)

        # Check if this is a hedged portfolio (weights have _long/_short suffixes)
        is_hedged = any("_long" in key or "_short" in key for key in optimal_weights.keys())

        if is_hedged:
            # For hedged portfolios, we only track net positions for simplicity
            # Combine long and short weights to get net position per asset
            net_weights = {}
            for key, weight in optimal_weights.items():
                if "_long" in key:
                    ticker = key.replace("_long", "")
                    if ticker not in net_weights:
                        net_weights[ticker] = 0.0
                    net_weights[ticker] += weight
                elif "_short" in key:
                    ticker = key.replace("_short", "")
                    if ticker not in net_weights:
                        net_weights[ticker] = 0.0
                    net_weights[ticker] -= weight  # Subtract short weight

            # Use net weights for position updates
            optimal_weights = net_weights

        # Calculate target dollar allocations
        target_values = {ticker: portfolio_value * weight
                        for ticker, weight in optimal_weights.items()}

        # Calculate current values
        current_values = {}
        for ticker in optimal_weights.keys():
            if ticker in self.positions:
                total_shares = sum(block["shares"] for block in self.positions[ticker])
                current_values[ticker] = total_shares * current_prices[ticker]
            else:
                current_values[ticker] = 0.0

        # First, process sells to free up capital
        for ticker in sorted(optimal_weights.keys()):
            current_value = current_values[ticker]
            target_value = target_values[ticker]

            if target_value < current_value:
                # Need to sell
                self._sell_shares(ticker, current_value - target_value, current_prices[ticker], current_date)

        # Then, process buys
        for ticker in sorted(optimal_weights.keys()):
            current_value = current_values[ticker]
            target_value = target_values[ticker]

            if target_value > current_value:
                # Need to buy
                amount_to_buy = min(target_value - current_value, self.cash)
                if amount_to_buy > 0:
                    self._buy_shares(ticker, amount_to_buy, current_prices[ticker], current_date)

    def _buy_shares(
        self,
        ticker: str,
        dollar_amount: float,
        price: float,
        purchase_date: datetime
    ):
        """
        Buy shares of an asset.

        Parameters:
        -----------
        ticker : str
            Asset ticker
        dollar_amount : float
            Dollar amount to invest
        price : float
            Current price per share
        purchase_date : datetime
            Date of purchase
        """
        if dollar_amount > self.cash:
            dollar_amount = self.cash

        if dollar_amount <= 0:
            return

        shares_to_buy = dollar_amount / price

        if ticker not in self.positions:
            self.positions[ticker] = []

        self.positions[ticker].append({
            "shares": shares_to_buy,
            "cost_basis": price,
            "purchase_date": purchase_date
        })

        self.cash -= dollar_amount

        # Log trade
        self.trades_history.append({
            "date": purchase_date,
            "ticker": ticker,
            "action": "BUY",
            "shares": shares_to_buy,
            "price": price,
            "amount": dollar_amount
        })

    def _sell_shares(
        self,
        ticker: str,
        dollar_amount: float,
        price: float,
        sale_date: datetime
    ):
        """
        Sell shares of an asset using LIFO or FIFO.

        Parameters:
        -----------
        ticker : str
            Asset ticker
        dollar_amount : float
            Dollar amount to sell
        price : float
            Current price per share
        sale_date : datetime
            Date of sale
        """
        if ticker not in self.positions or not self.positions[ticker]:
            return

        shares_to_sell = dollar_amount / price
        shares_sold = 0.0

        while shares_to_sell > 1e-6 and self.positions[ticker]:
            if self.cost_basis_method == "LIFO":
                block = self.positions[ticker][-1]
                index = -1
            else:  # FIFO
                block = self.positions[ticker][0]
                index = 0

            if block["shares"] <= shares_to_sell:
                # Sell entire block
                shares_sold += block["shares"]
                self.cash += block["shares"] * price
                shares_to_sell -= block["shares"]
                self.positions[ticker].pop(index)
            else:
                # Sell partial block
                block["shares"] -= shares_to_sell
                shares_sold += shares_to_sell
                self.cash += shares_to_sell * price
                shares_to_sell = 0

        # Log trade
        self.trades_history.append({
            "date": sale_date,
            "ticker": ticker,
            "action": "SELL",
            "shares": shares_sold,
            "price": price,
            "amount": shares_sold * price
        })

        # Clean up empty position
        if ticker in self.positions and not self.positions[ticker]:
            del self.positions[ticker]

    def calculate_metrics(self) -> Dict:
        """
        Calculate portfolio performance metrics.

        Returns:
        --------
        dict
            Dictionary containing performance metrics
        """
        if not self.portfolio_value_history:
            return {
                "total_return": 0.0,
                "max_drawdown": 0.0,
                "final_value": self.initial_capital,
                "num_trades": 0
            }

        # Extract portfolio values
        values = [entry["portfolio_value"] for entry in self.portfolio_value_history]

        # Total return
        final_value = values[-1]
        total_return = (final_value - self.initial_capital) / self.initial_capital

        # Maximum drawdown
        peak = values[0]
        max_drawdown = 0.0

        for value in values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            if drawdown > max_drawdown:
                max_drawdown = drawdown

        return {
            "total_return": total_return,
            "max_drawdown": max_drawdown,
            "final_value": final_value,
            "initial_capital": self.initial_capital,
            "num_trades": len(self.trades_history),
            "num_days": len(self.portfolio_value_history)
        }

    def run_backtest(
        self,
        data: pd.DataFrame,
        ticker: str,
        rebalance_frequency: int = 1,
        population_size: int = 100,
        num_samples: int = 1000
    ) -> Dict:
        """
        Run backtest on historical data.

        Parameters:
        -----------
        data : pd.DataFrame
            DataFrame with columns: date, Adj Close, predicted, std
        ticker : str
            Asset ticker symbol
        rebalance_frequency : int
            Number of days between rebalancing (default: 1 for daily)
        population_size : int
            Population size for genetic algorithm (default: 100)
        num_samples : int
            Number of Monte Carlo samples (default: 1000)

        Returns:
        --------
        dict
            Performance metrics
        """
        logging.info(f"Starting backtest for {ticker} with {len(data)} observations")
        logging.info(f"Initial capital: ${self.initial_capital:,.2f}")
        logging.info(f"Rebalance frequency: {rebalance_frequency} days")

        # Iterate through data chronologically
        for idx, row in data.iterrows():
            current_date = row['date']
            current_price = row['Adj Close']
            predicted_price = row['predicted']
            predicted_std = row['std']

            # Ensure we have valid data
            if pd.isna(current_price) or pd.isna(predicted_price) or pd.isna(predicted_std):
                continue

            # Rebalance at specified frequency
            if idx % rebalance_frequency == 0:
                # Get current portfolio value and cumulative return
                portfolio_value = self.get_portfolio_value({ticker: current_price})
                cumulative_return = self.get_current_cumulative_return()

                # Initialize optimizer with current state
                optimizer = GeneticOptimizer(
                    current_portfolio_value=portfolio_value,
                    current_cumulative_return=cumulative_return,
                    portfolio_type="long"
                )

                # Run optimization
                optimal_weights, fitness_score = optimizer.optimize(
                    current_prices={ticker: current_price},
                    predictions={ticker: {'predicted': predicted_price, 'std': predicted_std}},
                    population_size=population_size,
                    num_samples=num_samples
                )

                # Update positions
                self.update_positions(optimal_weights, {ticker: current_price}, current_date)

                # Record weights
                self.weights_history.append({
                    "date": current_date,
                    "weights": optimal_weights.copy(),
                    "fitness_score": fitness_score
                })

                if idx % 50 == 0:
                    logging.info(f"Day {idx}: Portfolio value: ${portfolio_value:,.2f}, "
                               f"Weight: {optimal_weights[ticker]:.3f}, Fitness: {fitness_score:.3f}")

            # Record portfolio value daily
            portfolio_value = self.get_portfolio_value({ticker: current_price})
            self.portfolio_value_history.append({
                "date": current_date,
                "portfolio_value": portfolio_value
            })

        # Calculate final metrics
        metrics = self.calculate_metrics()

        logging.info("\n" + "="*50)
        logging.info("BACKTEST RESULTS")
        logging.info("="*50)
        logging.info(f"Final Portfolio Value: ${metrics['final_value']:,.2f}")
        logging.info(f"Total Return: {metrics['total_return']*100:.2f}%")
        logging.info(f"Maximum Drawdown: {metrics['max_drawdown']*100:.2f}%")
        logging.info(f"Number of Trades: {metrics['num_trades']}")
        logging.info(f"Number of Days: {metrics['num_days']}")
        logging.info("="*50)

        return metrics

    def run_multi_asset_backtest(
        self,
        data_dict: Dict[str, pd.DataFrame],
        rebalance_frequency: int = 5,
        population_size: int = 50,
        num_samples: int = 500,
        portfolio_type: str = "long"
    ) -> Dict:
        """
        Run backtest with multiple assets.

        Parameters:
        -----------
        data_dict : dict
            Dictionary mapping tickers to their prediction DataFrames
        rebalance_frequency : int
            Number of days between rebalancing (default: 5)
        population_size : int
            Population size for genetic algorithm (default: 50)
        num_samples : int
            Number of Monte Carlo samples (default: 500)
        portfolio_type : str
            "long" for long-only or "short" for hedged portfolio (default: "long")

        Returns:
        --------
        dict
            Performance metrics
        """
        tickers = sorted(data_dict.keys())
        logging.info(f"Starting multi-asset backtest ({portfolio_type.upper()})")
        logging.info(f"Assets: {', '.join(tickers)}")
        logging.info(f"Initial capital: ${self.initial_capital:,.2f}")
        logging.info(f"Rebalance frequency: {rebalance_frequency} days")
        logging.info(f"Portfolio type: {portfolio_type}")

        # Verify all dataframes have the same dates
        all_dates = None
        for ticker, df in data_dict.items():
            df_dates = set(df['date'])
            if all_dates is None:
                all_dates = df_dates
            else:
                all_dates = all_dates & df_dates

        if not all_dates:
            raise ValueError("No common dates found across all assets")

        common_dates = sorted(all_dates)
        logging.info(f"Common dates: {len(common_dates)}")

        # Iterate through common dates
        for idx, current_date in enumerate(common_dates):
            # Get current data for all assets
            current_prices = {}
            predictions = {}

            for ticker in tickers:
                ticker_data = data_dict[ticker]
                row = ticker_data[ticker_data['date'] == current_date].iloc[0]

                current_price = row['Adj Close']
                predicted_price = row['predicted']
                predicted_std = row['std']

                # Skip if any data is invalid
                if pd.isna(current_price) or pd.isna(predicted_price) or pd.isna(predicted_std):
                    continue

                current_prices[ticker] = current_price
                predictions[ticker] = {'predicted': predicted_price, 'std': predicted_std}

            # Rebalance at specified frequency
            if idx % rebalance_frequency == 0 and current_prices:
                # Get current portfolio value and cumulative return
                portfolio_value = self.get_portfolio_value(current_prices)
                cumulative_return = self.get_current_cumulative_return()

                # Initialize optimizer with current state
                optimizer = GeneticOptimizer(
                    current_portfolio_value=portfolio_value,
                    current_cumulative_return=cumulative_return,
                    portfolio_type=portfolio_type
                )

                # Run optimization
                optimal_weights, fitness_score = optimizer.optimize(
                    current_prices=current_prices,
                    predictions=predictions,
                    population_size=population_size,
                    num_samples=num_samples
                )

                # Update positions
                self.update_positions(optimal_weights, current_prices, current_date)

                # Record weights
                self.weights_history.append({
                    "date": current_date,
                    "weights": optimal_weights.copy(),
                    "fitness_score": fitness_score
                })

                if idx % 50 == 0:
                    weight_str = ", ".join([f"{t}: {w:.3f}" for t, w in optimal_weights.items()])
                    logging.info(f"Day {idx}: Portfolio value: ${portfolio_value:,.2f}, "
                               f"Weights: [{weight_str}], Fitness: {fitness_score:.3f}")

            # Record portfolio value daily
            if current_prices:
                portfolio_value = self.get_portfolio_value(current_prices)
                self.portfolio_value_history.append({
                    "date": current_date,
                    "portfolio_value": portfolio_value
                })

        # Calculate final metrics
        metrics = self.calculate_metrics()
        metrics['tickers'] = tickers
        metrics['portfolio_type'] = portfolio_type

        logging.info("\n" + "="*50)
        logging.info(f"MULTI-ASSET BACKTEST RESULTS ({portfolio_type.upper()})")
        logging.info("="*50)
        logging.info(f"Assets: {', '.join(tickers)}")
        logging.info(f"Final Portfolio Value: ${metrics['final_value']:,.2f}")
        logging.info(f"Total Return: {metrics['total_return']*100:.2f}%")
        logging.info(f"Maximum Drawdown: {metrics['max_drawdown']*100:.2f}%")
        logging.info(f"Number of Trades: {metrics['num_trades']}")
        logging.info(f"Number of Days: {metrics['num_days']}")
        logging.info("="*50)

        return metrics
