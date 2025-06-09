"""
Parameter Optimizer Module

MVP implementation of parameter optimization for trading strategies.
Focuses on simple grid search with clear deployment recommendations.
"""

# Console formatting utilities
class Colors:
    """ANSI color codes for console output."""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'

def format_section(title: str) -> str:
    """Format section headers with colors."""
    return f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}\n{Colors.BOLD}{Colors.CYAN}{title}{Colors.END}\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}"

def format_metric(label: str, value: float, precision: int = 3) -> str:
    """Format metrics with appropriate colors."""
    if 'sharpe' in label.lower() or 'score' in label.lower():
        color = Colors.GREEN if value > 1.0 else Colors.YELLOW if value > 0.5 else Colors.RED
    elif 'stability' in label.lower():
        color = Colors.GREEN if value > 0.8 else Colors.YELLOW if value > 0.6 else Colors.RED
    else:
        color = Colors.CYAN
    return f"   {Colors.BOLD}{label}:{Colors.END} {color}{value:.{precision}f}{Colors.END}"

def format_recommendation(rec: str) -> str:
    """Format deployment recommendations with colors."""
    if rec == 'DEPLOY':
        return f"{Colors.BOLD}{Colors.GREEN}🟢 DEPLOY{Colors.END}"
    elif rec == 'TEST_MORE':
        return f"{Colors.BOLD}{Colors.YELLOW}🟡 TEST_MORE{Colors.END}"
    else:
        return f"{Colors.BOLD}{Colors.RED}🔴 REJECT{Colors.END}"

from typing import Dict, Any, List, Union, Tuple, Callable
import itertools
import pandas as pd
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
import time
from dataclasses import dataclass

from ..backtesting.engine import Backtester, BacktestResults


def _run_single_backtest(strategy_class, data: pd.DataFrame, params: Dict[str, Any], 
                        initial_capital: float, validation_split: float = 0.7) -> Dict[str, Any]:
    """
    Worker function to run a single backtest with given parameters.
    Must be at module level for multiprocessing pickle support.
    
    Args:
        strategy_class: Strategy class to instantiate
        data: Historical price data
        params: Strategy parameters
        initial_capital: Starting capital
        
    Returns:
        Dict with backtest results or None if failed
    """
    try:
        # Train/test split (70/30)
        split_idx = int(len(data) * validation_split)
        train_data, test_data = data.iloc[:split_idx], data.iloc[split_idx:]
        
        # Create strategy and run on train data
        strategy = strategy_class(**params)
        train_signals = strategy.generate_signals(train_data)
        backtester = Backtester(initial_capital=initial_capital)
        train_results = backtester.run(train_data, train_signals)
        
        # Run on test data for validation
        test_signals = strategy.generate_signals(test_data)
        test_results = backtester.run(test_data, test_signals)
        
        # Calculate stability (out-of-sample vs in-sample performance)
        stability = min(test_results.sharpe_ratio / max(train_results.sharpe_ratio, 0.01), 2.0) if train_results.sharpe_ratio > 0 else 0
        
        return {
            'sharpe_ratio': train_results.sharpe_ratio,
            'total_return': train_results.total_return,
            'max_drawdown': train_results.max_drawdown,
            'win_rate': train_results.win_rate,
            'total_trades': train_results.total_trades,
            'avg_win': train_results.avg_win,
            'avg_loss': train_results.avg_loss,
            'profit_factor': train_results.profit_factor,
            'stability': stability,
            'test_sharpe': test_results.sharpe_ratio
        }
        
    except Exception as e:
        # Return None for failed backtests - will be filtered out
        return None


@dataclass
class OptimizationResults:
    """Container for optimization results and deployment recommendations."""
    best_params: Dict[str, Any]
    best_sharpe: float
    deployment_score: float
    recommendation: str  # 'DEPLOY', 'TEST_MORE', 'REJECT'
    all_results: List[Dict[str, Any]]
    total_combinations: int
    stability: float = 0.0
    test_sharpe: float = 0.0


class ParameterOptimizer:
    """
    Simple parameter optimizer using grid search.
    
    MVP implementation that focuses on:
    - Grid search across parameter space
    - Sharpe ratio optimization
    - Simple deployment scoring
    - Clear recommendations
    """
    
    def __init__(self, strategy_class, data: pd.DataFrame, param_ranges: Dict[str, Any], 
                 n_jobs: int = None, initial_capital: float = 10000):
        """
        Initialize the parameter optimizer.
        
        Args:
            strategy_class: Strategy class to optimize (e.g., HMAWAEStrategy)
            data: Historical price data for backtesting
            param_ranges: Dict of parameter names to ranges/values
            n_jobs: Number of parallel processes (None = auto-detect)
            initial_capital: Starting capital for backtesting
        """
        self.strategy_class = strategy_class
        self.data = data
        self.param_ranges = param_ranges
        self.n_jobs = n_jobs or min(mp.cpu_count(), 8)  # Cap at 8 to avoid overwhelming
        self.initial_capital = initial_capital
        self.logger = logging.getLogger(__name__)
        self._validate_param_ranges()
        
    def _validate_param_ranges(self) -> None:
        """Validate parameter ranges are properly formatted."""
        if not self.param_ranges:
            raise ValueError("Parameter ranges cannot be empty")
            
        for param_name, param_range in self.param_ranges.items():
            if not param_name:
                raise ValueError("Parameter name cannot be empty")
            
            # Check if it's iterable (range, list, etc.)
            try:
                range_list = list(param_range)
                if len(range_list) == 0:
                    raise ValueError(f"Parameter range for '{param_name}' cannot be empty")
            except TypeError:
                raise ValueError(f"Parameter range for '{param_name}' must be iterable")
                
        # Validate data
        if self.data is None or self.data.empty:
            raise ValueError("Data cannot be None or empty")
            
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in self.data.columns]
        if missing_columns:
            raise ValueError(f"Data missing required columns: {missing_columns}")
            
        # Validate strategy class
        if not hasattr(self.strategy_class, '__init__'):
            raise ValueError("Strategy class must be a valid class with __init__ method")
            
        # Log validation success
        self.logger.info(f"✅ Validation passed: {len(self.param_ranges)} parameters, {len(self.data)} data points")
    
    def _generate_parameter_combinations(self) -> List[Dict[str, Any]]:
        """
        Generate all parameter combinations for grid search.
        
        Returns:
            List of parameter dictionaries for each combination
        """
        # Get parameter names and their values
        param_names = list(self.param_ranges.keys())
        param_values = [list(self.param_ranges[name]) for name in param_names]
        
        # Generate all combinations using cartesian product
        combinations = []
        for combination in itertools.product(*param_values):
            param_dict = dict(zip(param_names, combination))
            combinations.append(param_dict)
            
        return combinations
    
    def get_total_combinations(self) -> int:
        """Get total number of parameter combinations to test."""
        combinations = self._generate_parameter_combinations()
        return len(combinations)
    
    def optimize(self) -> OptimizationResults:
        """
        Run parameter optimization using parallel grid search.
        
        Returns:
            OptimizationResults with best parameters and deployment recommendation
        """
        # Generate all parameter combinations
        combinations = self._generate_parameter_combinations()
        
        if not combinations:
            raise ValueError("No parameter combinations generated")
        
        # Warn for large search spaces
        if len(combinations) > 1000:
            self.logger.warning(f"⚠️  Large search space: {len(combinations)} combinations may take a long time")
        
        print(format_section("PARAMETER OPTIMIZATION"))
        print(f"   {Colors.BOLD}Strategy:{Colors.END} {self.strategy_class.__name__}")
        print(f"   {Colors.BOLD}Combinations:{Colors.END} {len(combinations)}")
        print(f"   {Colors.BOLD}Processes:{Colors.END} {self.n_jobs}")
        print(f"   {Colors.BOLD}Data Points:{Colors.END} {len(self.data)}")
        self.logger.info(f"🚀 Starting optimization with {len(combinations)} combinations using {self.n_jobs} processes")
        
        # Run parallel backtests
        start_time = time.time()
        try:
            all_results = self._run_parallel_backtests(combinations)
        except Exception as e:
            self.logger.error(f"❌ Optimization failed during parallel execution: {str(e)}")
            raise
            
        elapsed_time = time.time() - start_time
        print(format_section("OPTIMIZATION RESULTS"))
        print(f"   {Colors.BOLD}Completed:{Colors.END} {len(all_results)}/{len(combinations)} tests")
        print(f"   {Colors.BOLD}Duration:{Colors.END} {elapsed_time:.1f}s")
        self.logger.info(f"✅ Optimization completed in {elapsed_time:.1f}s")
        
        if not all_results:
            raise RuntimeError("No successful backtest results generated - check strategy parameters and data")
        
        # Calculate success rate
        success_rate = len(all_results) / len(combinations)
        if success_rate < 0.5:
            self.logger.warning(f"⚠️  Low success rate: {success_rate:.1%} of backtests succeeded")
        
        # Find best result (highest Sharpe ratio)
        valid_results = [r for r in all_results if r.get('sharpe_ratio') is not None]
        if not valid_results:
            raise RuntimeError("No valid results with Sharpe ratio found")
            
        best_result = max(valid_results, key=lambda x: x.get('sharpe_ratio', -999))
        
        # Calculate deployment score
        deployment_score = self._calculate_deployment_score(best_result)
        
        # Generate recommendation
        recommendation = self._generate_recommendation(deployment_score, best_result)
        
        # Display formatted results
        print(format_metric("Best Sharpe Ratio", best_result['sharpe_ratio']))
        print(format_metric("Deployment Score", deployment_score))
        print(format_metric("Stability", best_result.get('stability', 0.0)))
        print(f"   {Colors.BOLD}Recommendation:{Colors.END} {format_recommendation(recommendation)}")
        
        self.logger.info(f"🎯 Best result: Sharpe={best_result['sharpe_ratio']:.2f}, Score={deployment_score:.2f}, Recommendation={recommendation}")
        
        return OptimizationResults(
            best_params=best_result['params'],
            best_sharpe=best_result['sharpe_ratio'],
            deployment_score=deployment_score,
            recommendation=recommendation,
            all_results=all_results,
            total_combinations=len(combinations),
            stability=best_result.get('stability', 0.0),
            test_sharpe=best_result.get('test_sharpe', 0.0)
        )
    
    def _run_parallel_backtests(self, combinations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Run backtests in parallel for all parameter combinations.
        
        Args:
            combinations: List of parameter dictionaries
            
        Returns:
            List of backtest results
        """
        all_results = []
        completed_count = 0
        
        try:
            with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
                # Submit all jobs
                future_to_params = {}
                for params in combinations:
                    future = executor.submit(_run_single_backtest, 
                                           self.strategy_class, 
                                           self.data, 
                                           params, 
                                           self.initial_capital)
                    future_to_params[future] = params
                
                # Collect results as they complete
                for future in as_completed(future_to_params):
                    params = future_to_params[future]
                    completed_count += 1
                    
                    try:
                        result = future.result(timeout=60)  # 60s timeout per backtest
                        if result:
                            result['params'] = params
                            all_results.append(result)
                        
                        # Progress logging every 10% or every 5 completions
                        progress_pct = (completed_count / len(combinations)) * 100
                        if completed_count % max(1, len(combinations) // 10) == 0 or completed_count % 5 == 0:
                            self.logger.info(f"⏳ Progress: {completed_count}/{len(combinations)} ({progress_pct:.1f}%)")
                            
                    except Exception as e:
                        self.logger.warning(f"❌ Backtest failed for params {params}: {str(e)}")
                        continue
                        
        except Exception as e:
            self.logger.error(f"❌ Parallel execution failed: {str(e)}")
            raise RuntimeError(f"Optimization failed: {str(e)}")
        
        self.logger.info(f"📊 Successful backtests: {len(all_results)}/{len(combinations)}")
        return all_results
    
    def _calculate_deployment_score(self, result: Dict[str, Any]) -> float:
        """
        Calculate deployment readiness score (0-1).
        
        Enhanced scoring: Sharpe + Win Rate + Drawdown + Stability
        """
        sharpe = result.get('sharpe_ratio', 0)
        win_rate = result.get('win_rate', 0)  
        max_drawdown = abs(result.get('max_drawdown', 1))
        total_trades = result.get('total_trades', 0)
        
        # Enhanced stability from train/test validation
        stability = result.get('stability', 0)
        
        # Normalize metrics
        sharpe_score = min(max(sharpe / 1.5, 0), 1.0)  # Target > 1.5
        winrate_score = max(min(win_rate, 1.0), 0)     # 0-1 range
        drawdown_score = max(1.0 - (max_drawdown / 0.25), 0.0)  # Target < 25%
        stability_score = min(max(stability, 0), 1.0)  # Out-of-sample performance ratio
        
        # Weighted scoring with stability
        return min((0.3 * sharpe_score + 0.25 * winrate_score + 0.25 * drawdown_score + 0.2 * stability_score), 1.0)
    
    def _generate_recommendation(self, deployment_score: float, result: Dict[str, Any]) -> str:
        """Enhanced recommendation logic with stability check."""
        sharpe = result.get('sharpe_ratio', 0)
        stability = result.get('stability', 0)
        
        if deployment_score >= 0.75 and sharpe >= 1.2 and stability >= 0.7:
            return 'DEPLOY'
        elif deployment_score >= 0.6 and sharpe >= 1.0:
            return 'TEST_MORE'
        else:
            return 'REJECT'