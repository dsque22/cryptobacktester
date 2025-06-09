"""
Parameter Optimizer Module

MVP implementation of parameter optimization for trading strategies.
Focuses on simple grid search with clear deployment recommendations.
"""

from typing import Dict, Any, List, Union, Tuple
import itertools
import pandas as pd
from dataclasses import dataclass


@dataclass
class OptimizationResults:
    """Container for optimization results and deployment recommendations."""
    best_params: Dict[str, Any]
    best_sharpe: float
    deployment_score: float
    recommendation: str  # 'DEPLOY', 'TEST_MORE', 'REJECT'
    all_results: List[Dict[str, Any]]
    total_combinations: int


class ParameterOptimizer:
    """
    Simple parameter optimizer using grid search.
    
    MVP implementation that focuses on:
    - Grid search across parameter space
    - Sharpe ratio optimization
    - Simple deployment scoring
    - Clear recommendations
    """
    
    def __init__(self, strategy_class, data: pd.DataFrame, param_ranges: Dict[str, Any]):
        """
        Initialize the parameter optimizer.
        
        Args:
            strategy_class: Strategy class to optimize (e.g., HMAWAEStrategy)
            data: Historical price data for backtesting
            param_ranges: Dict of parameter names to ranges/values
                Example: {
                    'hma_period': range(15, 30, 5),
                    'wae_sensitivity': range(100, 200, 25),
                    'trade_direction': ['long', 'short', 'both']
                }
        """
        self.strategy_class = strategy_class
        self.data = data
        self.param_ranges = param_ranges
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
                iter(param_range)
            except TypeError:
                raise ValueError(f"Parameter range for '{param_name}' must be iterable")
    
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
        Run parameter optimization using grid search.
        
        Returns:
            OptimizationResults with best parameters and deployment recommendation
        """
        # Generate all parameter combinations
        combinations = self._generate_parameter_combinations()
        
        if not combinations:
            raise ValueError("No parameter combinations generated")
        
        # Placeholder for actual optimization logic (will be implemented in Stage 2)
        # For Stage 1, we just return the structure with dummy data
        
        # Mock results for testing the foundation
        mock_results = []
        for i, params in enumerate(combinations[:3]):  # Just test first 3 for foundation
            mock_results.append({
                'params': params,
                'sharpe_ratio': 1.0 + (i * 0.1),  # Mock sharpe ratios
                'total_return': 0.5 + (i * 0.1),
                'max_drawdown': -0.15 - (i * 0.02),
                'win_rate': 0.6 + (i * 0.05)
            })
        
        # Find best result (highest Sharpe ratio)
        best_result = max(mock_results, key=lambda x: x['sharpe_ratio'])
        
        # Calculate deployment score (placeholder logic)
        deployment_score = self._calculate_deployment_score(best_result)
        
        # Generate recommendation
        recommendation = self._generate_recommendation(deployment_score, best_result)
        
        return OptimizationResults(
            best_params=best_result['params'],
            best_sharpe=best_result['sharpe_ratio'],
            deployment_score=deployment_score,
            recommendation=recommendation,
            all_results=mock_results,
            total_combinations=len(combinations)
        )
    
    def _calculate_deployment_score(self, result: Dict[str, Any]) -> float:
        """
        Calculate deployment readiness score (0-1).
        
        Simple scoring based on:
        - Sharpe ratio
        - Win rate
        - Max drawdown
        """
        sharpe = result['sharpe_ratio']
        win_rate = result['win_rate']
        max_drawdown = abs(result['max_drawdown'])
        
        # Normalize metrics (simple approach for MVP)
        sharpe_score = min(sharpe / 1.5, 1.0)  # Target Sharpe > 1.5
        winrate_score = win_rate  # Already 0-1
        drawdown_score = max(1.0 - (max_drawdown / 0.25), 0.0)  # Target DD < 25%
        
        # Weighted average
        deployment_score = (
            0.4 * sharpe_score +
            0.3 * winrate_score +
            0.3 * drawdown_score
        )
        
        return min(deployment_score, 1.0)
    
    def _generate_recommendation(self, deployment_score: float, result: Dict[str, Any]) -> str:
        """Generate deployment recommendation based on score and metrics."""
        sharpe = result['sharpe_ratio']
        
        if deployment_score >= 0.8 and sharpe >= 1.2:
            return 'DEPLOY'
        elif deployment_score >= 0.6 and sharpe >= 1.0:
            return 'TEST_MORE'
        else:
            return 'REJECT'