"""
Performance metrics calculation.
"""
import pandas as pd
import numpy as np
from typing import Dict

from utils import calculate_sharpe_ratio, calculate_max_drawdown

def calculate_metrics(results) -> Dict[str, float]:
    """Calculate comprehensive performance metrics."""
    metrics = {}
    
    # Basic metrics
    metrics['total_return'] = results.total_return
    metrics['total_trades'] = results.total_trades
    metrics['win_rate'] = results.win_rate
    metrics['profit_factor'] = results.profit_factor
    
    # Risk metrics
    if len(results.returns) > 0:
        metrics['volatility'] = results.returns.std() * np.sqrt(252)
        metrics['sharpe_ratio'] = calculate_sharpe_ratio(results.returns)
        metrics['max_drawdown'] = calculate_max_drawdown(results.equity_curve)
        
        # Additional metrics
        metrics['avg_return'] = results.returns.mean()
        metrics['best_day'] = results.returns.max()
        metrics['worst_day'] = results.returns.min()
        metrics['positive_days'] = (results.returns > 0).sum()
        metrics['negative_days'] = (results.returns < 0).sum()
        
        # Risk-adjusted metrics
        downside_returns = results.returns[results.returns < 0]
        if len(downside_returns) > 0:
            downside_std = downside_returns.std() * np.sqrt(252)
            metrics['sortino_ratio'] = (results.returns.mean() * 252) / downside_std
        else:
            metrics['sortino_ratio'] = float('inf')
        
        # Calmar ratio
        if metrics['max_drawdown'] != 0:
            annual_return = results.total_return * (252 / len(results.returns))
            metrics['calmar_ratio'] = annual_return / abs(metrics['max_drawdown'])
        else:
            metrics['calmar_ratio'] = float('inf')
    
    # Trade metrics
    if results.total_trades > 0:
        all_pnl = [t.pnl for t in results.trades]
        metrics['avg_trade'] = sum(all_pnl) / len(all_pnl)
        metrics['best_trade'] = max(all_pnl)
        metrics['worst_trade'] = min(all_pnl)
        
        # Trade durations
        durations = [(t.exit_time - t.entry_time).total_seconds() / 3600 
                    for t in results.trades]
        metrics['avg_duration_hours'] = sum(durations) / len(durations)
    
    return metrics

def format_metrics(metrics: Dict[str, float]) -> str:
    """Format metrics for display."""
    output = []
    output.append("=== Performance Metrics ===")
    output.append(f"Total Return: {metrics.get('total_return', 0):.2%}")
    output.append(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
    output.append(f"Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
    output.append(f"Volatility: {metrics.get('volatility', 0):.2%}")
    output.append(f"Win Rate: {metrics.get('win_rate', 0):.2%}")
    output.append(f"Profit Factor: {metrics.get('profit_factor', 0):.2f}")
    output.append(f"Total Trades: {metrics.get('total_trades', 0)}")
    
    return "\n".join(output)