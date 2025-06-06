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

def create_performance_table(results, strategy_name: str = "Strategy") -> pd.DataFrame:
    """Create a comprehensive performance metrics table."""
    metrics = calculate_metrics(results)
    
    # Create structured table data
    performance_data = [
        # Returns & Profitability
        ["Total Return", f"{metrics.get('total_return', 0):.2%}"],
        ["Annualized Return", f"{metrics.get('total_return', 0) * (252 / len(results.returns) if len(results.returns) > 0 else 1):.2%}"],
        ["Total Trades", f"{int(metrics.get('total_trades', 0))}"],
        ["Winning Trades", f"{results.winning_trades}"],
        ["Losing Trades", f"{results.losing_trades}"],
        ["Win Rate", f"{metrics.get('win_rate', 0):.2%}"],
        
        # Risk Metrics
        ["Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0):.2f}"],
        ["Sortino Ratio", f"{metrics.get('sortino_ratio', 0):.2f}"],
        ["Calmar Ratio", f"{metrics.get('calmar_ratio', 0):.2f}"],
        ["Max Drawdown", f"{metrics.get('max_drawdown', 0):.2%}"],
        ["Volatility (Ann.)", f"{metrics.get('volatility', 0):.2%}"],
        
        # Trade Analysis
        ["Profit Factor", f"{metrics.get('profit_factor', 0):.2f}"],
        ["Average Trade", f"${metrics.get('avg_trade', 0):.2f}"],
        ["Best Trade", f"${metrics.get('best_trade', 0):.2f}"],
        ["Worst Trade", f"${metrics.get('worst_trade', 0):.2f}"],
        ["Avg Trade Duration (hrs)", f"{metrics.get('avg_duration_hours', 0):.1f}"],
        
        # Daily Statistics
        ["Average Daily Return", f"{metrics.get('avg_return', 0):.4%}"],
        ["Best Day", f"{metrics.get('best_day', 0):.2%}"],
        ["Worst Day", f"{metrics.get('worst_day', 0):.2%}"],
        ["Positive Days", f"{int(metrics.get('positive_days', 0))}"],
        ["Negative Days", f"{int(metrics.get('negative_days', 0))}"],
    ]
    
    # Create DataFrame
    df = pd.DataFrame(performance_data, columns=['Metric', 'Value'])
    df['Strategy'] = strategy_name
    
    return df

def save_performance_table(results, strategy_name: str, filepath: str = None):
    """Save performance table to CSV file."""
    from config import RESULTS_DIR
    
    # Create table
    table = create_performance_table(results, strategy_name)
    
    # Default filepath
    if filepath is None:
        filepath = RESULTS_DIR / f"{strategy_name}_performance_metrics.csv"
    
    # Save to CSV
    table.to_csv(filepath, index=False)
    print(f"📊 Performance metrics saved to: {filepath}")
    
    return table

def create_trades_table(results, strategy_name: str = "Strategy") -> pd.DataFrame:
    """Create detailed trades table for export."""
    if not results.trades:
        print("No trades to export")
        return pd.DataFrame()
    
    trades_data = []
    for i, trade in enumerate(results.trades, 1):
        duration = (trade.exit_time - trade.entry_time).total_seconds() / 3600  # hours
        
        trades_data.append({
            'Trade #': i,
            'Strategy': strategy_name,
            'Entry Time': trade.entry_time.strftime('%Y-%m-%d %H:%M:%S'),
            'Exit Time': trade.exit_time.strftime('%Y-%m-%d %H:%M:%S'),
            'Side': trade.side.title(),
            'Entry Price': f"${trade.entry_price:.4f}",
            'Exit Price': f"${trade.exit_price:.4f}",
            'Position Size': f"{trade.position_size:.6f}",
            'P&L ($)': f"${trade.pnl:.2f}",
            'P&L (%)': f"{trade.pnl_percent:.2%}",
            'Commission': f"${trade.commission:.2f}",
            'Duration (hrs)': f"{duration:.1f}",
            'Result': 'Win' if trade.pnl > 0 else 'Loss'
        })
    
    return pd.DataFrame(trades_data)

def save_trades_table(results, strategy_name: str, filepath: str = None):
    """Save trades table to CSV file."""
    from config import RESULTS_DIR
    
    # Create table
    table = create_trades_table(results, strategy_name)
    
    if table.empty:
        return table
    
    # Default filepath
    if filepath is None:
        filepath = RESULTS_DIR / f"{strategy_name}_trades_detailed.csv"
    
    # Save to CSV
    table.to_csv(filepath, index=False)
    print(f"📋 Detailed trades saved to: {filepath}")
    print(f"Total trades exported: {len(table)}")
    
    return table