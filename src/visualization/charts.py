"""
Simple visualization for backtest results.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Optional

from config import RESULTS_DIR

# Set a consistent and modern style
plt.style.use('seaborn-v0_8-darkgrid')

# Define a color palette
PALETTE = {
    'green': '#28a745',
    'red': '#dc3545',
    'blue': '#007bff',
    'gray': '#f0f0f0',
    'black': '#000000',
    'white': '#ffffff',
    'yellow': '#ffc107',
    'orange': '#fd7e14',
    'purple': '#6f42c1',
    'cyan': '#17a2b8',
    'magenta': '#e83e8c'
}

def plot_equity_curve(results, title: str = "Equity Curve", save: bool = False):
    """Plot equity curve and drawdown."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), 
                                   gridspec_kw={'height_ratios': [3, 1]})
    
    # Equity curve
    ax1.plot(results.equity_curve.index, results.equity_curve.values, 
             linewidth=2, color='blue')
    ax1.set_title(title)
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.grid(True, alpha=0.3)
    
    # Drawdown
    drawdown = (results.equity_curve / results.equity_curve.cummax()) - 1
    ax2.fill_between(drawdown.index, drawdown.values * 100, 0, 
                     color='red', alpha=0.3)
    ax2.set_ylabel('Drawdown (%)')
    ax2.set_xlabel('Date')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save:
        plt.savefig(RESULTS_DIR / f"{title.replace(' ', '_')}_equity.png")
    
    plt.show()

def plot_returns_distribution(results, save: bool = False):
    """Plot returns distribution."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    returns = results.returns * 100  # Convert to percentage
    
    # Histogram
    n, bins, patches = ax.hist(returns, bins=50, alpha=0.7, 
                              color='blue', edgecolor='black')
    
    # Add normal distribution overlay
    mu, sigma = returns.mean(), returns.std()
    x = np.linspace(returns.min(), returns.max(), 100)
    ax.plot(x, ((1 / (np.sqrt(2 * np.pi) * sigma)) *
                np.exp(-0.5 * ((x - mu) / sigma) ** 2)) * len(returns) * (bins[1] - bins[0]),
            'r-', linewidth=2, label='Normal Distribution')
    
    ax.axvline(mu, color='red', linestyle='--', linewidth=2, label=f'Mean: {mu:.2f}%')
    ax.set_xlabel('Daily Returns (%)')
    ax.set_ylabel('Frequency')
    ax.set_title('Returns Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save:
        plt.savefig(RESULTS_DIR / "returns_distribution.png")
    
    plt.show()

def plot_trade_analysis(results, save: bool = False):
    """Plot trade analysis."""
    if not results.trades:
        print("No trades to analyze")
        return
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Extract trade data
    pnl = [t.pnl for t in results.trades]
    pnl_pct = [t.pnl_percent * 100 for t in results.trades]
    
    # 1. P&L per trade
    ax1.bar(range(len(pnl)), pnl, color=['green' if p > 0 else 'red' for p in pnl])
    ax1.set_title('P&L per Trade')
    ax1.set_xlabel('Trade Number')
    ax1.set_ylabel('P&L ($)')
    ax1.grid(True, alpha=0.3)
    
    # 2. Cumulative P&L
    cum_pnl = np.cumsum(pnl)
    ax2.plot(cum_pnl, linewidth=2)
    ax2.set_title('Cumulative P&L')
    ax2.set_xlabel('Trade Number')
    ax2.set_ylabel('Cumulative P&L ($)')
    ax2.grid(True, alpha=0.3)
    
    # 3. P&L distribution
    ax3.hist(pnl_pct, bins=20, alpha=0.7, color='blue', edgecolor='black')
    ax3.axvline(0, color='red', linestyle='--', linewidth=2)
    ax3.set_title('Trade P&L Distribution')
    ax3.set_xlabel('P&L (%)')
    ax3.set_ylabel('Frequency')
    ax3.grid(True, alpha=0.3)
    
    # 4. Win/Loss pie chart
    wins = sum(1 for p in pnl if p > 0)
    losses = len(pnl) - wins
    ax4.pie([wins, losses], labels=['Wins', 'Losses'], 
            colors=['green', 'red'], autopct='%1.1f%%')
    ax4.set_title('Win/Loss Ratio')
    
    plt.tight_layout()
    
    if save:
        plt.savefig(RESULTS_DIR / "trade_analysis.png")
    
    plt.show()

def plot_monthly_returns(results, save: bool = False):
    """Plot monthly returns heatmap."""
    # Calculate monthly returns
    monthly_returns = results.returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
    
    # Create pivot table
    monthly_data = pd.DataFrame({
        'Year': monthly_returns.index.year,
        'Month': monthly_returns.index.month,
        'Return': monthly_returns.values * 100
    })
    
    pivot = monthly_data.pivot(index='Year', columns='Month', values='Return')
    
    # Plot heatmap
    plt.figure(figsize=(12, 6))
    sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn', 
                center=0, cbar_kws={'label': 'Monthly Return (%)'})
    
    plt.title('Monthly Returns Heatmap')
    plt.xlabel('Month')
    plt.ylabel('Year')
    
    # Month labels
    plt.gca().set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    
    plt.tight_layout()
    
    if save:
        plt.savefig(RESULTS_DIR / "monthly_returns.png")
    
    plt.show()

def create_performance_report(results, strategy_name: str = "Strategy"):
    """Create complete performance report with all charts."""
    print(f"\nGenerating performance report for {strategy_name}...")
    
    plot_equity_curve(results, title=f"{strategy_name} - Equity Curve", save=True)
    plot_returns_distribution(results, save=True)
    plot_trade_analysis(results, save=True)
    plot_monthly_returns(results, save=True)
    
    print(f"Report saved to {RESULTS_DIR}")