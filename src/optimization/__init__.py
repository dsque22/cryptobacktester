"""
Parameter Optimization Module

This module provides parameter optimization capabilities for trading strategies.
MVP version focuses on simple grid search optimization with deployment scoring.
"""

from .optimizer import ParameterOptimizer

__all__ = ['ParameterOptimizer']