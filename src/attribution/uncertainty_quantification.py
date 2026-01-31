"""
Uncertainty Quantification Engine

Implements sensitivity analysis and uncertainty quantification per whitepaper:
1. α-sweep (blend parameter sensitivity)
2. λ-sweep (psychographic prior strength)
3. Bootstrap UQ (path sampling uncertainty)
4. Dirichlet UQ (transition parameter uncertainty)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import numpy as np
from copy import deepcopy
import logging

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from attribution.hybrid_engine import HybridAttributionEngine, HybridAttributionConfig

logger = logging.getLogger(__name__)


@dataclass
class SensitivityResult:
    """Results from sensitivity analysis."""
    parameter_name: str
    parameter_values: List[float]
    channel_sensitivities: Dict[str, Dict[float, float]]  # channel -> {param_value: attribution_share}
    
    def get_range(self, channel: str) -> Tuple[float, float]:
        """Get min/max attribution share for a channel across parameter sweep."""
        values = list(self.channel_sensitivities[channel].values())
        return min(values), max(values)
    
    def get_relative_range(self, channel: str) -> float:
        """Get relative range as percentage of mean."""
        min_val, max_val = self.get_range(channel)
        mean_val = np.mean(list(self.channel_sensitivities[channel].values()))
        if mean_val > 0:
            return (max_val - min_val) / mean_val
        return 0.0
    
    def get_rank_stability(self, channel: str, rank: int = 1) -> float:
        """
        Get percentage of parameter values where channel has given rank.
        
        Example: rank_stability('Search', 1) = 0.7 means Search was #1 in 70% of runs.
        """
        n_total = len(self.parameter_values)
        n_at_rank = 0
        
        for param_val in self.parameter_values:
            # Get attributions at this parameter value
            attrs = {
                ch: self.channel_sensitivities[ch][param_val]
                for ch in self.channel_sensitivities
            }
            
            # Sort by attribution
            sorted_channels = sorted(attrs.items(), key=lambda x: x[1], reverse=True)
            
            # Check if channel has desired rank
            if rank <= len(sorted_channels):
                if sorted_channels[rank - 1][0] == channel:
                    n_at_rank += 1
        
        return n_at_rank / n_total if n_total > 0 else 0.0


@dataclass
class UncertaintyResult:
    """Results from uncertainty quantification."""
    method: str  # 'bootstrap' or 'dirichlet'
    n_samples: int
    confidence_intervals: Dict[str, Dict[str, float]]  # channel -> {p05, p25, p50, p75, p95}
    rank_stability: Dict[str, Dict[str, float]]  # channel -> {top1, top2, top3}
    
    def get_summary(self) -> str:
        """Get human-readable summary."""
        lines = [
            f"\n{'='*70}",
            f"UNCERTAINTY QUANTIFICATION ({self.method.upper()})",
            f"{'='*70}",
            f"Samples: {self.n_samples}",
            "",
            "Confidence Intervals (90%):",
            "-" * 70,
            f"{'Channel':<20} {'p05':>10} {'p50':>10} {'p95':>10} {'Range':>10}",
            "-" * 70,
        ]
        
        for channel in sorted(self.confidence_intervals.keys()):
            ci = self.confidence_intervals[channel]
            p05 = ci.get('p05', 0)
            p50 = ci.get('p50', 0)
            p95 = ci.get('p95', 0)
            range_pct = (p95 - p05) / p50 * 100 if p50 > 0 else 0
            
            lines.append(
                f"{channel:<20} {p05:>9.2%} {p50:>9.2%} {p95:>9.2%} {range_pct:>9.1f}%"
            )
        
        lines.extend([
            "-" * 70,
            "",
            "Rank Stability (% of samples at rank):",
            "-" * 70,
            f"{'Channel':<20} {'#1':>10} {'#2':>10} {'#3':>10}",
            "-" * 70,
        ])
        
        for channel in sorted(self.rank_stability.keys()):
            rs = self.rank_stability[channel]
            lines.append(
                f"{channel:<20} {rs.get('top1', 0):>9.1%} "
                f"{rs.get('top2', 0):>9.1%} {rs.get('top3', 0):>9.1%}"
            )
        
        lines.extend([
            "-" * 70,
            "",
            "Interpretation:",
            "• Narrow intervals: High confidence in attribution",
            "• Wide intervals (>20%): High uncertainty, collect more data",
            "• Rank stability >70%: Confident in channel ranking",
            "• Rank stability <50%: Ranking is uncertain",
            f"{'='*70}",
        ])
        
        return "\n".join(lines)


class UncertaintyQuantificationEngine:
    """
    Comprehensive uncertainty quantification per whitepaper Section 8.
    """
    
    def __init__(self, base_engine: HybridAttributionEngine):
        self.base_engine = base_engine
        
    def alpha_sweep(
        self,
        alpha_grid: Optional[List[float]] = None
    ) -> SensitivityResult:
        """
        Sweep blend parameter α across [0, 1].
        
        Parameters
        ----------
        alpha_grid : List[float], optional
            Values of α to test. Default: 0.0 to 1.0 in 0.05 increments
        
        Returns
        -------
        SensitivityResult
            Sensitivity analysis results
        """
        if alpha_grid is None:
            alpha_grid = [i * 0.05 for i in range(21)]  # 0.0, 0.05, ..., 1.0
        
        logger.info(f"Running α-sweep with {len(alpha_grid)} values")
        
        channel_sensitivities = defaultdict(dict)
        
        for alpha in alpha_grid:
            # Create config with this alpha
            config = deepcopy(self.base_engine.config)
            config.alpha = alpha
            
            # Create new engine with this config
            engine = HybridAttributionEngine(config)
            engine.fit(
                self.base_engine.paths,
                self.base_engine.conversions,
                self.base_engine.path_values
            )
            
            # Compute attribution
            result = engine.compute_attribution()
            
            # Store shares
            for channel, share in result.hybrid_shares.items():
                channel_sensitivities[channel][alpha] = share
        
        return SensitivityResult(
            parameter_name="alpha",
            parameter_values=alpha_grid,
            channel_sensitivities=dict(channel_sensitivities)
        )
    
    def lambda_sweep(
        self,
        base_weights: Dict[str, float],
        lambda_grid: Optional[List[float]] = None
    ) -> SensitivityResult:
        """
        Sweep psychographic prior strength λ.
        
        Formula: w'(k) = 1 + λ × (w(k) - 1)
        
        Parameters
        ----------
        base_weights : Dict[str, float]
            Base psychographic weights
        lambda_grid : List[float], optional
            Values of λ to test. Default: [0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
        
        Returns
        -------
        SensitivityResult
            Sensitivity analysis results
        """
        if lambda_grid is None:
            lambda_grid = [0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
        
        logger.info(f"Running λ-sweep with {len(lambda_grid)} values")
        
        channel_sensitivities = defaultdict(dict)
        
        for lam in lambda_grid:
            # Transform weights: w'(k) = 1 + λ × (w(k) - 1)
            weighted_weights = {
                k: 1 + lam * (w - 1)
                for k, w in base_weights.items()
            }
            
            # Compute attribution with weighted transitions
            result = self.base_engine.compute_with_psychographics(weighted_weights)
            
            # Store shares
            for channel, share in result.hybrid_shares.items():
                channel_sensitivities[channel][lam] = share
        
        return SensitivityResult(
            parameter_name="lambda",
            parameter_values=lambda_grid,
            channel_sensitivities=dict(channel_sensitivities)
        )
    
    def bootstrap_uq(
        self,
        n_bootstrap: int = 100,
        confidence_level: float = 0.90
    ) -> UncertaintyResult:
        """
        Bootstrap uncertainty quantification by resampling paths.
        
        Procedure:
        1. For b = 1 to B:
           - Resample paths with replacement
           - Rebuild transition matrix
           - Recompute attribution
           - Record hybrid_value
        2. Compute percentiles
        
        Parameters
        ----------
        n_bootstrap : int
            Number of bootstrap samples (default: 100)
        confidence_level : float
            Confidence level for intervals (default: 0.90)
        
        Returns
        -------
        UncertaintyResult
            Bootstrap uncertainty quantification
        """
        logger.info(f"Running bootstrap UQ with {n_bootstrap} samples")
        
        n_paths = len(self.base_engine.paths)
        
        # Store attribution samples
        channel_samples = defaultdict(list)
        rank_samples = defaultdict(lambda: defaultdict(int))
        
        for b in range(n_bootstrap):
            # Resample paths with replacement
            indices = np.random.choice(n_paths, size=n_paths, replace=True)
            
            resampled_paths = [self.base_engine.paths[i] for i in indices]
            resampled_conversions = [self.base_engine.conversions[i] for i in indices]
            resampled_values = [self.base_engine.path_values[i] for i in indices]
            
            # Create new engine with resampled data
            engine = HybridAttributionEngine(self.base_engine.config)
            engine.fit(resampled_paths, resampled_conversions, resampled_values)
            
            try:
                result = engine.compute_attribution()
                
                # Record shares
                for channel, share in result.hybrid_shares.items():
                    channel_samples[channel].append(share)
                
                # Record ranks
                sorted_channels = sorted(
                    result.hybrid_shares.items(),
                    key=lambda x: x[1],
                    reverse=True
                )
                
                for rank, (channel, _) in enumerate(sorted_channels, 1):
                    if rank <= 3:
                        rank_samples[channel][f"top{rank}"] += 1
            
            except Exception as e:
                logger.warning(f"Bootstrap sample {b} failed: {e}")
                continue
        
        # Compute confidence intervals
        confidence_intervals = {}
        alpha = 1 - confidence_level
        lower_pct = alpha / 2 * 100
        upper_pct = (1 - alpha / 2) * 100
        
        for channel, samples in channel_samples.items():
            if samples:
                confidence_intervals[channel] = {
                    'p05': np.percentile(samples, 5),
                    'p25': np.percentile(samples, 25),
                    'p50': np.percentile(samples, 50),
                    'p75': np.percentile(samples, 75),
                    'p95': np.percentile(samples, 95),
                }
        
        # Compute rank stability
        rank_stability = {}
        for channel, rank_counts in rank_samples.items():
            rank_stability[channel] = {
                rank: count / n_bootstrap
                for rank, count in rank_counts.items()
            }
        
        return UncertaintyResult(
            method="bootstrap",
            n_samples=n_bootstrap,
            confidence_intervals=confidence_intervals,
            rank_stability=rank_stability
        )
    
    def dirichlet_uq(
        self,
        n_samples: int = 100,
        dirichlet_prior: float = 0.1,
        confidence_level: float = 0.90
    ) -> UncertaintyResult:
        """
        Dirichlet transition matrix uncertainty quantification.
        
        Bayesian posterior: T[i,·] ~ Dirichlet(α₀ + counts[i,·])
        
        Parameters
        ----------
        n_samples : int
            Number of Dirichlet samples
        dirichlet_prior : float
            Dirichlet concentration parameter (α₀)
        confidence_level : float
            Confidence level for intervals
        
        Returns
        -------
        UncertaintyResult
            Dirichlet uncertainty quantification
        """
        logger.info(f"Running Dirichlet UQ with {n_samples} samples")
        
        # Get base transition counts from Markov engine
        markov_engine = self.base_engine.markov_engine
        if markov_engine is None or markov_engine.T is None:
            raise ValueError("Base engine not fitted")
        
        # Convert transition probabilities back to counts (approximate)
        # This is a simplification - ideally we'd track actual counts
        T = markov_engine.T
        n_states = T.shape[0]
        
        # Estimate total observations per state
        # Assume at least 100 transitions per state
        estimated_counts = T * 100
        
        channel_samples = defaultdict(list)
        rank_samples = defaultdict(lambda: defaultdict(int))
        
        for _ in range(n_samples):
            # Sample new transition matrix from Dirichlet
            T_sample = np.zeros_like(T)
            
            for i in range(n_states):
                # Add prior to counts
                alpha = estimated_counts[i, :] + dirichlet_prior
                
                # Sample from Dirichlet
                T_sample[i, :] = np.random.dirichlet(alpha)
            
            # Build new engine with sampled matrix
            engine = HybridAttributionEngine(self.base_engine.config)
            engine.fit(
                self.base_engine.paths,
                self.base_engine.conversions,
                self.base_engine.path_values
            )
            
            # Replace transition matrix with sample
            engine.markov_engine.T = T_sample
            engine.markov_engine._partition_matrix()
            engine.markov_engine._compute_fundamental_matrix()
            
            try:
                result = engine.compute_attribution()
                
                # Record shares
                for channel, share in result.hybrid_shares.items():
                    channel_samples[channel].append(share)
                
                # Record ranks
                sorted_channels = sorted(
                    result.hybrid_shares.items(),
                    key=lambda x: x[1],
                    reverse=True
                )
                
                for rank, (channel, _) in enumerate(sorted_channels, 1):
                    if rank <= 3:
                        rank_samples[channel][f"top{rank}"] += 1
            
            except Exception as e:
                logger.warning(f"Dirichlet sample failed: {e}")
                continue
        
        # Compute confidence intervals
        confidence_intervals = {}
        for channel, samples in channel_samples.items():
            if samples:
                confidence_intervals[channel] = {
                    'p05': np.percentile(samples, 5),
                    'p25': np.percentile(samples, 25),
                    'p50': np.percentile(samples, 50),
                    'p75': np.percentile(samples, 75),
                    'p95': np.percentile(samples, 95),
                }
        
        # Compute rank stability
        rank_stability = {}
        for channel, rank_counts in rank_samples.items():
            rank_stability[channel] = {
                rank: count / n_samples
                for rank, count in rank_counts.items()
            }
        
        return UncertaintyResult(
            method="dirichlet",
            n_samples=n_samples,
            confidence_intervals=confidence_intervals,
            rank_stability=rank_stability
        )
    
    def compare_uq_methods(
        self,
        n_samples: int = 100
    ) -> Dict[str, Any]:
        """
        Compare Bootstrap vs Dirichlet UQ methods.
        
        Returns comparison table showing which uncertainty dominates.
        """
        bootstrap_result = self.bootstrap_uq(n_bootstrap=n_samples)
        dirichlet_result = self.dirichlet_uq(n_samples=n_samples)
        
        comparison = {}
        
        for channel in bootstrap_result.confidence_intervals:
            if channel in dirichlet_result.confidence_intervals:
                boot_ci = bootstrap_result.confidence_intervals[channel]
                dir_ci = dirichlet_result.confidence_intervals[channel]
                
                boot_range = boot_ci['p95'] - boot_ci['p05']
                dir_range = dir_ci['p95'] - dir_ci['p05']
                
                if boot_range < dir_range:
                    dominant = "path_variation"
                elif boot_range > dir_range:
                    dominant = "transition_uncertainty"
                else:
                    dominant = "balanced"
                
                comparison[channel] = {
                    'bootstrap_range': boot_range,
                    'dirichlet_range': dir_range,
                    'dominant_source': dominant,
                }
        
        return comparison


# Convenience functions

def run_full_uq_analysis(
    engine: HybridAttributionEngine,
    n_bootstrap: int = 100,
    n_dirichlet: int = 100
) -> Dict[str, Any]:
    """
    Run complete uncertainty quantification analysis.
    
    Returns all UQ results in a structured format.
    """
    uq_engine = UncertaintyQuantificationEngine(engine)
    
    results = {
        'alpha_sweep': uq_engine.alpha_sweep(),
        'bootstrap': uq_engine.bootstrap_uq(n_bootstrap),
        'dirichlet': uq_engine.dirichlet_uq(n_dirichlet),
    }
    
    # Compare methods
    results['comparison'] = uq_engine.compare_uq_methods(min(n_bootstrap, n_dirichlet))
    
    return results
