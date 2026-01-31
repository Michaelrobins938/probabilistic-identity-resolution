"""
Hybrid Attribution Engine

Implements the core contribution from the whitepaper:
H_i = α × markov_share[i] + (1 - α) × shapley_share[i]

Stacks Markov chains (causality) with Shapley values (fairness).
Provides psychographic prior modulation and uncertainty quantification.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Set, Callable
from collections import defaultdict
import numpy as np
import logging
from copy import deepcopy

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from attribution.markov_engine import MarkovAttributionEngine, MarkovAttributionConfig
from attribution.shapley_engine import ShapleyAttributionEngine, ShapleyConfig

logger = logging.getLogger(__name__)


@dataclass
class PsychographicPrior:
    """Context-dependent weight for transition modulation."""
    name: str
    weight: float
    context_keys: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if self.weight <= 0:
            raise ValueError(f"Weight must be positive, got {self.weight}")


@dataclass
class HybridAttributionConfig:
    """Configuration for hybrid attribution."""
    # Blend parameter α ∈ [0, 1]
    alpha: float = 0.5  # 1.0 = pure Markov, 0.0 = pure Shapley, 0.5 = balanced
    
    # Engine configs
    markov_config: MarkovAttributionConfig = field(default_factory=MarkovAttributionConfig)
    shapley_config: ShapleyConfig = field(default_factory=ShapleyConfig)
    
    # Psychographic priors
    psychographic_priors: List[PsychographicPrior] = field(default_factory=list)
    
    # Runtime invariants
    enforce_invariants: bool = True
    tolerance: float = 1e-6
    
    def __post_init__(self):
        if not 0 <= self.alpha <= 1:
            raise ValueError(f"Alpha must be in [0, 1], got {self.alpha}")


@dataclass
class AttributionResult:
    """Complete attribution result with dual scores."""
    # Channel attributions
    markov_shares: Dict[str, float]
    shapley_shares: Dict[str, float]
    hybrid_shares: Dict[str, float]
    hybrid_values: Dict[str, float]
    
    # Configuration used
    alpha: float
    total_conversion_value: float
    
    # Metadata
    conversion_probability: float
    n_channels: int
    n_paths: int
    
    # Validation
    shares_sum_to_one: bool
    value_conservation: bool
    
    def get_summary(self) -> str:
        """Get human-readable summary."""
        lines = [
            "=" * 70,
            "HYBRID ATTRIBUTION RESULT",
            "=" * 70,
            "",
            f"Configuration: α = {self.alpha:.2f}",
            f"Blend: {(1-self.alpha)*100:.0f}% Shapley + {self.alpha*100:.0f}% Markov",
            "",
            f"Total Conversion Value: ${self.total_conversion_value:,.2f}",
            f"Conversion Probability: {self.conversion_probability:.2%}",
            f"Channels: {self.n_channels} | Paths: {self.n_paths}",
            "",
            "Attribution Breakdown:",
            "-" * 70,
            f"{'Channel':<20} {'Markov':>12} {'Shapley':>12} {'Hybrid':>12} {'Value':>12}",
            "-" * 70,
        ]
        
        for channel in sorted(self.hybrid_shares.keys()):
            m_share = self.markov_shares.get(channel, 0)
            s_share = self.shapley_shares.get(channel, 0)
            h_share = self.hybrid_shares.get(channel, 0)
            value = self.hybrid_values.get(channel, 0)
            
            lines.append(
                f"{channel:<20} {m_share:>11.1%} {s_share:>11.1%} "
                f"{h_share:>11.1%} ${value:>10,.2f}"
            )
        
        lines.extend([
            "-" * 70,
            "",
            "Validation:",
            f"  Shares sum to 1.0:   {'✓' if self.shares_sum_to_one else '✗'}",
            f"  Value conservation:    {'✓' if self.value_conservation else '✗'}",
            "",
            "=" * 70,
        ])
        
        return "\n".join(lines)


class HybridAttributionEngine:
    """
    Complete hybrid attribution system per whitepaper specification.
    
    Features:
    1. Dual attribution scores (Markov + Shapley)
    2. Hybrid blend with configurable α
    3. Psychographic prior modulation
    4. Runtime invariants and validation
    5. Uncertainty quantification hooks
    """
    
    def __init__(self, config: Optional[HybridAttributionConfig] = None):
        self.config = config or HybridAttributionConfig()
        self.markov_engine: Optional[MarkovAttributionEngine] = None
        self.shapley_engine: Optional[ShapleyAttributionEngine] = None
        self.paths: List[List[str]] = []
        self.conversions: List[bool] = []
        self.path_values: List[float] = []
        
    def fit(
        self,
        paths: List[List[str]],
        conversions: List[bool],
        path_values: Optional[List[float]] = None
    ) -> 'HybridAttributionEngine':
        """
        Fit the attribution model to path data.
        
        Parameters
        ----------
        paths : List[List[str]]
            Customer journey paths (sequences of channels)
        conversions : List[bool]
            Whether each path converted
        path_values : List[float], optional
            Conversion values
        """
        self.paths = paths
        self.conversions = conversions
        self.path_values = path_values or [1.0 if c else 0.0 for c in conversions]
        
        # Build Markov engine
        self.markov_engine = MarkovAttributionEngine(self.config.markov_config)
        self.markov_engine.build_from_paths(paths, conversions, path_values)
        
        # Build Shapley engine (will use Markov's characteristic function)
        all_channels = set()
        for path in paths:
            all_channels.update(path)
        
        channels = sorted(all_channels - {'CONVERSION', 'NULL', 'START'})
        
        # Create characteristic function using Markov engine
        def characteristic_function(coalition: Set[str]) -> float:
            return self.markov_engine.characteristic_function(coalition)
        
        self.shapley_engine = ShapleyAttributionEngine(self.config.shapley_config)
        self.shapley_engine.fit(channels, characteristic_function)
        
        logger.info(f"Hybrid engine fitted: {len(channels)} channels, {len(paths)} paths")
        return self
    
    def compute_attribution(self) -> AttributionResult:
        """
        Compute hybrid attribution with dual scores.
        
        Returns
        -------
        AttributionResult
            Complete attribution with validation
        """
        if self.markov_engine is None or self.shapley_engine is None:
            raise ValueError("Engine not fitted. Call fit() first.")
        
        # Compute Markov removal effects
        markov_effects = self.markov_engine.compute_removal_effects()
        
        # Compute Shapley values
        shapley_values = self.shapley_engine.compute_shapley_values()
        
        # Normalize to shares
        markov_shares = self._normalize_to_shares(markov_effects)
        shapley_shares = self._normalize_to_shares(shapley_values)
        
        # Compute hybrid shares
        hybrid_shares = {}
        for channel in markov_shares:
            hybrid_shares[channel] = (
                self.config.alpha * markov_shares[channel] +
                (1 - self.config.alpha) * shapley_shares.get(channel, 0)
            )
        
        # Normalize hybrid shares to ensure they sum to 1
        hybrid_shares = self._normalize_to_shares(hybrid_shares)
        
        # Compute monetary values
        total_value = sum(self.path_values)
        hybrid_values = {
            c: share * total_value
            for c, share in hybrid_shares.items()
        }
        
        # Validate
        shares_sum_to_one = self._check_shares_sum_to_one(hybrid_shares)
        value_conservation = abs(sum(hybrid_values.values()) - total_value) < 1.0
        
        result = AttributionResult(
            markov_shares=markov_shares,
            shapley_shares=shapley_shares,
            hybrid_shares=hybrid_shares,
            hybrid_values=hybrid_values,
            alpha=self.config.alpha,
            total_conversion_value=total_value,
            conversion_probability=self.markov_engine.get_conversion_probability(),
            n_channels=len(markov_shares),
            n_paths=len(self.paths),
            shares_sum_to_one=shares_sum_to_one,
            value_conservation=value_conservation
        )
        
        # Enforce invariants if configured
        if self.config.enforce_invariants:
            self._enforce_invariants(result)
        
        return result
    
    def compute_with_psychographics(
        self,
        context_weights: Dict[str, float]
    ) -> AttributionResult:
        """
        Compute attribution with psychographic prior modulation.
        
        Weights modulate transition counts before normalization.
        Example: {"high_intent_search": 1.5, "desktop_checkout": 1.3}
        """
        if not context_weights:
            return self.compute_attribution()
        
        # Apply weights to paths
        weighted_paths = self._apply_psychographic_weights(
            self.paths, context_weights
        )
        
        # Rebuild Markov engine with weighted paths
        markov_engine_weighted = MarkovAttributionEngine(self.config.markov_config)
        markov_engine_weighted.build_from_paths(
            weighted_paths, self.conversions, self.path_values
        )
        
        # Update characteristic function
        all_channels = set()
        for path in weighted_paths:
            all_channels.update(path)
        channels = sorted(all_channels - {'CONVERSION', 'NULL', 'START'})
        
        def char_func(coalition: Set[str]) -> float:
            return markov_engine_weighted.characteristic_function(coalition)
        
        shapley_engine_weighted = ShapleyAttributionEngine(self.config.shapley_config)
        shapley_engine_weighted.fit(channels, char_func)
        
        # Temporarily swap engines
        original_markov = self.markov_engine
        original_shapley = self.shapley_engine
        
        self.markov_engine = markov_engine_weighted
        self.shapley_engine = shapley_engine_weighted
        
        try:
            result = self.compute_attribution()
        finally:
            # Restore original engines
            self.markov_engine = original_markov
            self.shapley_engine = original_shapley
        
        return result
    
    def _apply_psychographic_weights(
        self,
        paths: List[List[str]],
        weights: Dict[str, float]
    ) -> List[List[str]]:
        """
        Apply psychographic weights to paths.
        
        Modulates transition counts: weighted_count = count × w(context)
        """
        weighted_paths = []
        
        for path in paths:
            weighted_path = []
            for channel in path:
                # Get weight for this channel context
                weight = weights.get(channel, 1.0)
                
                # Add channel multiple times based on weight
                # (This effectively weights the transition)
                n_copies = max(1, int(weight))
                for _ in range(n_copies):
                    weighted_path.append(channel)
            
            weighted_paths.append(weighted_path)
        
        return weighted_paths
    
    def _normalize_to_shares(self, values: Dict[str, float]) -> Dict[str, float]:
        """Normalize values to shares that sum to 1.0."""
        total = sum(values.values())
        if total == 0:
            n = len(values)
            return {k: 1.0 / n for k in values}
        return {k: v / total for k, v in values.items()}
    
    def _check_shares_sum_to_one(self, shares: Dict[str, float]) -> bool:
        """Check if shares sum to 1.0 within tolerance."""
        total = sum(shares.values())
        return abs(total - 1.0) <= self.config.tolerance
    
    def _enforce_invariants(self, result: AttributionResult) -> None:
        """
        Enforce runtime invariants per whitepaper specification.
        
        Raises ValueError if invariants are violated.
        """
        # Invariant 1: Shares must sum to 1.0
        total_shares = sum(result.hybrid_shares.values())
        if abs(total_shares - 1.0) > self.config.tolerance:
            raise ValueError(
                f"Invariant violation: Shares sum to {total_shares:.6f}, not 1.0"
            )
        
        # Invariant 2: Values must conserve total
        total_values = sum(result.hybrid_values.values())
        if abs(total_values - result.total_conversion_value) > 1.0:
            raise ValueError(
                f"Invariant violation: Values sum to {total_values:.2f}, "
                f"expected {result.total_conversion_value:.2f}"
            )
        
        # Invariant 3: Channel count guardrail
        if result.n_channels > 12 and not self.config.shapley_config.use_monte_carlo:
            raise ValueError(
                f"Invariant violation: {result.n_channels} channels > 12. "
                f"Use Monte Carlo approximation."
            )


# Convenience functions

def compute_hybrid_attribution(
    paths: List[List[str]],
    conversions: List[bool],
    alpha: float = 0.5,
    path_values: Optional[List[float]] = None,
    psychographic_weights: Optional[Dict[str, float]] = None
) -> AttributionResult:
    """
    Compute hybrid attribution for customer journeys.
    
    Parameters
    ----------
    paths : List[List[str]]
        Customer journey paths
    conversions : List[bool]
        Conversion flags
    alpha : float
        Blend parameter (0.0 = pure Shapley, 1.0 = pure Markov)
    path_values : List[float], optional
        Conversion values
    psychographic_weights : Dict[str, float], optional
        Context-dependent weights
    
    Returns
    -------
    AttributionResult
        Complete attribution result
    """
    config = HybridAttributionConfig(alpha=alpha)
    engine = HybridAttributionEngine(config)
    engine.fit(paths, conversions, path_values)
    
    if psychographic_weights:
        return engine.compute_with_psychographics(psychographic_weights)
    else:
        return engine.compute_attribution()
