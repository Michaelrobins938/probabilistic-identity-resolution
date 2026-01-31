"""
First-Principles Markov Attribution Engine

Implements the exact mathematical framework from the whitepaper:
- Absorbing Markov chains with Q/R/I partitioning
- Fundamental matrix N = (I-Q)^-1
- Characteristic function v(S) = P(CONVERSION | S)
- Markov removal effects M_i = v(N) - v(N \\ {i})

This is the rigorous mathematical foundation for attribution.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set, Any
from collections import defaultdict
import numpy as np
from itertools import combinations
import logging

logger = logging.getLogger(__name__)


@dataclass
class MarkovState:
    """A state in the Markov chain."""
    name: str
    state_type: str  # 'start', 'channel', 'conversion', 'null'
    index: int


@dataclass
class MarkovAttributionConfig:
    """Configuration for Markov attribution."""
    # State definitions
    start_state: str = "START"
    conversion_state: str = "CONVERSION"
    null_state: str = "NULL"
    
    # Removal policy when calculating v(N \\ {i})
    removal_policy: str = "redirect_to_null"  # or "exclude"
    
    # Numerical tolerances
    tolerance: float = 1e-6
    
    # Maximum channels for exact computation
    max_channels: int = 12


class MarkovAttributionEngine:
    """
    First-principles Markov chain attribution engine.
    
    Implements the mathematical framework:
    1. Build transition matrix T = [Q R; 0 I]
    2. Compute fundamental matrix N = (I-Q)^-1
    3. Compute absorption probabilities B = NR
    4. Characteristic function v(S) = P(CONVERSION | only S active)
    5. Removal effect M_i = v(N) - v(N \\ {i})
    """
    
    def __init__(self, config: Optional[MarkovAttributionConfig] = None):
        self.config = config or MarkovAttributionConfig()
        self.states: List[MarkovState] = []
        self.channels: Set[str] = set()
        self.T: Optional[np.ndarray] = None  # Full transition matrix
        self.Q: Optional[np.ndarray] = None  # Transient-to-transient
        self.R: Optional[np.ndarray] = None  # Transient-to-absorbing
        self.N: Optional[np.ndarray] = None  # Fundamental matrix
        self.B: Optional[np.ndarray] = None  # Absorption probabilities
        
    def build_from_paths(
        self,
        paths: List[List[str]],
        conversions: List[bool],
        path_values: Optional[List[float]] = None
    ) -> 'MarkovAttributionEngine':
        """
        Build Markov chain from customer journey paths.
        
        Parameters
        ----------
        paths : List[List[str]]
            Each path is a sequence of channel touchpoints
        conversions : List[bool]
            Whether each path ended in conversion
        path_values : List[float], optional
            Conversion values for each path
            
        Returns
        -------
        MarkovAttributionEngine
            Self for chaining
        """
        # Extract unique channels
        all_channels = set()
        for path in paths:
            all_channels.update(path)
        
        self.channels = all_channels - {self.config.conversion_state, self.config.null_state}
        
        # Validate channel count
        if len(self.channels) > self.config.max_channels:
            raise ValueError(
                f"Channel count {len(self.channels)} exceeds max {self.config.max_channels}. "
                f"Use Monte Carlo Shapley approximation."
            )
        
        # Build state space
        self._build_state_space()
        
        # Build transition counts
        transition_counts = self._build_transition_counts(paths, conversions)
        
        # Normalize to probabilities
        self._build_transition_matrix(transition_counts)
        
        # Compute fundamental matrix and absorption probabilities
        self._compute_fundamental_matrix()
        
        return self
    
    def _build_state_space(self) -> None:
        """Construct the state space: START, channels, CONVERSION, NULL."""
        self.states = []
        idx = 0
        
        # Start state
        self.states.append(MarkovState(
            name=self.config.start_state,
            state_type='start',
            index=idx
        ))
        idx += 1
        
        # Channel states (transient)
        for channel in sorted(self.channels):
            self.states.append(MarkovState(
                name=channel,
                state_type='channel',
                index=idx
            ))
            idx += 1
        
        # Absorbing states
        self.states.append(MarkovState(
            name=self.config.conversion_state,
            state_type='conversion',
            index=idx
        ))
        idx += 1
        
        self.states.append(MarkovState(
            name=self.config.null_state,
            state_type='null',
            index=idx
        ))
        
        logger.info(f"Built state space with {len(self.states)} states: "
                   f"1 start, {len(self.channels)} channels, 2 absorbing")
    
    def _build_transition_counts(
        self,
        paths: List[List[str]],
        conversions: List[bool]
    ) -> Dict[Tuple[str, str], int]:
        """Count transitions between states."""
        counts = defaultdict(int)
        
        for path, converted in zip(paths, conversions):
            # Add START state at beginning
            full_path = [self.config.start_state] + path
            
            # Add absorbing state at end
            if converted:
                full_path.append(self.config.conversion_state)
            else:
                full_path.append(self.config.null_state)
            
            # Count transitions
            for i in range(len(full_path) - 1):
                from_state = full_path[i]
                to_state = full_path[i + 1]
                counts[(from_state, to_state)] += 1
        
        return dict(counts)
    
    def _build_transition_matrix(self, counts: Dict[Tuple[str, str], int]) -> None:
        """Build row-stochastic transition matrix T."""
        n = len(self.states)
        self.T = np.zeros((n, n))
        
        # Map state names to indices
        state_indices = {s.name: s.index for s in self.states}
        
        # Fill in transition counts
        for (from_state, to_state), count in counts.items():
            i = state_indices[from_state]
            j = state_indices[to_state]
            self.T[i, j] = count
        
        # Normalize rows to make it stochastic
        for i in range(n):
            row_sum = self.T[i, :].sum()
            if row_sum > 0:
                self.T[i, :] /= row_sum
            else:
                # Isolated state - set self-loop probability 1
                self.T[i, i] = 1.0
        
        # Partition into Q, R, 0, I
        self._partition_matrix()
        
        logger.info(f"Built transition matrix: shape {self.T.shape}")
    
    def _partition_matrix(self) -> None:
        """
        Partition T into canonical form:
        T = [Q R; 0 I]
        Where Q = transient-to-transient, R = transient-to-absorbing
        """
        # Identify transient and absorbing states
        transient_indices = [s.index for s in self.states 
                            if s.state_type in ['start', 'channel']]
        absorbing_indices = [s.index for s in self.states 
                            if s.state_type in ['conversion', 'null']]
        
        n_transient = len(transient_indices)
        n_absorbing = len(absorbing_indices)
        
        # Extract Q and R
        self.Q = self.T[np.ix_(transient_indices, transient_indices)]
        self.R = self.T[np.ix_(transient_indices, absorbing_indices)]
        
        logger.info(f"Partitioned matrix: Q={self.Q.shape}, R={self.R.shape}")
    
    def _compute_fundamental_matrix(self) -> None:
        """
        Compute fundamental matrix N = (I - Q)^-1.
        
        N[i,j] = expected number of visits to transient state j 
                 starting from transient state i.
        """
        n = self.Q.shape[0]
        I = np.eye(n)
        
        try:
            # N = (I - Q)^-1
            self.N = np.linalg.inv(I - self.Q)
        except np.linalg.LinAlgError:
            logger.error("Matrix inversion failed - singular matrix")
            # Use pseudo-inverse as fallback
            self.N = np.linalg.pinv(I - self.Q)
        
        # Absorption probabilities B = NR
        self.B = self.N @ self.R
        
        # Map absorbing states
        absorbing_states = [s for s in self.states if s.state_type in ['conversion', 'null']]
        self.conversion_idx = next(i for i, s in enumerate(absorbing_states) 
                                   if s.name == self.config.conversion_state)
        self.null_idx = next(i for i, s in enumerate(absorbing_states) 
                            if s.name == self.config.null_state)
        
        logger.info(f"Computed fundamental matrix N and absorption probabilities B")
    
    def characteristic_function(self, coalition: Set[str]) -> float:
        """
        Compute v(S) = P(CONVERSION | only channels in S active).
        
        This is the value function for the cooperative game.
        Channels not in S are effectively removed.
        """
        if not coalition:
            # Empty coalition - return baseline conversion probability
            return self.B[0, self.conversion_idx] if self.B is not None else 0.0
        
        if self.B is None:
            raise ValueError("Model not fitted. Call build_from_paths() first.")
        
        # Build restricted transition matrix with only coalition channels
        T_restricted = self._build_restricted_transition_matrix(coalition)
        
        # Compute absorption probability with restricted matrix
        return self._compute_absorption_probability(T_restricted)
    
    def _build_restricted_transition_matrix(
        self,
        coalition: Set[str]
    ) -> np.ndarray:
        """
        Build transition matrix where channels not in coalition are removed.
        
        Removal policy: redirect_to_null (transitions go to NULL instead)
        """
        # Start with full matrix
        T_restricted = self.T.copy()
        
        # For each channel not in coalition
        for state in self.states:
            if state.state_type == 'channel' and state.name not in coalition:
                # Get index of this channel
                channel_idx = state.index
                
                # Get index of NULL state
                null_idx = next(s.index for s in self.states 
                               if s.name == self.config.null_state)
                
                if self.config.removal_policy == "redirect_to_null":
                    # Redirect all outgoing transitions to NULL
                    T_restricted[channel_idx, :] = 0
                    T_restricted[channel_idx, null_idx] = 1.0
                else:  # exclude
                    # Remove the state entirely (more complex - skip for now)
                    pass
        
        # Renormalize rows
        for i in range(T_restricted.shape[0]):
            row_sum = T_restricted[i, :].sum()
            if row_sum > 0:
                T_restricted[i, :] /= row_sum
        
        return T_restricted
    
    def _compute_absorption_probability(self, T: np.ndarray) -> float:
        """Compute P(CONVERSION | START) from transition matrix."""
        # Partition new matrix
        n = T.shape[0]
        
        # Identify transient and absorbing indices
        transient_indices = [s.index for s in self.states 
                            if s.state_type in ['start', 'channel']]
        absorbing_indices = [s.index for s in self.states 
                            if s.state_type in ['conversion', 'null']]
        
        # Extract Q and R
        Q_new = T[np.ix_(transient_indices, transient_indices)]
        R_new = T[np.ix_(transient_indices, absorbing_indices)]
        
        # Compute fundamental matrix
        I = np.eye(Q_new.shape[0])
        try:
            N_new = np.linalg.inv(I - Q_new)
        except np.linalg.LinAlgError:
            N_new = np.linalg.pinv(I - Q_new)
        
        # Absorption probabilities
        B_new = N_new @ R_new
        
        # Find conversion column
        conversion_col = next(i for i, s in enumerate([self.states[i] for i in absorbing_indices])
                             if s.name == self.config.conversion_state)
        
        # Return P(CONVERSION | START)
        return B_new[0, conversion_col]
    
    def compute_removal_effects(self) -> Dict[str, float]:
        """
        Compute Markov removal effects: M_i = v(N) - v(N \\ {i}).
        
        Interpretation: How much does conversion probability drop
        if channel i is removed?
        """
        if self.B is None:
            raise ValueError("Model not fitted")
        
        all_channels = self.channels
        v_full = self.characteristic_function(all_channels)
        
        removal_effects = {}
        for channel in all_channels:
            coalition_without = all_channels - {channel}
            v_without = self.characteristic_function(coalition_without)
            
            removal_effects[channel] = v_full - v_without
        
        logger.info(f"Computed removal effects for {len(removal_effects)} channels")
        return removal_effects
    
    def get_conversion_probability(self) -> float:
        """Get P(CONVERSION | START) from full model."""
        if self.B is None:
            raise ValueError("Model not fitted")
        return self.B[0, self.conversion_idx]
    
    def get_state_visits(self, start_from: str = "START") -> Dict[str, float]:
        """
        Get expected number of visits to each transient state
        starting from given state.
        
        Uses fundamental matrix N.
        """
        if self.N is None:
            raise ValueError("Model not fitted")
        
        start_idx = next(i for i, s in enumerate(self.states) 
                        if s.name == start_from and s.state_type in ['start', 'channel'])
        
        # Map to transient index
        transient_states = [s for s in self.states if s.state_type in ['start', 'channel']]
        transient_idx = next(i for i, s in enumerate(transient_states) if s.index == start_idx)
        
        visits = {}
        for i, state in enumerate(transient_states):
            visits[state.name] = self.N[transient_idx, i]
        
        return visits


# Convenience functions

def compute_markov_attribution(
    paths: List[List[str]],
    conversions: List[bool],
    removal_policy: str = "redirect_to_null"
) -> Dict[str, float]:
    """
    Compute Markov removal effect attribution for paths.
    
    Parameters
    ----------
    paths : List[List[str]]
        Customer journey paths
    conversions : List[bool]
        Conversion flags
    removal_policy : str
        How to handle removed channels
    
    Returns
    -------
    Dict[str, float]
        Attribution scores (not normalized)
    """
    config = MarkovAttributionConfig(removal_policy=removal_policy)
    engine = MarkovAttributionEngine(config)
    engine.build_from_paths(paths, conversions)
    
    return engine.compute_removal_effects()
