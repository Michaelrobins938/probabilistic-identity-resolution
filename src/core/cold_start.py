"""
Cold Start Strategy

Handles new users and accounts with insufficient data (<10 sessions).
Provides graceful degradation from person-level to account-level attribution
until enough data is collected for reliable clustering.

Strategies:
1. Heuristic Persona Assignment (rules-based fallback)
2. Account-Level Attribution (treat as single person)
3. Confidence Threshold Gating (low confidence = no assignment)
4. Probabilistic Priors (Bayesian approach with population averages)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class ColdStartConfig:
    """Configuration for cold start handling."""
    # Thresholds
    min_sessions_for_clustering: int = 10
    min_sessions_for_confident_assignment: int = 20
    min_session_duration_total: float = 300  # 5 hours total
    
    # Strategy
    default_strategy: str = "heuristic"  # "heuristic", "account_level", "probabilistic_prior"
    
    # Confidence levels
    low_confidence_threshold: float = 0.5
    medium_confidence_threshold: float = 0.7
    high_confidence_threshold: float = 0.9
    
    # Heuristic weights (for heuristic strategy)
    time_weight: float = 2.0
    device_weight: float = 1.5
    genre_weight: float = 1.0


@dataclass
class ColdStartAssignment:
    """Assignment result during cold start period."""
    person_id: str
    confidence: float
    strategy_used: str
    is_placeholder: bool
    
    # Heuristic info
    detected_persona: Optional[str] = None
    match_score: float = 0.0
    
    # Recommended action
    needs_more_data: bool = True
    recommended_sessions: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "person_id": self.person_id,
            "confidence": self.confidence,
            "strategy": self.strategy_used,
            "is_placeholder": self.is_placeholder,
            "persona": self.detected_persona,
            "needs_more_data": self.needs_more_data,
        }


class ColdStartHandler:
    """
    Handles attribution for new users with limited history.
    
    Progression:
    1. Sessions 1-9: Cold start mode (heuristic/account-level)
    2. Sessions 10-19: Initial clustering (low confidence)
    3. Sessions 20+: Full clustering (high confidence)
    
    Prevents poor assignments that could poison the identity graph.
    """
    
    def __init__(self, config: Optional[ColdStartConfig] = None):
        self.config = config or ColdStartConfig()
        
        # Population priors for Bayesian approach
        self.population_priors = self._load_population_priors()
        
        logger.info(f"ColdStartHandler initialized (min_sessions={self.config.min_sessions_for_clustering})")
    
    def _load_population_priors(self) -> Dict[str, Any]:
        """Load population-level behavioral priors."""
        return {
            "primary_adult": {
                "peak_hours": [20, 21, 22, 23, 19],
                "genres": ["Drama", "Documentary", "Thriller"],
                "devices": ["tv", "desktop"],
                "prior_probability": 0.4
            },
            "secondary_adult": {
                "peak_hours": [18, 19, 20, 21, 14],
                "genres": ["Comedy", "Reality", "Romance"],
                "devices": ["tv", "tablet"],
                "prior_probability": 0.25
            },
            "teen": {
                "peak_hours": [21, 22, 23, 0, 1],
                "genres": ["Action", "SciFi", "Comedy"],
                "devices": ["mobile", "tablet"],
                "prior_probability": 0.20
            },
            "child": {
                "peak_hours": [15, 16, 17, 10, 11],
                "genres": ["Animation", "Kids", "Family"],
                "devices": ["tablet", "tv"],
                "prior_probability": 0.15
            }
        }
    
    def handle_cold_start(
        self,
        account_id: str,
        sessions: List[Any],
        session_features: np.ndarray,
        existing_persons: Optional[List[str]] = None
    ) -> ColdStartAssignment:
        """
        Handle assignment for account with insufficient data.
        
        Parameters
        ----------
        account_id : str
            Account identifier
        sessions : List
            Sessions for this account (n < 10)
        session_features : np.ndarray
            Feature vector for current session
        existing_persons : List[str], optional
            If household has some established persons
        
        Returns
        -------
        ColdStartAssignment
            Assignment with confidence and strategy info
        """
        n_sessions = len(sessions)
        
        # Check if we have enough data for clustering
        if n_sessions >= self.config.min_sessions_for_clustering:
            # Not actually cold start
            return ColdStartAssignment(
                person_id=f"{account_id}_person_0",
                confidence=0.5,
                strategy_used="ready_for_clustering",
                is_placeholder=False,
                needs_more_data=False
            )
        
        # Select strategy based on data availability
        if n_sessions < 3:
            # Very limited data - use account-level
            return self._account_level_strategy(account_id, n_sessions)
        
        elif n_sessions < 7:
            # Some data - use probabilistic priors
            return self._probabilistic_prior_strategy(
                account_id, sessions, session_features
            )
        
        else:
            # Approaching threshold - use heuristic
            return self._heuristic_strategy(
                account_id, sessions, session_features, existing_persons
            )
    
    def _account_level_strategy(self, account_id: str, n_sessions: int) -> ColdStartAssignment:
        """
        Treat entire account as single person.
        
        Safest approach when data is extremely limited.
        """
        logger.debug(f"Account-level attribution for {account_id} ({n_sessions} sessions)")
        
        return ColdStartAssignment(
            person_id=f"{account_id}_unified",
            confidence=0.3,  # Low confidence
            strategy_used="account_level",
            is_placeholder=True,
            detected_persona=None,
            needs_more_data=True,
            recommended_sessions=self.config.min_sessions_for_clustering - n_sessions
        )
    
    def _probabilistic_prior_strategy(
        self,
        account_id: str,
        sessions: List[Any],
        session_features: np.ndarray
    ) -> ColdStartAssignment:
        """
        Use Bayesian approach with population priors.
        
        P(persona | session) ∝ P(session | persona) × P(persona)
        """
        # Extract session characteristics
        session_hour = self._extract_hour(sessions[-1]) if sessions else 20
        session_device = self._extract_device(sessions[-1]) if sessions else "tv"
        session_genres = self._extract_genres(sessions[-1]) if sessions else []
        
        # Compute likelihood for each persona
        scores = {}
        for persona, prior in self.population_priors.items():
            score = prior["prior_probability"]
            
            # Hour match
            if session_hour in prior["peak_hours"]:
                score *= 2.0
            
            # Device match
            if session_device in prior["devices"]:
                score *= 1.5
            
            # Genre match
            if any(g in prior["genres"] for g in session_genres):
                score *= 1.5
            
            scores[persona] = score
        
        # Normalize
        total_score = sum(scores.values())
        if total_score > 0:
            scores = {k: v / total_score for k, v in scores.items()}
        
        # Select best match
        best_persona = max(scores.keys(), key=lambda k: scores[k])
        confidence = scores[best_persona]
        
        return ColdStartAssignment(
            person_id=f"{account_id}_{best_persona}_probabilistic",
            confidence=confidence,
            strategy_used="probabilistic_prior",
            is_placeholder=True,
            detected_persona=best_persona,
            match_score=confidence,
            needs_more_data=True,
            recommended_sessions=self.config.min_sessions_for_clustering - len(sessions)
        )
    
    def _heuristic_strategy(
        self,
        account_id: str,
        sessions: List[Any],
        session_features: np.ndarray,
        existing_persons: Optional[List[str]] = None
    ) -> ColdStartAssignment:
        """
        Use rule-based heuristics to assign session.
        
        Looks for simple patterns:
        - Time of day → Persona
        - Device type → Persona
        - Content genre → Persona
        """
        # Aggregate session statistics
        all_hours = []
        all_devices = []
        all_genres = []
        
        for session in sessions:
            if hasattr(session, 'start_time') and session.start_time:
                all_hours.append(session.start_time.hour)
            if hasattr(session, 'device_type'):
                all_devices.append(session.device_type)
            if hasattr(session, 'genres_watched'):
                all_genres.extend(session.genres_watched.keys())
        
        # Compute mode (most common)
        from collections import Counter
        
        mode_hour = Counter(all_hours).most_common(1)[0][0] if all_hours else 20
        mode_device = Counter(all_devices).most_common(1)[0][0] if all_devices else "tv"
        mode_genre = Counter(all_genres).most_common(1)[0][0] if all_genres else "Drama"
        
        # Apply rules
        detected_persona = self._apply_heuristic_rules(mode_hour, mode_device, mode_genre)
        
        # Calculate confidence based on consistency
        hour_consistency = len(set(all_hours)) / len(all_hours) if all_hours else 1.0
        confidence = 0.5 + (1 - hour_consistency) * 0.3  # More consistent = higher confidence
        
        return ColdStartAssignment(
            person_id=f"{account_id}_{detected_persona}_heuristic",
            confidence=confidence,
            strategy_used="heuristic",
            is_placeholder=True,
            detected_persona=detected_persona,
            match_score=confidence,
            needs_more_data=True,
            recommended_sessions=self.config.min_sessions_for_clustering - len(sessions)
        )
    
    def _apply_heuristic_rules(self, hour: int, device: str, genre: str) -> str:
        """Apply simple heuristic rules to detect persona."""
        # Child rules (highest priority)
        if hour in range(10, 18) and genre in ["Kids", "Animation", "Family"]:
            return "child"
        
        # Teen rules
        if hour in range(21, 24) or hour in range(0, 2):
            if device in ["mobile", "tablet"]:
                return "teen"
            if genre in ["Action", "SciFi", "Fantasy"]:
                return "teen"
        
        # Adult rules
        if device == "tv" and hour in range(19, 24):
            return "primary_adult"
        
        if genre in ["Documentary", "News", "Classics"]:
            return "secondary_adult"
        
        # Default
        return "primary_adult"
    
    def should_use_cold_start(
        self,
        account_id: str,
        sessions: List[Any],
        current_confidence: float = 0.0
    ) -> bool:
        """
        Determine if cold start mode should be used.
        
        Returns True if:
        - Less than min_sessions_for_clustering
        - OR confidence is very low (< 0.5)
        """
        n_sessions = len(sessions)
        
        if n_sessions < self.config.min_sessions_for_clustering:
            return True
        
        if current_confidence < self.config.low_confidence_threshold:
            return True
        
        return False
    
    def get_progress_toward_clustering(
        self,
        sessions: List[Any]
    ) -> Dict[str, Any]:
        """
        Get progress information for cold start accounts.
        
        Shows how close account is to full clustering.
        """
        n_sessions = len(sessions)
        min_required = self.config.min_sessions_for_clustering
        
        progress_pct = min(100, (n_sessions / min_required) * 100)
        
        # Estimate remaining sessions needed
        remaining = max(0, min_required - n_sessions)
        
        # Estimate time to full clustering (assume 2 sessions/day)
        days_remaining = remaining / 2
        
        return {
            "current_sessions": n_sessions,
            "required_sessions": min_required,
            "progress_percentage": progress_pct,
            "sessions_remaining": remaining,
            "estimated_days_remaining": days_remaining,
            "can_cluster_now": n_sessions >= min_required,
            "recommended_action": (
                "Ready for clustering" if n_sessions >= min_required
                else f"Need {remaining} more sessions (~{days_remaining:.0f} days)"
            )
        }
    
    def merge_cold_start_assignments(
        self,
        assignments: List[ColdStartAssignment],
        clustering_result: Dict[str, Any]
    ) -> Dict[str, str]:
        """
        Merge cold start placeholder assignments with actual clustering.
        
        When account graduates from cold start, map heuristic assignments
        to real cluster IDs.
        """
        # Create mapping from heuristic IDs to real cluster IDs
        mapping = {}
        
        # Try to match based on persona type
        for assignment in assignments:
            if assignment.detected_persona:
                # Find cluster with matching persona
                for cluster_id, cluster_info in clustering_result.items():
                    if cluster_info.get("persona") == assignment.detected_persona:
                        mapping[assignment.person_id] = cluster_id
                        break
        
        return mapping
    
    def _extract_hour(self, session: Any) -> int:
        """Extract hour from session."""
        if hasattr(session, 'start_time') and session.start_time:
            return session.start_time.hour
        if hasattr(session, 'hour_of_day'):
            return session.hour_of_day
        return 20
    
    def _extract_device(self, session: Any) -> str:
        """Extract device type from session."""
        if hasattr(session, 'device_type'):
            return session.device_type
        if hasattr(session, 'device_fingerprint'):
            # Infer from fingerprint
            fp = session.device_fingerprint.lower()
            if 'tv' in fp or 'roku' in fp or 'apple_tv' in fp:
                return "tv"
            elif 'mobile' in fp or 'iphone' in fp or 'android' in fp:
                return "mobile"
        return "tv"
    
    def _extract_genres(self, session: Any) -> List[str]:
        """Extract genres from session."""
        if hasattr(session, 'genres_watched'):
            return list(session.genres_watched.keys())
        if hasattr(session, 'content_genre'):
            return [session.content_genre] if session.content_genre else []
        return []


# Convenience functions

def handle_new_user(
    account_id: str,
    sessions: List[Any],
    session_features: np.ndarray
) -> ColdStartAssignment:
    """
    Quick function to handle new user assignment.
    
    Example:
    ```python
    assignment = handle_new_user(
        account_id="account_123",
        sessions=[session1, session2],  # Only 2 sessions
        session_features=features
    )
    # Returns: person_id="account_123_primary_adult_heuristic", confidence=0.6
    ```
    """
    handler = ColdStartHandler()
    return handler.handle_cold_start(account_id, sessions, session_features)


def get_cold_start_status(account_id: str, sessions: List[Any]) -> Dict[str, Any]:
    """Get cold start status for an account."""
    handler = ColdStartHandler()
    return handler.get_progress_toward_clustering(sessions)
