"""
Drift Detection and Management System

Monitors behavioral patterns for significant changes that indicate:
1. Person behavior evolution (child grows up, new job schedule)
2. Household composition changes (new baby, teen goes to college)
3. Data quality issues (missing events, tracking problems)
4. Concept drift (seasonal changes, life events)

Triggers automatic re-clustering when drift exceeds threshold.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Set
from datetime import datetime, timedelta
from collections import defaultdict
import numpy as np
import logging
from enum import Enum

logger = logging.getLogger(__name__)


class DriftType(Enum):
    """Types of drift that can be detected."""
    BEHAVIORAL = "behavioral"  # Same person, changed habits
    COMPOSITION = "composition"  # New/removed person
    SEASONAL = "seasonal"  # Predictable time-based changes
    GRADUAL = "gradual"  # Slow evolution over time
    SUDDEN = "sudden"  # Immediate shift (life event)
    MISSING_DATA = "missing_data"  # Tracking issues


@dataclass
class DriftEvent:
    """Represents a detected drift event."""
    drift_id: str
    account_id: str
    drift_type: DriftType
    severity: float  # 0.0 - 1.0
    timestamp: datetime
    
    # Affected entities
    affected_persons: List[str] = field(default_factory=list)
    affected_clusters: List[str] = field(default_factory=list)
    
    # Metrics
    drift_score: float = 0.0
    previous_confidence: float = 0.0
    current_confidence: float = 0.0
    
    # Context
    description: str = ""
    recommended_action: str = ""
    
    # Resolution
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    resolution_action: str = ""


@dataclass
class DriftDetectorConfig:
    """Configuration for drift detection."""
    # Window sizes
    reference_window_days: int = 30  # Baseline period
    detection_window_days: int = 7   # Current period
    
    # Thresholds
    drift_threshold: float = 2.0  # Z-score threshold
    confidence_drop_threshold: float = 0.2  # 20% drop triggers alert
    
    # Check frequency
    check_interval_hours: int = 24  # Check once per day
    
    # Auto-reclustering
    auto_recluster_threshold: float = 0.7  # Severity > 0.7 triggers re-clustering
    min_sessions_for_detection: int = 50
    
    # Seasonal adjustments
    enable_seasonal_detection: bool = True
    seasonal_periods: List[int] = field(default_factory=lambda: [7, 30, 365])  # Week, month, year


class DriftDetector:
    """
    Automated drift detection for identity resolution.
    
    Monitors multiple signals:
    1. Centroid movement (cluster drift)
    2. Assignment confidence degradation
    3. Silhouette score changes
    4. Data volume anomalies
    5. Behavioral pattern shifts
    
    Triggers re-clustering or alerts when drift detected.
    """
    
    def __init__(self, config: Optional[DriftDetectorConfig] = None):
        self.config = config or DriftDetectorConfig()
        self.drift_history: Dict[str, List[DriftEvent]] = defaultdict(list)
        self.baselines: Dict[str, Dict] = {}  # account_id -> baseline stats
        self.last_check: Dict[str, datetime] = {}
        
        logger.info("DriftDetector initialized")
    
    def establish_baseline(
        self,
        account_id: str,
        sessions: List[Any],
        cluster_assignments: Dict[str, str]
    ) -> None:
        """
        Establish baseline behavior for an account.
        
        Parameters
        ----------
        account_id : str
            Account identifier
        sessions : List
            Historical sessions (reference window)
        cluster_assignments : Dict[str, str]
            Session ID -> Cluster ID mapping
        """
        if len(sessions) < self.config.min_sessions_for_detection:
            logger.warning(f"Insufficient sessions for baseline: {len(sessions)}")
            return
        
        # Compute baseline statistics
        baseline = {
            "established_at": datetime.now(),
            "n_sessions": len(sessions),
            "cluster_distribution": self._compute_cluster_distribution(cluster_assignments),
            "temporal_patterns": self._compute_temporal_patterns(sessions),
            "genre_preferences": self._compute_genre_preferences(sessions),
            "avg_confidence": 0.8,  # Default
            "silhouette_score": 0.5,  # Default
        }
        
        self.baselines[account_id] = baseline
        logger.info(f"Baseline established for {account_id}: {baseline['n_sessions']} sessions")
    
    def check_for_drift(
        self,
        account_id: str,
        recent_sessions: List[Any],
        cluster_stats: Dict[str, Any]
    ) -> Optional[DriftEvent]:
        """
        Check for drift in recent data.
        
        Returns DriftEvent if drift detected, None otherwise.
        """
        # Check if enough time has passed since last check
        now = datetime.now()
        if account_id in self.last_check:
            elapsed = now - self.last_check[account_id]
            if elapsed < timedelta(hours=self.config.check_interval_hours):
                return None
        
        self.last_check[account_id] = now
        
        if account_id not in self.baselines:
            logger.warning(f"No baseline for {account_id}, skipping drift check")
            return None
        
        baseline = self.baselines[account_id]
        
        # Check various drift signals
        drift_signals = []
        
        # 1. Centroid drift
        centroid_drift = self._check_centroid_drift(baseline, cluster_stats)
        if centroid_drift:
            drift_signals.append(("centroid", centroid_drift))
        
        # 2. Confidence degradation
        confidence_drift = self._check_confidence_drift(baseline, recent_sessions)
        if confidence_drift:
            drift_signals.append(("confidence", confidence_drift))
        
        # 3. Cluster composition change
        composition_drift = self._check_composition_drift(baseline, cluster_stats)
        if composition_drift:
            drift_signals.append(("composition", composition_drift))
        
        # 4. Temporal pattern shift
        temporal_drift = self._check_temporal_drift(baseline, recent_sessions)
        if temporal_drift:
            drift_signals.append(("temporal", temporal_drift))
        
        # 5. Data volume anomaly
        volume_drift = self._check_volume_drift(baseline, recent_sessions)
        if volume_drift:
            drift_signals.append(("volume", volume_drift))
        
        if not drift_signals:
            return None
        
        # Combine signals into drift event
        max_drift_signal = max(drift_signals, key=lambda x: x[1]["severity"])
        drift_type, drift_info = max_drift_signal
        
        drift_event = DriftEvent(
            drift_id=f"drift_{account_id}_{now.strftime('%Y%m%d_%H%M%S')}",
            account_id=account_id,
            drift_type=drift_info["type"],
            severity=drift_info["severity"],
            timestamp=now,
            affected_clusters=drift_info.get("affected_clusters", []),
            drift_score=drift_info["score"],
            description=drift_info["description"],
            recommended_action=drift_info["recommended_action"]
        )
        
        # Store in history
        self.drift_history[account_id].append(drift_event)
        
        logger.warning(f"Drift detected for {account_id}: {drift_event.drift_type.value} "
                      f"(severity={drift_event.severity:.2f})")
        
        return drift_event
    
    def should_trigger_reclustering(self, drift_event: DriftEvent) -> bool:
        """Determine if drift severity warrants immediate re-clustering."""
        if drift_event.severity >= self.config.auto_recluster_threshold:
            return True
        
        # Also trigger for composition changes
        if drift_event.drift_type == DriftType.COMPOSITION:
            return True
        
        # Trigger for sudden drift
        if drift_event.drift_type == DriftType.SUDDEN and drift_event.severity > 0.5:
            return True
        
        return False
    
    def _check_centroid_drift(
        self,
        baseline: Dict,
        cluster_stats: Dict[str, Any]
    ) -> Optional[Dict]:
        """Check if cluster centroids have moved significantly."""
        if "centroids" not in cluster_stats:
            return None
        
        current_centroids = cluster_stats["centroids"]
        baseline_temporal = baseline.get("temporal_patterns", {})
        
        # Compare temporal centroids (peak hours)
        current_hours = current_centroids.get("peak_hours", [])
        baseline_hours = baseline_temporal.get("peak_hours", [])
        
        if not current_hours or not baseline_hours:
            return None
        
        # Calculate hour shift
        hour_diffs = []
        for c_hour in current_hours[:3]:  # Top 3 hours
            closest_baseline = min(baseline_hours, key=lambda x: abs(x - c_hour))
            # Handle circular time (23 vs 0)
            diff = abs(c_hour - closest_baseline)
            diff = min(diff, 24 - diff)
            hour_diffs.append(diff)
        
        avg_shift = np.mean(hour_diffs)
        
        # Drift if shift > 3 hours
        if avg_shift > 3:
            return {
                "type": DriftType.BEHAVIORAL,
                "severity": min(avg_shift / 6, 1.0),  # Normalize to 0-1
                "score": avg_shift,
                "affected_clusters": list(current_centroids.keys()),
                "description": f"Peak viewing time shifted by {avg_shift:.1f} hours",
                "recommended_action": "Re-cluster with updated temporal features"
            }
        
        return None
    
    def _check_confidence_drift(
        self,
        baseline: Dict,
        recent_sessions: List[Any]
    ) -> Optional[Dict]:
        """Check if assignment confidence has degraded."""
        baseline_confidence = baseline.get("avg_confidence", 0.8)
        
        # Compute current confidence
        if not recent_sessions:
            return None
        
        current_confidences = [
            getattr(s, "assignment_confidence", 0.5)
            for s in recent_sessions
            if hasattr(s, "assignment_confidence")
        ]
        
        if not current_confidences:
            return None
        
        avg_current = np.mean(current_confidences)
        confidence_drop = baseline_confidence - avg_current
        
        if confidence_drop > self.config.confidence_drop_threshold:
            return {
                "type": DriftType.GRADUAL,
                "severity": min(confidence_drop * 2, 1.0),
                "score": confidence_drop,
                "description": f"Assignment confidence dropped {confidence_drop:.1%}",
                "recommended_action": "Review feature weights and re-cluster"
            }
        
        return None
    
    def _check_composition_drift(
        self,
        baseline: Dict,
        cluster_stats: Dict[str, Any]
    ) -> Optional[Dict]:
        """Check if cluster composition has changed (new/removed people)."""
        baseline_dist = baseline.get("cluster_distribution", {})
        current_dist = cluster_stats.get("cluster_distribution", {})
        
        baseline_clusters = set(baseline_dist.keys())
        current_clusters = set(current_dist.keys())
        
        # Check for new clusters
        new_clusters = current_clusters - baseline_clusters
        removed_clusters = baseline_clusters - current_clusters
        
        if new_clusters or removed_clusters:
            severity = 0.7 if new_clusters else 0.5
            
            description_parts = []
            if new_clusters:
                description_parts.append(f"New clusters detected: {new_clusters}")
            if removed_clusters:
                description_parts.append(f"Clusters disappeared: {removed_clusters}")
            
            return {
                "type": DriftType.COMPOSITION,
                "severity": severity,
                "score": len(new_clusters) + len(removed_clusters),
                "affected_clusters": list(new_clusters | removed_clusters),
                "description": "; ".join(description_parts),
                "recommended_action": "Immediate re-clustering required"
            }
        
        # Check for significant distribution changes
        if baseline_dist and current_dist:
            # Compute KL divergence or chi-square
            chi_square = 0
            for cluster in baseline_clusters:
                baseline_pct = baseline_dist.get(cluster, 0)
                current_pct = current_dist.get(cluster, 0)
                if baseline_pct > 0:
                    chi_square += ((current_pct - baseline_pct) ** 2) / baseline_pct
            
            if chi_square > 0.5:  # Threshold
                return {
                    "type": DriftType.BEHAVIORAL,
                    "severity": min(chi_square / 2, 1.0),
                    "score": chi_square,
                    "description": f"Cluster usage distribution changed (χ²={chi_square:.2f})",
                    "recommended_action": "Consider re-clustering"
                }
        
        return None
    
    def _check_temporal_drift(
        self,
        baseline: Dict,
        recent_sessions: List[Any]
    ) -> Optional[Dict]:
        """Check for seasonal or temporal pattern changes."""
        baseline_temporal = baseline.get("temporal_patterns", {})
        
        if not recent_sessions:
            return None
        
        # Compute current temporal patterns
        current_temporal = self._compute_temporal_patterns(recent_sessions)
        
        # Compare weekend vs weekday ratios
        baseline_weekend_ratio = baseline_temporal.get("weekend_ratio", 0.3)
        current_weekend_ratio = current_temporal.get("weekend_ratio", 0.3)
        
        ratio_change = abs(current_weekend_ratio - baseline_weekend_ratio)
        
        if ratio_change > 0.15:  # 15% change
            drift_type = DriftType.SEASONAL if self._is_seasonal_time() else DriftType.BEHAVIORAL
            
            return {
                "type": drift_type,
                "severity": min(ratio_change * 3, 1.0),
                "score": ratio_change,
                "description": f"Weekend viewing ratio changed from {baseline_weekend_ratio:.1%} to {current_weekend_ratio:.1%}",
                "recommended_action": "Update baseline for seasonal adjustment"
            }
        
        return None
    
    def _check_volume_drift(
        self,
        baseline: Dict,
        recent_sessions: List[Any]
    ) -> Optional[Dict]:
        """Check for data volume anomalies (missing data)."""
        baseline_n = baseline.get("n_sessions", 0)
        current_n = len(recent_sessions)
        
        # Expected sessions in detection window (proportional)
        expected_n = baseline_n * (self.config.detection_window_days / self.config.reference_window_days)
        
        if expected_n > 0:
            volume_ratio = current_n / expected_n
            
            if volume_ratio < 0.5:  # Less than 50% of expected
                return {
                    "type": DriftType.MISSING_DATA,
                    "severity": 1.0 - volume_ratio,
                    "score": volume_ratio,
                    "description": f"Data volume dropped to {volume_ratio:.1%} of expected ({current_n}/{expected_n:.0f} sessions)",
                    "recommended_action": "Check tracking and data pipeline"
                }
        
        return None
    
    def _compute_cluster_distribution(self, assignments: Dict[str, str]) -> Dict[str, float]:
        """Compute percentage distribution across clusters."""
        if not assignments:
            return {}
        
        counts = defaultdict(int)
        for session_id, cluster_id in assignments.items():
            counts[cluster_id] += 1
        
        total = sum(counts.values())
        return {k: v / total for k, v in counts.items()}
    
    def _compute_temporal_patterns(self, sessions: List[Any]) -> Dict[str, Any]:
        """Extract temporal patterns from sessions."""
        if not sessions:
            return {}
        
        hours = []
        weekdays = []
        
        for session in sessions:
            if hasattr(session, 'start_time') and session.start_time:
                hours.append(session.start_time.hour)
                weekdays.append(session.start_time.weekday())
        
        if not hours:
            return {}
        
        # Peak hours
        hour_counts = defaultdict(int)
        for h in hours:
            hour_counts[h] += 1
        peak_hours = sorted(hour_counts.keys(), key=lambda x: hour_counts[x], reverse=True)[:3]
        
        # Weekend ratio
        weekend_count = sum(1 for d in weekdays if d >= 5)
        weekend_ratio = weekend_count / len(weekdays) if weekdays else 0.3
        
        return {
            "peak_hours": peak_hours,
            "weekend_ratio": weekend_ratio,
            "avg_hour": np.mean(hours),
            "hour_std": np.std(hours)
        }
    
    def _compute_genre_preferences(self, sessions: List[Any]) -> Dict[str, float]:
        """Compute genre distribution."""
        genre_counts = defaultdict(float)
        
        for session in sessions:
            if hasattr(session, 'genres_watched'):
                for genre, duration in session.genres_watched.items():
                    genre_counts[genre] += duration
        
        total = sum(genre_counts.values())
        if total == 0:
            return {}
        
        return {k: v / total for k, v in genre_counts.items()}
    
    def _is_seasonal_time(self) -> bool:
        """Check if current time is a seasonal transition."""
        now = datetime.now()
        month = now.month
        
        # Summer break (June-August)
        if month in [6, 7, 8]:
            return True
        
        # Winter break (December-January)
        if month in [12, 1]:
            return True
        
        # Spring break (March)
        if month == 3:
            return True
        
        return False
    
    def resolve_drift(
        self,
        drift_id: str,
        resolution_action: str,
        affected_clusters: List[str] = None
    ) -> bool:
        """Mark a drift event as resolved."""
        for account_id, events in self.drift_history.items():
            for event in events:
                if event.drift_id == drift_id:
                    event.resolved = True
                    event.resolved_at = datetime.now()
                    event.resolution_action = resolution_action
                    
                    if affected_clusters:
                        event.affected_clusters = affected_clusters
                    
                    logger.info(f"Drift {drift_id} resolved: {resolution_action}")
                    return True
        
        return False
    
    def get_active_drifts(self, account_id: str) -> List[DriftEvent]:
        """Get unresolved drift events for an account."""
        return [
            event for event in self.drift_history[account_id]
            if not event.resolved
        ]
    
    def get_drift_summary(self, account_id: str) -> Dict[str, Any]:
        """Get summary of drift history for an account."""
        events = self.drift_history[account_id]
        
        if not events:
            return {"status": "stable", "total_drifts": 0}
        
        active = self.get_active_drifts(account_id)
        
        return {
            "status": "stable" if not active else "drifting",
            "total_drifts": len(events),
            "active_drifts": len(active),
            "last_drift": events[-1].timestamp.isoformat() if events else None,
            "drift_types": list(set(e.drift_type.value for e in events)),
            "avg_severity": np.mean([e.severity for e in events]) if events else 0.0,
        }


# Convenience functions

def create_drift_detector(config: Optional[DriftDetectorConfig] = None) -> DriftDetector:
    """Create drift detector with configuration."""
    return DriftDetector(config)


def quick_drift_check(
    sessions: List[Any],
    baseline_sessions: List[Any]
) -> Optional[str]:
    """
    Quick check for drift without full detector setup.
    
    Returns drift description if detected, None otherwise.
    """
    if len(sessions) < 10 or len(baseline_sessions) < 10:
        return None
    
    # Simple temporal comparison
    baseline_hours = [s.start_time.hour for s in baseline_sessions if hasattr(s, 'start_time')]
    current_hours = [s.start_time.hour for s in sessions if hasattr(s, 'start_time')]
    
    if not baseline_hours or not current_hours:
        return None
    
    baseline_mean = np.mean(baseline_hours)
    current_mean = np.mean(current_hours)
    
    # Circular difference
    diff = abs(current_mean - baseline_mean)
    diff = min(diff, 24 - diff)
    
    if diff > 3:
        return f"Viewing time shifted by {diff:.1f} hours"
    
    return None
