"""
Enhanced Drift Detection with Model Snapshots

Provides drift detection with automatic model snapshotting and rollback
capabilities. When drift is detected, the system:
1. Captures pre-drift model state
2. Evaluates new regime against holdout metrics
3. Allows quick rollback if performance degrades
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import numpy as np
import json
import hashlib
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ModelSnapshot:
    """Snapshot of model state for rollback capability."""
    snapshot_id: str
    account_id: str
    timestamp: datetime
    
    # Model parameters
    centroids: np.ndarray
    cluster_counts: Dict[str, int]
    feature_names: List[str]
    
    # Performance metrics at snapshot time
    brier_score: float
    assignment_accuracy: float
    avg_confidence: float
    
    # Drift context
    drift_score: float
    drift_type: str
    
    # Metadata
    total_sessions: int
    version: str = "1.0"
    
    def to_dict(self) -> Dict:
        """Convert snapshot to dictionary for serialization."""
        return {
            "snapshot_id": self.snapshot_id,
            "account_id": self.account_id,
            "timestamp": self.timestamp.isoformat(),
            "centroids": self.centroids.tobytes().hex(),
            "centroids_shape": self.centroids.shape,
            "centroids_dtype": str(self.centroids.dtype),
            "cluster_counts": self.cluster_counts,
            "feature_names": self.feature_names,
            "brier_score": self.brier_score,
            "assignment_accuracy": self.assignment_accuracy,
            "avg_confidence": self.avg_confidence,
            "drift_score": self.drift_score,
            "drift_type": self.drift_type,
            "total_sessions": self.total_sessions,
            "version": self.version
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ModelSnapshot':
        """Create snapshot from dictionary."""
        # Reconstruct centroids
        centroid_bytes = bytes.fromhex(data["centroids"])
        centroids = np.frombuffer(
            centroid_bytes,
            dtype=data["centroids_dtype"]
        ).reshape(data["centroids_shape"])
        
        return cls(
            snapshot_id=data["snapshot_id"],
            account_id=data["account_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            centroids=centroids,
            cluster_counts=data["cluster_counts"],
            feature_names=data["feature_names"],
            brier_score=data["brier_score"],
            assignment_accuracy=data["assignment_accuracy"],
            avg_confidence=data["avg_confidence"],
            drift_score=data["drift_score"],
            drift_type=data["drift_type"],
            total_sessions=data["total_sessions"],
            version=data.get("version", "1.0")
        )


@dataclass
class DriftConfig:
    """Configuration for drift detection."""
    # Thresholds
    drift_threshold: float = 2.0  # Standard deviations
    rollback_threshold: float = 0.15  # Brier score increase to trigger rollback
    
    # Window sizes
    drift_window_size: int = 50
    evaluation_window_size: int = 100
    
    # Snapshot settings
    max_snapshots_per_account: int = 5
    snapshot_retention_days: int = 30
    
    # Evaluation settings
    holdout_ratio: float = 0.1  # 10% of sessions for holdout
    min_holdout_size: int = 20
    
    # Auto-rollback
    enable_auto_rollback: bool = True
    evaluation_period_seconds: int = 3600  # 1 hour evaluation window


class DriftDetectorWithSnapshots:
    """
    Enhanced drift detection with model snapshotting and rollback.
    
    This class extends basic drift detection to:
    1. Capture model snapshots before reclustering
    2. Maintain holdout set for online evaluation
    3. Compare pre/post drift performance
    4. Automatic rollback if new regime is worse
    """
    
    def __init__(self, config: Optional[DriftConfig] = None, snapshot_dir: str = "./snapshots"):
        self.config = config or DriftConfig()
        self.snapshot_dir = Path(snapshot_dir)
        self.snapshot_dir.mkdir(exist_ok=True)
        
        # In-memory storage
        self.snapshots: Dict[str, List[ModelSnapshot]] = {}  # account_id -> snapshots
        self.holdout_sessions: Dict[str, List[Dict]] = {}  # account_id -> holdout sessions
        self.drift_history: Dict[str, List[Dict]] = {}  # account_id -> drift events
        
        # Current evaluation period
        self.evaluation_start: Dict[str, datetime] = {}
        self.post_drift_metrics: Dict[str, Dict] = {}
    
    def detect_drift(
        self,
        account_id: str,
        recent_distances: List[float],
        baseline_distances: List[float]
    ) -> Tuple[bool, float, str]:
        """
        Detect if drift has occurred.
        
        Args:
            account_id: Account identifier
            recent_distances: Distances from recent sessions
            baseline_distances: Baseline distances for comparison
        
        Returns:
            (is_drift, drift_score, drift_type)
        """
        if len(recent_distances) < self.config.drift_window_size:
            return False, 0.0, "insufficient_data"
        
        # Calculate drift score (z-score)
        baseline_mean = np.mean(baseline_distances)
        baseline_std = np.std(baseline_distances)
        
        if baseline_std == 0:
            return False, 0.0, "no_variation"
        
        recent_mean = np.mean(recent_distances[-self.config.drift_window_size:])
        drift_score = abs(recent_mean - baseline_mean) / baseline_std
        
        # Classify drift type
        drift_type = self._classify_drift(
            recent_distances, baseline_distances, drift_score
        )
        
        is_drift = drift_score > self.config.drift_threshold
        
        if is_drift:
            logger.warning(
                f"Drift detected for {account_id}: "
                f"score={drift_score:.2f}, type={drift_type}"
            )
            
            # Record drift event
            if account_id not in self.drift_history:
                self.drift_history[account_id] = []
            
            self.drift_history[account_id].append({
                "timestamp": datetime.now().isoformat(),
                "drift_score": drift_score,
                "drift_type": drift_type,
                "recent_mean": recent_mean,
                "baseline_mean": baseline_mean,
                "baseline_std": baseline_std
            })
        
        return is_drift, drift_score, drift_type
    
    def _classify_drift(
        self,
        recent: List[float],
        baseline: List[float],
        drift_score: float
    ) -> str:
        """Classify type of drift."""
        recent_mean = np.mean(recent[-self.config.drift_window_size:])
        baseline_mean = np.mean(baseline)
        
        # Check trend
        if len(recent) >= 20:
            first_half = np.mean(recent[-20:-10])
            second_half = np.mean(recent[-10:])
            
            if abs(second_half - first_half) / max(abs(first_half), 0.001) > 0.2:
                if second_half > first_half:
                    return "gradual_increase"
                else:
                    return "gradual_decrease"
        
        # Check sudden change
        if drift_score > 3.0:
            if recent_mean > baseline_mean * 1.5:
                return "sudden_expansion"
            elif recent_mean < baseline_mean * 0.5:
                return "sudden_contraction"
            else:
                return "sudden_shift"
        
        # Check recurring pattern
        if len(recent) >= 100:
            # Look for cyclical pattern (weekly)
            try:
                from scipy import signal
                autocorr = np.correlate(recent[-100:], recent[-100:], mode='full')
                autocorr = autocorr[len(autocorr)//2:]
                peaks = signal.find_peaks(autocorr[:50], height=np.max(autocorr)*0.3)[0]
                if len(peaks) > 0 and peaks[0] >= 7:
                    return "recurring_cyclical"
            except ImportError:
                pass
        
        return "general_drift"
    
    def create_snapshot(
        self,
        account_id: str,
        centroids: np.ndarray,
        cluster_counts: Dict[str, int],
        feature_names: List[str],
        current_metrics: Dict[str, float],
        drift_score: float,
        drift_type: str
    ) -> ModelSnapshot:
        """
        Create model snapshot before reclustering.
        
        Args:
            account_id: Account identifier
            centroids: Current cluster centroids
            cluster_counts: Session counts per cluster
            feature_names: Feature names
            current_metrics: Current performance metrics
            drift_score: Drift score that triggered snapshot
            drift_type: Type of drift detected
        
        Returns:
            ModelSnapshot object
        """
        snapshot_id = f"{account_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}_{hashlib.sha256(centroids.tobytes()).hexdigest()[:8]}"
        
        snapshot = ModelSnapshot(
            snapshot_id=snapshot_id,
            account_id=account_id,
            timestamp=datetime.now(),
            centroids=centroids.copy(),
            cluster_counts=cluster_counts.copy(),
            feature_names=feature_names.copy(),
            brier_score=current_metrics.get("brier_score", 0.15),
            assignment_accuracy=current_metrics.get("assignment_accuracy", 0.8),
            avg_confidence=current_metrics.get("avg_confidence", 0.75),
            drift_score=drift_score,
            drift_type=drift_type,
            total_sessions=sum(cluster_counts.values())
        )
        
        # Store in memory
        if account_id not in self.snapshots:
            self.snapshots[account_id] = []
        
        self.snapshots[account_id].append(snapshot)
        
        # Limit snapshots per account
        if len(self.snapshots[account_id]) > self.config.max_snapshots_per_account:
            self.snapshots[account_id].pop(0)
        
        # Persist to disk
        self._persist_snapshot(snapshot)
        
        logger.info(f"Created snapshot {snapshot_id} for {account_id}")
        
        return snapshot
    
    def _persist_snapshot(self, snapshot: ModelSnapshot):
        """Save snapshot to disk."""
        filepath = self.snapshot_dir / f"{snapshot.snapshot_id}.json"
        
        try:
            with open(filepath, 'w') as f:
                json.dump(snapshot.to_dict(), f, indent=2)
        except Exception as e:
            logger.error(f"Failed to persist snapshot: {e}")
    
    def load_snapshot(self, snapshot_id: str) -> Optional[ModelSnapshot]:
        """Load snapshot from disk."""
        filepath = self.snapshot_dir / f"{snapshot_id}.json"
        
        if not filepath.exists():
            return None
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            return ModelSnapshot.from_dict(data)
        except Exception as e:
            logger.error(f"Failed to load snapshot: {e}")
            return None
    
    def get_latest_snapshot(self, account_id: str) -> Optional[ModelSnapshot]:
        """Get most recent snapshot for account."""
        if account_id not in self.snapshots or not self.snapshots[account_id]:
            return None
        
        return self.snapshots[account_id][-1]
    
    def add_holdout_session(
        self,
        account_id: str,
        session_features: np.ndarray,
        true_person_id: str  # Ground truth for evaluation
    ):
        """
        Add session to holdout set for online evaluation.
        
        Args:
            account_id: Account identifier
            session_features: Feature vector
            true_person_id: Known person ID (for synthetic validation)
        """
        if account_id not in self.holdout_sessions:
            self.holdout_sessions[account_id] = []
        
        self.holdout_sessions[account_id].append({
            "features": session_features,
            "true_person_id": true_person_id,
            "timestamp": datetime.now().isoformat()
        })
        
        # Limit holdout size
        max_size = max(
            self.config.min_holdout_size,
            int(self.holdout_sessions[account_id].__len__() * (1 + self.config.holdout_ratio))
        )
        
        if len(self.holdout_sessions[account_id]) > max_size * 2:
            # Trim oldest sessions
            self.holdout_sessions[account_id] = self.holdout_sessions[account_id][-max_size:]
    
    def evaluate_new_regime(
        self,
        account_id: str,
        new_centroids: np.ndarray,
        new_cluster_ids: List[str]
    ) -> Dict[str, float]:
        """
        Evaluate new model against holdout set.
        
        Args:
            account_id: Account identifier
            new_centroids: New cluster centroids
            new_cluster_ids: New cluster IDs
        
        Returns:
            Performance metrics dict
        """
        if account_id not in self.holdout_sessions:
            return {"error": "no_holdout_data"}
        
        holdout = self.holdout_sessions[account_id]
        
        if len(holdout) < self.config.min_holdout_size:
            return {"error": "insufficient_holdout", "size": len(holdout)}
        
        # Evaluate assignments
        correct = 0
        total = 0
        confidences = []
        brier_scores = []
        
        for session in holdout:
            features = session["features"]
            true_person = session["true_person_id"]
            
            # Find closest centroid
            distances = np.linalg.norm(new_centroids - features, axis=1)
            predicted_idx = np.argmin(distances)
            predicted_person = new_cluster_ids[predicted_idx]
            confidence = 1.0 / (1.0 + distances[predicted_idx])
            
            confidences.append(confidence)
            
            # Brier score
            is_correct = 1.0 if predicted_person == true_person else 0.0
            brier = (confidence - is_correct) ** 2
            brier_scores.append(brier)
            
            if predicted_person == true_person:
                correct += 1
            total += 1
        
        metrics = {
            "accuracy": correct / total if total > 0 else 0,
            "avg_confidence": np.mean(confidences) if confidences else 0,
            "brier_score": np.mean(brier_scores) if brier_scores else 0.25,
            "holdout_size": total,
            "evaluation_time": datetime.now().isoformat()
        }
        
        self.post_drift_metrics[account_id] = metrics
        
        return metrics
    
    def should_rollback(self, account_id: str) -> Tuple[bool, str]:
        """
        Determine if rollback to previous snapshot is needed.
        
        Args:
            account_id: Account identifier
        
        Returns:
            (should_rollback, reason)
        """
        if not self.config.enable_auto_rollback:
            return False, "auto_rollback_disabled"
        
        snapshot = self.get_latest_snapshot(account_id)
        if snapshot is None:
            return False, "no_snapshot_available"
        
        metrics = self.post_drift_metrics.get(account_id)
        if metrics is None:
            return False, "no_evaluation_metrics"
        
        if "error" in metrics:
            return False, f"evaluation_error: {metrics['error']}"
        
        # Check if new regime is worse
        brier_degradation = metrics["brier_score"] - snapshot.brier_score
        accuracy_degradation = snapshot.assignment_accuracy - metrics["accuracy"]
        
        if brier_degradation > self.config.rollback_threshold:
            return True, f"brier_degraded_by_{brier_degradation:.3f}"
        
        if accuracy_degradation > 0.1:  # 10% accuracy drop
            return True, f"accuracy_dropped_by_{accuracy_degradation:.3f}"
        
        return False, "new_regime_acceptable"
    
    def get_rollback_snapshot(self, account_id: str) -> Optional[ModelSnapshot]:
        """Get snapshot for rollback."""
        return self.get_latest_snapshot(account_id)
    
    def get_drift_report(self, account_id: str) -> Dict[str, Any]:
        """Generate comprehensive drift report."""
        report = {
            "account_id": account_id,
            "generated_at": datetime.now().isoformat(),
            "drift_history": self.drift_history.get(account_id, []),
            "snapshots": [
                {
                    "id": s.snapshot_id,
                    "timestamp": s.timestamp.isoformat(),
                    "drift_score": s.drift_score,
                    "drift_type": s.drift_type,
                    "brier_score": s.brier_score,
                    "accuracy": s.assignment_accuracy
                }
                for s in self.snapshots.get(account_id, [])
            ],
            "post_drift_metrics": self.post_drift_metrics.get(account_id, {}),
            "holdout_size": len(self.holdout_sessions.get(account_id, []))
        }
        
        return report
    
    def cleanup_old_snapshots(self, account_id: Optional[str] = None):
        """Remove snapshots older than retention period."""
        cutoff = datetime.now() - timedelta(days=self.config.snapshot_retention_days)
        
        if account_id:
            accounts = [account_id]
        else:
            accounts = list(self.snapshots.keys())
        
        for acc_id in accounts:
            # Filter in-memory snapshots
            if acc_id in self.snapshots:
                self.snapshots[acc_id] = [
                    s for s in self.snapshots[acc_id]
                    if s.timestamp > cutoff
                ]
            
            # Clean up disk
            for snapshot_file in self.snapshot_dir.glob(f"{acc_id}_*.json"):
                try:
                    with open(snapshot_file, 'r') as f:
                        data = json.load(f)
                    
                    snapshot_time = datetime.fromisoformat(data["timestamp"])
                    if snapshot_time < cutoff:
                        snapshot_file.unlink()
                        logger.info(f"Deleted old snapshot: {snapshot_file}")
                except Exception as e:
                    logger.warning(f"Failed to cleanup snapshot {snapshot_file}: {e}")


# Convenience function
def get_drift_detector(snapshot_dir: str = "./snapshots") -> DriftDetectorWithSnapshots:
    """Get or create drift detector instance."""
    return DriftDetectorWithSnapshots(snapshot_dir=snapshot_dir)
