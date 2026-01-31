"""
Incremental (Online) K-Means Clustering

Real-time clustering that updates centroids as new sessions arrive.
Eliminates batch processing latency - target <100ms per update.

Algorithm: Mini-batch K-Means with adaptive learning rate
- Processes sessions one-at-a-time or in micro-batches
- Updates centroids using learning rate α = 1 / (n + 1)
- Maintains running statistics (counts, means, variances)

Reference: Sculley (2010) - Web-scale k-means clustering
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import numpy as np
import logging
from datetime import datetime, timedelta
import hashlib
import json

logger = logging.getLogger(__name__)


@dataclass
class IncrementalClusterConfig:
    """Configuration for incremental clustering."""
    max_k: int = 6  # Maximum clusters (people)
    min_k: int = 1
    feature_dim: int = 20
    
    # Learning rate schedule
    learning_rate_base: float = 0.1
    learning_rate_decay: float = 0.99
    min_learning_rate: float = 0.001
    
    # Drift detection
    drift_threshold: float = 2.0  # Standard deviations
    drift_window_size: int = 50  # Sessions to monitor
    
    # Cluster management
    max_age_hours: int = 168  # 1 week - deprecate old clusters
    min_sessions_to_cluster: int = 10
    
    # Performance
    batch_size: int = 10  # Micro-batch size for efficiency
    

@dataclass
class ClusterState:
    """State of a single cluster (person) in incremental learning."""
    cluster_id: str
    centroid: np.ndarray
    count: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    
    # Running statistics
    sum_features: np.ndarray = field(default=None)
    sum_squared_features: np.ndarray = field(default=None)
    variance: np.ndarray = field(default=None)
    
    # Session tracking
    recent_sessions: List[Dict] = field(default_factory=list)
    session_history_size: int = 100
    
    # Drift detection
    recent_distances: List[float] = field(default_factory=list)
    drift_score: float = 0.0
    
    def __post_init__(self):
        if self.sum_features is None:
            self.sum_features = np.zeros_like(self.centroid)
        if self.sum_squared_features is None:
            self.sum_squared_features = np.zeros_like(self.centroid)
        if self.variance is None:
            self.variance = np.ones_like(self.centroid) * 0.1
    
    def update(self, features: np.ndarray, learning_rate: float) -> float:
        """
        Update cluster centroid with new session features.
        
        Parameters
        ----------
        features : np.ndarray
            Feature vector of new session
        learning_rate : float
            Step size for update (α)
        
        Returns
        -------
        float
            Distance before update
        """
        distance = np.linalg.norm(features - self.centroid)
        
        # Update centroid: θ_new = θ_old + α(x - θ_old)
        self.centroid = self.centroid + learning_rate * (features - self.centroid)
        
        # Update statistics
        self.count += 1
        self.sum_features += features
        self.sum_squared_features += features ** 2
        
        # Update variance (Welford's online algorithm)
        if self.count > 1:
            mean = self.sum_features / self.count
            variance = (self.sum_squared_features / self.count) - (mean ** 2)
            self.variance = np.maximum(variance, 0.001)  # Prevent zero variance
        
        # Track for drift detection
        self.recent_distances.append(distance)
        if len(self.recent_distances) > self.session_history_size:
            self.recent_distances.pop(0)
        
        # Calculate drift score (z-score of recent distances)
        if len(self.recent_distances) >= 10:
            recent_mean = np.mean(self.recent_distances[-10:])
            recent_std = np.std(self.recent_distances[-10:])
            if recent_std > 0:
                self.drift_score = abs(distance - recent_mean) / recent_std
        
        self.last_updated = datetime.now()
        
        return distance
    
    def is_deprecated(self, max_age_hours: int = 168) -> bool:
        """Check if cluster is deprecated (no activity)."""
        age = datetime.now() - self.last_updated
        return age > timedelta(hours=max_age_hours)
    
    def get_confidence_radius(self) -> float:
        """Get confidence radius based on variance."""
        return np.mean(np.sqrt(self.variance))


class IncrementalKMeans:
    """
    Online K-Means clustering for real-time identity resolution.
    
    Processes sessions incrementally, updating centroids immediately.
    Eliminates batch processing delays - <100ms latency per session.
    
    Features:
    - Micro-batch processing for efficiency
    - Automatic k adjustment (split/merge clusters)
    - Drift detection and re-clustering triggers
    - Deprecated cluster cleanup
    """
    
    def __init__(self, config: Optional[IncrementalClusterConfig] = None):
        self.config = config or IncrementalClusterConfig()
        self.clusters: Dict[str, ClusterState] = {}
        self.account_id: Optional[str] = None
        
        # Pending sessions for micro-batching
        self.pending_sessions: List[Tuple[str, np.ndarray]] = []
        
        # Global statistics
        self.total_sessions = 0
        self.created_at = datetime.now()
        self.last_reclustering = datetime.now()
        
        logger.info(f"IncrementalKMeans initialized (max_k={self.config.max_k})")
    
    def initialize(self, account_id: str, initial_sessions: List[Tuple[str, np.ndarray]]) -> None:
        """
        Initialize with initial batch of sessions.
        
        Parameters
        ----------
        account_id : str
            Account identifier
        initial_sessions : List[Tuple[str, np.ndarray]]
            List of (session_id, features) for initialization
        """
        self.account_id = account_id
        
        if len(initial_sessions) < self.config.min_sessions_to_cluster:
            logger.warning(f"Too few sessions ({len(initial_sessions)}) for clustering")
            # Create single cluster with mean of all sessions
            if initial_sessions:
                features = np.array([s[1] for s in initial_sessions])
                centroid = np.mean(features, axis=0)
                self._create_cluster(centroid)
            return
        
        # Run mini-batch K-means to get initial centroids
        from sklearn.cluster import MiniBatchKMeans
        
        features = np.array([s[1] for s in initial_sessions])
        
        # Determine optimal k using silhouette (simplified)
        best_k = 1
        best_score = -1
        
        for k in range(1, min(self.config.max_k + 1, len(features))):
            if k == 1:
                score = 0.0  # Single cluster baseline
            else:
                kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, n_init=3)
                labels = kmeans.fit_predict(features)
                
                # Compute silhouette-like score (simplified)
                score = self._compute_cohesion(features, labels)
            
            if score > best_score:
                best_score = score
                best_k = k
        
        # Create clusters
        kmeans = MiniBatchKMeans(n_clusters=best_k, random_state=42, n_init=3)
        kmeans.fit(features)
        
        for i, centroid in enumerate(kmeans.cluster_centers_):
            cluster_id = f"{account_id}_person_{i}"
            self.clusters[cluster_id] = ClusterState(
                cluster_id=cluster_id,
                centroid=centroid,
                count=0
            )
        
        # Assign initial sessions
        for session_id, features in initial_sessions:
            self.assign_session(session_id, features, immediate=True)
        
        self.total_sessions = len(initial_sessions)
        logger.info(f"Initialized {len(self.clusters)} clusters for {account_id}")
    
    def assign_session(
        self,
        session_id: str,
        features: np.ndarray,
        immediate: bool = False
    ) -> Tuple[str, float]:
        """
        Assign session to cluster incrementally.
        
        Parameters
        ----------
        session_id : str
            Session identifier
        features : np.ndarray
            Feature vector
        immediate : bool
            If True, update immediately; if False, buffer for micro-batch
        
        Returns
        -------
        Tuple[str, float]
            (cluster_id, confidence)
        """
        if not immediate:
            # Buffer for micro-batch processing
            self.pending_sessions.append((session_id, features))
            
            if len(self.pending_sessions) >= self.config.batch_size:
                return self._process_batch()
            else:
                # Return best current assignment without updating
                return self._find_best_cluster(features)
        
        # Immediate update
        return self._update_and_assign(session_id, features)
    
    def _update_and_assign(self, session_id: str, features: np.ndarray) -> Tuple[str, float]:
        """Update clusters and assign session."""
        if not self.clusters:
            # Create first cluster
            cluster_id = self._create_cluster(features)
            return cluster_id, 1.0
        
        # Find closest cluster
        best_cluster = None
        best_distance = float('inf')
        
        for cluster in self.clusters.values():
            distance = np.linalg.norm(features - cluster.centroid)
            if distance < best_distance:
                best_distance = distance
                best_cluster = cluster
        
        # Calculate confidence
        confidence_radius = best_cluster.get_confidence_radius()
        confidence = np.exp(-best_distance / (confidence_radius + 0.1))
        
        # Adaptive learning rate
        lr = max(
            self.config.min_learning_rate,
            self.config.learning_rate_base / (1 + 0.01 * best_cluster.count)
        )
        
        # Update cluster
        best_cluster.update(features, lr)
        self.total_sessions += 1
        
        # Check for drift
        if best_cluster.drift_score > self.config.drift_threshold:
            logger.warning(f"Drift detected in cluster {best_cluster.cluster_id}: "
                         f"score={best_cluster.drift_score:.2f}")
        
        return best_cluster.cluster_id, confidence
    
    def _process_batch(self) -> Tuple[str, float]:
        """Process pending sessions in micro-batch."""
        if not self.pending_sessions:
            return None, 0.0
        
        # Process all pending sessions
        results = []
        for session_id, features in self.pending_sessions:
            result = self._update_and_assign(session_id, features)
            results.append(result)
        
        self.pending_sessions = []
        
        # Return last assignment
        return results[-1] if results else (None, 0.0)
    
    def _find_best_cluster(self, features: np.ndarray) -> Tuple[str, float]:
        """Find best cluster without updating."""
        if not self.clusters:
            return None, 0.0
        
        best_cluster = None
        best_distance = float('inf')
        
        for cluster in self.clusters.values():
            distance = np.linalg.norm(features - cluster.centroid)
            if distance < best_distance:
                best_distance = distance
                best_cluster = cluster
        
        confidence_radius = best_cluster.get_confidence_radius() if best_cluster else 1.0
        confidence = np.exp(-best_distance / (confidence_radius + 0.1))
        
        return best_cluster.cluster_id if best_cluster else None, confidence
    
    def _create_cluster(self, centroid: np.ndarray) -> str:
        """Create new cluster."""
        cluster_id = f"{self.account_id}_person_{len(self.clusters)}"
        
        # Check max clusters
        if len(self.clusters) >= self.config.max_k:
            # Merge with closest instead of creating new
            return self._merge_with_closest(centroid)
        
        self.clusters[cluster_id] = ClusterState(
            cluster_id=cluster_id,
            centroid=centroid.copy(),
            count=0
        )
        
        logger.info(f"Created new cluster {cluster_id}")
        return cluster_id
    
    def _merge_with_closest(self, features: np.ndarray) -> str:
        """Merge features with closest existing cluster."""
        best_cluster = None
        best_distance = float('inf')
        
        for cluster in self.clusters.values():
            distance = np.linalg.norm(features - cluster.centroid)
            if distance < best_distance:
                best_distance = distance
                best_cluster = cluster
        
        if best_cluster:
            # Update with high learning rate to absorb new pattern
            best_cluster.update(features, learning_rate=0.5)
            return best_cluster.cluster_id
        
        return list(self.clusters.keys())[0]
    
    def get_cluster_probabilities(self, features: np.ndarray) -> Dict[str, float]:
        """Get soft assignment probabilities across all clusters."""
        if not self.clusters:
            return {}
        
        distances = {}
        for cluster_id, cluster in self.clusters.items():
            distance = np.linalg.norm(features - cluster.centroid)
            confidence_radius = cluster.get_confidence_radius()
            # Use Gaussian-like probability
            distances[cluster_id] = np.exp(-distance ** 2 / (2 * confidence_radius ** 2))
        
        # Softmax
        total = sum(distances.values())
        if total == 0:
            return {cid: 1.0 / len(self.clusters) for cid in self.clusters}
        
        return {cid: prob / total for cid, prob in distances.items()}
    
    def detect_drift(self) -> List[str]:
        """Detect clusters showing significant drift."""
        drifting_clusters = []
        
        for cluster_id, cluster in self.clusters.items():
            if cluster.drift_score > self.config.drift_threshold:
                drifting_clusters.append(cluster_id)
        
        return drifting_clusters
    
    def cleanup_deprecated(self) -> int:
        """Remove deprecated clusters (no activity)."""
        deprecated = [
            cid for cid, cluster in self.clusters.items()
            if cluster.is_deprecated(self.config.max_age_hours)
        ]
        
        for cid in deprecated:
            del self.clusters[cid]
            logger.info(f"Removed deprecated cluster {cid}")
        
        return len(deprecated)
    
    def recluster_if_needed(self) -> bool:
        """Trigger full re-clustering if drift detected."""
        drifting = self.detect_drift()
        
        if len(drifting) >= max(1, len(self.clusters) * 0.3):  # 30% of clusters drifting
            logger.info(f"Triggering re-clustering: {len(drifting)} clusters drifting")
            # Signal to calling system that re-clustering is needed
            return True
        
        return False
    
    def _compute_cohesion(self, features: np.ndarray, labels: np.ndarray) -> float:
        """Compute cluster cohesion (simplified silhouette)."""
        n_samples = len(features)
        if n_samples == 0:
            return 0.0
        
        # Within-cluster sum of squares
        wcss = 0.0
        for label in np.unique(labels):
            mask = labels == label
            if np.sum(mask) > 1:
                cluster_points = features[mask]
                centroid = np.mean(cluster_points, axis=0)
                wcss += np.sum((cluster_points - centroid) ** 2)
        
        # Normalize by number of clusters
        k = len(np.unique(labels))
        return -wcss / (n_samples * k)  # Higher is better
    
    def get_stats(self) -> Dict[str, Any]:
        """Get clustering statistics."""
        return {
            "n_clusters": len(self.clusters),
            "total_sessions": self.total_sessions,
            "avg_cluster_size": self.total_sessions / max(1, len(self.clusters)),
            "deprecated_clusters": sum(1 for c in self.clusters.values() if c.is_deprecated()),
            "drifting_clusters": len(self.detect_drift()),
            "last_reclustering": self.last_reclustering.isoformat(),
        }
