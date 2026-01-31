"""
Cross-Device Linker

Links devices belonging to the same person across platforms.

Solves the cross-device attribution problem:
- Same person uses mobile, desktop, tablet, TV
- Need to unify these into a single identity for attribution
- Uses probabilistic matching based on multiple signals
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set, Any
from datetime import datetime, timedelta
import math
import hashlib

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from models.device_profile import DeviceProfile
from models.streaming_event import Session
from core.identity_graph import IdentityGraph


@dataclass
class LinkingConfig:
    """Configuration for cross-device linking."""
    # Probability thresholds
    link_threshold: float = 0.7        # Minimum probability to create link
    strong_link_threshold: float = 0.9  # Threshold for high-confidence links

    # Signal weights
    ip_weight: float = 3.0             # Same network (strong signal)
    temporal_weight: float = 2.0       # Events close in time
    behavioral_weight: float = 1.5     # Similar content preferences
    login_weight: float = 5.0          # Deterministic login match

    # Temporal settings
    temporal_window_minutes: int = 30  # Window for temporal correlation

    # Pruning
    max_links_per_device: int = 5      # Limit links per device
    min_sessions_for_linking: int = 5  # Minimum sessions to consider linking


@dataclass
class DeviceLink:
    """A link between two devices."""
    device_a: str
    device_b: str
    probability: float
    signals: Dict[str, float]  # Signal name -> contribution
    created_at: datetime = field(default_factory=datetime.now)

    @property
    def link_id(self) -> str:
        """Deterministic ID based on device pair using secure hashing."""
        sorted_devices = sorted([self.device_a, self.device_b])
        return hashlib.sha256(f"{sorted_devices[0]}_{sorted_devices[1]}".encode()).hexdigest()[:16]


class CrossDeviceLinker:
    """
    Links devices belonging to the same person.

    Uses multiple signals to compute P(same_person | signals):
    1. IP Address: Same network suggests same household/person
    2. Temporal Correlation: Events close in time on different devices
    3. Behavioral Similarity: Similar content preferences
    4. Deterministic Links: Logged-in user IDs

    The output is a graph where edges represent same-person probability.
    """

    def __init__(self, config: Optional[LinkingConfig] = None):
        """
        Initialize the cross-device linker.

        Parameters
        ----------
        config : LinkingConfig, optional
            Configuration for linking parameters
        """
        self.config = config or LinkingConfig()

    def link_devices(
        self,
        device_profiles: List[DeviceProfile],
        sessions: Optional[List[Session]] = None
    ) -> List[DeviceLink]:
        """
        Find links between devices.

        Parameters
        ----------
        device_profiles : List[DeviceProfile]
            Device profiles to link
        sessions : List[Session], optional
            Sessions for temporal analysis

        Returns
        -------
        List[DeviceLink]
            Identified device links with probabilities
        """
        links = []

        # Build session index for temporal analysis
        session_index = self._build_session_index(sessions) if sessions else {}

        # Compare all pairs
        n = len(device_profiles)
        for i in range(n):
            for j in range(i + 1, n):
                device_a = device_profiles[i]
                device_b = device_profiles[j]

                # Skip if insufficient data
                if (device_a.total_sessions < self.config.min_sessions_for_linking or
                    device_b.total_sessions < self.config.min_sessions_for_linking):
                    continue

                # Compute link probability
                probability, signals = self.compute_link_probability(
                    device_a, device_b, session_index
                )

                if probability >= self.config.link_threshold:
                    links.append(DeviceLink(
                        device_a=device_a.fingerprint_id,
                        device_b=device_b.fingerprint_id,
                        probability=probability,
                        signals=signals
                    ))

        # Prune excess links
        links = self._prune_links(links)

        return links

    def compute_link_probability(
        self,
        device_a: DeviceProfile,
        device_b: DeviceProfile,
        session_index: Optional[Dict[str, List[Session]]] = None
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute probability that two devices belong to the same person.

        Uses Bayesian combination of multiple signals.

        Parameters
        ----------
        device_a : DeviceProfile
            First device
        device_b : DeviceProfile
            Second device
        session_index : Dict, optional
            Sessions indexed by device fingerprint

        Returns
        -------
        Tuple[float, Dict[str, float]]
            (probability, signal_contributions)
        """
        signals = {}
        weights = []
        scores = []

        # 1. IP Address Match
        ip_score = self._compute_ip_similarity(device_a, device_b)
        if ip_score > 0:
            signals["ip_match"] = ip_score
            scores.append(ip_score)
            weights.append(self.config.ip_weight)

        # 2. Temporal Correlation
        if session_index:
            temporal_score = self._compute_temporal_correlation(
                device_a.fingerprint_id,
                device_b.fingerprint_id,
                session_index
            )
            if temporal_score > 0:
                signals["temporal_correlation"] = temporal_score
                scores.append(temporal_score)
                weights.append(self.config.temporal_weight)

        # 3. Behavioral Similarity
        behavioral_score = self._compute_behavioral_similarity(device_a, device_b)
        if behavioral_score > 0:
            signals["behavioral_similarity"] = behavioral_score
            scores.append(behavioral_score)
            weights.append(self.config.behavioral_weight)

        # 4. Deterministic Login Match
        login_score = self._compute_login_match(device_a, device_b)
        if login_score > 0:
            signals["login_match"] = login_score
            scores.append(login_score)
            weights.append(self.config.login_weight)

        if not scores:
            return 0.0, signals

        # Weighted combination
        total_weight = sum(weights)
        weighted_sum = sum(s * w for s, w in zip(scores, weights))
        probability = weighted_sum / total_weight

        # Apply sigmoid to get smoother probability
        probability = self._sigmoid(probability * 2 - 1)  # Center around 0.5

        return probability, signals

    def _compute_ip_similarity(
        self,
        device_a: DeviceProfile,
        device_b: DeviceProfile
    ) -> float:
        """
        Compute IP-based similarity score.

        Same IP strongly suggests same person or household.
        """
        if not device_a.ip_hashes or not device_b.ip_hashes:
            return 0.0

        # Count overlapping IPs
        overlap = len(set(device_a.ip_hashes) & set(device_b.ip_hashes))
        total = max(len(device_a.ip_hashes), len(device_b.ip_hashes))

        if total == 0:
            return 0.0

        return overlap / total

    def _compute_temporal_correlation(
        self,
        device_a_id: str,
        device_b_id: str,
        session_index: Dict[str, List[Session]]
    ) -> float:
        """
        Compute temporal correlation between devices.

        High correlation = events on one device shortly before/after the other.
        This suggests same person switching devices.
        """
        sessions_a = session_index.get(device_a_id, [])
        sessions_b = session_index.get(device_b_id, [])

        if not sessions_a or not sessions_b:
            return 0.0

        window = timedelta(minutes=self.config.temporal_window_minutes)
        correlation_count = 0
        total_comparisons = 0

        for sa in sessions_a:
            for sb in sessions_b:
                if not hasattr(sa.start_time, 'timestamp') or not hasattr(sb.start_time, 'timestamp'):
                    continue

                # Check if sessions are within window
                time_diff = abs((sa.start_time - sb.start_time).total_seconds())

                if time_diff <= window.total_seconds():
                    # Weight by closeness
                    closeness = 1.0 - (time_diff / window.total_seconds())
                    correlation_count += closeness

                total_comparisons += 1

        if total_comparisons == 0:
            return 0.0

        # Normalize by expected correlation under random hypothesis
        # (If devices are independent, we'd expect few close events)
        expected = min(len(sessions_a), len(sessions_b)) * 0.1
        observed = correlation_count

        if expected == 0:
            return 0.0

        # Ratio of observed to expected, capped at 1.0
        score = min(1.0, observed / expected / 10)

        return score

    def _compute_behavioral_similarity(
        self,
        device_a: DeviceProfile,
        device_b: DeviceProfile
    ) -> float:
        """
        Compute behavioral similarity between devices.

        Similar content preferences suggest same person.
        """
        scores = []

        # Genre similarity
        if device_a.genre_affinities and device_b.genre_affinities:
            genre_sim = self._cosine_similarity(
                device_a.genre_affinities,
                device_b.genre_affinities
            )
            scores.append(genre_sim)

        # Hour distribution similarity
        if device_a.hour_distribution and device_b.hour_distribution:
            hour_sim = self._cosine_similarity(
                device_a.hour_distribution,
                device_b.hour_distribution
            )
            scores.append(hour_sim)

        # Session duration similarity
        if device_a.avg_session_duration > 0 and device_b.avg_session_duration > 0:
            duration_ratio = min(device_a.avg_session_duration, device_b.avg_session_duration) / \
                           max(device_a.avg_session_duration, device_b.avg_session_duration)
            scores.append(duration_ratio)

        if not scores:
            return 0.0

        return sum(scores) / len(scores)

    def _compute_login_match(
        self,
        device_a: DeviceProfile,
        device_b: DeviceProfile
    ) -> float:
        """
        Check for deterministic login matches.

        If both devices have the same account_id, they're likely same person.
        """
        if not device_a.account_ids or not device_b.account_ids:
            return 0.0

        overlap = len(set(device_a.account_ids) & set(device_b.account_ids))
        total = max(len(device_a.account_ids), len(device_b.account_ids))

        if total == 0:
            return 0.0

        return overlap / total

    def _cosine_similarity(self, d1: Dict, d2: Dict) -> float:
        """Compute cosine similarity between two dictionaries."""
        keys = set(d1.keys()) | set(d2.keys())
        if not keys:
            return 0.0

        dot = sum(d1.get(k, 0) * d2.get(k, 0) for k in keys)
        norm1 = math.sqrt(sum(v ** 2 for v in d1.values()))
        norm2 = math.sqrt(sum(v ** 2 for v in d2.values()))

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot / (norm1 * norm2)

    def _sigmoid(self, x: float) -> float:
        """Sigmoid function for smooth probability."""
        return 1.0 / (1.0 + math.exp(-x))

    def _build_session_index(
        self,
        sessions: List[Session]
    ) -> Dict[str, List[Session]]:
        """Index sessions by device fingerprint."""
        index = {}
        for session in sessions:
            fp = session.device_fingerprint
            if fp not in index:
                index[fp] = []
            index[fp].append(session)
        return index

    def _prune_links(self, links: List[DeviceLink]) -> List[DeviceLink]:
        """
        Prune links to keep only the strongest for each device.

        Prevents a device from having too many links.
        """
        # Count links per device
        device_link_counts: Dict[str, int] = {}
        device_links: Dict[str, List[DeviceLink]] = {}

        for link in links:
            for device in [link.device_a, link.device_b]:
                if device not in device_links:
                    device_links[device] = []
                device_links[device].append(link)

        # Keep only top k links per device
        kept_links = set()

        for device, d_links in device_links.items():
            # Sort by probability
            d_links.sort(key=lambda l: l.probability, reverse=True)

            # Keep top k
            for link in d_links[:self.config.max_links_per_device]:
                kept_links.add(link.link_id)

        return [l for l in links if l.link_id in kept_links]

    def cluster_devices_to_persons(
        self,
        links: List[DeviceLink],
        threshold: Optional[float] = None
    ) -> List[Set[str]]:
        """
        Cluster devices into persons based on links.

        Uses connected components with threshold-based edge pruning.

        Parameters
        ----------
        links : List[DeviceLink]
            Device links
        threshold : float, optional
            Minimum link probability to include

        Returns
        -------
        List[Set[str]]
            Clusters of device fingerprints (each cluster = one person)
        """
        threshold = threshold or self.config.link_threshold

        # Build adjacency list
        adjacency: Dict[str, Set[str]] = {}

        for link in links:
            if link.probability >= threshold:
                if link.device_a not in adjacency:
                    adjacency[link.device_a] = set()
                if link.device_b not in adjacency:
                    adjacency[link.device_b] = set()

                adjacency[link.device_a].add(link.device_b)
                adjacency[link.device_b].add(link.device_a)

        # Find connected components (BFS)
        visited = set()
        clusters = []

        for device in adjacency:
            if device in visited:
                continue

            cluster = set()
            queue = [device]

            while queue:
                current = queue.pop(0)
                if current in visited:
                    continue

                visited.add(current)
                cluster.add(current)

                for neighbor in adjacency.get(current, []):
                    if neighbor not in visited:
                        queue.append(neighbor)

            if cluster:
                clusters.append(cluster)

        return clusters

    def add_links_to_graph(
        self,
        graph: IdentityGraph,
        links: List[DeviceLink]
    ) -> None:
        """
        Add device links to an identity graph.

        Parameters
        ----------
        graph : IdentityGraph
            The graph to add links to
        links : List[DeviceLink]
            Device links to add
        """
        for link in links:
            graph.add_edge(
                source_id=link.device_a,
                target_id=link.device_b,
                edge_type=IdentityGraph.EDGE_SAME_PERSON,
                weight=link.probability,
                confidence=link.probability,
                signals=link.signals
            )
