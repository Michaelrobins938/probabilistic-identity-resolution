"""
Household Inference Engine

Solves the Netflix Co-Viewing Problem:
- One account, multiple people
- Need to infer WHO is watching and attribute accordingly
- Uses device + time + content patterns to cluster sessions into persons

This is the core differentiator for streaming attribution.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from datetime import datetime
import math
import hashlib

# Import models
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from models.streaming_event import Session, StreamingEvent, group_events_into_sessions
from models.household_profile import HouseholdProfile, PersonProfile


@dataclass
class ClusteringConfig:
    """Configuration for household member clustering."""
    min_sessions_per_person: int = 10  # Minimum sessions to identify a person
    max_household_size: int = 6        # Maximum people per account
    silhouette_threshold: float = 0.3  # Minimum silhouette score to accept clustering
    time_weight: float = 1.5           # Weight for time-of-day features
    device_weight: float = 1.2         # Weight for device features
    genre_weight: float = 1.0          # Weight for content preferences
    random_seed: int = 42


class HouseholdInferenceEngine:
    """
    Infers household composition and assigns sessions to persons.

    The Netflix Co-Viewing Problem:
    - Multiple people share one streaming account
    - We need to attribute viewing (and conversions) to individuals
    - Use behavioral patterns to infer WHO is watching each session

    Approach:
    1. Extract features from each session (time, device, content)
    2. Cluster sessions using K-means with silhouette-based k selection
    3. Each cluster represents a distinct "person"
    4. For each new session, compute probability distribution over persons
    """

    def __init__(self, config: Optional[ClusteringConfig] = None):
        """
        Initialize the household inference engine.

        Parameters
        ----------
        config : ClusteringConfig, optional
            Configuration for clustering parameters
        """
        self.config = config or ClusteringConfig()
        self._rng = np.random.default_rng(self.config.random_seed)

        # Cache for trained models per account
        self._account_models: Dict[str, Dict] = {}

    def analyze_household(
        self,
        sessions: List[Session],
        account_id: Optional[str] = None
    ) -> HouseholdProfile:
        """
        Analyze sessions to infer household composition.

        This is the main entry point for household inference.

        Parameters
        ----------
        sessions : List[Session]
            All sessions from a single account
        account_id : str, optional
            Account identifier

        Returns
        -------
        HouseholdProfile
            Complete household analysis with member profiles
        """
        if not sessions:
            return self._empty_household(account_id or "unknown")

        # Get or infer account ID
        account_id = account_id or sessions[0].account_id

        # Step 1: Extract features from sessions
        features, feature_names = self._extract_session_features(sessions)

        # Step 2: Determine optimal number of clusters (persons)
        optimal_k, silhouette = self._find_optimal_k(features)

        # Step 3: Cluster sessions
        labels, centroids = self._cluster_sessions(features, optimal_k)

        # Step 4: Build person profiles from clusters
        members = self._build_person_profiles(sessions, labels, optimal_k, account_id)

        # Step 5: Create household profile
        household = self._build_household_profile(
            account_id=account_id,
            members=members,
            sessions=sessions,
            n_clusters=optimal_k,
            silhouette=silhouette
        )

        # Cache the model for future session assignment
        self._account_models[account_id] = {
            "centroids": centroids,
            "feature_names": feature_names,
            "members": [m.person_id for m in members]
        }

        return household

    def assign_session_to_person(
        self,
        session: Session,
        account_id: str
    ) -> Dict[str, float]:
        """
        Assign a single session to household members with probabilities.

        Uses the trained model to compute P(person | session_features).

        Parameters
        ----------
        session : Session
            Session to assign
        account_id : str
            Account this session belongs to

        Returns
        -------
        Dict[str, float]
            person_id -> probability mapping
        """
        if account_id not in self._account_models:
            # No trained model - return uniform distribution
            return {}

        model = self._account_models[account_id]
        centroids = model["centroids"]
        members = model["members"]

        # Extract features for this session
        features = self._session_to_feature_vector(session)

        # Compute distance to each centroid
        distances = []
        for centroid in centroids:
            dist = np.linalg.norm(features - centroid)
            distances.append(dist)

        # Convert distances to probabilities (softmax with temperature)
        distances = np.array(distances)
        temperature = 0.5  # Lower = more confident
        exp_neg_dist = np.exp(-distances / temperature)
        probabilities = exp_neg_dist / exp_neg_dist.sum()

        # Map to person IDs
        result = {members[i]: float(probabilities[i]) for i in range(len(members))}

        # Also update the session object
        session.person_probabilities = result
        session.assigned_person_id = max(result, key=result.get)
        session.assignment_confidence = max(result.values())

        return result

    def infer_household_size(self, sessions: List[Session]) -> Tuple[int, float]:
        """
        Estimate the number of distinct people using this account.

        Uses silhouette analysis to determine optimal cluster count.

        Parameters
        ----------
        sessions : List[Session]
            Sessions from the account

        Returns
        -------
        Tuple[int, float]
            (estimated_size, confidence)
        """
        if len(sessions) < self.config.min_sessions_per_person:
            return 1, 0.5  # Default to single person with low confidence

        features, _ = self._extract_session_features(sessions)
        optimal_k, silhouette = self._find_optimal_k(features)

        # Confidence based on silhouette score
        confidence = min(1.0, max(0.0, silhouette / 0.5))

        return optimal_k, confidence

    def _extract_session_features(
        self,
        sessions: List[Session]
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Extract feature matrix from sessions for clustering.

        Features include:
        - Time of day (cyclical encoding)
        - Day of week (cyclical encoding)
        - Device type (one-hot)
        - Content genre preferences
        - Session duration

        Returns
        -------
        Tuple[np.ndarray, List[str]]
            (feature_matrix, feature_names)
        """
        feature_dicts = [self._session_to_feature_dict(s) for s in sessions]

        # Get all feature names
        all_features = set()
        for fd in feature_dicts:
            all_features.update(fd.keys())
        feature_names = sorted(all_features)

        # Build matrix
        n_sessions = len(sessions)
        n_features = len(feature_names)
        X = np.zeros((n_sessions, n_features))

        for i, fd in enumerate(feature_dicts):
            for j, name in enumerate(feature_names):
                X[i, j] = fd.get(name, 0.0)

        # Apply feature weights
        for j, name in enumerate(feature_names):
            if name.startswith("hour_") or name.startswith("day_"):
                X[:, j] *= self.config.time_weight
            elif name.startswith("device_"):
                X[:, j] *= self.config.device_weight
            elif name.startswith("genre_"):
                X[:, j] *= self.config.genre_weight

        # Normalize features
        X = self._normalize_features(X)

        return X, feature_names

    def _session_to_feature_dict(self, session: Session) -> Dict[str, float]:
        """Convert a session to a feature dictionary."""
        features = {}

        # Time features (cyclical encoding)
        if hasattr(session.start_time, 'hour'):
            hour = session.start_time.hour
            features["hour_sin"] = math.sin(2 * math.pi * hour / 24)
            features["hour_cos"] = math.cos(2 * math.pi * hour / 24)

            day = session.start_time.weekday()
            features["day_sin"] = math.sin(2 * math.pi * day / 7)
            features["day_cos"] = math.cos(2 * math.pi * day / 7)

            # Weekend indicator
            features["is_weekend"] = 1.0 if day >= 5 else 0.0
        else:
            features["hour_sin"] = 0.0
            features["hour_cos"] = 1.0
            features["day_sin"] = 0.0
            features["day_cos"] = 1.0
            features["is_weekend"] = 0.0

        # Device type one-hot
        device_types = ['tv', 'desktop', 'mobile', 'tablet']
        for dt in device_types:
            features[f"device_{dt}"] = 1.0 if session.device_type == dt else 0.0

        # Genre preferences (normalized)
        genres = ['Drama', 'Comedy', 'Action', 'Documentary', 'Kids',
                  'Animation', 'Reality', 'Thriller', 'Romance', 'SciFi']
        total_genre_time = sum(session.genres_watched.values()) or 1.0

        for genre in genres:
            features[f"genre_{genre.lower()}"] = \
                session.genres_watched.get(genre, 0) / total_genre_time

        # Duration (log-scaled)
        features["duration_log"] = math.log1p(session.total_duration / 60)  # Convert to hours

        # Event density
        if session.total_duration > 0:
            features["event_density"] = session.event_count / (session.total_duration / 60)
        else:
            features["event_density"] = 0.0

        return features

    def _session_to_feature_vector(self, session: Session) -> np.ndarray:
        """Convert session to feature vector matching trained model."""
        fd = self._session_to_feature_dict(session)

        # Use cached feature names if available
        if session.account_id in self._account_models:
            feature_names = self._account_models[session.account_id]["feature_names"]
        else:
            feature_names = sorted(fd.keys())

        return np.array([fd.get(name, 0.0) for name in feature_names])

    def _normalize_features(self, X: np.ndarray) -> np.ndarray:
        """Normalize features to zero mean, unit variance."""
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        std[std == 0] = 1.0  # Avoid division by zero
        return (X - mean) / std

    def _find_optimal_k(self, X: np.ndarray) -> Tuple[int, float]:
        """
        Find optimal number of clusters using silhouette analysis.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix

        Returns
        -------
        Tuple[int, float]
            (optimal_k, best_silhouette_score)
        """
        n_samples = X.shape[0]

        if n_samples < self.config.min_sessions_per_person * 2:
            return 1, 0.0  # Not enough data for multiple people

        # Try different values of k
        max_k = min(
            self.config.max_household_size,
            n_samples // self.config.min_sessions_per_person
        )
        max_k = max(2, max_k)

        best_k = 1
        best_silhouette = -1.0

        for k in range(2, max_k + 1):
            labels, _ = self._cluster_sessions(X, k)
            silhouette = self._compute_silhouette(X, labels)

            if silhouette > best_silhouette:
                best_silhouette = silhouette
                best_k = k

        # Only accept clustering if silhouette is above threshold
        if best_silhouette < self.config.silhouette_threshold:
            return 1, best_silhouette

        return best_k, best_silhouette

    def _cluster_sessions(
        self,
        X: np.ndarray,
        k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Cluster sessions using K-means.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix
        k : int
            Number of clusters

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (labels, centroids)
        """
        n_samples = X.shape[0]

        if k == 1:
            return np.zeros(n_samples, dtype=int), X.mean(axis=0, keepdims=True)

        # K-means++ initialization
        centroids = self._kmeans_init(X, k)

        # Iterate
        max_iter = 100
        for _ in range(max_iter):
            # Assign to nearest centroid
            labels = self._assign_to_centroids(X, centroids)

            # Update centroids
            new_centroids = np.zeros_like(centroids)
            for i in range(k):
                mask = labels == i
                if mask.sum() > 0:
                    new_centroids[i] = X[mask].mean(axis=0)
                else:
                    new_centroids[i] = centroids[i]

            # Check convergence
            if np.allclose(centroids, new_centroids):
                break

            centroids = new_centroids

        return labels, centroids

    def _kmeans_init(self, X: np.ndarray, k: int) -> np.ndarray:
        """K-means++ initialization."""
        n_samples = X.shape[0]
        centroids = np.zeros((k, X.shape[1]))

        # First centroid: random
        idx = self._rng.integers(n_samples)
        centroids[0] = X[idx]

        # Remaining centroids: proportional to squared distance
        for i in range(1, k):
            distances = np.min([
                np.sum((X - centroids[j]) ** 2, axis=1)
                for j in range(i)
            ], axis=0)

            probs = distances / distances.sum()
            idx = self._rng.choice(n_samples, p=probs)
            centroids[i] = X[idx]

        return centroids

    def _assign_to_centroids(
        self,
        X: np.ndarray,
        centroids: np.ndarray
    ) -> np.ndarray:
        """Assign each point to nearest centroid."""
        distances = np.array([
            np.sum((X - c) ** 2, axis=1)
            for c in centroids
        ]).T
        return np.argmin(distances, axis=1)

    def _compute_silhouette(
        self,
        X: np.ndarray,
        labels: np.ndarray
    ) -> float:
        """
        Compute silhouette score for clustering.

        Returns
        -------
        float
            Silhouette score (-1 to 1, higher is better)
        """
        n_samples = X.shape[0]
        unique_labels = np.unique(labels)

        if len(unique_labels) < 2:
            return 0.0

        silhouettes = []

        for i in range(n_samples):
            # a(i): average distance to same cluster
            same_cluster = X[labels == labels[i]]
            if len(same_cluster) > 1:
                a_i = np.mean(np.linalg.norm(X[i] - same_cluster, axis=1))
            else:
                a_i = 0.0

            # b(i): minimum average distance to other clusters
            b_i = float('inf')
            for label in unique_labels:
                if label != labels[i]:
                    other_cluster = X[labels == label]
                    avg_dist = np.mean(np.linalg.norm(X[i] - other_cluster, axis=1))
                    b_i = min(b_i, avg_dist)

            if b_i == float('inf'):
                b_i = 0.0

            # Silhouette for this point
            if max(a_i, b_i) > 0:
                s_i = (b_i - a_i) / max(a_i, b_i)
            else:
                s_i = 0.0

            silhouettes.append(s_i)

        return np.mean(silhouettes)

    def _build_person_profiles(
        self,
        sessions: List[Session],
        labels: np.ndarray,
        n_clusters: int,
        account_id: str
    ) -> List[PersonProfile]:
        """Build person profiles from cluster assignments."""
        members = []

        for cluster_id in range(n_clusters):
            cluster_sessions = [s for s, l in zip(sessions, labels) if l == cluster_id]

            if not cluster_sessions:
                continue

            # Generate person ID using SHA-256 (secure hashing)
            person_id = hashlib.sha256(
                f"{account_id}_person_{cluster_id}".encode()
            ).hexdigest()[:16]

            # Build profile
            profile = self._build_single_person_profile(
                person_id=person_id,
                household_id=account_id,
                sessions=cluster_sessions,
                cluster_id=cluster_id
            )

            members.append(profile)

        # Sort by engagement (descending)
        members.sort(key=lambda m: m.session_count, reverse=True)

        # Assign labels (Person A, B, C...)
        for i, member in enumerate(members):
            member.label = f"Person {chr(65 + i)}"

        return members

    def _build_single_person_profile(
        self,
        person_id: str,
        household_id: str,
        sessions: List[Session],
        cluster_id: int
    ) -> PersonProfile:
        """Build profile for a single person from their sessions."""
        profile = PersonProfile(
            person_id=person_id,
            household_id=household_id
        )

        # Session stats
        profile.session_count = len(sessions)
        profile.total_viewing_time = sum(s.total_duration for s in sessions)
        profile.avg_session_duration = profile.total_viewing_time / len(sessions)

        # Time patterns
        hour_counts = {}
        day_counts = {}

        for session in sessions:
            if hasattr(session.start_time, 'hour'):
                hour = session.start_time.hour
                hour_counts[hour] = hour_counts.get(hour, 0) + 1

                day = session.start_time.weekday()
                day_counts[day] = day_counts.get(day, 0) + 1

        # Hour distribution
        total = sum(hour_counts.values()) or 1
        profile.hour_distribution = {h: c / total for h, c in hour_counts.items()}
        profile.typical_hours = sorted(hour_counts, key=hour_counts.get, reverse=True)[:4]

        # Day distribution
        total = sum(day_counts.values()) or 1
        profile.day_distribution = {d: c / total for d, c in day_counts.items()}
        profile.is_weekday_viewer = sum(day_counts.get(d, 0) for d in range(5)) > 0
        profile.is_weekend_viewer = sum(day_counts.get(d, 0) for d in [5, 6]) > 0

        # Device preferences
        device_counts = {}
        for session in sessions:
            dt = session.device_type
            device_counts[dt] = device_counts.get(dt, 0) + 1

        if device_counts:
            profile.primary_device_type = max(device_counts, key=device_counts.get)
            total = sum(device_counts.values())
            profile.device_distribution = {d: c / total for d, c in device_counts.items()}

        # Genre preferences
        genre_times = {}
        for session in sessions:
            for genre, time in session.genres_watched.items():
                genre_times[genre] = genre_times.get(genre, 0) + time

        if genre_times:
            total = sum(genre_times.values())
            profile.genre_affinities = {g: t / total for g, t in genre_times.items()}
            profile.top_genres = sorted(genre_times, key=genre_times.get, reverse=True)[:3]

        # Infer persona type
        profile.persona_type = self._infer_persona_type(profile)

        # Conversion attribution
        profile.attributed_sessions = sum(1 for s in sessions if s.has_conversion)
        profile.attributed_value = sum(s.conversion_value for s in sessions)

        return profile

    def _infer_persona_type(self, profile: PersonProfile) -> str:
        """Infer persona type from profile characteristics."""
        # Check for child indicators
        kids_genres = {'Animation', 'Kids', 'Family'}
        kids_affinity = sum(profile.genre_affinities.get(g, 0) for g in kids_genres)

        if kids_affinity > 0.5:
            # Check for after-school viewing pattern
            afternoon_hours = [14, 15, 16, 17, 18]
            afternoon_viewing = sum(profile.hour_distribution.get(h, 0) for h in afternoon_hours)
            if afternoon_viewing > 0.3:
                return "child"

        # Check for teen indicators
        teen_genres = {'Action', 'SciFi', 'Fantasy', 'Animation', 'Comedy'}
        teen_affinity = sum(profile.genre_affinities.get(g, 0) for g in teen_genres)

        if teen_affinity > 0.6:
            # Evening/night viewing suggests teen
            evening_hours = list(range(19, 24)) + [0, 1]
            evening_viewing = sum(profile.hour_distribution.get(h, 0) for h in evening_hours)
            if evening_viewing > 0.5 and profile.primary_device_type in ['mobile', 'tablet']:
                return "teen"

        # Primary vs secondary adult
        if profile.session_count > 50:
            return "primary_adult"
        else:
            return "secondary_adult"

    def _build_household_profile(
        self,
        account_id: str,
        members: List[PersonProfile],
        sessions: List[Session],
        n_clusters: int,
        silhouette: float
    ) -> HouseholdProfile:
        """Build complete household profile."""
        household = HouseholdProfile(
            household_id=account_id,
            account_id=account_id,
            estimated_size=len(members),
            size_confidence=min(1.0, max(0.0, silhouette / 0.5))
        )

        # Add members
        for member in members:
            household.add_member(member)

        # Aggregate stats
        household.total_sessions = len(sessions)
        household.total_viewing_time = sum(s.total_duration for s in sessions)
        household.total_conversion_value = sum(s.conversion_value for s in sessions)

        # Devices
        household.devices = list(set(s.device_fingerprint for s in sessions))
        household.device_count = len(household.devices)

        # Peak hours
        all_hours = []
        for session in sessions:
            if hasattr(session.start_time, 'hour'):
                all_hours.append(session.start_time.hour)

        if all_hours:
            hour_counts = {}
            for h in all_hours:
                hour_counts[h] = hour_counts.get(h, 0) + 1
            household.peak_hours = sorted(hour_counts, key=hour_counts.get, reverse=True)[:4]

        # Primary device
        device_counts = {}
        for session in sessions:
            dt = session.device_type
            device_counts[dt] = device_counts.get(dt, 0) + 1
        if device_counts:
            household.primary_device_type = max(device_counts, key=device_counts.get)

        # Timestamps
        timestamps = [s.start_time for s in sessions if s.start_time]
        if timestamps:
            household.first_activity = min(timestamps)
            household.last_activity = max(timestamps)

        # Compute attribution shares
        household.compute_attribution_shares()

        return household

    def _empty_household(self, account_id: str) -> HouseholdProfile:
        """Create an empty household profile."""
        return HouseholdProfile(
            household_id=account_id,
            account_id=account_id,
            estimated_size=1,
            size_confidence=0.0
        )


# Convenience function for quick analysis
def analyze_household_from_events(
    events: List[StreamingEvent],
    session_gap_minutes: int = 30,
    config: Optional[ClusteringConfig] = None
) -> HouseholdProfile:
    """
    Convenience function to analyze a household from raw events.

    Parameters
    ----------
    events : List[StreamingEvent]
        Raw streaming events
    session_gap_minutes : int
        Gap threshold for session grouping
    config : ClusteringConfig, optional
        Clustering configuration

    Returns
    -------
    HouseholdProfile
        Complete household analysis
    """
    sessions = group_events_into_sessions(events, session_gap_minutes)
    engine = HouseholdInferenceEngine(config)
    return engine.analyze_household(sessions)
