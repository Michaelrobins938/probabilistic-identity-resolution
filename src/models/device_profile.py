"""
Device Profile Model

Represents a device's fingerprint and behavioral signature for identity resolution.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import hashlib
import math


@dataclass
class DeviceProfile:
    """
    A device's fingerprint and behavioral signature.

    Used for:
    1. Device identification (fingerprinting)
    2. Person inference (behavioral patterns suggest WHO uses this device)
    3. Cross-device linking (same person on different devices)
    """
    fingerprint_id: str

    # Hardware signals (for fingerprinting)
    device_type: str = "unknown"       # 'tv', 'desktop', 'mobile', 'tablet'
    os_family: str = "unknown"         # 'iOS', 'Android', 'Windows', 'macOS', 'Linux'
    browser_family: str = "unknown"    # 'Chrome', 'Safari', 'Firefox'
    screen_resolution: str = "unknown" # '1920x1080', '2560x1440'

    # Network signals
    ip_hashes: List[str] = field(default_factory=list)  # Multiple IPs (home, work, mobile)
    primary_ip_hash: Optional[str] = None

    # Temporal patterns (for person inference)
    typical_hours: List[int] = field(default_factory=list)  # Most common hours [8, 9, 20, 21]
    hour_distribution: Dict[int, float] = field(default_factory=dict)  # Hour -> frequency
    day_distribution: Dict[int, float] = field(default_factory=dict)   # Day -> frequency

    # Content preferences (for person inference)
    genre_affinities: Dict[str, float] = field(default_factory=dict)  # Genre -> affinity score
    avg_session_duration: float = 0.0
    avg_events_per_session: float = 0.0

    # Interaction patterns
    interaction_speed: float = 0.0     # Events per minute
    scroll_depth: float = 0.0          # Average scroll depth 0-1
    click_patterns: Dict[str, float] = field(default_factory=dict)

    # Household context
    room_inference: str = "unknown"    # 'living_room', 'bedroom', 'office', 'kitchen'
    is_shared_device: bool = False     # True if multiple persons use this device

    # Usage statistics
    first_seen: Optional[datetime] = None
    last_seen: Optional[datetime] = None
    total_sessions: int = 0
    total_events: int = 0

    # Linked identities
    account_ids: List[str] = field(default_factory=list)
    likely_person_ids: Dict[str, float] = field(default_factory=dict)  # person_id -> probability

    def update_from_sessions(self, sessions: List["Session"]) -> None:
        """
        Update device profile from observed sessions.

        Parameters
        ----------
        sessions : List[Session]
            Sessions observed on this device
        """
        if not sessions:
            return

        # Update temporal patterns
        hour_counts = {}
        day_counts = {}
        total_duration = 0.0
        total_events = 0
        genre_durations = {}

        for session in sessions:
            # Hour distribution
            if hasattr(session.start_time, 'hour'):
                hour = session.start_time.hour
                hour_counts[hour] = hour_counts.get(hour, 0) + 1

            # Day distribution
            if hasattr(session.start_time, 'weekday'):
                day = session.start_time.weekday()
                day_counts[day] = day_counts.get(day, 0) + 1

            # Duration and events
            total_duration += session.total_duration
            total_events += session.event_count

            # Genre preferences
            for genre, duration in session.genres_watched.items():
                genre_durations[genre] = genre_durations.get(genre, 0) + duration

        # Normalize distributions
        total_sessions = len(sessions)
        self.hour_distribution = {h: c / total_sessions for h, c in hour_counts.items()}
        self.day_distribution = {d: c / total_sessions for d, c in day_counts.items()}

        # Top hours
        sorted_hours = sorted(hour_counts.items(), key=lambda x: x[1], reverse=True)
        self.typical_hours = [h for h, _ in sorted_hours[:4]]

        # Averages
        self.avg_session_duration = total_duration / total_sessions if total_sessions > 0 else 0
        self.avg_events_per_session = total_events / total_sessions if total_sessions > 0 else 0
        self.total_sessions = total_sessions
        self.total_events = total_events

        # Genre affinities (normalized)
        total_genre_time = sum(genre_durations.values()) or 1.0
        self.genre_affinities = {g: d / total_genre_time for g, d in genre_durations.items()}

        # Timestamps
        timestamps = [s.start_time for s in sessions if s.start_time]
        if timestamps:
            self.first_seen = min(timestamps)
            self.last_seen = max(timestamps)

        # Account IDs
        self.account_ids = list(set(s.account_id for s in sessions))

        # Room inference based on device type and usage patterns
        self._infer_room()

        # Detect if shared device (multiple distinct usage patterns)
        self._detect_shared_device()

    def _infer_room(self) -> None:
        """Infer room placement based on device type and usage patterns."""
        if self.device_type == 'tv':
            # TV in evening = living room, late night = bedroom
            evening_usage = sum(self.hour_distribution.get(h, 0) for h in range(18, 23))
            late_night_usage = sum(self.hour_distribution.get(h, 0) for h in list(range(23, 24)) + list(range(0, 6)))

            if late_night_usage > evening_usage * 0.5:
                self.room_inference = 'bedroom'
            else:
                self.room_inference = 'living_room'

        elif self.device_type == 'desktop':
            # Desktop during work hours = office, otherwise bedroom/office
            work_hours_usage = sum(self.hour_distribution.get(h, 0) for h in range(9, 18))
            if work_hours_usage > 0.4:
                self.room_inference = 'office'
            else:
                self.room_inference = 'bedroom'

        elif self.device_type in ['mobile', 'tablet']:
            self.room_inference = 'portable'  # Moves around

        else:
            self.room_inference = 'unknown'

    def _detect_shared_device(self) -> None:
        """
        Detect if multiple persons likely use this device.

        Indicators:
        - Bimodal time distribution
        - Diverse genre preferences
        - High variance in session characteristics
        """
        # Check for bimodal hour distribution
        if len(self.hour_distribution) > 0:
            hours_used = len([h for h, f in self.hour_distribution.items() if f > 0.05])
            peak_hours = len([h for h, f in self.hour_distribution.items() if f > 0.1])

            # Wide spread with multiple peaks suggests multiple users
            if hours_used > 12 and peak_hours > 3:
                self.is_shared_device = True

        # Diverse genre preferences
        if len(self.genre_affinities) > 0:
            # High entropy in genre distribution suggests multiple people
            entropy = -sum(p * math.log(p + 1e-10) for p in self.genre_affinities.values())
            max_entropy = math.log(len(self.genre_affinities) + 1e-10)
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0

            if normalized_entropy > 0.7:  # High genre diversity
                self.is_shared_device = True

    def get_similarity_to(self, other: "DeviceProfile") -> float:
        """
        Compute similarity score to another device profile.

        Used for cross-device linking (same person, different devices).

        Returns
        -------
        float
            Similarity score 0.0 - 1.0
        """
        scores = []
        weights = []

        # IP overlap (strong signal)
        if self.ip_hashes and other.ip_hashes:
            ip_overlap = len(set(self.ip_hashes) & set(other.ip_hashes))
            ip_score = ip_overlap / max(len(self.ip_hashes), len(other.ip_hashes))
            scores.append(ip_score)
            weights.append(3.0)  # High weight

        # Temporal pattern similarity
        if self.hour_distribution and other.hour_distribution:
            hour_sim = self._cosine_similarity(self.hour_distribution, other.hour_distribution)
            scores.append(hour_sim)
            weights.append(1.0)

        # Genre preference similarity
        if self.genre_affinities and other.genre_affinities:
            genre_sim = self._cosine_similarity(self.genre_affinities, other.genre_affinities)
            scores.append(genre_sim)
            weights.append(2.0)  # Medium-high weight

        # Session behavior similarity
        if self.avg_session_duration > 0 and other.avg_session_duration > 0:
            duration_ratio = min(self.avg_session_duration, other.avg_session_duration) / \
                           max(self.avg_session_duration, other.avg_session_duration)
            scores.append(duration_ratio)
            weights.append(0.5)

        if not scores:
            return 0.0

        # Weighted average
        return sum(s * w for s, w in zip(scores, weights)) / sum(weights)

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

    def get_feature_vector(self) -> Dict[str, float]:
        """
        Extract feature vector for ML-based matching.

        Returns
        -------
        Dict[str, float]
            Normalized feature dictionary
        """
        features = {}

        # Device type one-hot
        device_types = ['tv', 'desktop', 'mobile', 'tablet', 'unknown']
        for dt in device_types:
            features[f"device_{dt}"] = 1.0 if self.device_type == dt else 0.0

        # Hour distribution (as features)
        for h in range(24):
            features[f"hour_{h}"] = self.hour_distribution.get(h, 0.0)

        # Genre affinities
        genres = ['Drama', 'Comedy', 'Action', 'Documentary', 'Kids',
                  'Animation', 'Reality', 'Thriller', 'Romance', 'SciFi']
        for g in genres:
            features[f"genre_{g.lower()}"] = self.genre_affinities.get(g, 0.0)

        # Session characteristics
        features["avg_duration_log"] = math.log1p(self.avg_session_duration)
        features["avg_events_log"] = math.log1p(self.avg_events_per_session)
        features["is_shared"] = 1.0 if self.is_shared_device else 0.0

        return features

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "fingerprint_id": self.fingerprint_id,
            "device_type": self.device_type,
            "os_family": self.os_family,
            "browser_family": self.browser_family,
            "screen_resolution": self.screen_resolution,
            "ip_hashes": self.ip_hashes,
            "primary_ip_hash": self.primary_ip_hash,
            "typical_hours": self.typical_hours,
            "hour_distribution": self.hour_distribution,
            "day_distribution": self.day_distribution,
            "genre_affinities": self.genre_affinities,
            "avg_session_duration": self.avg_session_duration,
            "avg_events_per_session": self.avg_events_per_session,
            "room_inference": self.room_inference,
            "is_shared_device": self.is_shared_device,
            "first_seen": self.first_seen.isoformat() if self.first_seen else None,
            "last_seen": self.last_seen.isoformat() if self.last_seen else None,
            "total_sessions": self.total_sessions,
            "total_events": self.total_events,
            "account_ids": self.account_ids,
            "likely_person_ids": self.likely_person_ids,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DeviceProfile":
        """Deserialize from dictionary."""
        if data.get("first_seen"):
            data["first_seen"] = datetime.fromisoformat(data["first_seen"])
        if data.get("last_seen"):
            data["last_seen"] = datetime.fromisoformat(data["last_seen"])
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


def create_device_fingerprint(
    device_type: str,
    os_family: str,
    browser_family: str = "",
    screen_resolution: str = "",
    user_agent: str = ""
) -> str:
    """
    Create a device fingerprint from device signals.

    This is a privacy-preserving identifier that doesn't use PII.

    Parameters
    ----------
    device_type : str
        Type of device
    os_family : str
        Operating system
    browser_family : str
        Browser (if applicable)
    screen_resolution : str
        Screen resolution
    user_agent : str
        Full user agent string (optional)

    Returns
    -------
    str
        16-character hex fingerprint
    """
    components = [
        device_type.lower(),
        os_family.lower(),
        browser_family.lower(),
        screen_resolution,
    ]

    if user_agent:
        components.append(user_agent)

    fingerprint_string = "|".join(components)
    return hashlib.sha256(fingerprint_string.encode()).hexdigest()[:16]
