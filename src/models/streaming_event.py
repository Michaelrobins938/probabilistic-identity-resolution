"""
Streaming Event and Session Models

Netflix-style event schema for household identity resolution.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
import hashlib
import json


@dataclass
class StreamingEvent:
    """
    A single streaming/viewing event from a platform like Netflix.

    This is the raw input format - one event per user action.
    Events are grouped into Sessions for analysis.
    """
    # Core identifiers
    event_id: str
    account_id: str                    # The shared account (e.g., Netflix account)
    device_fingerprint: str            # Device identifier
    timestamp: datetime

    # Event details
    event_type: str                    # 'play', 'pause', 'browse', 'search', 'conversion'
    content_id: Optional[str] = None   # What content was interacted with
    content_title: Optional[str] = None
    content_genre: Optional[str] = None
    duration_seconds: float = 0.0      # How long the event lasted

    # Device signals (for fingerprinting)
    device_type: str = "unknown"       # 'tv', 'desktop', 'mobile', 'tablet'
    os_family: str = "unknown"
    browser_family: str = "unknown"
    screen_resolution: str = "unknown"
    ip_hash: Optional[str] = None      # Hashed IP for network detection

    # Behavioral signals (for person inference)
    hour_of_day: int = 0               # 0-23
    day_of_week: int = 0               # 0-6 (Monday=0)

    # Conversion tracking
    conversion_value: float = 0.0
    conversion_type: Optional[str] = None  # 'subscription', 'upgrade', 'purchase'

    # Marketing channel (for attribution)
    channel: Optional[str] = None
    campaign_id: Optional[str] = None

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Extract hour and day from timestamp if not set."""
        if isinstance(self.timestamp, str):
            self.timestamp = datetime.fromisoformat(self.timestamp.replace('Z', '+00:00'))

        if self.hour_of_day == 0 and self.timestamp:
            self.hour_of_day = self.timestamp.hour
        if self.day_of_week == 0 and self.timestamp:
            self.day_of_week = self.timestamp.weekday()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "event_id": self.event_id,
            "account_id": self.account_id,
            "device_fingerprint": self.device_fingerprint,
            "timestamp": self.timestamp.isoformat() if isinstance(self.timestamp, datetime) else self.timestamp,
            "event_type": self.event_type,
            "content_id": self.content_id,
            "content_title": self.content_title,
            "content_genre": self.content_genre,
            "duration_seconds": self.duration_seconds,
            "device_type": self.device_type,
            "os_family": self.os_family,
            "browser_family": self.browser_family,
            "screen_resolution": self.screen_resolution,
            "ip_hash": self.ip_hash,
            "hour_of_day": self.hour_of_day,
            "day_of_week": self.day_of_week,
            "conversion_value": self.conversion_value,
            "conversion_type": self.conversion_type,
            "channel": self.channel,
            "campaign_id": self.campaign_id,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StreamingEvent":
        """Deserialize from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def get_feature_vector(self) -> Dict[str, float]:
        """
        Extract features for clustering/ML.

        Returns normalized feature dictionary for household inference.
        """
        # Cyclical encoding for hour (captures 11pm being close to 1am)
        import math
        hour_sin = math.sin(2 * math.pi * self.hour_of_day / 24)
        hour_cos = math.cos(2 * math.pi * self.hour_of_day / 24)

        # Day encoding
        day_sin = math.sin(2 * math.pi * self.day_of_week / 7)
        day_cos = math.cos(2 * math.pi * self.day_of_week / 7)

        # Device type one-hot
        device_types = ['tv', 'desktop', 'mobile', 'tablet', 'unknown']
        device_features = {f"device_{dt}": 1.0 if self.device_type == dt else 0.0
                          for dt in device_types}

        return {
            "hour_sin": hour_sin,
            "hour_cos": hour_cos,
            "day_sin": day_sin,
            "day_cos": day_cos,
            "duration_log": math.log1p(self.duration_seconds),
            **device_features,
        }


@dataclass
class Session:
    """
    A session groups consecutive events from the same device within a time window.

    Sessions are the unit of analysis for household inference -
    we assign each session to a probable person.
    """
    session_id: str
    account_id: str
    device_fingerprint: str

    # Session timing
    start_time: datetime
    end_time: datetime

    # Aggregated session features
    events: List[StreamingEvent] = field(default_factory=list)
    total_duration: float = 0.0        # Sum of event durations
    event_count: int = 0

    # Content profile (aggregated across session)
    genres_watched: Dict[str, float] = field(default_factory=dict)  # genre -> seconds
    primary_genre: Optional[str] = None

    # Device info (from events)
    device_type: str = "unknown"

    # Identity assignment (filled by inference engine)
    assigned_person_id: Optional[str] = None
    person_probabilities: Dict[str, float] = field(default_factory=dict)
    assignment_confidence: float = 0.0

    # Conversion tracking
    has_conversion: bool = False
    conversion_value: float = 0.0

    # Marketing channel (for attribution)
    channels: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Compute derived fields from events."""
        if self.events:
            self._compute_aggregates()

    def _compute_aggregates(self):
        """Compute session-level aggregates from events."""
        if not self.events:
            return

        # Sort events by timestamp
        sorted_events = sorted(self.events, key=lambda e: e.timestamp)

        # Timing
        self.start_time = sorted_events[0].timestamp
        self.end_time = sorted_events[-1].timestamp
        self.event_count = len(sorted_events)

        # Duration and genres
        self.total_duration = sum(e.duration_seconds for e in sorted_events)

        for event in sorted_events:
            if event.content_genre:
                self.genres_watched[event.content_genre] = \
                    self.genres_watched.get(event.content_genre, 0) + event.duration_seconds

        if self.genres_watched:
            self.primary_genre = max(self.genres_watched, key=self.genres_watched.get)

        # Device
        self.device_type = sorted_events[0].device_type

        # Conversions
        for event in sorted_events:
            if event.conversion_value > 0:
                self.has_conversion = True
                self.conversion_value += event.conversion_value

        # Channels
        self.channels = list(set(e.channel for e in sorted_events if e.channel))

    def add_event(self, event: StreamingEvent):
        """Add an event to this session and recompute aggregates."""
        self.events.append(event)
        self._compute_aggregates()

    def get_feature_vector(self) -> Dict[str, float]:
        """
        Extract session-level features for clustering.

        Used by household inference to cluster sessions into persons.
        """
        import math

        # Time features (cyclical)
        if isinstance(self.start_time, datetime):
            hour = self.start_time.hour
            day = self.start_time.weekday()
        else:
            hour = 12
            day = 0

        hour_sin = math.sin(2 * math.pi * hour / 24)
        hour_cos = math.cos(2 * math.pi * hour / 24)
        day_sin = math.sin(2 * math.pi * day / 7)
        day_cos = math.cos(2 * math.pi * day / 7)

        # Device features
        device_types = ['tv', 'desktop', 'mobile', 'tablet', 'unknown']
        device_features = {f"device_{dt}": 1.0 if self.device_type == dt else 0.0
                          for dt in device_types}

        # Genre features (normalized)
        all_genres = ['Drama', 'Comedy', 'Action', 'Documentary', 'Kids',
                      'Animation', 'Reality', 'Thriller', 'Romance', 'SciFi']
        total_genre_time = sum(self.genres_watched.values()) or 1.0
        genre_features = {f"genre_{g.lower()}": self.genres_watched.get(g, 0) / total_genre_time
                         for g in all_genres}

        # Duration features
        duration_log = math.log1p(self.total_duration)
        event_count_log = math.log1p(self.event_count)

        return {
            "hour_sin": hour_sin,
            "hour_cos": hour_cos,
            "day_sin": day_sin,
            "day_cos": day_cos,
            "duration_log": duration_log,
            "event_count_log": event_count_log,
            **device_features,
            **genre_features,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Serialize session to dictionary."""
        return {
            "session_id": self.session_id,
            "account_id": self.account_id,
            "device_fingerprint": self.device_fingerprint,
            "start_time": self.start_time.isoformat() if isinstance(self.start_time, datetime) else self.start_time,
            "end_time": self.end_time.isoformat() if isinstance(self.end_time, datetime) else self.end_time,
            "total_duration": self.total_duration,
            "event_count": self.event_count,
            "genres_watched": self.genres_watched,
            "primary_genre": self.primary_genre,
            "device_type": self.device_type,
            "assigned_person_id": self.assigned_person_id,
            "person_probabilities": self.person_probabilities,
            "assignment_confidence": self.assignment_confidence,
            "has_conversion": self.has_conversion,
            "conversion_value": self.conversion_value,
            "channels": self.channels,
        }


def group_events_into_sessions(
    events: List[StreamingEvent],
    session_gap_minutes: int = 30
) -> List[Session]:
    """
    Group events into sessions based on time gaps.

    Events on the same device within `session_gap_minutes` of each other
    are grouped into the same session.

    Parameters
    ----------
    events : List[StreamingEvent]
        Raw events to group
    session_gap_minutes : int
        Maximum gap between events in same session

    Returns
    -------
    List[Session]
        Grouped sessions
    """
    if not events:
        return []

    # Sort by account, device, then timestamp
    sorted_events = sorted(events, key=lambda e: (e.account_id, e.device_fingerprint, e.timestamp))

    sessions = []
    current_session_events = [sorted_events[0]]

    for i in range(1, len(sorted_events)):
        prev = sorted_events[i - 1]
        curr = sorted_events[i]

        # Check if same account and device
        same_context = (prev.account_id == curr.account_id and
                       prev.device_fingerprint == curr.device_fingerprint)

        # Check time gap
        if isinstance(prev.timestamp, datetime) and isinstance(curr.timestamp, datetime):
            gap_seconds = (curr.timestamp - prev.timestamp).total_seconds()
        else:
            gap_seconds = float('inf')

        within_gap = gap_seconds <= session_gap_minutes * 60

        if same_context and within_gap:
            # Continue current session
            current_session_events.append(curr)
        else:
            # End current session and start new one
            session_id = hashlib.md5(
                f"{current_session_events[0].account_id}_{current_session_events[0].device_fingerprint}_{current_session_events[0].timestamp}".encode()
            ).hexdigest()[:16]

            session = Session(
                session_id=session_id,
                account_id=current_session_events[0].account_id,
                device_fingerprint=current_session_events[0].device_fingerprint,
                start_time=current_session_events[0].timestamp,
                end_time=current_session_events[-1].timestamp,
                events=current_session_events,
            )
            sessions.append(session)

            current_session_events = [curr]

    # Don't forget the last session
    if current_session_events:
        session_id = hashlib.md5(
            f"{current_session_events[0].account_id}_{current_session_events[0].device_fingerprint}_{current_session_events[0].timestamp}".encode()
        ).hexdigest()[:16]

        session = Session(
            session_id=session_id,
            account_id=current_session_events[0].account_id,
            device_fingerprint=current_session_events[0].device_fingerprint,
            start_time=current_session_events[0].timestamp,
            end_time=current_session_events[-1].timestamp,
            events=current_session_events,
        )
        sessions.append(session)

    return sessions
