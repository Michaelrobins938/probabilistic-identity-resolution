"""
Household Profile Model

Represents a household's composition and viewing patterns.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import math


@dataclass
class PersonProfile:
    """
    Profile of an inferred person within a household.

    This is the output of household inference - a cluster of sessions
    that we believe belong to the same person.
    """
    person_id: str
    household_id: str

    # Inferred attributes
    persona_type: str = "unknown"  # 'primary_adult', 'secondary_adult', 'child', 'teen', 'unknown'
    label: str = "Person"  # Human-readable label like "Person A"

    # Temporal patterns
    typical_hours: List[int] = field(default_factory=list)  # Peak viewing hours
    hour_distribution: Dict[int, float] = field(default_factory=dict)
    day_distribution: Dict[int, float] = field(default_factory=dict)
    is_weekday_viewer: bool = True
    is_weekend_viewer: bool = True

    # Device preferences
    primary_device_type: str = "unknown"
    device_distribution: Dict[str, float] = field(default_factory=dict)  # device_type -> share

    # Content preferences
    genre_affinities: Dict[str, float] = field(default_factory=dict)
    top_genres: List[str] = field(default_factory=list)
    avg_session_duration: float = 0.0

    # Engagement metrics
    session_count: int = 0
    total_viewing_time: float = 0.0
    engagement_score: float = 0.0  # 0-1, relative to household average

    # Conversion attribution
    attributed_sessions: int = 0
    attributed_value: float = 0.0
    attribution_share: float = 0.0  # Share of household value

    # Confidence
    confidence: float = 0.0  # How confident are we this is a distinct person?

    def get_time_of_day_label(self) -> str:
        """Get human-readable time of day preference."""
        if not self.typical_hours:
            return "Unknown"

        avg_hour = sum(self.typical_hours) / len(self.typical_hours)

        if 5 <= avg_hour < 12:
            return "Morning Viewer"
        elif 12 <= avg_hour < 17:
            return "Afternoon Viewer"
        elif 17 <= avg_hour < 21:
            return "Evening Viewer"
        else:
            return "Night Owl"

    def get_description(self) -> str:
        """Generate human-readable description of this person."""
        parts = []

        # Persona
        if self.persona_type != "unknown":
            parts.append(self.persona_type.replace("_", " ").title())

        # Viewing time
        parts.append(self.get_time_of_day_label())

        # Primary device
        if self.primary_device_type != "unknown":
            parts.append(f"on {self.primary_device_type}")

        # Top genres
        if self.top_genres:
            genres = ", ".join(self.top_genres[:2])
            parts.append(f"watches {genres}")

        return " | ".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "person_id": self.person_id,
            "household_id": self.household_id,
            "persona_type": self.persona_type,
            "label": self.label,
            "typical_hours": self.typical_hours,
            "hour_distribution": self.hour_distribution,
            "day_distribution": self.day_distribution,
            "is_weekday_viewer": self.is_weekday_viewer,
            "is_weekend_viewer": self.is_weekend_viewer,
            "primary_device_type": self.primary_device_type,
            "device_distribution": self.device_distribution,
            "genre_affinities": self.genre_affinities,
            "top_genres": self.top_genres,
            "avg_session_duration": self.avg_session_duration,
            "session_count": self.session_count,
            "total_viewing_time": self.total_viewing_time,
            "engagement_score": self.engagement_score,
            "attributed_sessions": self.attributed_sessions,
            "attributed_value": self.attributed_value,
            "attribution_share": self.attribution_share,
            "confidence": self.confidence,
            "description": self.get_description(),
        }


@dataclass
class HouseholdProfile:
    """
    Complete profile of a household's composition and viewing patterns.

    This is the main output of household inference - understanding WHO
    is using a shared account and HOW they use it.
    """
    household_id: str
    account_id: str

    # Household composition
    estimated_size: int = 1  # Number of distinct people
    size_confidence: float = 0.0  # Confidence in size estimate

    # Member profiles
    members: List[PersonProfile] = field(default_factory=list)

    # Household type inference
    household_type: str = "unknown"  # 'single', 'couple', 'family', 'roommates'
    has_children: bool = False
    has_teens: bool = False

    # Device landscape
    devices: List[str] = field(default_factory=list)  # Device fingerprints
    device_count: int = 0
    primary_device_type: str = "unknown"  # Most used device type

    # Aggregate patterns
    total_sessions: int = 0
    total_viewing_time: float = 0.0
    peak_hours: List[int] = field(default_factory=list)
    active_days_per_week: float = 0.0

    # Shared viewing
    shared_viewing_sessions: int = 0  # Sessions with likely co-viewing
    shared_viewing_rate: float = 0.0  # Percentage of shared sessions

    # Conversion/subscription
    total_conversion_value: float = 0.0
    subscription_tier: Optional[str] = None

    # Timestamps
    first_activity: Optional[datetime] = None
    last_activity: Optional[datetime] = None
    analysis_date: Optional[datetime] = None

    def __post_init__(self):
        """Set analysis date if not provided."""
        if not self.analysis_date:
            self.analysis_date = datetime.now()

    def add_member(self, member: PersonProfile) -> None:
        """Add a member to the household."""
        member.household_id = self.household_id
        self.members.append(member)
        self._update_household_type()

    def _update_household_type(self) -> None:
        """Infer household type from member composition."""
        n_members = len(self.members)

        if n_members == 1:
            self.household_type = "single"
            return

        # Count by persona type
        adults = [m for m in self.members if 'adult' in m.persona_type.lower()]
        children = [m for m in self.members if 'child' in m.persona_type.lower()]
        teens = [m for m in self.members if 'teen' in m.persona_type.lower()]

        self.has_children = len(children) > 0
        self.has_teens = len(teens) > 0

        if children or teens:
            self.household_type = "family"
        elif len(adults) == 2:
            self.household_type = "couple"
        elif len(adults) > 2:
            self.household_type = "roommates"
        else:
            self.household_type = "unknown"

    def get_member_by_id(self, person_id: str) -> Optional[PersonProfile]:
        """Get a member by their ID."""
        for member in self.members:
            if member.person_id == person_id:
                return member
        return None

    def get_primary_member(self) -> Optional[PersonProfile]:
        """Get the primary household member (highest engagement)."""
        if not self.members:
            return None

        return max(self.members, key=lambda m: m.session_count)

    def compute_attribution_shares(self) -> Dict[str, float]:
        """
        Compute how to split attribution across household members.

        Returns
        -------
        Dict[str, float]
            person_id -> share (sums to 1.0)
        """
        if not self.members:
            return {}

        total_engagement = sum(m.session_count * m.avg_session_duration for m in self.members)

        if total_engagement == 0:
            # Equal split
            n = len(self.members)
            return {m.person_id: 1.0 / n for m in self.members}

        shares = {}
        for member in self.members:
            engagement = member.session_count * member.avg_session_duration
            shares[member.person_id] = engagement / total_engagement
            member.attribution_share = shares[member.person_id]

        return shares

    def get_summary(self) -> str:
        """Generate human-readable summary of the household."""
        lines = [
            f"Household Analysis: {self.account_id}",
            "=" * 50,
            "",
            f"Inferred Household Size: {self.estimated_size} people "
            f"(confidence: {self.size_confidence:.0%})",
            f"Household Type: {self.household_type.title()}",
            "",
        ]

        for i, member in enumerate(self.members, 1):
            label = member.label or f"Person {chr(64 + i)}"
            lines.append(f"{label} ({member.persona_type.replace('_', ' ').title()}):")
            lines.append(f"  - Devices: {member.primary_device_type}")
            lines.append(f"  - Peak hours: {member.typical_hours}")
            lines.append(f"  - Genres: {', '.join(member.top_genres[:3])}")
            lines.append(f"  - Sessions: {member.session_count}")
            lines.append(f"  - Attribution: {member.attribution_share:.0%}")
            lines.append("")

        if self.total_conversion_value > 0:
            lines.append("Attribution Impact:")
            lines.append(f"  - Traditional: 1 conversion = ${self.total_conversion_value:.2f}")
            lines.append("  - Identity-Resolved:")
            for member in self.members:
                value = member.attribution_share * self.total_conversion_value
                lines.append(f"    - {member.label}: {member.attribution_share:.0%} = ${value:.2f}")

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "household_id": self.household_id,
            "account_id": self.account_id,
            "estimated_size": self.estimated_size,
            "size_confidence": self.size_confidence,
            "members": [m.to_dict() for m in self.members],
            "household_type": self.household_type,
            "has_children": self.has_children,
            "has_teens": self.has_teens,
            "devices": self.devices,
            "device_count": self.device_count,
            "primary_device_type": self.primary_device_type,
            "total_sessions": self.total_sessions,
            "total_viewing_time": self.total_viewing_time,
            "peak_hours": self.peak_hours,
            "active_days_per_week": self.active_days_per_week,
            "shared_viewing_sessions": self.shared_viewing_sessions,
            "shared_viewing_rate": self.shared_viewing_rate,
            "total_conversion_value": self.total_conversion_value,
            "subscription_tier": self.subscription_tier,
            "first_activity": self.first_activity.isoformat() if self.first_activity else None,
            "last_activity": self.last_activity.isoformat() if self.last_activity else None,
            "analysis_date": self.analysis_date.isoformat() if self.analysis_date else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HouseholdProfile":
        """Deserialize from dictionary."""
        # Handle nested members
        members_data = data.pop("members", [])
        members = [PersonProfile(**m) for m in members_data]

        # Handle timestamps
        for field_name in ["first_activity", "last_activity", "analysis_date"]:
            if data.get(field_name):
                data[field_name] = datetime.fromisoformat(data[field_name])

        profile = cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
        profile.members = members
        return profile
