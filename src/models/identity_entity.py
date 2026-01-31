"""
Identity Entity Model

Represents a resolved identity (person, device, or household).
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Set
from enum import Enum
import uuid


class EntityType(Enum):
    """Types of identity entities."""
    PERSON = "person"
    DEVICE = "device"
    HOUSEHOLD = "household"
    ACCOUNT = "account"


@dataclass
class IdentityEntity:
    """
    A resolved identity entity in the identity graph.

    Can represent:
    - A person (inferred from behavioral patterns)
    - A device (physical device fingerprint)
    - A household (collection of people sharing an account)
    - An account (the subscription/login identity)
    """
    entity_id: str
    entity_type: EntityType

    # Confidence in this entity's existence/resolution
    confidence: float = 1.0  # 0.0 - 1.0

    # Labels/attributes
    label: Optional[str] = None  # Human-readable label ("Person A", "Living Room TV")
    persona_type: Optional[str] = None  # 'primary_adult', 'secondary_adult', 'child', 'teen'

    # Linked identifiers
    device_fingerprints: List[str] = field(default_factory=list)
    account_ids: List[str] = field(default_factory=list)
    session_ids: List[str] = field(default_factory=list)

    # For PERSON entities: probability distribution over household members
    # (when we're uncertain about which person in the household)
    person_probabilities: Dict[str, float] = field(default_factory=dict)

    # For DEVICE entities: which persons likely use this device
    device_user_probabilities: Dict[str, float] = field(default_factory=dict)

    # For HOUSEHOLD entities: member person IDs
    household_members: List[str] = field(default_factory=list)

    # Temporal metadata
    first_seen: Optional[datetime] = None
    last_seen: Optional[datetime] = None
    session_count: int = 0
    event_count: int = 0

    # Behavioral profile (for persons)
    typical_hours: List[int] = field(default_factory=list)
    genre_preferences: Dict[str, float] = field(default_factory=dict)
    device_preferences: Dict[str, float] = field(default_factory=dict)  # device_type -> affinity

    # Conversion/value data (for attribution)
    total_conversion_value: float = 0.0
    conversion_count: int = 0

    # Graph edges (relationships to other entities)
    edges: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # edge_id -> {target_entity_id, edge_type, weight, metadata}

    def __post_init__(self):
        """Generate entity ID if not provided."""
        if not self.entity_id:
            self.entity_id = str(uuid.uuid4())[:12]

        if isinstance(self.entity_type, str):
            self.entity_type = EntityType(self.entity_type)

    def add_edge(
        self,
        target_id: str,
        edge_type: str,
        weight: float = 1.0,
        **metadata
    ) -> str:
        """
        Add an edge to another entity.

        Parameters
        ----------
        target_id : str
            ID of the target entity
        edge_type : str
            Type of relationship: 'uses_device', 'member_of', 'same_person', etc.
        weight : float
            Edge weight (probability/confidence)
        **metadata
            Additional edge metadata

        Returns
        -------
        str
            Edge ID
        """
        edge_id = f"{self.entity_id}_{target_id}_{edge_type}"
        self.edges[edge_id] = {
            "target_entity_id": target_id,
            "edge_type": edge_type,
            "weight": weight,
            **metadata
        }
        return edge_id

    def get_edges_by_type(self, edge_type: str) -> List[Dict[str, Any]]:
        """Get all edges of a specific type."""
        return [e for e in self.edges.values() if e["edge_type"] == edge_type]

    def get_connected_entities(self) -> Set[str]:
        """Get IDs of all connected entities."""
        return {e["target_entity_id"] for e in self.edges.values()}

    def merge_with(self, other: "IdentityEntity") -> "IdentityEntity":
        """
        Merge another entity into this one.

        Used when we determine two entities are actually the same.
        The merged entity keeps this entity's ID.

        Parameters
        ----------
        other : IdentityEntity
            Entity to merge in

        Returns
        -------
        IdentityEntity
            The merged entity (self, modified)
        """
        # Merge device fingerprints
        self.device_fingerprints = list(set(self.device_fingerprints + other.device_fingerprints))

        # Merge account IDs
        self.account_ids = list(set(self.account_ids + other.account_ids))

        # Merge session IDs
        self.session_ids = list(set(self.session_ids + other.session_ids))

        # Update confidence (average weighted by session count)
        total_sessions = self.session_count + other.session_count
        if total_sessions > 0:
            self.confidence = (self.confidence * self.session_count +
                             other.confidence * other.session_count) / total_sessions

        # Update timestamps
        if other.first_seen:
            if not self.first_seen or other.first_seen < self.first_seen:
                self.first_seen = other.first_seen
        if other.last_seen:
            if not self.last_seen or other.last_seen > self.last_seen:
                self.last_seen = other.last_seen

        # Sum counts
        self.session_count += other.session_count
        self.event_count += other.event_count
        self.total_conversion_value += other.total_conversion_value
        self.conversion_count += other.conversion_count

        # Merge genre preferences (average)
        for genre, score in other.genre_preferences.items():
            if genre in self.genre_preferences:
                self.genre_preferences[genre] = (self.genre_preferences[genre] + score) / 2
            else:
                self.genre_preferences[genre] = score

        # Merge edges
        self.edges.update(other.edges)

        # Merge household members if applicable
        if self.entity_type == EntityType.HOUSEHOLD:
            self.household_members = list(set(self.household_members + other.household_members))

        return self

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "entity_id": self.entity_id,
            "entity_type": self.entity_type.value,
            "confidence": self.confidence,
            "label": self.label,
            "persona_type": self.persona_type,
            "device_fingerprints": self.device_fingerprints,
            "account_ids": self.account_ids,
            "session_ids": self.session_ids,
            "person_probabilities": self.person_probabilities,
            "device_user_probabilities": self.device_user_probabilities,
            "household_members": self.household_members,
            "first_seen": self.first_seen.isoformat() if self.first_seen else None,
            "last_seen": self.last_seen.isoformat() if self.last_seen else None,
            "session_count": self.session_count,
            "event_count": self.event_count,
            "typical_hours": self.typical_hours,
            "genre_preferences": self.genre_preferences,
            "device_preferences": self.device_preferences,
            "total_conversion_value": self.total_conversion_value,
            "conversion_count": self.conversion_count,
            "edges": self.edges,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IdentityEntity":
        """Deserialize from dictionary."""
        if data.get("first_seen"):
            data["first_seen"] = datetime.fromisoformat(data["first_seen"])
        if data.get("last_seen"):
            data["last_seen"] = datetime.fromisoformat(data["last_seen"])
        if data.get("entity_type"):
            data["entity_type"] = EntityType(data["entity_type"])
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class PersonEntity(IdentityEntity):
    """
    A person entity - an individual inferred from behavioral patterns.

    Extends IdentityEntity with person-specific attributes.
    """
    # Person-specific attributes
    age_bracket: Optional[str] = None  # 'child', 'teen', 'adult', 'senior'
    gender_inference: Optional[str] = None  # Inferred, not collected
    role_in_household: Optional[str] = None  # 'primary', 'secondary', 'child'

    # Content profile
    binge_tendency: float = 0.0  # 0-1, how often they binge watch
    variety_seeking: float = 0.0  # 0-1, genre diversity
    completion_rate: float = 0.0  # 0-1, how often they finish content

    # Engagement metrics
    avg_session_duration: float = 0.0
    sessions_per_week: float = 0.0

    def __post_init__(self):
        super().__post_init__()
        self.entity_type = EntityType.PERSON

    def infer_age_bracket_from_content(self) -> str:
        """Infer age bracket from genre preferences."""
        kids_genres = {'Animation', 'Kids', 'Family'}
        teen_genres = {'Action', 'SciFi', 'Fantasy', 'Animation'}
        adult_genres = {'Drama', 'Documentary', 'Thriller', 'Romance'}

        kids_affinity = sum(self.genre_preferences.get(g, 0) for g in kids_genres)
        teen_affinity = sum(self.genre_preferences.get(g, 0) for g in teen_genres)
        adult_affinity = sum(self.genre_preferences.get(g, 0) for g in adult_genres)

        if kids_affinity > teen_affinity and kids_affinity > adult_affinity:
            return 'child'
        elif teen_affinity > adult_affinity:
            return 'teen'
        else:
            return 'adult'


@dataclass
class HouseholdEntity(IdentityEntity):
    """
    A household entity - a group of people sharing an account.

    Represents the Netflix "household" or streaming account unit.
    """
    # Household-specific attributes
    estimated_size: int = 1  # Number of people in household
    household_type: Optional[str] = None  # 'family', 'couple', 'roommates', 'single'

    # Member breakdown
    adult_count: int = 0
    child_count: int = 0
    teen_count: int = 0

    # Household viewing patterns
    shared_viewing_rate: float = 0.0  # How often multiple people watch together
    peak_concurrent_devices: int = 1

    # Subscription info
    subscription_tier: Optional[str] = None
    subscription_value: float = 0.0

    def __post_init__(self):
        super().__post_init__()
        self.entity_type = EntityType.HOUSEHOLD

    def add_member(self, person_id: str, role: str = "member") -> None:
        """Add a person to this household."""
        if person_id not in self.household_members:
            self.household_members.append(person_id)

        # Add edge
        self.add_edge(person_id, "has_member", weight=1.0, role=role)

    def get_attribution_split(self) -> Dict[str, float]:
        """
        Get attribution split across household members.

        Returns how to distribute conversion value across members.
        """
        if not self.household_members:
            return {}

        # For now, simple proportional split
        # In practice, would use session engagement to weight
        n_members = len(self.household_members)
        return {member: 1.0 / n_members for member in self.household_members}
