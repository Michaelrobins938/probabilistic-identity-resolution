"""
Probabilistic Identity Resolver

Main orchestrator that combines:
- Household inference (who is watching on a shared account)
- Cross-device linking (same person on different devices)
- Behavioral fingerprinting (pattern-based identity)

Outputs a unified identity graph with confidence scores.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set, Any
from datetime import datetime
import hashlib

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from models.streaming_event import StreamingEvent, Session, group_events_into_sessions
from models.device_profile import DeviceProfile
from models.identity_entity import IdentityEntity, EntityType, PersonEntity, HouseholdEntity
from models.household_profile import HouseholdProfile, PersonProfile
from core.identity_graph import IdentityGraph
from core.household_inference import HouseholdInferenceEngine, ClusteringConfig
from core.cross_device_linker import CrossDeviceLinker, LinkingConfig, DeviceLink


@dataclass
class ResolverConfig:
    """Configuration for the identity resolver."""
    # Session grouping
    session_gap_minutes: int = 30

    # Household inference
    clustering_config: ClusteringConfig = field(default_factory=ClusteringConfig)

    # Cross-device linking
    linking_config: LinkingConfig = field(default_factory=LinkingConfig)

    # Output options
    include_device_profiles: bool = True
    include_session_assignments: bool = True


@dataclass
class ResolutionResult:
    """Complete result of identity resolution."""
    # Core outputs
    identity_graph: IdentityGraph
    households: List[HouseholdProfile]

    # Device profiles
    device_profiles: Dict[str, DeviceProfile]

    # Cross-device links
    device_links: List[DeviceLink]

    # Statistics
    total_events: int
    total_sessions: int
    total_accounts: int
    total_persons: int
    total_devices: int

    # Timing
    resolution_time_seconds: float
    resolved_at: datetime

    def get_summary(self) -> str:
        """Get human-readable summary of resolution."""
        lines = [
            "=" * 60,
            "IDENTITY RESOLUTION SUMMARY",
            "=" * 60,
            "",
            f"Events processed:     {self.total_events:,}",
            f"Sessions created:     {self.total_sessions:,}",
            f"Accounts analyzed:    {self.total_accounts:,}",
            f"Persons identified:   {self.total_persons:,}",
            f"Devices profiled:     {self.total_devices:,}",
            f"Cross-device links:   {len(self.device_links):,}",
            "",
            f"Resolution time:      {self.resolution_time_seconds:.2f}s",
            "",
        ]

        # Household summaries
        if self.households:
            lines.append("HOUSEHOLD BREAKDOWN:")
            lines.append("-" * 40)
            for hh in self.households[:5]:  # Show first 5
                lines.append(f"  {hh.account_id}: {hh.estimated_size} people ({hh.household_type})")
            if len(self.households) > 5:
                lines.append(f"  ... and {len(self.households) - 5} more")

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "identity_graph": self.identity_graph.to_dict(),
            "households": [h.to_dict() for h in self.households],
            "device_profiles": {k: v.to_dict() for k, v in self.device_profiles.items()},
            "device_links": [
                {
                    "device_a": l.device_a,
                    "device_b": l.device_b,
                    "probability": l.probability,
                    "signals": l.signals
                }
                for l in self.device_links
            ],
            "statistics": {
                "total_events": self.total_events,
                "total_sessions": self.total_sessions,
                "total_accounts": self.total_accounts,
                "total_persons": self.total_persons,
                "total_devices": self.total_devices,
            },
            "resolution_time_seconds": self.resolution_time_seconds,
            "resolved_at": self.resolved_at.isoformat(),
        }


class ProbabilisticIdentityResolver:
    """
    Main identity resolution engine.

    Combines:
    1. Household Inference: WHO is watching on each shared account
    2. Cross-Device Linking: Same person across devices
    3. Unified Graph: All identities and relationships

    Usage:
    ```python
    resolver = ProbabilisticIdentityResolver()
    result = resolver.resolve(events)
    print(result.get_summary())
    ```
    """

    def __init__(self, config: Optional[ResolverConfig] = None):
        """
        Initialize the identity resolver.

        Parameters
        ----------
        config : ResolverConfig, optional
            Configuration for resolution
        """
        self.config = config or ResolverConfig()

        # Initialize sub-engines
        self.household_engine = HouseholdInferenceEngine(self.config.clustering_config)
        self.device_linker = CrossDeviceLinker(self.config.linking_config)

    def resolve(
        self,
        events: List[StreamingEvent]
    ) -> ResolutionResult:
        """
        Perform full identity resolution on events.

        This is the main entry point.

        Parameters
        ----------
        events : List[StreamingEvent]
            Raw streaming events to analyze

        Returns
        -------
        ResolutionResult
            Complete resolution with graph, households, and statistics
        """
        start_time = datetime.now()

        # Step 1: Group events into sessions
        sessions = group_events_into_sessions(
            events,
            session_gap_minutes=self.config.session_gap_minutes
        )

        # Step 2: Build device profiles
        device_profiles = self._build_device_profiles(sessions)

        # Step 3: Group sessions by account
        sessions_by_account = self._group_sessions_by_account(sessions)

        # Step 4: Analyze each household
        households = []
        all_person_profiles = []

        for account_id, account_sessions in sessions_by_account.items():
            household = self.household_engine.analyze_household(
                account_sessions,
                account_id
            )
            households.append(household)
            all_person_profiles.extend(household.members)

        # Step 5: Cross-device linking
        device_links = self.device_linker.link_devices(
            list(device_profiles.values()),
            sessions
        )

        # Step 6: Build identity graph
        graph = self._build_identity_graph(
            households,
            device_profiles,
            device_links,
            sessions
        )

        # Step 7: Assign sessions to persons (if enabled)
        if self.config.include_session_assignments:
            self._assign_sessions_to_persons(sessions, households)

        # Compute timing
        end_time = datetime.now()
        resolution_time = (end_time - start_time).total_seconds()

        return ResolutionResult(
            identity_graph=graph,
            households=households,
            device_profiles=device_profiles,
            device_links=device_links,
            total_events=len(events),
            total_sessions=len(sessions),
            total_accounts=len(sessions_by_account),
            total_persons=sum(h.estimated_size for h in households),
            total_devices=len(device_profiles),
            resolution_time_seconds=resolution_time,
            resolved_at=end_time
        )

    def get_attribution_ready_events(
        self,
        result: ResolutionResult,
        events: List[StreamingEvent]
    ) -> List[Dict[str, Any]]:
        """
        Convert resolved events to attribution-ready format.

        This is the integration point with the attribution engine.

        Parameters
        ----------
        result : ResolutionResult
            Resolution result
        events : List[StreamingEvent]
            Original events

        Returns
        -------
        List[Dict]
            Events with resolved person IDs and confidence
        """
        # Group events by session
        sessions = group_events_into_sessions(
            events,
            session_gap_minutes=self.config.session_gap_minutes
        )

        # Build session lookup
        session_lookup = {}
        for session in sessions:
            for event in session.events:
                session_lookup[event.event_id] = session

        # Transform events
        attribution_events = []

        for event in events:
            session = session_lookup.get(event.event_id)

            if session and session.assigned_person_id:
                resolved_person_id = session.assigned_person_id
                confidence = session.assignment_confidence
            else:
                # Fallback to account-level
                resolved_person_id = event.account_id
                confidence = 0.5

            # Get household info
            household_id = None
            for hh in result.households:
                if hh.account_id == event.account_id:
                    household_id = hh.household_id
                    break

            # Build attribution event
            attr_event = {
                # Original fields
                "event_id": event.event_id,
                "timestamp": event.timestamp.isoformat() if isinstance(event.timestamp, datetime) else event.timestamp,
                "channel": event.channel,
                "conversion_value": event.conversion_value,
                "conversion_type": event.conversion_type,

                # Resolved identity
                "user_id": resolved_person_id,  # Now resolved to person, not account
                "original_account_id": event.account_id,
                "identity_confidence": confidence,
                "household_id": household_id,

                # Context for segmentation
                "context_key": self._build_context_key(event, session),
                "device_type": event.device_type,
                "device_fingerprint": event.device_fingerprint,

                # Metadata
                "resolution_method": "household_inference" if confidence > 0.7 else "fallback",
            }

            attribution_events.append(attr_event)

        return attribution_events

    def _build_device_profiles(
        self,
        sessions: List[Session]
    ) -> Dict[str, DeviceProfile]:
        """Build device profiles from sessions."""
        # Group sessions by device
        sessions_by_device: Dict[str, List[Session]] = {}

        for session in sessions:
            fp = session.device_fingerprint
            if fp not in sessions_by_device:
                sessions_by_device[fp] = []
            sessions_by_device[fp].append(session)

        # Build profiles
        profiles = {}

        for fp, device_sessions in sessions_by_device.items():
            profile = DeviceProfile(fingerprint_id=fp)

            # Get device type from first session
            if device_sessions:
                profile.device_type = device_sessions[0].device_type

            # Update from sessions
            profile.update_from_sessions(device_sessions)

            profiles[fp] = profile

        return profiles

    def _group_sessions_by_account(
        self,
        sessions: List[Session]
    ) -> Dict[str, List[Session]]:
        """Group sessions by account ID."""
        by_account: Dict[str, List[Session]] = {}

        for session in sessions:
            account_id = session.account_id
            if account_id not in by_account:
                by_account[account_id] = []
            by_account[account_id].append(session)

        return by_account

    def _build_identity_graph(
        self,
        households: List[HouseholdProfile],
        device_profiles: Dict[str, DeviceProfile],
        device_links: List[DeviceLink],
        sessions: List[Session]
    ) -> IdentityGraph:
        """Build the complete identity graph."""
        graph = IdentityGraph()

        # Add household entities
        for household in households:
            hh_entity = HouseholdEntity(
                entity_id=household.household_id,
                entity_type=EntityType.HOUSEHOLD,
                confidence=household.size_confidence,
                label=f"Household {household.account_id}",
                account_ids=[household.account_id],
                household_members=[m.person_id for m in household.members],
                estimated_size=household.estimated_size,
                household_type=household.household_type,
            )
            graph.add_node(hh_entity)

            # Add person entities
            for member in household.members:
                person_entity = PersonEntity(
                    entity_id=member.person_id,
                    entity_type=EntityType.PERSON,
                    confidence=member.confidence,
                    label=member.label,
                    persona_type=member.persona_type,
                    account_ids=[household.account_id],
                    typical_hours=member.typical_hours,
                    genre_preferences=member.genre_affinities,
                    device_preferences=member.device_distribution,
                    session_count=member.session_count,
                    total_conversion_value=member.attributed_value,
                )
                graph.add_node(person_entity)

                # Add edge: person -> household
                graph.add_edge(
                    source_id=member.person_id,
                    target_id=household.household_id,
                    edge_type=IdentityGraph.EDGE_MEMBER_OF,
                    weight=1.0,
                    confidence=member.confidence
                )

        # Add device entities
        for fp, profile in device_profiles.items():
            device_entity = IdentityEntity(
                entity_id=fp,
                entity_type=EntityType.DEVICE,
                confidence=1.0,  # Devices are deterministic
                label=f"{profile.device_type} ({profile.room_inference})",
                device_fingerprints=[fp],
                account_ids=profile.account_ids,
                typical_hours=profile.typical_hours,
                genre_preferences=profile.genre_affinities,
            )
            graph.add_node(device_entity)

            # Add edges: person -> device (uses_device)
            for person_id, prob in profile.likely_person_ids.items():
                graph.add_edge(
                    source_id=person_id,
                    target_id=fp,
                    edge_type=IdentityGraph.EDGE_USES_DEVICE,
                    weight=prob,
                    confidence=prob
                )

        # Add cross-device links
        self.device_linker.add_links_to_graph(graph, device_links)

        return graph

    def _assign_sessions_to_persons(
        self,
        sessions: List[Session],
        households: List[HouseholdProfile]
    ) -> None:
        """Assign each session to a person using the household engine."""
        # Build account -> household lookup
        household_lookup = {h.account_id: h for h in households}

        for session in sessions:
            household = household_lookup.get(session.account_id)

            if household and household.members:
                # Use household engine to assign
                probabilities = self.household_engine.assign_session_to_person(
                    session,
                    session.account_id
                )

                if probabilities:
                    session.person_probabilities = probabilities
                    session.assigned_person_id = max(probabilities, key=probabilities.get)
                    session.assignment_confidence = max(probabilities.values())

    def _build_context_key(
        self,
        event: StreamingEvent,
        session: Optional[Session]
    ) -> str:
        """Build context key for attribution segmentation."""
        parts = []

        # Device type
        parts.append(event.device_type or "unknown")

        # Intent signal (based on engagement)
        if session:
            if session.event_count > 10 or session.total_duration > 3600:
                parts.append("high_intent")
            elif session.event_count > 3:
                parts.append("medium_intent")
            else:
                parts.append("low_intent")
        else:
            parts.append("unknown_intent")

        # Person type (if assigned)
        if session and session.assigned_person_id:
            # Get persona from household
            parts.append(f"person_{session.assigned_person_id[:4]}")

        return "_".join(parts)


# Convenience function
def resolve_identities(
    events: List[StreamingEvent],
    config: Optional[ResolverConfig] = None
) -> ResolutionResult:
    """
    Convenience function for identity resolution.

    Parameters
    ----------
    events : List[StreamingEvent]
        Events to analyze
    config : ResolverConfig, optional
        Configuration

    Returns
    -------
    ResolutionResult
        Resolution result
    """
    resolver = ProbabilisticIdentityResolver(config)
    return resolver.resolve(events)
