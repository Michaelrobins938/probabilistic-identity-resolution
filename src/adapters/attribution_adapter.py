"""
Attribution Adapter

Transforms identity-resolved data into format compatible with
the first-principles-attribution engine.

This is the integration layer that connects:
- Identity Resolution (this project)
- Attribution Engine (first-principles-attribution-repo)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
import json

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from models.streaming_event import StreamingEvent, Session
from models.household_profile import HouseholdProfile
from core.probabilistic_resolver import ResolutionResult


@dataclass
class AttributionEvent:
    """
    Event format compatible with first-principles-attribution engine.

    Maps to the UniversalEvent schema from the attribution engine.
    """
    # Core fields (required by attribution engine)
    timestamp: str                     # ISO8601
    user_id: str                       # Resolved person ID (not account!)
    channel: str                       # Marketing channel
    context_key: str                   # Psychographic context

    # Conversion tracking
    conversion_value: float = 0.0
    event_type: str = "touchpoint"     # 'touchpoint' or 'conversion'

    # Identity resolution metadata
    identity_confidence: float = 1.0   # How confident in person assignment
    household_id: Optional[str] = None
    original_account_id: Optional[str] = None
    device_fingerprint: Optional[str] = None

    # Additional context
    device_type: str = "unknown"
    persona_type: Optional[str] = None  # 'primary_adult', 'child', etc.

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for attribution engine."""
        return {
            "timestamp": self.timestamp,
            "user_id": self.user_id,
            "channel": self.channel,
            "context_key": self.context_key,
            "conversion_value": self.conversion_value,
            "event_type": self.event_type,
            "identity_confidence": self.identity_confidence,
            "household_id": self.household_id,
            "original_account_id": self.original_account_id,
            "device_fingerprint": self.device_fingerprint,
            "device_type": self.device_type,
            "persona_type": self.persona_type,
            "metadata": self.metadata,
        }

    def to_attribution_engine_format(self) -> Dict[str, Any]:
        """
        Convert to exact format expected by attribution.js.

        Matches the schema in first-principles-attribution-repo.
        """
        return {
            "timestamp": self.timestamp,
            "user_id": self.user_id,
            "channel": self.channel or "Direct",
            "context_key": self.context_key,
            "conversion_value": self.conversion_value,
            "fingerprint": self.device_fingerprint,
            # Extension fields
            "identity_metadata": {
                "confidence": self.identity_confidence,
                "household_id": self.household_id,
                "original_account_id": self.original_account_id,
                "persona_type": self.persona_type,
                "resolution_method": "probabilistic_identity_resolution",
            }
        }


class AttributionAdapter:
    """
    Adapter to transform identity-resolved data for attribution analysis.

    Provides two main functions:
    1. Convert resolved identities to attribution events
    2. Run segment-specific attribution (by persona, device, household)
    """

    def __init__(self, resolution_result: ResolutionResult):
        """
        Initialize adapter with resolution result.

        Parameters
        ----------
        resolution_result : ResolutionResult
            Output from ProbabilisticIdentityResolver.resolve()
        """
        self.result = resolution_result
        self._household_lookup = {h.account_id: h for h in resolution_result.households}
        self._person_lookup = {}

        # Build person lookup
        for household in resolution_result.households:
            for member in household.members:
                self._person_lookup[member.person_id] = {
                    "household_id": household.household_id,
                    "persona_type": member.persona_type,
                    "attribution_share": member.attribution_share,
                }

    def to_attribution_events(
        self,
        events: List[StreamingEvent],
        sessions: List[Session]
    ) -> List[AttributionEvent]:
        """
        Convert streaming events to attribution-ready format.

        Parameters
        ----------
        events : List[StreamingEvent]
            Original streaming events
        sessions : List[Session]
            Sessions with person assignments

        Returns
        -------
        List[AttributionEvent]
            Events ready for attribution engine
        """
        # Build session lookup by event
        event_to_session: Dict[str, Session] = {}
        for session in sessions:
            for event in session.events:
                event_to_session[event.event_id] = session

        attribution_events = []

        for event in events:
            session = event_to_session.get(event.event_id)

            # Determine resolved person
            if session and session.assigned_person_id:
                resolved_person_id = session.assigned_person_id
                confidence = session.assignment_confidence
            else:
                # Fallback to account
                resolved_person_id = event.account_id
                confidence = 0.5

            # Get person info
            person_info = self._person_lookup.get(resolved_person_id, {})

            # Build context key
            context_key = self._build_context_key(
                event=event,
                session=session,
                person_info=person_info
            )

            # Create attribution event
            attr_event = AttributionEvent(
                timestamp=event.timestamp.isoformat() if isinstance(event.timestamp, datetime) else str(event.timestamp),
                user_id=resolved_person_id,
                channel=event.channel or "Direct",
                context_key=context_key,
                conversion_value=event.conversion_value,
                event_type="conversion" if event.conversion_value > 0 else "touchpoint",
                identity_confidence=confidence,
                household_id=person_info.get("household_id"),
                original_account_id=event.account_id,
                device_fingerprint=event.device_fingerprint,
                device_type=event.device_type,
                persona_type=person_info.get("persona_type"),
                metadata={
                    "content_genre": event.content_genre,
                    "content_id": event.content_id,
                    "hour_of_day": event.hour_of_day,
                    "day_of_week": event.day_of_week,
                }
            )

            attribution_events.append(attr_event)

        return attribution_events

    def _build_context_key(
        self,
        event: StreamingEvent,
        session: Optional[Session],
        person_info: Dict[str, Any]
    ) -> str:
        """
        Build context key for psychographic weighting.

        Format: {device}_{intent}_{persona}
        Example: "desktop_high_intent_primary_adult"
        """
        parts = []

        # Device type
        device = event.device_type or "unknown"
        parts.append(device)

        # Intent signal (based on engagement)
        if session:
            if session.event_count > 10 or session.total_duration > 3600:
                intent = "high_intent"
            elif session.event_count > 3:
                intent = "medium_intent"
            else:
                intent = "low_intent"
        else:
            intent = "unknown_intent"
        parts.append(intent)

        # Persona type
        persona = person_info.get("persona_type", "unknown")
        parts.append(persona)

        return "_".join(parts)

    def get_segment_attribution_data(self) -> Dict[str, List[AttributionEvent]]:
        """
        Get attribution data segmented by persona type.

        Returns events grouped by persona for segment-specific analysis.

        Returns
        -------
        Dict[str, List[AttributionEvent]]
            Persona type -> events for that segment
        """
        segments: Dict[str, List[AttributionEvent]] = {}

        # This would need the original events to work
        # For now, return structure for illustration
        for household in self.result.households:
            for member in household.members:
                persona = member.persona_type
                if persona not in segments:
                    segments[persona] = []

        return segments

    def get_household_attribution_summary(self) -> List[Dict[str, Any]]:
        """
        Get attribution summary by household.

        Shows how conversion value should be split across household members.

        Returns
        -------
        List[Dict]
            Per-household attribution breakdown
        """
        summaries = []

        for household in self.result.households:
            summary = {
                "household_id": household.household_id,
                "account_id": household.account_id,
                "household_type": household.household_type,
                "total_conversion_value": household.total_conversion_value,
                "estimated_size": household.estimated_size,
                "members": []
            }

            for member in household.members:
                member_summary = {
                    "person_id": member.person_id,
                    "label": member.label,
                    "persona_type": member.persona_type,
                    "attribution_share": member.attribution_share,
                    "attributed_value": member.attribution_share * household.total_conversion_value,
                    "session_count": member.session_count,
                    "primary_device": member.primary_device_type,
                    "top_genres": member.top_genres[:3],
                }
                summary["members"].append(member_summary)

            summaries.append(summary)

        return summaries

    def export_for_attribution_engine(
        self,
        events: List[StreamingEvent],
        sessions: List[Session],
        output_path: Optional[str] = None
    ) -> str:
        """
        Export events in JSON format for attribution engine.

        Parameters
        ----------
        events : List[StreamingEvent]
            Original events
        sessions : List[Session]
            Sessions with assignments
        output_path : str, optional
            Path to write JSON file

        Returns
        -------
        str
            JSON string of attribution events
        """
        attribution_events = self.to_attribution_events(events, sessions)

        output = {
            "events": [e.to_attribution_engine_format() for e in attribution_events],
            "metadata": {
                "total_events": len(attribution_events),
                "resolution_method": "probabilistic_identity_resolution",
                "resolved_at": self.result.resolved_at.isoformat(),
                "statistics": {
                    "total_households": len(self.result.households),
                    "total_persons": self.result.total_persons,
                    "total_devices": self.result.total_devices,
                }
            }
        }

        json_str = json.dumps(output, indent=2, default=str)

        if output_path:
            with open(output_path, 'w') as f:
                f.write(json_str)

        return json_str


def convert_to_attribution_format(
    resolution_result: ResolutionResult,
    events: List[StreamingEvent],
    sessions: List[Session]
) -> List[Dict[str, Any]]:
    """
    Convenience function to convert resolved data to attribution format.

    Parameters
    ----------
    resolution_result : ResolutionResult
        Output from identity resolution
    events : List[StreamingEvent]
        Original events
    sessions : List[Session]
        Sessions with person assignments

    Returns
    -------
    List[Dict]
        Events in attribution engine format
    """
    adapter = AttributionAdapter(resolution_result)
    attribution_events = adapter.to_attribution_events(events, sessions)
    return [e.to_attribution_engine_format() for e in attribution_events]
