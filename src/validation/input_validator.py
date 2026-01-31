"""
Input Validation Layer

Comprehensive input validation for identity resolution system.
Prevents silent failures and provides clear error messages.
"""

from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from dataclasses import dataclass
import re

from models.streaming_event import StreamingEvent, Session
from models.household_profile import HouseholdProfile
from models.device_profile import DeviceProfile


class ValidationError(Exception):
    """Raised when input validation fails."""
    pass


@dataclass
class ValidationResult:
    """Result of input validation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    
    def __init__(self):
        self.is_valid = True
        self.errors = []
        self.warnings = []
    
    def add_error(self, message: str):
        self.is_valid = False
        self.errors.append(message)
    
    def add_warning(self, message: str):
        self.warnings.append(message)


class StreamingEventValidator:
    """Validates streaming events."""
    
    REQUIRED_FIELDS = [
        'event_id', 'account_id', 'device_fingerprint', 
        'timestamp', 'event_type', 'content_id'
    ]
    
    VALID_EVENT_TYPES = [
        'play', 'pause', 'browse', 'search', 'click', 
        'conversion', 'impression', 'ad_click'
    ]
    
    VALID_DEVICE_TYPES = ['tv', 'desktop', 'mobile', 'tablet']
    
    @classmethod
    def validate(cls, event: StreamingEvent) -> ValidationResult:
        """Validate a single streaming event."""
        result = ValidationResult()
        
        # Check required fields
        for field in cls.REQUIRED_FIELDS:
            value = getattr(event, field, None)
            if value is None or value == '':
                result.add_error(f"Missing required field: {field}")
        
        # Validate event_type
        if event.event_type not in cls.VALID_EVENT_TYPES:
            result.add_warning(
                f"Unknown event_type: {event.event_type}. "
                f"Expected one of: {cls.VALID_EVENT_TYPES}"
            )
        
        # Validate device_type
        if event.device_type not in cls.VALID_DEVICE_TYPES:
            result.add_warning(
                f"Unknown device_type: {event.device_type}. "
                f"Expected one of: {cls.VALID_DEVICE_TYPES}"
            )
        
        # Validate timestamp
        if event.timestamp:
            if not isinstance(event.timestamp, datetime):
                result.add_error(f"timestamp must be datetime, got {type(event.timestamp)}")
            elif event.timestamp.year < 2020 or event.timestamp.year > 2030:
                result.add_warning(
                    f"Suspicious timestamp year: {event.timestamp.year}"
                )
        
        # Validate duration
        if event.duration_seconds is not None:
            if event.duration_seconds < 0:
                result.add_error(f"duration_seconds cannot be negative: {event.duration_seconds}")
            elif event.duration_seconds > 86400:  # 24 hours
                result.add_warning(
                    f"Unusually long duration: {event.duration_seconds}s (>24h)"
                )
        
        # Validate IDs are not empty and reasonable length
        if event.event_id and len(event.event_id) > 256:
            result.add_error(f"event_id too long: {len(event.event_id)} chars (max 256)")
        
        if event.account_id and len(event.account_id) > 128:
            result.add_error(f"account_id too long: {len(event.account_id)} chars (max 128)")
        
        # Check for suspicious patterns
        if event.account_id and re.search(r'[<>\"\'&]', event.account_id):
            result.add_error("account_id contains suspicious characters (possible injection)")
        
        return result
    
    @classmethod
    def validate_batch(cls, events: List[StreamingEvent]) -> ValidationResult:
        """Validate a batch of events."""
        result = ValidationResult()
        
        if not events:
            result.add_error("Empty event list")
            return result
        
        if len(events) > 1000000:
            result.add_warning(f"Very large batch: {len(events)} events (may impact performance)")
        
        # Check for duplicate event IDs
        event_ids = [e.event_id for e in events if e.event_id]
        duplicates = set([x for x in event_ids if event_ids.count(x) > 1])
        if duplicates:
            result.add_warning(f"Duplicate event IDs found: {len(duplicates)} duplicates")
        
        # Validate each event
        for i, event in enumerate(events):
            event_result = cls.validate(event)
            if not event_result.is_valid:
                for error in event_result.errors:
                    result.add_error(f"Event {i} ({event.event_id}): {error}")
            for warning in event_result.warnings:
                result.add_warning(f"Event {i}: {warning}")
        
        return result


class SessionValidator:
    """Validates sessions."""
    
    @classmethod
    def validate(cls, session: Session) -> ValidationResult:
        """Validate a single session."""
        result = ValidationResult()
        
        # Check required fields
        if not session.session_id:
            result.add_error("Missing session_id")
        
        if not session.account_id:
            result.add_error("Missing account_id")
        
        if not session.device_fingerprint:
            result.add_error("Missing device_fingerprint")
        
        # Validate timestamps
        if session.start_time and session.end_time:
            if session.end_time < session.start_time:
                result.add_error("end_time before start_time")
            
            duration = (session.end_time - session.start_time).total_seconds()
            if duration > 86400 * 7:  # 7 days
                result.add_warning(f"Very long session: {duration/3600:.1f} hours")
        
        # Validate event count
        if session.event_count < 0:
            result.add_error(f"event_count cannot be negative: {session.event_count}")
        
        if session.event_count > 10000:
            result.add_warning(f"Very high event count: {session.event_count}")
        
        # Validate genres_watched
        if session.genres_watched:
            total_genre_time = sum(session.genres_watched.values())
            if total_genre_time > session.total_duration * 10:  # More than 10x duration
                result.add_warning(
                    f"Genre times ({total_genre_time}) much larger than session duration "
                    f"({session.total_duration})"
                )
        
        # Validate person probabilities if present
        if session.person_probabilities:
            probs = list(session.person_probabilities.values())
            if abs(sum(probs) - 1.0) > 0.01:
                result.add_error(
                    f"Person probabilities don't sum to 1.0: {sum(probs):.3f}"
                )
            
            for person_id, prob in session.person_probabilities.items():
                if prob < 0 or prob > 1:
                    result.add_error(
                        f"Invalid probability for {person_id}: {prob} (must be 0-1)"
                    )
        
        # Validate assignment confidence
        if session.assignment_confidence is not None:
            if session.assignment_confidence < 0 or session.assignment_confidence > 1:
                result.add_error(
                    f"assignment_confidence must be 0-1: {session.assignment_confidence}"
                )
        
        return result
    
    @classmethod
    def validate_batch(cls, sessions: List[Session]) -> ValidationResult:
        """Validate a batch of sessions."""
        result = ValidationResult()
        
        if not sessions:
            result.add_error("Empty session list")
            return result
        
        # Check for duplicate session IDs
        session_ids = [s.session_id for s in sessions if s.session_id]
        duplicates = set([x for x in session_ids if session_ids.count(x) > 1])
        if duplicates:
            result.add_error(f"Duplicate session IDs: {len(duplicates)} duplicates")
        
        # Validate each session
        for i, session in enumerate(sessions):
            session_result = cls.validate(session)
            if not session_result.is_valid:
                for error in session_result.errors:
                    result.add_error(f"Session {i} ({session.session_id}): {error}")
            for warning in session_result.warnings:
                result.add_warning(f"Session {i}: {warning}")
        
        # Check account consistency
        accounts = set(s.account_id for s in sessions if s.account_id)
        if len(accounts) > 1:
            result.add_warning(
                f"Sessions from multiple accounts: {len(accounts)} accounts"
            )
        
        return result


class HouseholdProfileValidator:
    """Validates household profiles."""
    
    @classmethod
    def validate(cls, household: HouseholdProfile) -> ValidationResult:
        """Validate a household profile."""
        result = ValidationResult()
        
        if not household.household_id:
            result.add_error("Missing household_id")
        
        if not household.account_id:
            result.add_error("Missing account_id")
        
        # Validate size
        if household.estimated_size < 1:
            result.add_error(f"estimated_size must be >= 1: {household.estimated_size}")
        elif household.estimated_size > 20:
            result.add_warning(
                f"Very large household: {household.estimated_size} members"
            )
        
        # Validate confidence
        if household.size_confidence < 0 or household.size_confidence > 1:
            result.add_error(
                f"size_confidence must be 0-1: {household.size_confidence}"
            )
        
        # Check member consistency
        actual_members = len(household.members)
        if actual_members != household.estimated_size:
            result.add_warning(
                f"Member count mismatch: estimated={household.estimated_size}, "
                f"actual={actual_members}"
            )
        
        # Validate member profiles
        for member in household.members:
            if not member.person_id:
                result.add_error("Member missing person_id")
            
            if member.household_id != household.household_id:
                result.add_error(
                    f"Member household_id mismatch: {member.household_id} != "
                    f"{household.household_id}"
                )
            
            if member.session_count < 0:
                result.add_error(
                    f"Member {member.person_id} has negative session_count"
                )
        
        # Check device count
        if household.device_count > household.estimated_size * 5:
            result.add_warning(
                f"High device-to-person ratio: {household.device_count} devices / "
                f"{household.estimated_size} persons"
            )
        
        # Validate attribution shares sum to 1
        if household.attribution_shares:
            total_share = sum(household.attribution_shares.values())
            if abs(total_share - 1.0) > 0.01:
                result.add_error(
                    f"Attribution shares don't sum to 1.0: {total_share:.3f}"
                )
        
        return result


class DeviceProfileValidator:
    """Validates device profiles."""
    
    @classmethod
    def validate(cls, device: DeviceProfile) -> ValidationResult:
        """Validate a device profile."""
        result = ValidationResult()
        
        if not device.fingerprint_id:
            result.add_error("Missing fingerprint_id")
        
        if device.total_sessions < 0:
            result.add_error(f"total_sessions cannot be negative: {device.total_sessions}")
        
        # Validate time distributions sum to ~1
        if device.hour_distribution:
            total = sum(device.hour_distribution.values())
            if abs(total - 1.0) > 0.1:
                result.add_warning(f"Hour distribution doesn't sum to 1.0: {total:.3f}")
        
        if device.day_distribution:
            total = sum(device.day_distribution.values())
            if abs(total - 1.0) > 0.1:
                result.add_warning(f"Day distribution doesn't sum to 1.0: {total:.3f}")
        
        if device.genre_affinities:
            total = sum(device.genre_affinities.values())
            if abs(total - 1.0) > 0.1:
                result.add_warning(f"Genre affinities don't sum to 1.0: {total:.3f}")
        
        return result


# Convenience functions

def validate_events(events: List[StreamingEvent], raise_on_error: bool = False) -> ValidationResult:
    """
    Validate a list of streaming events.
    
    Parameters
    ----------
    events : List[StreamingEvent]
        Events to validate
    raise_on_error : bool
        If True, raise ValidationError on invalid input
    
    Returns
    -------
    ValidationResult
        Validation results with errors and warnings
    
    Raises
    ------
    ValidationError
        If raise_on_error=True and validation fails
    """
    result = StreamingEventValidator.validate_batch(events)
    
    if raise_on_error and not result.is_valid:
        raise ValidationError(
            f"Validation failed with {len(result.errors)} errors: "
            f"{'; '.join(result.errors[:3])}"
        )
    
    return result


def validate_sessions(sessions: List[Session], raise_on_error: bool = False) -> ValidationResult:
    """Validate a list of sessions."""
    result = SessionValidator.validate_batch(sessions)
    
    if raise_on_error and not result.is_valid:
        raise ValidationError(
            f"Validation failed with {len(result.errors)} errors"
        )
    
    return result


def validate_household(household: HouseholdProfile, raise_on_error: bool = False) -> ValidationResult:
    """Validate a household profile."""
    result = HouseholdProfileValidator.validate(household)
    
    if raise_on_error and not result.is_valid:
        raise ValidationError(
            f"Validation failed with {len(result.errors)} errors"
        )
    
    return result
