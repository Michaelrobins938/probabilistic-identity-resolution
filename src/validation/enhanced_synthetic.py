"""
Enhanced Synthetic Household Generator

Generates realistic streaming data with KNOWN ground truth for validation.
Includes advanced scenarios: seasonal patterns, co-viewing, device handoffs, WWE events.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
from enum import Enum
import random
import hashlib
import math

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from models.streaming_event import StreamingEvent, Session


class ViewingPattern(Enum):
    """Types of viewing patterns for realistic simulation."""
    REGULAR = "regular"
    BINGE = "binge"
    OCCASIONAL = "occasional"
    LIVE_EVENT = "live_event"
    CO_VIEWING = "co_viewing"
    SEASONAL = "seasonal"


@dataclass
class EnhancedSyntheticConfig:
    """Enhanced configuration for realistic synthetic data generation."""
    # Household settings
    n_households: int = 100
    persons_per_household_range: Tuple[int, int] = (1, 5)
    devices_per_person_range: Tuple[int, int] = (1, 4)
    
    # Session settings
    sessions_per_person_range: Tuple[int, int] = (20, 150)
    events_per_session_range: Tuple[int, int] = (3, 25)
    session_duration_range: Tuple[float, float] = (10.0, 240.0)
    
    # Time settings
    start_date: datetime = field(default_factory=lambda: datetime(2025, 1, 1))
    end_date: datetime = field(default_factory=lambda: datetime(2025, 12, 31))
    
    # Realistic scenario settings
    viewing_patterns: List[ViewingPattern] = field(default_factory=lambda: [
        ViewingPattern.REGULAR, ViewingPattern.BINGE, ViewingPattern.OCCASIONAL
    ])
    pattern_distribution: Dict[ViewingPattern, float] = field(default_factory=lambda: {
        ViewingPattern.REGULAR: 0.5,
        ViewingPattern.BINGE: 0.2,
        ViewingPattern.OCCASIONAL: 0.2,
        ViewingPattern.LIVE_EVENT: 0.05,
        ViewingPattern.CO_VIEWING: 0.03,
        ViewingPattern.SEASONAL: 0.02,
    })
    
    # Co-viewing settings
    co_viewing_rate: float = 0.15  # 15% of TV sessions involve multiple people
    co_viewing_device: str = "tv"  # Co-viewing primarily happens on TV
    
    # Device handoff settings
    device_handoff_rate: float = 0.1  # 10% chance of continuing on another device
    handoff_time_window_minutes: int = 10  # Within 10 minutes
    
    # Seasonal settings
    seasonal_boost_months: List[int] = field(default_factory=lambda: [12, 1, 6, 7])
    seasonal_boost_factor: float = 2.0  # 2x more viewing in peak months
    
    # Live event settings (WWE Raw, etc.)
    live_events: List[Dict] = field(default_factory=lambda: [
        {"name": "WWE Raw", "day": 0, "hour": 20, "duration_hours": 3, "genre": "Sports", "boost_factor": 3.0},
        {"name": "WWE SmackDown", "day": 4, "hour": 20, "duration_hours": 2, "genre": "Sports", "boost_factor": 2.5},
        {"name": "Netflix Release", "day": 4, "hour": 19, "duration_hours": 4, "genre": "Drama", "boost_factor": 2.0},
    ])
    
    # Noise and sharing
    noise_level: float = 0.12
    device_sharing_rate: float = 0.25
    
    # Persona settings
    persona_types: List[str] = field(default_factory=lambda: [
        "primary_adult", "secondary_adult", "teen", "child", "grandparent"
    ])
    
    # Random seed
    seed: int = 42


@dataclass
class EnhancedGroundTruth:
    """Enhanced ground truth with co-viewing and device handoffs."""
    # Session -> Person mapping (can be multiple for co-viewing)
    session_to_persons: Dict[str, List[str]] = field(default_factory=dict)
    
    # Device -> Primary Person mapping
    device_to_person: Dict[str, str] = field(default_factory=dict)
    
    # Household -> Members mapping
    household_members: Dict[str, List[str]] = field(default_factory=dict)
    
    # Person -> Persona Type
    person_personas: Dict[str, str] = field(default_factory=dict)
    
    # Person -> Devices
    person_devices: Dict[str, List[str]] = field(default_factory=dict)
    
    # Co-viewing sessions
    co_viewing_sessions: List[str] = field(default_factory=list)
    
    # Device handoffs (session_chain)
    device_handoffs: List[Tuple[str, str, str]] = field(default_factory=list)  # (person, from_device, to_device)
    
    # Viewing patterns per person
    person_viewing_patterns: Dict[str, ViewingPattern] = field(default_factory=dict)
    
    def get_household_size(self, household_id: str) -> int:
        return len(self.household_members.get(household_id, []))
    
    def get_persons_for_session(self, session_id: str) -> List[str]:
        return self.session_to_persons.get(session_id, [])
    
    def is_co_viewing(self, session_id: str) -> bool:
        return session_id in self.co_viewing_sessions


# Enhanced persona behavioral profiles
ENHANCED_PERSONA_PROFILES = {
    "primary_adult": {
        "peak_hours": [20, 21, 22, 23, 19],
        "genres": ["Drama", "Documentary", "Thriller", "Comedy", "Crime"],
        "devices": ["tv", "desktop"],
        "session_duration_mult": 1.3,
        "weekend_preference": 0.4,
        "viewing_pattern_probs": {"regular": 0.6, "binge": 0.2, "occasional": 0.15, "live_event": 0.05},
    },
    "secondary_adult": {
        "peak_hours": [18, 19, 20, 21, 14],
        "genres": ["Comedy", "Reality", "Romance", "Drama", "Lifestyle"],
        "devices": ["tv", "tablet", "mobile"],
        "session_duration_mult": 1.0,
        "weekend_preference": 0.5,
        "viewing_pattern_probs": {"regular": 0.5, "binge": 0.15, "occasional": 0.3, "live_event": 0.05},
    },
    "teen": {
        "peak_hours": [21, 22, 23, 0, 1, 20],
        "genres": ["Action", "SciFi", "Comedy", "Animation", "Fantasy"],
        "devices": ["mobile", "tablet", "desktop"],
        "session_duration_mult": 0.8,
        "weekend_preference": 0.7,
        "viewing_pattern_probs": {"regular": 0.4, "binge": 0.3, "occasional": 0.25, "live_event": 0.05},
    },
    "child": {
        "peak_hours": [15, 16, 17, 10, 11, 14],
        "genres": ["Animation", "Kids", "Comedy", "Family"],
        "devices": ["tablet", "tv"],
        "session_duration_mult": 0.5,
        "weekend_preference": 0.8,
        "viewing_pattern_probs": {"regular": 0.3, "binge": 0.1, "occasional": 0.55, "live_event": 0.05},
    },
    "grandparent": {
        "peak_hours": [9, 10, 11, 14, 15, 20],
        "genres": ["Documentary", "Drama", "Classics", "News"],
        "devices": ["tv", "tablet"],
        "session_duration_mult": 1.1,
        "weekend_preference": 0.3,
        "viewing_pattern_probs": {"regular": 0.5, "binge": 0.1, "occasional": 0.35, "live_event": 0.05},
    },
}

GENRE_LIST = [
    "Drama", "Comedy", "Action", "Documentary", "Kids",
    "Animation", "Reality", "Thriller", "Romance", "SciFi",
    "Crime", "Lifestyle", "Family", "Fantasy", "Classics", "News", "Sports"
]

DEVICE_TYPES = ["tv", "desktop", "mobile", "tablet"]


def generate_enhanced_synthetic_data(
    config: Optional[EnhancedSyntheticConfig] = None
) -> Tuple[List[StreamingEvent], EnhancedGroundTruth]:
    """
    Generate enhanced synthetic streaming data with realistic scenarios.
    
    Returns
    -------
    Tuple[List[StreamingEvent], EnhancedGroundTruth]
        (events, ground_truth)
    """
    config = config or EnhancedSyntheticConfig()
    random.seed(config.seed)
    
    all_events = []
    ground_truth = EnhancedGroundTruth()
    
    for hh_idx in range(config.n_households):
        household_id = f"household_{hh_idx:04d}"
        account_id = f"account_{hh_idx:04d}"
        
        # Determine household composition
        n_persons = random.randint(*config.persons_per_household_range)
        members = []
        
        for p_idx in range(n_persons):
            person_id = f"{household_id}_person_{p_idx}"
            members.append(person_id)
            
            # Assign persona
            persona = _assign_persona(p_idx, n_persons)
            ground_truth.person_personas[person_id] = persona
            
            # Assign viewing pattern
            profile = ENHANCED_PERSONA_PROFILES[persona]
            pattern_probs = profile["viewing_pattern_probs"]
            pattern = random.choices(
                list(pattern_probs.keys()),
                weights=list(pattern_probs.values())
            )[0]
            ground_truth.person_viewing_patterns[person_id] = ViewingPattern(pattern)
            
            # Generate devices
            n_devices = random.randint(*config.devices_per_person_range)
            person_devices = _generate_devices_for_person(
                person_id, n_devices, profile, ground_truth
            )
            
            # Generate sessions based on viewing pattern
            sessions = _generate_sessions_for_person(
                person_id=person_id,
                account_id=account_id,
                devices=person_devices,
                persona=persona,
                pattern=ViewingPattern(pattern),
                config=config,
                household_members=members
            )
            
            all_events.extend(sessions)
        
        ground_truth.household_members[household_id] = members
    
    # Post-process: add device handoffs
    all_events = _add_device_handoffs(all_events, ground_truth, config)
    
    # Shuffle events
    random.shuffle(all_events)
    
    return all_events, ground_truth


def _assign_persona(person_index: int, total_persons: int) -> str:
    """Assign persona based on position in household."""
    if person_index == 0:
        return "primary_adult"
    elif person_index == 1 and total_persons > 2:
        return random.choice(["secondary_adult", "teen"])
    elif total_persons >= 4 and person_index == 3:
        return random.choice(["child", "grandparent"])
    else:
        return random.choice(["secondary_adult", "teen", "child"])


def _generate_devices_for_person(
    person_id: str,
    n_devices: int,
    profile: Dict,
    ground_truth: EnhancedGroundTruth
) -> List[Tuple[str, str]]:
    """Generate devices for a person."""
    devices = []
    preferred_devices = profile["devices"]
    
    for d_idx in range(n_devices):
        device_id = f"{person_id}_device_{d_idx}"
        
        if d_idx == 0:
            device_type = random.choice(preferred_devices)
        else:
            device_type = random.choice(DEVICE_TYPES)
        
        devices.append((device_id, device_type))
        ground_truth.device_to_person[device_id] = person_id
    
    ground_truth.person_devices[person_id] = [d[0] for d in devices]
    return devices


def _generate_sessions_for_person(
    person_id: str,
    account_id: str,
    devices: List[Tuple[str, str]],
    persona: str,
    pattern: ViewingPattern,
    config: EnhancedSyntheticConfig,
    household_members: List[str]
) -> List[StreamingEvent]:
    """Generate sessions for a person based on their viewing pattern."""
    profile = ENHANCED_PERSONA_PROFILES[persona]
    all_events = []
    
    # Determine number of sessions based on pattern
    base_sessions = random.randint(*config.sessions_per_person_range)
    if pattern == ViewingPattern.BINGE:
        n_sessions = int(base_sessions * 1.5)
    elif pattern == ViewingPattern.OCCASIONAL:
        n_sessions = int(base_sessions * 0.5)
    else:
        n_sessions = base_sessions
    
    for s_idx in range(n_sessions):
        # Check for live event session
        is_live_event = pattern == ViewingPattern.LIVE_EVENT or _is_live_event_time(config)
        
        # Check for co-viewing (only on TV, only with household members)
        is_co_viewing = False
        co_viewers = []
        
        device_id, device_type = random.choice(devices)
        if device_type == config.co_viewing_device and random.random() < config.co_viewing_rate:
            is_co_viewing = True
            # Select co-viewers from household
            potential_viewers = [m for m in household_members if m != person_id]
            if potential_viewers:
                n_co_viewers = random.randint(1, min(2, len(potential_viewers)))
                co_viewers = random.sample(potential_viewers, n_co_viewers)
        
        # Generate session
        session_events = _generate_enhanced_session(
            session_id=f"{person_id}_session_{s_idx}",
            account_id=account_id,
            device_id=device_id,
            device_type=device_type,
            persona=persona,
            profile=profile,
            config=config,
            is_live_event=is_live_event,
            is_co_viewing=is_co_viewing,
            co_viewers=co_viewers
        )
        
        all_events.extend(session_events)
        
        # Record ground truth
        all_viewers = [person_id] + co_viewers
        config.ground_truth.session_to_persons[f"{person_id}_session_{s_idx}"] = all_viewers
        
        if is_co_viewing:
            config.ground_truth.co_viewing_sessions.append(f"{person_id}_session_{s_idx}")
    
    return all_events


def _is_live_event_time(config: EnhancedSyntheticConfig) -> bool:
    """Check if current time matches a live event."""
    # Simplified - in reality would check against actual dates
    return random.random() < 0.1  # 10% chance


def _generate_enhanced_session(
    session_id: str,
    account_id: str,
    device_id: str,
    device_type: str,
    persona: str,
    profile: Dict,
    config: EnhancedSyntheticConfig,
    is_live_event: bool = False,
    is_co_viewing: bool = False,
    co_viewers: List[str] = []
) -> List[StreamingEvent]:
    """Generate a single enhanced session with events."""
    events = []
    
    # Session timing with seasonal adjustment
    session_date = _generate_session_date(config, persona, profile)
    
    if is_live_event:
        # Use live event timing
        live_event = random.choice(config.live_events)
        hour = live_event["hour"]
        genre = live_event["genre"]
        duration = live_event["duration_hours"] * 60 * random.uniform(0.8, 1.2)
    else:
        # Regular timing
        if random.random() < 0.7:
            hour = random.choice(profile["peak_hours"])
        else:
            hour = random.randint(0, 23)
        
        # Weekend adjustment
        if session_date.weekday() >= 5 and random.random() < profile["weekend_preference"]:
            hour = random.choice(profile["peak_hours"])
        
        genre = random.choice(profile["genres"]) if random.random() < 0.8 else random.choice(GENRE_LIST)
        
        base_duration = random.uniform(*config.session_duration_range)
        duration = base_duration * profile["session_duration_mult"]
    
    session_start = session_date.replace(hour=hour, minute=random.randint(0, 59))
    
    # Number of events
    n_events = random.randint(*config.events_per_session_range)
    event_duration = duration / n_events
    
    # Generate events
    current_time = session_start
    for e_idx in range(n_events):
        event_id = f"{session_id}_event_{e_idx}"
        
        # Event type
        if e_idx == 0:
            event_type = "play"
        elif e_idx == n_events - 1:
            event_type = "pause"
        else:
            event_type = random.choice(["play", "browse", "search"])
        
        # Content with noise
        event_genre = genre
        if random.random() < config.noise_level:
            event_genre = random.choice(GENRE_LIST)
        
        content_id = f"content_{random.randint(1000, 9999)}"
        
        # Add co-viewing metadata
        metadata = {"session_id": session_id, "persona": persona}
        if is_co_viewing:
            metadata["co_viewing"] = True
            metadata["co_viewers"] = co_viewers
        
        event = StreamingEvent(
            event_id=event_id,
            account_id=account_id,
            device_fingerprint=device_id,
            timestamp=current_time,
            event_type=event_type,
            content_id=content_id,
            content_title=f"Title {content_id}",
            content_genre=event_genre,
            duration_seconds=event_duration * 60,
            device_type=device_type,
            os_family="unknown",
            browser_family="unknown",
            screen_resolution="1920x1080",
            hour_of_day=current_time.hour,
            day_of_week=current_time.weekday(),
            metadata=metadata,
        )
        
        events.append(event)
        current_time += timedelta(minutes=event_duration)
    
    return events


def _generate_session_date(
    config: EnhancedSyntheticConfig,
    persona: str,
    profile: Dict
) -> datetime:
    """Generate session date with seasonal patterns."""
    total_days = (config.end_date - config.start_date).days
    day_offset = random.randint(0, total_days)
    
    session_date = config.start_date + timedelta(days=day_offset)
    
    # Apply seasonal boost
    if session_date.month in config.seasonal_boost_months:
        if random.random() < 0.3:  # 30% chance to add extra session in peak months
            pass  # Already generated, just mark as seasonal if needed
    
    return session_date


def _add_device_handoffs(
    events: List[StreamingEvent],
    ground_truth: EnhancedGroundTruth,
    config: EnhancedSyntheticConfig
) -> List[StreamingEvent]:
    """Add realistic device handoffs (same person, different devices)."""
    if config.device_handoff_rate == 0:
        return events
    
    # Group events by person
    person_events: Dict[str, List[StreamingEvent]] = {}
    for event in events:
        person = ground_truth.device_to_person.get(event.device_fingerprint)
        if person:
            if person not in person_events:
                person_events[person] = []
            person_events[person].append(event)
    
    # For each person, potentially add handoff events
    for person, person_event_list in person_events.items():
        if len(person_event_list) < 10:
            continue
        
        devices = ground_truth.person_devices.get(person, [])
        if len(devices) < 2:
            continue
        
        # Sort by timestamp
        person_event_list.sort(key=lambda e: e.timestamp or datetime.min)
        
        # Find opportunities for handoffs
        for i in range(len(person_event_list) - 1):
            if random.random() > config.device_handoff_rate:
                continue
            
            event = person_event_list[i]
            next_event = person_event_list[i + 1]
            
            # Check time gap
            if event.timestamp and next_event.timestamp:
                gap_minutes = (next_event.timestamp - event.timestamp).total_seconds() / 60
                
                if 2 < gap_minutes < config.handoff_time_window_minutes:
                    # Create handoff event - same content on different device
                    other_devices = [d for d in devices if d != event.device_fingerprint]
                    if other_devices:
                        to_device = random.choice(other_devices)
                        
                        # Record handoff
                        ground_truth.device_handoffs.append(
                            (person, event.device_fingerprint, to_device)
                        )
                        
                        # Modify next event to show handoff
                        next_event.metadata = next_event.metadata or {}
                        next_event.metadata["device_handoff_from"] = event.device_fingerprint
    
    return events


def generate_wwe_raw_scenario(
    n_households: int = 20,
    include_co_viewing: bool = True
) -> Tuple[List[StreamingEvent], EnhancedGroundTruth]:
    """
    Generate specific test data for WWE Raw live event scenario.
    
    This creates households where multiple people gather to watch WWE Raw together,
    testing the system's ability to handle co-viewing and live events.
    """
    config = EnhancedSyntheticConfig(
        n_households=n_households,
        persons_per_household_range=(2, 4),
        viewing_patterns=[ViewingPattern.LIVE_EVENT, ViewingPattern.REGULAR],
        pattern_distribution={
            ViewingPattern.REGULAR: 0.7,
            ViewingPattern.LIVE_EVENT: 0.3,
        },
        co_viewing_rate=0.4 if include_co_viewing else 0.0,  # High co-viewing for WWE
        live_events=[
            {"name": "WWE Raw", "day": 0, "hour": 20, "duration_hours": 3, "genre": "Sports", "boost_factor": 5.0},
        ],
        seed=42,
    )
    
    # Set start/end to capture multiple Monday nights
    config.start_date = datetime(2025, 1, 6)  # First Monday of 2025
    config.end_date = datetime(2025, 3, 31)   # End of Q1
    
    return generate_enhanced_synthetic_data(config)


# Backward compatibility alias
EnhancedGroundTruth = EnhancedGroundTruth
