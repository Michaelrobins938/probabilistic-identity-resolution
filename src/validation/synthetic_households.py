"""
Synthetic Household Data Generator

Generates realistic streaming data with KNOWN ground truth for validation.

Key insight: We create data where we KNOW:
- How many people are in each household
- Which sessions belong to which person
- Which devices belong to which person

Then we measure how well our inference recovers this ground truth.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import random
import hashlib
import math

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from models.streaming_event import StreamingEvent, Session


@dataclass
class SyntheticConfig:
    """Configuration for synthetic data generation."""
    # Household settings
    n_households: int = 50
    persons_per_household_range: Tuple[int, int] = (1, 4)  # min, max
    devices_per_person_range: Tuple[int, int] = (1, 3)

    # Session settings
    sessions_per_person_range: Tuple[int, int] = (30, 100)
    events_per_session_range: Tuple[int, int] = (3, 20)
    session_duration_range: Tuple[float, float] = (15.0, 180.0)  # minutes

    # Time settings
    start_date: datetime = field(default_factory=lambda: datetime(2025, 1, 1))
    end_date: datetime = field(default_factory=lambda: datetime(2025, 6, 30))

    # Noise settings
    noise_level: float = 0.15  # Probability of random perturbation
    device_sharing_rate: float = 0.2  # Rate of shared device usage

    # Persona settings
    persona_types: List[str] = field(default_factory=lambda: [
        "primary_adult", "secondary_adult", "teen", "child"
    ])

    # Random seed
    seed: int = 42


@dataclass
class GroundTruth:
    """Ground truth for validation."""
    # Session -> Person mapping
    session_to_person: Dict[str, str] = field(default_factory=dict)

    # Device -> Primary Person mapping
    device_to_person: Dict[str, str] = field(default_factory=dict)

    # Household -> Members mapping
    household_members: Dict[str, List[str]] = field(default_factory=dict)

    # Person -> Persona Type
    person_personas: Dict[str, str] = field(default_factory=dict)

    # Cross-device links (person uses multiple devices)
    person_devices: Dict[str, List[str]] = field(default_factory=dict)

    def get_household_size(self, household_id: str) -> int:
        """Get true household size."""
        return len(self.household_members.get(household_id, []))

    def get_person_for_session(self, session_id: str) -> Optional[str]:
        """Get true person for a session."""
        return self.session_to_person.get(session_id)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize ground truth."""
        return {
            "session_to_person": self.session_to_person,
            "device_to_person": self.device_to_person,
            "household_members": self.household_members,
            "person_personas": self.person_personas,
            "person_devices": self.person_devices,
        }


# Persona behavioral profiles
PERSONA_PROFILES = {
    "primary_adult": {
        "peak_hours": [20, 21, 22, 23],
        "genres": ["Drama", "Documentary", "Thriller", "Comedy"],
        "devices": ["tv", "desktop"],
        "session_duration_mult": 1.2,
        "weekend_preference": 0.4,
    },
    "secondary_adult": {
        "peak_hours": [18, 19, 20, 21],
        "genres": ["Comedy", "Reality", "Romance", "Drama"],
        "devices": ["tv", "tablet"],
        "session_duration_mult": 1.0,
        "weekend_preference": 0.5,
    },
    "teen": {
        "peak_hours": [21, 22, 23, 0, 1],
        "genres": ["Action", "SciFi", "Comedy", "Animation"],
        "devices": ["mobile", "tablet", "desktop"],
        "session_duration_mult": 0.8,
        "weekend_preference": 0.6,
    },
    "child": {
        "peak_hours": [15, 16, 17, 18, 10, 11],
        "genres": ["Animation", "Kids", "Comedy"],
        "devices": ["tablet", "tv"],
        "session_duration_mult": 0.6,
        "weekend_preference": 0.7,
    },
}

GENRE_LIST = [
    "Drama", "Comedy", "Action", "Documentary", "Kids",
    "Animation", "Reality", "Thriller", "Romance", "SciFi"
]

DEVICE_TYPES = ["tv", "desktop", "mobile", "tablet"]


def generate_synthetic_household_data(
    config: Optional[SyntheticConfig] = None
) -> Tuple[List[StreamingEvent], GroundTruth]:
    """
    Generate synthetic streaming data with known ground truth.

    Parameters
    ----------
    config : SyntheticConfig, optional
        Configuration for generation

    Returns
    -------
    Tuple[List[StreamingEvent], GroundTruth]
        (events, ground_truth)
    """
    config = config or SyntheticConfig()
    random.seed(config.seed)

    all_events = []
    ground_truth = GroundTruth()

    for hh_idx in range(config.n_households):
        household_id = f"household_{hh_idx:04d}"
        account_id = f"account_{hh_idx:04d}"

        # Determine household composition
        n_persons = random.randint(*config.persons_per_household_range)
        members = []

        for p_idx in range(n_persons):
            person_id = f"{household_id}_person_{p_idx}"
            members.append(person_id)

            # Assign persona (first person is primary adult, etc.)
            if p_idx == 0:
                persona = "primary_adult"
            elif p_idx == 1 and n_persons > 2:
                persona = "secondary_adult"
            elif random.random() < 0.3:
                persona = "child"
            else:
                persona = random.choice(["secondary_adult", "teen"])

            ground_truth.person_personas[person_id] = persona

            # Generate devices for this person
            n_devices = random.randint(*config.devices_per_person_range)
            person_devices = []

            profile = PERSONA_PROFILES[persona]
            preferred_devices = profile["devices"]

            for d_idx in range(n_devices):
                device_id = f"{person_id}_device_{d_idx}"

                # Prefer persona's typical devices
                if d_idx == 0:
                    device_type = random.choice(preferred_devices)
                else:
                    device_type = random.choice(DEVICE_TYPES)

                person_devices.append((device_id, device_type))
                ground_truth.device_to_person[device_id] = person_id

            ground_truth.person_devices[person_id] = [d[0] for d in person_devices]

            # Generate sessions for this person
            n_sessions = random.randint(*config.sessions_per_person_range)

            for s_idx in range(n_sessions):
                session_id = f"{person_id}_session_{s_idx}"

                # Choose device (with possible sharing noise)
                if random.random() < config.device_sharing_rate and n_persons > 1:
                    # Use another person's device (shared device scenario)
                    other_persons = [m for m in members if m != person_id]
                    if other_persons:
                        other = random.choice(other_persons)
                        other_devices = ground_truth.person_devices.get(other, [])
                        if other_devices:
                            device_id = random.choice(other_devices)
                            device_type = "tv"  # Shared devices are usually TVs
                        else:
                            device_id, device_type = random.choice(person_devices)
                    else:
                        device_id, device_type = random.choice(person_devices)
                else:
                    device_id, device_type = random.choice(person_devices)

                # Record ground truth
                ground_truth.session_to_person[session_id] = person_id

                # Generate session events
                session_events = _generate_session_events(
                    session_id=session_id,
                    account_id=account_id,
                    device_id=device_id,
                    device_type=device_type,
                    persona=persona,
                    config=config
                )

                all_events.extend(session_events)

        ground_truth.household_members[household_id] = members

    # Shuffle events to simulate real data arrival
    random.shuffle(all_events)

    return all_events, ground_truth


def _generate_session_events(
    session_id: str,
    account_id: str,
    device_id: str,
    device_type: str,
    persona: str,
    config: SyntheticConfig
) -> List[StreamingEvent]:
    """Generate events for a single session."""
    profile = PERSONA_PROFILES[persona]
    events = []

    # Session timing
    session_date = config.start_date + timedelta(
        seconds=random.randint(0, int((config.end_date - config.start_date).total_seconds()))
    )

    # Pick hour based on persona preferences
    if random.random() < 0.7:  # 70% chance of preferred hour
        hour = random.choice(profile["peak_hours"])
    else:
        hour = random.randint(0, 23)

    # Weekend adjustment
    if session_date.weekday() >= 5:  # Weekend
        if random.random() < profile["weekend_preference"]:
            hour = random.choice(profile["peak_hours"])

    session_start = session_date.replace(hour=hour, minute=random.randint(0, 59))

    # Session duration
    base_duration = random.uniform(*config.session_duration_range)
    duration = base_duration * profile["session_duration_mult"]

    # Number of events
    n_events = random.randint(*config.events_per_session_range)

    # Pick genres for this session
    if random.random() < 0.8:  # 80% preferred genres
        session_genres = random.sample(
            profile["genres"],
            min(len(profile["genres"]), random.randint(1, 3))
        )
    else:
        session_genres = random.sample(GENRE_LIST, random.randint(1, 3))

    # Generate individual events
    current_time = session_start
    event_duration = duration / n_events

    for e_idx in range(n_events):
        event_id = f"{session_id}_event_{e_idx}"

        # Event type
        if e_idx == 0:
            event_type = "play"
        elif e_idx == n_events - 1:
            event_type = "pause"
        else:
            event_type = random.choice(["play", "browse", "search"])

        # Content
        genre = random.choice(session_genres)
        content_id = f"content_{random.randint(1000, 9999)}"

        # Add noise
        if random.random() < config.noise_level:
            genre = random.choice(GENRE_LIST)

        event = StreamingEvent(
            event_id=event_id,
            account_id=account_id,
            device_fingerprint=device_id,
            timestamp=current_time,
            event_type=event_type,
            content_id=content_id,
            content_title=f"Title {content_id}",
            content_genre=genre,
            duration_seconds=event_duration * 60,  # Convert to seconds
            device_type=device_type,
            os_family="unknown",
            browser_family="unknown",
            screen_resolution="1920x1080",
            hour_of_day=current_time.hour,
            day_of_week=current_time.weekday(),
            metadata={"session_id": session_id, "persona": persona},
        )

        events.append(event)
        current_time += timedelta(minutes=event_duration)

    return events


def generate_cross_device_test_data(
    n_persons: int = 20,
    devices_per_person: int = 3,
    sessions_per_device: int = 30,
    seed: int = 42
) -> Tuple[List[StreamingEvent], Dict[str, List[str]]]:
    """
    Generate data specifically for cross-device linking tests.

    Returns
    -------
    Tuple[List[StreamingEvent], Dict[str, List[str]]]
        (events, person_to_devices mapping)
    """
    random.seed(seed)

    all_events = []
    person_to_devices = {}

    for p_idx in range(n_persons):
        person_id = f"person_{p_idx:04d}"
        account_id = f"account_{p_idx:04d}"

        # Create devices for this person
        devices = []
        for d_idx in range(devices_per_person):
            device_id = f"{person_id}_device_{d_idx}"
            device_type = random.choice(DEVICE_TYPES)
            devices.append((device_id, device_type))

        person_to_devices[person_id] = [d[0] for d in devices]

        # Generate consistent behavior across devices
        preferred_genres = random.sample(GENRE_LIST, 3)
        preferred_hours = random.sample(range(24), 4)

        # Generate sessions on each device
        for device_id, device_type in devices:
            for s_idx in range(sessions_per_device):
                # Use consistent preferences (with some variation)
                session_start = datetime(2025, 1, 1) + timedelta(days=random.randint(0, 180))
                hour = random.choice(preferred_hours) if random.random() < 0.7 else random.randint(0, 23)
                session_start = session_start.replace(hour=hour)

                n_events = random.randint(3, 10)

                for e_idx in range(n_events):
                    genre = random.choice(preferred_genres) if random.random() < 0.7 else random.choice(GENRE_LIST)

                    event = StreamingEvent(
                        event_id=f"{device_id}_s{s_idx}_e{e_idx}",
                        account_id=account_id,
                        device_fingerprint=device_id,
                        timestamp=session_start + timedelta(minutes=e_idx * 5),
                        event_type="play",
                        content_genre=genre,
                        duration_seconds=300,
                        device_type=device_type,
                        hour_of_day=hour,
                        day_of_week=session_start.weekday(),
                    )
                    all_events.append(event)

    random.shuffle(all_events)
    return all_events, person_to_devices
