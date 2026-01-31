"""
Enhanced Synthetic Data Generator with Ground Truth

Creates realistic household viewing patterns with known person labels.
Used for:
- Accuracy validation
- Demo visualizations  
- Benchmark testing
- Training data generation
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class PersonaProfile:
    """Definition of a household member's viewing patterns."""
    name: str
    age_group: str  # child, teen, adult, senior
    
    # Temporal patterns
    typical_hours: List[int]  # Hours of day (0-23) when usually watching
    weekend_bias: float  # 0=weekdays only, 1=weekends only
    session_frequency: float  # Sessions per day
    
    # Device preferences
    preferred_devices: List[str]  # ['tv', 'mobile', 'tablet', 'desktop']
    device_weights: List[float]  # Probability distribution
    
    # Content preferences  
    genre_preferences: Dict[str, float]  # Genre -> probability
    content_duration_avg: float  # Average session duration in seconds
    content_duration_std: float  # Standard deviation
    
    # Behavioral quirks
    binge_watching_probability: float  # 0-1 likelihood of long sessions
    channel_switching_rate: float  # Events per hour
    
    # Ground truth ID
    person_id: int = field(default_factory=lambda: np.random.randint(1000, 9999))


@dataclass  
class SyntheticHousehold:
    """A generated household with ground truth labels."""
    account_id: str
    personas: List[PersonaProfile]
    sessions: List[Dict] = field(default_factory=list)
    ground_truth: Dict[str, Any] = field(default_factory=dict)
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame."""
        return pd.DataFrame(self.sessions)
    
    def get_accuracy_report(self, predictions: Dict[str, str]) -> Dict:
        """Compare predictions to ground truth."""
        correct = 0
        total = 0
        
        for session_id, true_person_id in self.ground_truth.items():
            if session_id in predictions:
                pred_person_id = predictions[session_id]
                if pred_person_id == true_person_id:
                    correct += 1
                total += 1
        
        return {
            "accuracy": correct / total if total > 0 else 0,
            "correct": correct,
            "total": total,
            "account_id": self.account_id
        }


class HouseholdDataGenerator:
    """
    Generates realistic synthetic household streaming data.
    
    Creates households with multiple members, each with distinct:
    - Viewing time patterns (adults at night, kids after school)
    - Device preferences (TV vs mobile)
    - Content preferences (drama vs cartoons)
    - Session behaviors (binge watching vs short sessions)
    """
    
    # Pre-defined realistic personas
    PERSONA_TEMPLATES = {
        "primary_adult": PersonaProfile(
            name="Primary Adult",
            age_group="adult",
            typical_hours=[19, 20, 21, 22, 23],  # Evening prime time
            weekend_bias=0.3,
            session_frequency=1.2,
            preferred_devices=["tv", "desktop"],
            device_weights=[0.7, 0.3],
            genre_preferences={
                "Drama": 0.30,
                "Documentary": 0.20,
                "Thriller": 0.15,
                "Comedy": 0.15,
                "Action": 0.10,
                "News": 0.10
            },
            content_duration_avg=7200,  # 2 hours
            content_duration_std=1800,
            binge_watching_probability=0.4,
            channel_switching_rate=5.0
        ),
        
        "secondary_adult": PersonaProfile(
            name="Secondary Adult",
            age_group="adult",
            typical_hours=[20, 21, 22],  # Later evening
            weekend_bias=0.5,
            session_frequency=0.8,
            preferred_devices=["tablet", "tv"],
            device_weights=[0.6, 0.4],
            genre_preferences={
                "Romance": 0.25,
                "Comedy": 0.25,
                "Drama": 0.20,
                "Reality": 0.15,
                "Documentary": 0.15
            },
            content_duration_avg=5400,  # 1.5 hours
            content_duration_std=1200,
            binge_watching_probability=0.25,
            channel_switching_rate=4.0
        ),
        
        "teen": PersonaProfile(
            name="Teen",
            age_group="teen",
            typical_hours=[15, 16, 20, 21, 22, 23],  # After school + late night
            weekend_bias=0.6,
            session_frequency=2.5,
            preferred_devices=["mobile", "tablet"],
            device_weights=[0.7, 0.3],
            genre_preferences={
                "SciFi": 0.30,
                "Action": 0.25,
                "Animation": 0.15,
                "Comedy": 0.20,
                "Fantasy": 0.10
            },
            content_duration_avg=3600,  # 1 hour
            content_duration_std=900,
            binge_watching_probability=0.35,
            channel_switching_rate=8.0
        ),
        
        "child": PersonaProfile(
            name="Child",
            age_group="child",
            typical_hours=[16, 17, 18, 19],  # After school, before dinner
            weekend_bias=0.7,
            session_frequency=2.0,
            preferred_devices=["tablet"],
            device_weights=[1.0],
            genre_preferences={
                "Animation": 0.40,
                "Kids": 0.35,
                "Comedy": 0.15,
                "Family": 0.10
            },
            content_duration_avg=1800,  # 30 minutes
            content_duration_std=600,
            binge_watching_probability=0.15,
            channel_switching_rate=3.0
        ),
        
        "senior": PersonaProfile(
            name="Senior",
            age_group="senior",
            typical_hours=[14, 15, 16, 20, 21],  # Afternoon + evening
            weekend_bias=0.4,
            session_frequency=1.5,
            preferred_devices=["tv"],
            device_weights=[1.0],
            genre_preferences={
                "News": 0.30,
                "Documentary": 0.25,
                "Drama": 0.20,
                "Classic": 0.15,
                "Romance": 0.10
            },
            content_duration_avg=5400,  # 1.5 hours
            content_duration_std=900,
            binge_watching_probability=0.2,
            channel_switching_rate=2.0
        )
    }
    
    def __init__(self, random_seed: int = 42):
        self.rng = np.random.default_rng(random_seed)
    
    def generate_household(
        self,
        account_id: Optional[str] = None,
        num_persons: Optional[int] = None,
        persona_types: Optional[List[str]] = None,
        days_of_history: int = 30
    ) -> SyntheticHousehold:
        """
        Generate a household with realistic viewing patterns.
        
        Args:
            account_id: Optional account ID (generated if not provided)
            num_persons: Number of people in household (2-5 typical)
            persona_types: Specific persona types to use
            days_of_history: How many days of viewing history
        
        Returns:
            SyntheticHousehold with sessions and ground truth
        """
        if account_id is None:
            account_id = f"household_{self.rng.integers(10000, 99999)}"
        
        # Determine household composition
        if persona_types:
            selected_personas = [
                self.PERSONA_TEMPLATES[pt] 
                for pt in persona_types 
                if pt in self.PERSONA_TEMPLATES
            ]
        else:
            if num_persons is None:
                num_persons = self.rng.integers(2, 5)
            
            # Select realistic combinations
            available = list(self.PERSONA_TEMPLATES.keys())
            
            # Ensure at least one adult if we have children
            selected_types = self.rng.choice(available, size=num_persons, replace=False)
            selected_personas = [self.PERSONA_TEMPLATES[st] for st in selected_types]
        
        # Assign unique person IDs
        for i, persona in enumerate(selected_personas):
            persona.person_id = i
        
        # Generate sessions
        sessions = []
        ground_truth = {}
        
        for day_offset in range(days_of_history):
            date = datetime.now() - timedelta(days=day_offset)
            
            for persona in selected_personas:
                # Determine if this person watches today
                expected_sessions = persona.session_frequency
                actual_sessions = self.rng.poisson(expected_sessions)
                
                for _ in range(actual_sessions):
                    session = self._generate_session(
                        account_id=account_id,
                        persona=persona,
                        date=date
                    )
                    
                    sessions.append(session)
                    ground_truth[session["session_id"]] = {
                        "person_id": persona.person_id,
                        "persona_type": persona.name,
                        "persona_age": persona.age_group
                    }
        
        # Sort by timestamp
        sessions.sort(key=lambda x: x["timestamp"])
        
        return SyntheticHousehold(
            account_id=account_id,
            personas=selected_personas,
            sessions=sessions,
            ground_truth=ground_truth
        )
    
    def _generate_session(
        self,
        account_id: str,
        persona: PersonaProfile,
        date: datetime
    ) -> Dict:
        """Generate a single viewing session for a persona."""
        
        # Choose time
        is_weekend = date.weekday() >= 5
        if self.rng.random() < persona.weekend_bias and is_weekend:
            # More likely to watch on weekends
            hour = self.rng.choice([10, 11, 14, 15, 16, 20, 21])
        else:
            hour = self.rng.choice(persona.typical_hours)
        
        minute = self.rng.integers(0, 60)
        timestamp = date.replace(hour=hour, minute=minute, second=0, microsecond=0)
        
        # Choose device
        device = self.rng.choice(
            persona.preferred_devices,
            p=persona.device_weights
        )
        
        # Determine if binge watching
        is_binge = self.rng.random() < persona.binge_watching_probability
        if is_binge:
            duration = persona.content_duration_avg * self.rng.uniform(1.5, 3.0)
        else:
            duration = self.rng.normal(
                persona.content_duration_avg,
                persona.content_duration_std
            )
        
        duration = max(300, duration)  # Minimum 5 minutes
        
        # Generate content viewed
        genres = list(persona.genre_preferences.keys())
        probabilities = list(persona.genre_preferences.values())
        
        # Select primary genre
        primary_genre = self.rng.choice(genres, p=probabilities)
        
        # Generate genre distribution
        genre_times = {}
        remaining_duration = duration
        
        # Primary genre gets majority
        primary_time = remaining_duration * self.rng.uniform(0.5, 0.8)
        genre_times[primary_genre] = primary_time
        remaining_duration -= primary_time
        
        # Add secondary genres
        while remaining_duration > 60 and self.rng.random() < 0.7:
            secondary_genre = self.rng.choice(genres, p=probabilities)
            secondary_time = min(remaining_duration * 0.3, self.rng.uniform(300, 900))
            genre_times[secondary_genre] = genre_times.get(secondary_genre, 0) + secondary_time
            remaining_duration -= secondary_time
        
        # Event count based on duration and switching rate
        events = int((duration / 3600) * persona.channel_switching_rate)
        events = max(1, events + self.rng.poisson(2))
        
        return {
            "session_id": f"sess_{account_id}_{persona.person_id}_{self.rng.integers(1000000)}",
            "account_id": account_id,
            "person_id": persona.person_id,  # Ground truth
            "timestamp": timestamp.isoformat(),
            "hour": hour,
            "day_of_week": date.weekday(),
            "is_weekend": is_weekend,
            "device_type": device,
            "device_fingerprint": f"fp_{device}_{account_id}",
            "duration_seconds": duration,
            "genre_distribution": genre_times,
            "primary_genre": primary_genre,
            "event_count": events,
            "is_binge": is_binge
        }
    
    def generate_dataset(
        self,
        num_households: int = 100,
        output_file: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Generate multiple households and combine into dataset.
        
        Args:
            num_households: Number of households to generate
            output_file: Optional file to save dataset
        
        Returns:
            DataFrame with all sessions and ground truth
        """
        all_sessions = []
        
        for i in range(num_households):
            household = self.generate_household(
                account_id=f"household_{i:04d}",
                days_of_history=30
            )
            all_sessions.extend(household.sessions)
        
        df = pd.DataFrame(all_sessions)
        
        if output_file:
            df.to_csv(output_file, index=False)
            logger.info(f"Dataset saved to {output_file}")
        
        return df
    
    def get_dataset_statistics(self, households: List[SyntheticHousehold]) -> Dict:
        """Compute statistics across generated households."""
        
        stats = {
            "num_households": len(households),
            "total_sessions": sum(len(h.sessions) for h in households),
            "avg_sessions_per_household": np.mean([len(h.sessions) for h in households]),
            "avg_persons_per_household": np.mean([len(h.personas) for h in households]),
            "persona_distribution": {}
        }
        
        # Count personas
        persona_counts = {}
        for h in households:
            for p in h.personas:
                persona_counts[p.name] = persona_counts.get(p.name, 0) + 1
        
        stats["persona_distribution"] = persona_counts
        
        return stats


# Convenience function
def generate_demo_household() -> SyntheticHousehold:
    """Generate a demo household for visualizations."""
    generator = HouseholdDataGenerator(random_seed=42)
    return generator.generate_household(
        account_id="demo_household_001",
        persona_types=["primary_adult", "teen", "child"],
        days_of_history=14
    )


if __name__ == "__main__":
    # Generate demo data
    generator = HouseholdDataGenerator(random_seed=42)
    
    # Single household
    household = generator.generate_household(
        account_id="test_household",
        num_persons=3,
        days_of_history=7
    )
    
    print(f"Generated household: {household.account_id}")
    print(f"Persons: {len(household.personas)}")
    print(f"Sessions: {len(household.sessions)}")
    
    for persona in household.personas:
        persona_sessions = [s for s in household.sessions if s["person_id"] == persona.person_id]
        print(f"  {persona.name}: {len(persona_sessions)} sessions")
    
    # Save to file
    df = household.to_dataframe()
    df.to_csv("demo_household.csv", index=False)
    print("\nSaved to demo_household.csv")
