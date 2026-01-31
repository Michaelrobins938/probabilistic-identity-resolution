"""
Behavioral Profiling Attribution

Extends the existing identity resolution system with:
- Psychographic priors and behavioral segmentation
- Segment-specific attribution analysis
- Heterogeneous treatment effects
- Visual comparison with traditional attribution
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from core.probabilistic_resolver import ResolutionResult
from models.streaming_event import StreamingEvent, Session
from models.household_profile import HouseholdProfile, PersonProfile
from validation.synthetic_households import PERSONA_PROFILES, GENRE_LIST


@dataclass
class BehavioralProfile:
    """Enhanced behavioral profile with psychographic priors."""
    persona_type: str
    intent_score: float  # 0-1 scale
    engagement_depth: float  # 0-1 scale
    device_preference_score: Dict[str, float]  # TV, Desktop, Mobile, Tablet
    genre_affinities: Dict[str, float]  # Weighted genre preferences
    time_of_day_preference: List[float]  # 24-hour preference curve
    weekend_vs_weekday: float  # 0-1 scale
    price_sensitivity: float  # 0-1 scale (higher = more price sensitive)
    content_discovery_method: str  # "search", "browse", "recommendation"
    viewing_context: str  # "solo", "co-viewing", "background"


@dataclass
class BehavioralSegment:
    """A behavioral segment with attribution characteristics."""
    segment_id: str
    description: str
    behavioral_profile: BehavioralProfile
    
    # Attribution metrics
    conversion_rate: float
    average_order_value: float
    attribution_share: float  # Share of total conversions
    roi_multiplier: float  # vs baseline
    
    # Channel effectiveness
    channel_efficacy: Dict[str, float]  # Channel -> effectiveness score
    optimal_frequency: int  # Optimal ad frequency per week
    
    # Confidence metrics
    sample_size: int
    confidence_interval: Tuple[float, float]


@dataclass
class BehavioralAttributionResult:
    """Results of behavioral segmentation attribution."""
    segments: List[BehavioralSegment]
    segment_attribution: Dict[str, Dict[str, float]]  # segment_id -> channel -> attribution_share
    traditional_attribution: Dict[str, float]  # channel -> attribution_share
    behavioral_enhancement: float  # % improvement over traditional
    segment_recommendations: Dict[str, Dict[str, Any]]  # segment_id -> recommendations


class BehavioralProfilingEngine:
    """
    Behavioral Profiling and Segmentation Engine.
    
    Extends identity resolution with psychographic priors and behavioral segmentation.
    """
    
    def __init__(self, resolution_result: ResolutionResult):
        self.result = resolution_result
        self._person_profiles = self._build_person_profiles()
        self._household_profiles = self._build_household_profiles()
        self._segments = self._create_behavioral_segments()
    
    def _build_person_profiles(self) -> Dict[str, PersonProfile]:
        """Build enhanced person profiles with behavioral metrics."""
        profiles = {}
        
        for household in self.result.households:
            for member in household.members:
                profile = self._create_enhanced_person_profile(member, household)
                profiles[member.person_id] = profile
        
        return profiles
    
    def _create_enhanced_person_profile(self, member: PersonProfile, household: HouseholdProfile) -> PersonProfile:
        """Create enhanced person profile with behavioral metrics."""
        # Base profile
        profile = member
        
        # Add behavioral metrics
        persona_type = member.persona_type
        
        # Intent score based on engagement
        engagement_score = self._calculate_engagement_score(member)
        intent_score = self._calculate_intent_score(engagement_score, persona_type)
        
        # Device preferences
        device_preferences = self._calculate_device_preferences(member)
        
        # Genre affinities (weighted by viewing time)
        genre_affinities = self._calculate_genre_affinities(member)
        
        # Time of day preference
        time_preferences = self._calculate_time_preferences(member)
        
        # Price sensitivity (persona-based)
        price_sensitivity = self._calculate_price_sensitivity(persona_type)
        
        # Content discovery method
        discovery_method = self._infer_discovery_method(member)
        
        # Viewing context
        viewing_context = self._infer_viewing_context(member, household)
        
        # Engagement depth
        engagement_depth = self._calculate_engagement_depth(member)
        
        # Update profile with behavioral fields
        profile.behavioral_metrics = {
            'intent_score': intent_score,
            'engagement_depth': engagement_depth,
            'device_preferences': device_preferences,
            'genre_affinities': genre_affinities,
            'time_preferences': time_preferences,
            'price_sensitivity': price_sensitivity,
            'discovery_method': discovery_method,
            'viewing_context': viewing_context,
            'persona_type': persona_type,
        }
        
        return profile
    
    def _build_household_profiles(self) -> Dict[str, HouseholdProfile]:
        """Build household profiles with aggregated behavioral metrics."""
        profiles = {}
        
        for household in self.result.households:
            aggregated_metrics = self._aggregate_household_metrics(household)
            household.behavioral_metrics = aggregated_metrics
            profiles[household.household_id] = household
        
        return profiles
    
    def _create_behavioral_segments(self) -> List[BehavioralSegment]:
        """Create behavioral segments based on personas and engagement."""
        segments = []
        
        # Define segment templates
        segment_templates = self._get_segment_templates()
        
        # Create segments from personas
        for persona_type in set(p.behavioral_metrics['persona_type'] for p in self._person_profiles.values()):
            for template in segment_templates:
                if template['persona_filter'](persona_type):
                    segment = self._create_segment_from_template(persona_type, template)
                    segments.append(segment)
        
        return segments
    
    def _get_segment_templates(self) -> List[Dict]:
        """Define behavioral segment templates."""
        return [
            {
                'segment_id': 'high_intent_desktop_adult',
                'description': 'High-intent desktop users (primary decision makers)',
                'persona_filter': lambda p: p in ['primary_adult', 'secondary_adult'],
                'intent_threshold': 0.7,
                'device_preference': 'desktop',
                'channel_efficacy': {'Email': 0.8, 'Search': 0.7, 'Display': 0.4},
                'optimal_frequency': 3,
                'roi_multiplier': 1.5
            },
            {
                'segment_id': 'low_intent_mobile_teen',
                'description': 'Low-intent mobile users (exploratory behavior)',
                'persona_filter': lambda p: p in ['teen'],
                'intent_threshold': 0.3,
                'device_preference': 'mobile',
                'channel_efficacy': {'Social': 0.6, 'Display': 0.3, 'Video': 0.5},
                'optimal_frequency': 5,
                'roi_multiplier': 1.2
            },
            {
                'segment_id': 'family_co_viewing',
                'description': 'Family co-viewing sessions',
                'persona_filter': lambda p: True,  # All personas can be in family viewing
                'intent_threshold': 0.5,
                'device_preference': 'tv',
                'channel_efficacy': {'TV': 0.9, 'Display': 0.2, 'Social': 0.1},
                'optimal_frequency': 2,
                'roi_multiplier': 1.8
            },
            {
                'segment_id': 'price_sensitive_bargain_hunter',
                'description': 'Price-sensitive users looking for deals',
                'persona_filter': lambda p: True,
                'intent_threshold': 0.4,
                'device_preference': 'mobile',
                'channel_efficacy': {'Email': 0.7, 'Search': 0.6, 'Display': 0.3},
                'optimal_frequency': 4,
                'roi_multiplier': 1.3
            }
        ]
    
    def _create_segment_from_template(self, persona_type: str, template: Dict) -> BehavioralSegment:
        """Create a behavioral segment from template and persona data."""
        # Find matching profiles
        matching_profiles = [
            p for p in self._person_profiles.values() 
            if template['persona_filter'](p.behavioral_metrics['persona_type'])
            and p.behavioral_metrics['intent_score'] >= template['intent_threshold']
        ]
        
        if not matching_profiles:
            return None
        
        # Calculate segment metrics
        n_profiles = len(matching_profiles)
        total_conversion_value = sum(p.attributed_value for p in matching_profiles)
        avg_conversion_value = total_conversion_value / n_profiles if n_profiles > 0 else 0
        
        # Create behavioral profile
        sample_profile = matching_profiles[0].behavioral_metrics
        behavioral_profile = BehavioralProfile(
            persona_type=persona_type,
            intent_score=np.mean([p.behavioral_metrics['intent_score'] for p in matching_profiles]),
            engagement_depth=np.mean([p.behavioral_metrics['engagement_depth'] for p in matching_profiles]),
            device_preference_score=sample_profile['device_preferences'],
            genre_affinities=sample_profile['genre_affinities'],
            time_of_day_preference=sample_profile['time_preferences'],
            weekend_vs_weekday=sample_profile['weekend_vs_weekday'],
            price_sensitivity=sample_profile['price_sensitivity'],
            content_discovery_method=sample_profile['discovery_method'],
            viewing_context=sample_profile['viewing_context']
        )
        
        # Create segment
        segment = BehavioralSegment(
            segment_id=template['segment_id'],
            description=template['description'],
            behavioral_profile=behavioral_profile,
            conversion_rate=total_conversion_value / self.result.total_events if self.result.total_events > 0 else 0,
            average_order_value=avg_conversion_value,
            attribution_share=len(matching_profiles) / self.result.total_persons if self.result.total_persons > 0 else 0,
            roi_multiplier=template['roi_multiplier'],
            channel_efficacy=template['channel_efficacy'],
            optimal_frequency=template['optimal_frequency'],
            sample_size=n_profiles,
            confidence_interval=(0.05, 0.15)  # Example confidence interval
        )
        
        return segment
    
    def analyze_segment_attribution(self) -> BehavioralAttributionResult:
        """Analyze attribution across behavioral segments."""
        # Calculate traditional attribution (baseline)
        traditional_attribution = self._calculate_traditional_attribution()
        
        # Calculate segment-specific attribution
        segment_attribution = self._calculate_segment_attribution()
        
        # Calculate behavioral enhancement
        behavioral_enhancement = self._calculate_behavioral_enhancement(traditional_attribution, segment_attribution)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(segment_attribution)
        
        return BehavioralAttributionResult(
            segments=self._segments,
            segment_attribution=segment_attribution,
            traditional_attribution=traditional_attribution,
            behavioral_enhancement=behavioral_enhancement,
            segment_recommendations=recommendations
        )
    
    def _calculate_traditional_attribution(self) -> Dict[str, float]:
        """Calculate traditional attribution (baseline)."""
        # This would use the existing attribution adapter
        # For now, return mock data
        return {
            'Email': 0.25,
            'Search': 0.20,
            'Display': 0.15,
            'Social': 0.18,
            'TV': 0.12,
            'Direct': 0.10
        }
    
    def _calculate_segment_attribution(self) -> Dict[str, Dict[str, float]]:
        """Calculate attribution per segment."""
        segment_attribution = {}
        
        for segment in self._segments:
            if segment:
                # Calculate segment's channel attribution
                segment_share = segment.attribution_share
                channel_attribution = {}
                
                for channel, efficacy in segment.channel_efficacy.items():
                    channel_attribution[channel] = segment_share * efficacy
                
                segment_attribution[segment.segment_id] = channel_attribution
        
        return segment_attribution
    
    def _calculate_behavioral_enhancement(self, traditional: Dict[str, float], segment: Dict[str, Dict[str, float]]) -> float:
        """Calculate improvement from behavioral segmentation."""
        # Calculate traditional ROI
        traditional_roi = sum(traditional.values())  # Simplified
        
        # Calculate behavioral ROI
        behavioral_roi = 0
        for segment_data in segment.values():
            behavioral_roi += sum(segment_data.values())
        
        # Calculate enhancement
        if traditional_roi > 0:
            enhancement = (behavioral_roi - traditional_roi) / traditional_roi
        else:
            enhancement = 0
        
        return enhancement * 100  # Return as percentage
    
    def _generate_recommendations(self, segment_attribution: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, Any]]:
        """Generate channel recommendations per segment."""
        recommendations = {}
        
        for segment_id, attribution in segment_attribution.items():
            # Find top channels
            sorted_channels = sorted(attribution.items(), key=lambda x: x[1], reverse=True)
            top_channels = sorted_channels[:3]
            
            recommendations[segment_id] = {
                'top_channels': [ch[0] for ch in top_channels],
                'channel_weights': {ch[0]: ch[1] for ch in top_channels},
                'budget_allocation': self._calculate_budget_allocation(top_channels),
                'frequency_recommendation': self._get_frequency_recommendation(segment_id),
                'creative_recommendation': self._get_creative_recommendation(segment_id)
            }
        
        return recommendations
    
    def _calculate_budget_allocation(self, top_channels: List[Tuple[str, float]]) -> Dict[str, float]:
        """Calculate budget allocation across top channels."""
        total = sum(weight for _, weight in top_channels)
        return {channel: weight / total for channel, weight in top_channels}
    
    def _get_frequency_recommendation(self, segment_id: str) -> int:
        """Get frequency recommendation for segment."""
        segment = next((s for s in self._segments if s and s.segment_id == segment_id), None)
        return segment.optimal_frequency if segment else 3
    
    def _get_creative_recommendation(self, segment_id: str) -> str:
        """Get creative recommendation for segment."""
        # Simple persona-based recommendations
        if 'adult' in segment_id:
            return 'professional, value-focused messaging'
        elif 'teen' in segment_id:
            return 'trendy, social-proof messaging'
        elif 'family' in segment_id:
            return 'family-oriented, bundle-focused messaging'
        else:
            return 'personalized, context-aware messaging'
    
    def _calculate_engagement_score(self, member: PersonProfile) -> float:
        """Calculate engagement score based on viewing behavior."""
        session_count = member.session_count
        total_duration = member.total_duration_hours
        content_diversity = len(member.top_genres)
        
        # Simple engagement formula
        engagement = (session_count * 0.3 + total_duration * 0.5 + content_diversity * 0.2) / 3
        return min(max(engagement, 0), 1)
    
    def _calculate_intent_score(self, engagement_score: float, persona_type: str) -> float:
        """Calculate intent score based on engagement and persona."""
        base_intent = {
            'primary_adult': 0.7,
            'secondary_adult': 0.6,
            'teen': 0.4,
            'child': 0.2
        }.get(persona_type, 0.5)
        
        # Adjust based on engagement
        adjusted_intent = base_intent + (engagement_score - 0.5) * 0.3
        return min(max(adjusted_intent, 0), 1)
    
    def _calculate_device_preferences(self, member: PersonProfile) -> Dict[str, float]:
        """Calculate device preferences based on usage patterns."""
        # This would use actual device usage data
        # For now, use persona-based defaults
        persona_devices = PERSONA_PROFILES[member.persona_type]['devices']
        
        preferences = {device: 0.2 for device in ['tv', 'desktop', 'mobile', 'tablet']}
        
        for device in persona_devices:
            preferences[device] = 0.3
        
        # Normalize
        total = sum(preferences.values())
        return {k: v / total for k, v in preferences.items()}
    
    def _calculate_genre_affinities(self, member: PersonProfile) -> Dict[str, float]:
        """Calculate genre affinities based on viewing history."""
        # This would use actual viewing data
        # For now, use persona-based defaults
        persona_genres = PERSONA_PROFILES[member.persona_type]['genres']
        
        affinities = {genre: 0.1 for genre in GENRE_LIST}
        
        for genre in persona_genres:
            affinities[genre] = 0.3
        
        # Normalize
        total = sum(affinities.values())
        return {k: v / total for k, v in affinities.items()}
    
    def _calculate_time_preferences(self, member: PersonProfile) -> List[float]:
        """Calculate 24-hour time preferences."""
        # This would use actual viewing time data
        # For now, use persona-based defaults
        persona_hours = PERSONA_PROFILES[member.persona_type]['peak_hours']
        
        preferences = [0.1] * 24
        
        for hour in persona_hours:
            preferences[hour] = 0.3
        
        return preferences
    
    def _calculate_price_sensitivity(self, persona_type: str) -> float:
        """Calculate price sensitivity based on persona."""
        sensitivity = {
            'primary_adult': 0.3,
            'secondary_adult': 0.4,
            'teen': 0.6,
            'child': 0.2
        }.get(persona_type, 0.5)
        
        return sensitivity
    
    def _infer_discovery_method(self, member: PersonProfile) -> str:
        """Infer content discovery method based on behavior."""
        # This would use actual discovery data
        # For now, use persona-based inference
        if member.persona_type in ['teen', 'child']:
            return 'recommendation'
        elif member.persona_type == 'primary_adult':
            return 'search'
        else:
            return 'browse'
    
    def _infer_viewing_context(self, member: PersonProfile, household: HouseholdProfile) -> str:
        """Infer viewing context (solo, co-viewing, background)."""
        # This would use actual session data
        # For now, use household composition
        if household.estimated_size > 1:
            return 'co-viewing'
        else:
            return 'solo'
    
    def _calculate_engagement_depth(self, member: PersonProfile) -> float:
        """Calculate engagement depth (how deeply users engage with content)."""
        # This would use actual engagement metrics
        # For now, use session duration and content diversity
        session_count = member.session_count
        total_duration = member.total_duration_hours
        content_diversity = len(member.top_genres)
        
        depth = (session_count * 0.2 + total_duration * 0.5 + content_diversity * 0.3) / 3
        return min(max(depth, 0), 1)
    
    def _aggregate_household_metrics(self, household: HouseholdProfile) -> Dict[str, float]:
        """Aggregate behavioral metrics across household members."""
        # This would aggregate actual metrics
        # For now, return placeholder
        return {
            'avg_intent_score': 0.5,
            'avg_engagement_depth': 0.6,
            'price_sensitivity': 0.4,
            'co_viewing_rate': 0.7 if household.estimated_size > 1 else 0.2,
            'content_diversity': 0.8
        }


def analyze_behavioral_attribution(
    events: List[StreamingEvent],
    sessions: List[Session],
    config: Optional[ResolverConfig] = None
) -> BehavioralAttributionResult:
    """
    Main function to analyze behavioral attribution.
    
    Parameters
    ----------
    events : List[StreamingEvent]
        Streaming events to analyze
    sessions : List[Session]
        Sessions with person assignments
    config : ResolverConfig, optional
        Configuration for resolution
    
    Returns
    -------
    BehavioralAttributionResult
        Behavioral attribution analysis
    """
    # Run identity resolution
    resolver = ProbabilisticIdentityResolver(config)
    result = resolver.resolve(events)
    
    # Run behavioral profiling
    engine = BehavioralProfilingEngine(result)
    analysis = engine.analyze_segment_attribution()
    
    return analysis


# Example usage
if __name__ == "__main__":
    from validation.synthetic_households import generate_synthetic_household_data
    
    # Generate test data
    events, ground_truth = generate_synthetic_household_data()
    
    # Group into sessions
    from models.streaming_event import group_events_into_sessions
    sessions = group_events_into_sessions(events)
    
    # Analyze behavioral attribution
    result = analyze_behavioral_attribution(events, sessions)
    
    # Print results
    print("Behavioral Attribution Analysis")
    print("=" * 60)
    print(f"Segments identified: {len(result.segments)}")
    print(f"Behavioral enhancement: {result.behavioral_enhancement:.1f}%")
    
    for segment in result.segments:
        if segment:
            print(f"\nSegment: {segment.segment_id}")
            print(f"  Description: {segment.description}")
            print(f"  Conversion Rate: {segment.conversion_rate:.2%}")
            print(f"  ROI Multiplier: {segment.roi_multiplier}x")
            print(f"  Top Channels: {', '.join(segment.channel_efficacy.keys())}")
            print(f"  Optimal Frequency: {segment.optimal_frequency}/week")
            print(f"  Sample Size: {segment.sample_size} users")