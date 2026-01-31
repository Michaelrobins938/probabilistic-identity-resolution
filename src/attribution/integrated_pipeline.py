"""
Integration: Identity Resolution + Attribution

Connects the identity resolution system with the rigorous attribution engine.
Transforms resolved identities into first-principles attribution-ready format.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import logging

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from models.streaming_event import StreamingEvent, Session
from models.household_profile import HouseholdProfile
from core.probabilistic_resolver import ResolutionResult
from attribution.hybrid_engine import HybridAttributionEngine, HybridAttributionConfig, AttributionResult
from attribution.uncertainty_quantification import UncertaintyQuantificationEngine, run_full_uq_analysis
from adapters.attribution_adapter import AttributionAdapter, AttributionEvent

logger = logging.getLogger(__name__)


@dataclass
class IntegratedAttributionResult:
    """Complete result from identity resolution + attribution pipeline."""
    
    # Identity resolution results
    n_households: int
    n_persons: int
    n_sessions: int
    
    # Attribution results
    attribution: AttributionResult
    
    # Per-persona breakdown
    persona_attributions: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Per-device breakdown
    device_attributions: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Uncertainty quantification
    uncertainty_analysis: Optional[Dict[str, Any]] = None
    
    def get_summary(self) -> str:
        """Get human-readable summary."""
        lines = [
            "=" * 80,
            "INTEGRATED IDENTITY RESOLUTION + ATTRIBUTION RESULT",
            "=" * 80,
            "",
            "IDENTITY RESOLUTION",
            f"  Households Resolved:     {self.n_households}",
            f"  Persons Identified:      {self.n_persons}",
            f"  Sessions Analyzed:       {self.n_sessions}",
            "",
        ]
        
        lines.append(self.attribution.get_summary())
        
        if self.persona_attributions:
            lines.extend([
                "",
                "PERSONA-SPECIFIC ATTRIBUTION:",
                "-" * 80,
            ])
            for persona, attrs in sorted(self.persona_attributions.items()):
                total = sum(attrs.values())
                lines.append(f"  {persona:20s}: ${total:,.2f}")
        
        return "\n".join(lines)


class IntegratedAttributionPipeline:
    """
    Complete pipeline integrating identity resolution with attribution.
    
    Flow:
    1. Identity Resolution: events → sessions → persons/households
    2. Attribution: paths → Markov chains → Shapley values → hybrid scores
    3. Uncertainty Quantification: bootstrap + Dirichlet analysis
    4. Segmentation: breakdown by persona, device, household
    """
    
    def __init__(
        self,
        attribution_config: Optional[HybridAttributionConfig] = None,
        enable_uq: bool = True
    ):
        self.attribution_config = attribution_config or HybridAttributionConfig()
        self.enable_uq = enable_uq
        
    def run(
        self,
        resolution_result: ResolutionResult,
        events: List[StreamingEvent],
        sessions: List[Session]
    ) -> IntegratedAttributionResult:
        """
        Run complete integrated attribution pipeline.
        
        Parameters
        ----------
        resolution_result : ResolutionResult
            Output from identity resolution
        events : List[StreamingEvent]
            Original streaming events
        sessions : List[Session]
            Sessions with person assignments
        
        Returns
        -------
        IntegratedAttributionResult
            Complete attribution with uncertainty quantification
        """
        logger.info("Starting integrated attribution pipeline...")
        
        # Step 1: Convert to attribution paths
        paths, conversions, path_values = self._extract_attribution_paths(
            events, sessions
        )
        
        logger.info(f"Extracted {len(paths)} attribution paths")
        
        # Step 2: Run hybrid attribution
        attribution_engine = HybridAttributionEngine(self.attribution_config)
        attribution_engine.fit(paths, conversions, path_values)
        attribution_result = attribution_engine.compute_attribution()
        
        logger.info("Hybrid attribution computed")
        
        # Step 3: Compute per-persona attributions
        persona_attributions = self._compute_persona_attributions(
            resolution_result, sessions, attribution_result
        )
        
        # Step 4: Compute per-device attributions
        device_attributions = self._compute_device_attributions(
            sessions, attribution_result
        )
        
        # Step 5: Run uncertainty quantification (if enabled)
        uncertainty_analysis = None
        if self.enable_uq:
            logger.info("Running uncertainty quantification...")
            uncertainty_analysis = run_full_uq_analysis(attribution_engine)
        
        # Build result
        result = IntegratedAttributionResult(
            n_households=len(resolution_result.households),
            n_persons=sum(len(h.members) for h in resolution_result.households),
            n_sessions=len(sessions),
            attribution=attribution_result,
            persona_attributions=persona_attributions,
            device_attributions=device_attributions,
            uncertainty_analysis=uncertainty_analysis
        )
        
        logger.info("Integrated attribution pipeline complete")
        return result
    
    def _extract_attribution_paths(
        self,
        events: List[StreamingEvent],
        sessions: List[Session]
    ) -> Tuple[List[List[str]], List[bool], List[float]]:
        """
        Extract attribution paths from sessions.
        
        Converts sessions to channel sequences for attribution.
        """
        # Group sessions by person
        person_sessions = defaultdict(list)
        for session in sessions:
            person_id = session.assigned_person_id or session.account_id
            person_sessions[person_id].append(session)
        
        paths = []
        conversions = []
        path_values = []
        
        for person_id, person_sess_list in person_sessions.items():
            # Sort by time
            person_sess_list.sort(key=lambda s: s.start_time or s.end_time)
            
            # Extract path (channels)
            path = []
            for session in person_sess_list:
                # Use device_type or channel from events
                if session.events:
                    # Get channel from first event
                    channel = session.events[0].channel or session.device_type
                else:
                    channel = session.device_type or "Direct"
                
                path.append(channel)
            
            # Check for conversion
            has_conversion = any(s.has_conversion for s in person_sess_list)
            conversion_value = sum(s.conversion_value for s in person_sess_list)
            
            if path:  # Only add non-empty paths
                paths.append(path)
                conversions.append(has_conversion)
                path_values.append(conversion_value)
        
        return paths, conversions, path_values
    
    def _compute_persona_attributions(
        self,
        resolution_result: ResolutionResult,
        sessions: List[Session],
        attribution_result: AttributionResult
    ) -> Dict[str, Dict[str, float]]:
        """Compute attribution breakdown by persona type."""
        persona_conversions = defaultdict(float)
        
        # Build person → persona mapping
        person_persona = {}
        for household in resolution_result.households:
            for member in household.members:
                person_persona[member.person_id] = member.persona_type
        
        # Aggregate by persona
        for session in sessions:
            if not session.has_conversion:
                continue
            
            persona = person_persona.get(session.assigned_person_id, "unknown")
            
            # Split value among channels based on hybrid shares
            # (Simplified - assumes session contributed to all channels proportionally)
            persona_conversions[persona] += session.conversion_value
        
        return {k: {"total_value": v} for k, v in persona_conversions.items()}
    
    def _compute_device_attributions(
        self,
        sessions: List[Session],
        attribution_result: AttributionResult
    ) -> Dict[str, Dict[str, float]]:
        """Compute attribution breakdown by device type."""
        device_conversions = defaultdict(float)
        
        for session in sessions:
            if not session.has_conversion:
                continue
            
            device_conversions[session.device_type] += session.conversion_value
        
        return {k: {"total_value": v} for k, v in device_conversions.items()}
    
    def run_with_psychographics(
        self,
        resolution_result: ResolutionResult,
        events: List[StreamingEvent],
        sessions: List[Session],
        psychographic_weights: Dict[str, float]
    ) -> IntegratedAttributionResult:
        """
        Run attribution with psychographic prior modulation.
        
        Example weights: {"high_intent_search": 1.5, "desktop_checkout": 1.3}
        """
        logger.info("Running integrated attribution with psychographics...")
        
        # Run base pipeline
        result = self.run(resolution_result, events, sessions)
        
        # Re-run attribution with psychographics
        paths, conversions, path_values = self._extract_attribution_paths(events, sessions)
        
        engine = HybridAttributionEngine(self.attribution_config)
        engine.fit(paths, conversions, path_values)
        
        # Compute with psychographics
        psych_result = engine.compute_with_psychographics(psychographic_weights)
        
        # Update result
        result.attribution = psych_result
        
        return result


# Convenience functions

def run_integrated_attribution(
    resolution_result: ResolutionResult,
    events: List[StreamingEvent],
    sessions: List[Session],
    alpha: float = 0.5,
    enable_uq: bool = True
) -> IntegratedAttributionResult:
    """
    Run complete identity resolution + attribution pipeline.
    
    Parameters
    ----------
    resolution_result : ResolutionResult
        Output from ProbabilisticIdentityResolver
    events : List[StreamingEvent]
        Original events
    sessions : List[Session]
        Resolved sessions
    alpha : float
        Blend parameter (0.0 = pure Shapley, 1.0 = pure Markov)
    enable_uq : bool
        Run uncertainty quantification
    
    Returns
    -------
    IntegratedAttributionResult
        Complete integrated results
    """
    config = HybridAttributionConfig(alpha=alpha)
    pipeline = IntegratedAttributionPipeline(config, enable_uq)
    return pipeline.run(resolution_result, events, sessions)
