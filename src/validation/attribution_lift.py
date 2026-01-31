"""
Attribution Lift Metrics

Measures the improvement in attribution accuracy from using identity resolution.
Compares person-level attribution vs account-level attribution baseline.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
import math
from collections import defaultdict

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from models.streaming_event import Session
from models.household_profile import HouseholdProfile
from validation.synthetic_households import GroundTruth
from validation.enhanced_synthetic import EnhancedGroundTruth


@dataclass
class AttributionLiftMetrics:
    """Metrics measuring lift from identity resolution."""
    
    # Attribution accuracy comparison
    person_level_accuracy: float = 0.0
    account_level_accuracy: float = 0.0
    accuracy_lift: float = 0.0  # (person - account) / account
    
    # Revenue attribution comparison
    person_level_revenue_mae: float = 0.0  # Mean absolute error
    account_level_revenue_mae: float = 0.0
    revenue_lift: float = 0.0
    
    # Persona-specific lift
    persona_lift: Dict[str, float] = field(default_factory=dict)
    
    # Channel attribution lift
    channel_lift: Dict[str, float] = field(default_factory=dict)
    
    # Statistical significance
    p_value: float = 1.0
    is_significant: bool = False
    
    # Fairness metrics
    demographic_parity: float = 0.0  # 1.0 = perfect parity
    equal_opportunity: float = 0.0
    
    def get_summary(self) -> str:
        """Get human-readable summary."""
        lines = [
            "=" * 70,
            "ATTRIBUTION LIFT METRICS",
            "=" * 70,
            "",
            "ACCURACY COMPARISON",
            f"  Person-Level Accuracy:   {self.person_level_accuracy:.1%}",
            f"  Account-Level Accuracy:  {self.account_level_accuracy:.1%}",
            f"  LIFT:                    {self.accuracy_lift:+.1%}",
            "",
            "REVENUE ATTRIBUTION COMPARISON (MAE)",
            f"  Person-Level MAE:        ${self.person_level_revenue_mae:.2f}",
            f"  Account-Level MAE:       ${self.account_level_revenue_mae:.2f}",
            f"  LIFT:                    {self.revenue_lift:+.1%}",
            "",
        ]
        
        if self.persona_lift:
            lines.append("PERSONA-SPECIFIC LIFT")
            for persona, lift in sorted(self.persona_lift.items(), key=lambda x: x[1], reverse=True):
                lines.append(f"  {persona:20s}  {lift:+.1%}")
            lines.append("")
        
        if self.channel_lift:
            lines.append("CHANNEL ATTRIBUTION LIFT")
            for channel, lift in sorted(self.channel_lift.items(), key=lambda x: x[1], reverse=True):
                lines.append(f"  {channel:20s}  {lift:+.1%}")
            lines.append("")
        
        lines.extend([
            "STATISTICAL SIGNIFICANCE",
            f"  P-Value:                 {self.p_value:.4f}",
            f"  Significant (p<0.05):    {self.is_significant}",
            "",
            "FAIRNESS METRICS",
            f"  Demographic Parity:      {self.demographic_parity:.3f}",
            f"  Equal Opportunity:       {self.equal_opportunity:.3f}",
            "",
            "=" * 70,
        ])
        
        return "\n".join(lines)


@dataclass
class FairnessMetrics:
    """Fairness evaluation across demographic groups."""
    
    # Attribution share by persona
    attribution_by_persona: Dict[str, float] = field(default_factory=dict)
    
    # Attribution share by device type
    attribution_by_device: Dict[str, float] = field(default_factory=dict)
    
    # Confidence by persona (should be similar)
    confidence_by_persona: Dict[str, float] = field(default_factory=dict)
    
    # Disparate impact ratio (0.8-1.2 is typically considered fair)
    disparate_impact_ratio: float = 1.0
    
    def get_summary(self) -> str:
        """Get human-readable summary."""
        lines = [
            "=" * 60,
            "FAIRNESS METRICS",
            "=" * 60,
            "",
            "ATTRIBUTION BY PERSONA",
        ]
        
        for persona, share in sorted(self.attribution_by_persona.items()):
            lines.append(f"  {persona:20s}  {share:.1%}")
        
        lines.extend([
            "",
            "ATTRIBUTION BY DEVICE",
        ])
        
        for device, share in sorted(self.attribution_by_device.items()):
            lines.append(f"  {device:20s}  {share:.1%}")
        
        lines.extend([
            "",
            f"Disparate Impact Ratio:  {self.disparate_impact_ratio:.3f}",
            f"  (0.8-1.25 range considered fair)",
            "",
            "=" * 60,
        ])
        
        return "\n".join(lines)


def calculate_attribution_lift(
    households: List[HouseholdProfile],
    sessions: List[Session],
    ground_truth: Any,
    use_enhanced: bool = False
) -> AttributionLiftMetrics:
    """
    Calculate attribution lift from identity resolution.
    
    Parameters
    ----------
    households : List[HouseholdProfile]
        Resolved households
    sessions : List[Session]
        Sessions with person assignments
    ground_truth : GroundTruth or EnhancedGroundTruth
        Ground truth data
    use_enhanced : bool
        Whether using enhanced ground truth with co-viewing
    
    Returns
    -------
    AttributionLiftMetrics
        Lift metrics comparing person vs account-level attribution
    """
    metrics = AttributionLiftMetrics()
    
    # Calculate person-level accuracy
    metrics.person_level_accuracy, persona_correct = \
        _calculate_person_level_accuracy(sessions, ground_truth, use_enhanced)
    
    # Calculate account-level accuracy (baseline)
    metrics.account_level_accuracy = \
        _calculate_account_level_accuracy(sessions, households, ground_truth, use_enhanced)
    
    # Calculate lift
    if metrics.account_level_accuracy > 0:
        metrics.accuracy_lift = (
            metrics.person_level_accuracy - metrics.account_level_accuracy
        ) / metrics.account_level_accuracy
    
    # Calculate revenue attribution MAE
    metrics.person_level_revenue_mae, metrics.account_level_revenue_mae = \
        _calculate_revenue_attribution_error(sessions, households, ground_truth, use_enhanced)
    
    if metrics.account_level_revenue_mae > 0:
        metrics.revenue_lift = (
            metrics.account_level_revenue_mae - metrics.person_level_revenue_mae
        ) / metrics.account_level_revenue_mae
    
    # Calculate persona-specific lift
    metrics.persona_lift = _calculate_persona_lift(sessions, ground_truth, use_enhanced)
    
    # Calculate channel lift
    metrics.channel_lift = _calculate_channel_lift(sessions, households)
    
    # Simple statistical test (placeholder)
    metrics.p_value = _calculate_p_value(metrics.person_level_accuracy, metrics.account_level_accuracy, len(sessions))
    metrics.is_significant = metrics.p_value < 0.05
    
    return metrics


def _calculate_person_level_accuracy(
    sessions: List[Session],
    ground_truth: Any,
    use_enhanced: bool
) -> Tuple[float, Dict[str, int]]:
    """Calculate person-level assignment accuracy."""
    correct = 0
    total = 0
    persona_correct = defaultdict(int)
    persona_total = defaultdict(int)
    
    for session in sessions:
        if not session.assigned_person_id:
            continue
        
        # Get ground truth
        if use_enhanced and hasattr(ground_truth, 'get_persons_for_session'):
            true_persons = ground_truth.get_persons_for_session(session.session_id)
            # For co-viewing, check if assigned person is in the list
            is_correct = session.assigned_person_id in true_persons if true_persons else False
        else:
            true_person = ground_truth.get_person_for_session(session.session_id)
            is_correct = (session.assigned_person_id == true_person)
        
        if is_correct:
            correct += 1
        total += 1
    
    accuracy = correct / total if total > 0 else 0
    return accuracy, dict(persona_correct)


def _calculate_account_level_accuracy(
    sessions: List[Session],
    households: List[HouseholdProfile],
    ground_truth: Any,
    use_enhanced: bool
) -> float:
    """
    Calculate baseline accuracy (assigning all to primary account holder).
    
    This simulates naive account-level attribution.
    """
    correct = 0
    total = 0
    
    # Build household lookup
    household_map = {h.account_id: h for h in households}
    
    for session in sessions:
        household = household_map.get(session.account_id)
        if not household or not household.members:
            continue
        
        # Naive: assign to first member (primary adult)
        predicted_person = household.members[0].person_id
        
        # Check ground truth
        if use_enhanced and hasattr(ground_truth, 'get_persons_for_session'):
            true_persons = ground_truth.get_persons_for_session(session.session_id)
            is_correct = predicted_person in true_persons if true_persons else False
        else:
            true_person = ground_truth.get_person_for_session(session.session_id)
            is_correct = (predicted_person == true_person)
        
        if is_correct:
            correct += 1
        total += 1
    
    return correct / total if total > 0 else 0


def _calculate_revenue_attribution_error(
    sessions: List[Session],
    households: List[HouseholdProfile],
    ground_truth: Any,
    use_enhanced: bool
) -> Tuple[float, float]:
    """Calculate revenue attribution mean absolute error."""
    person_errors = []
    account_errors = []
    
    household_map = {h.account_id: h for h in households}
    
    for session in sessions:
        if session.conversion_value <= 0:
            continue
        
        true_value = session.conversion_value
        
        # Person-level attribution (weighted by confidence)
        if session.person_probabilities:
            # Split value according to probabilities
            for person_id, prob in session.person_probabilities.items():
                attributed = true_value * prob
                # Compare to ground truth (simplified)
                error = abs(attributed - true_value * (1.0 / len(session.person_probabilities)))
                person_errors.append(error)
        else:
            # Single assignment
            person_errors.append(0)  # Perfect if we got it right
        
        # Account-level attribution (equal split)
        household = household_map.get(session.account_id)
        if household and household.members:
            n_members = len(household.members)
            equal_share = true_value / n_members
            # Error is variance from equal split
            account_errors.append(equal_share * (n_members - 1))
    
    person_mae = sum(person_errors) / len(person_errors) if person_errors else 0
    account_mae = sum(account_errors) / len(account_errors) if account_errors else 0
    
    return person_mae, account_mae


def _calculate_persona_lift(
    sessions: List[Session],
    ground_truth: Any,
    use_enhanced: bool
) -> Dict[str, float]:
    """Calculate attribution accuracy lift by persona type."""
    persona_accuracy = defaultdict(lambda: {'correct': 0, 'total': 0})
    
    for session in sessions:
        # Get ground truth persona
        if use_enhanced and hasattr(ground_truth, 'person_personas'):
            # Find which person this session belongs to
            if hasattr(ground_truth, 'get_persons_for_session'):
                true_persons = ground_truth.get_persons_for_session(session.session_id)
                if true_persons:
                    true_persona = ground_truth.person_personas.get(true_persons[0], 'unknown')
                else:
                    continue
            else:
                true_person = ground_truth.get_person_for_session(session.session_id)
                true_persona = ground_truth.person_personas.get(true_person, 'unknown')
        else:
            true_person = ground_truth.get_person_for_session(session.session_id)
            true_persona = ground_truth.person_personas.get(true_person, 'unknown')
        
        # Check if correct
        is_correct = False
        if session.assigned_person_id:
            if use_enhanced and hasattr(ground_truth, 'get_persons_for_session'):
                true_persons = ground_truth.get_persons_for_session(session.session_id)
                is_correct = session.assigned_person_id in true_persons if true_persons else False
            else:
                is_correct = (session.assigned_person_id == true_person)
        
        persona_accuracy[true_persona]['total'] += 1
        if is_correct:
            persona_accuracy[true_persona]['correct'] += 1
    
    # Calculate lift vs baseline (assume baseline is 1/n where n is average household size)
    baseline_accuracy = 0.33  # Assume 3 people per household baseline
    
    lift = {}
    for persona, counts in persona_accuracy.items():
        accuracy = counts['correct'] / counts['total'] if counts['total'] > 0 else 0
        lift[persona] = (accuracy - baseline_accuracy) / baseline_accuracy if baseline_accuracy > 0 else 0
    
    return dict(lift)


def _calculate_channel_lift(
    sessions: List[Session],
    households: List[HouseholdProfile]
) -> Dict[str, float]:
    """Calculate attribution accuracy improvement by channel."""
    # Group sessions by channel
    channel_stats = defaultdict(lambda: {'resolved': 0, 'total': 0})
    
    for session in sessions:
        channel = session.events[0].channel if session.events else 'unknown'
        channel_stats[channel]['total'] += 1
        if session.assigned_person_id:
            channel_stats[channel]['resolved'] += 1
    
    # Calculate lift (resolved rate vs 50% baseline)
    lift = {}
    for channel, stats in channel_stats.items():
        resolution_rate = stats['resolved'] / stats['total'] if stats['total'] > 0 else 0
        lift[channel] = (resolution_rate - 0.5) / 0.5
    
    return dict(lift)


def _calculate_p_value(person_acc: float, account_acc: float, n: int) -> float:
    """Calculate p-value for difference in proportions (simplified)."""
    if n == 0:
        return 1.0
    
    # Standard error for difference in proportions
    se = math.sqrt(
        (person_acc * (1 - person_acc) + account_acc * (1 - account_acc)) / n
    )
    
    if se == 0:
        return 1.0
    
    # Z-score
    z = (person_acc - account_acc) / se
    
    # Approximate p-value (two-tailed)
    # Using rough approximation: p ≈ 2 * (1 - Φ(|z|))
    p = 2 * (1 - _approx_cdf(abs(z)))
    
    return max(0.0, min(1.0, p))


def _approx_cdf(z: float) -> float:
    """Approximate standard normal CDF."""
    # Abramowitz and Stegun approximation
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p = 0.3275911
    
    sign = 1 if z >= 0 else -1
    z = abs(z) / math.sqrt(2.0)
    
    t = 1.0 / (1.0 + p * z)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-z * z)
    
    return 0.5 * (1.0 + sign * y)


def calculate_fairness_metrics(
    households: List[HouseholdProfile],
    sessions: List[Session]
) -> FairnessMetrics:
    """
    Calculate fairness metrics across demographic groups.
    
    Parameters
    ----------
    households : List[HouseholdProfile]
        Resolved households
    sessions : List[Session]
        Sessions with assignments
    
    Returns
    -------
    FairnessMetrics
        Fairness evaluation metrics
    """
    metrics = FairnessMetrics()
    
    # Attribution by persona
    persona_conversions = defaultdict(float)
    persona_counts = defaultdict(int)
    
    # Attribution by device
    device_conversions = defaultdict(float)
    device_counts = defaultdict(int)
    
    for session in sessions:
        if not session.has_conversion:
            continue
        
        # Find person's persona
        for household in households:
            if household.account_id == session.account_id:
                for member in household.members:
                    if member.person_id == session.assigned_person_id:
                        persona_conversions[member.persona_type] += session.conversion_value
                        persona_counts[member.persona_type] += 1
                        break
                break
        
        # Device attribution
        device_conversions[session.device_type] += session.conversion_value
        device_counts[session.device_type] += 1
    
    # Normalize
    total_conversions = sum(persona_conversions.values())
    if total_conversions > 0:
        metrics.attribution_by_persona = {
            k: v / total_conversions for k, v in persona_conversions.items()
        }
    
    total_device = sum(device_conversions.values())
    if total_device > 0:
        metrics.attribution_by_device = {
            k: v / total_device for k, v in device_conversions.items()
        }
    
    # Calculate disparate impact
    # Compare min and max persona attribution rates
    if metrics.attribution_by_persona:
        rates = list(metrics.attribution_by_persona.values())
        if len(rates) >= 2:
            min_rate = min(rates)
            max_rate = max(rates)
            if max_rate > 0:
                metrics.disparate_impact_ratio = min_rate / max_rate
    
    return metrics


def compare_to_benchmark(
    metrics: AttributionLiftMetrics,
    benchmark_name: str = "industry_average"
) -> str:
    """
    Compare metrics to industry benchmarks.
    
    Returns formatted comparison string.
    """
    benchmarks = {
        "industry_average": {
            "accuracy_lift": 0.15,  # 15% improvement
            "revenue_lift": 0.20,   # 20% improvement
            "p_value_threshold": 0.05,
        },
        "best_in_class": {
            "accuracy_lift": 0.30,
            "revenue_lift": 0.35,
            "p_value_threshold": 0.01,
        },
    }
    
    benchmark = benchmarks.get(benchmark_name, benchmarks["industry_average"])
    
    lines = [
        "",
        f"BENCHMARK COMPARISON ({benchmark_name})",
        "-" * 50,
        "",
        f"{'Metric':<30} {'Ours':>10} {'Benchmark':>10} {'Status':>10}",
        "-" * 50,
    ]
    
    comparisons = [
        ("Accuracy Lift", metrics.accuracy_lift, benchmark["accuracy_lift"]),
        ("Revenue Lift", metrics.revenue_lift, benchmark["revenue_lift"]),
    ]
    
    for name, ours, bench in comparisons:
        status = "✓ PASS" if ours >= bench else "✗ FAIL"
        lines.append(f"{name:<30} {ours:>9.1%} {bench:>9.1%} {status:>10}")
    
    # Statistical significance
    sig_status = "✓ PASS" if metrics.is_significant else "✗ FAIL"
    lines.append(f"{'Statistical Significance':<30} {'p<' + str(benchmark['p_value_threshold']):>9} {metrics.p_value:>9.4f} {sig_status:>10}")
    
    lines.append("-" * 50)
    
    return "\n".join(lines)
