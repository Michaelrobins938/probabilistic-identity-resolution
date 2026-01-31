"""
Integration Test: WWE Raw Live Event Scenario

Tests identity resolution during high-traffic live events with co-viewing.
This is a critical use case for Netflix-style streaming attribution.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from typing import List, Dict, Any, Tuple
from datetime import datetime
from dataclasses import dataclass

from models.streaming_event import StreamingEvent, Session, group_events_into_sessions
from models.household_profile import HouseholdProfile
from core.probabilistic_resolver import ProbabilisticIdentityResolver, ResolverConfig
from core.household_inference import HouseholdInferenceEngine
from core.cross_device_linker import CrossDeviceLinker
from validation.enhanced_synthetic import generate_wwe_raw_scenario, EnhancedGroundTruth
from validation.resolution_metrics import evaluate_resolution, ResolutionMetrics
from validation.attribution_lift import calculate_attribution_lift, calculate_fairness_metrics, AttributionLiftMetrics
from adapters.attribution_adapter import AttributionAdapter


@dataclass
class WWETestResult:
    """Results from WWE Raw scenario test."""
    test_name: str
    n_households: int
    n_events: int
    n_sessions: int
    
    # Resolution accuracy
    household_size_accuracy: float
    person_assignment_accuracy: float
    co_viewing_detection_rate: float
    
    # Attribution lift
    accuracy_lift: float
    revenue_lift: float
    
    # Performance
    processing_time_seconds: float
    
    # Co-viewing metrics
    co_viewing_sessions_total: int
    co_viewing_detected: int
    
    # Device linking
    cross_device_links_found: int
    
    passed: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "test_name": self.test_name,
            "n_households": self.n_households,
            "n_events": self.n_events,
            "n_sessions": self.n_sessions,
            "household_size_accuracy": self.household_size_accuracy,
            "person_assignment_accuracy": self.person_assignment_accuracy,
            "co_viewing_detection_rate": self.co_viewing_detection_rate,
            "accuracy_lift": self.accuracy_lift,
            "revenue_lift": self.revenue_lift,
            "processing_time_seconds": self.processing_time_seconds,
            "co_viewing_sessions_total": self.co_viewing_sessions_total,
            "co_viewing_detected": self.co_viewing_detected,
            "cross_device_links_found": self.cross_device_links_found,
            "passed": self.passed,
        }
    
    def get_summary(self) -> str:
        return f"""
{'='*70}
WWE RAW LIVE EVENT TEST RESULTS
{'='*70}

Test Configuration:
  Households:              {self.n_households}
  Total Events:            {self.n_events}
  Sessions Generated:      {self.n_sessions}

Accuracy Metrics:
  Household Size Accuracy: {self.household_size_accuracy:.1%}
  Person Assignment Acc:   {self.person_assignment_accuracy:.1%}
  Co-Viewing Detection:    {self.co_viewing_detection_rate:.1%}

Attribution Lift:
  Accuracy Lift:           {self.accuracy_lift:+.1%}
  Revenue Attribution:     {self.revenue_lift:+.1%}

Co-Viewing Analysis:
  Co-Viewing Sessions:     {self.co_viewing_sessions_total}
  Correctly Detected:      {self.co_viewing_detected}

Cross-Device Linking:
  Links Found:             {self.cross_device_links_found}

Performance:
  Processing Time:         {self.processing_time_seconds:.2f}s

Status: {'✓ PASSED' if self.passed else '✗ FAILED'}
{'='*70}
"""


def run_wwe_raw_integration_test(
    n_households: int = 20,
    verbose: bool = True
) -> WWETestResult:
    """
    Run integration test for WWE Raw live event scenario.
    
    This tests:
    1. High-traffic live event handling
    2. Co-viewing detection (multiple people watching together)
    3. Cross-device linking during event
    4. Attribution accuracy improvement
    
    Parameters
    ----------
    n_households : int
        Number of households to simulate
    verbose : bool
        Print detailed progress
    
    Returns
    -------
    WWETestResult
        Test results with metrics
    """
    import time
    
    if verbose:
        print(f"\n{'='*70}")
        print("WWE RAW LIVE EVENT INTEGRATION TEST")
        print(f"{'='*70}\n")
        print(f"Generating synthetic data for {n_households} households...")
    
    start_time = time.time()
    
    # Step 1: Generate WWE Raw scenario data
    events, ground_truth = generate_wwe_raw_scenario(
        n_households=n_households,
        include_co_viewing=True
    )
    
    if verbose:
        print(f"  Generated {len(events)} events")
        print(f"  Co-viewing sessions: {len(ground_truth.co_viewing_sessions)}")
        print(f"\nGrouping events into sessions...")
    
    # Step 2: Group events into sessions
    sessions = group_events_into_sessions(events, session_gap_minutes=30)
    
    if verbose:
        print(f"  Created {len(sessions)} sessions")
        print(f"\nRunning identity resolution...")
    
    # Step 3: Run identity resolution
    config = ResolverConfig(
        enable_household_inference=True,
        enable_cross_device_linking=True,
        enable_behavioral_segmentation=True,
        confidence_threshold=0.6,
    )
    
    resolver = ProbabilisticIdentityResolver(config)
    resolution_result = resolver.resolve(sessions, events)
    
    processing_time = time.time() - start_time
    
    if verbose:
        print(f"  Resolved {len(resolution_result.households)} households")
        print(f"  Found {len(resolution_result.cross_device_links)} cross-device links")
        print(f"\nEvaluating results...")
    
    # Step 4: Evaluate resolution accuracy
    metrics = evaluate_resolution(
        households=resolution_result.households,
        sessions=sessions,
        ground_truth=ground_truth,
        predicted_device_links=[
            (link.device_a, link.device_b) 
            for link in resolution_result.cross_device_links
        ]
    )
    
    # Step 5: Calculate attribution lift
    lift_metrics = calculate_attribution_lift(
        households=resolution_result.households,
        sessions=sessions,
        ground_truth=ground_truth,
        use_enhanced=True
    )
    
    # Step 6: Calculate fairness metrics
    fairness = calculate_fairness_metrics(
        households=resolution_result.households,
        sessions=sessions
    )
    
    # Step 7: Analyze co-viewing detection
    co_viewing_detected = _analyze_co_viewing_detection(
        sessions, ground_truth, resolution_result.households
    )
    
    # Step 8: Build result
    result = WWETestResult(
        test_name="WWE Raw Live Event",
        n_households=n_households,
        n_events=len(events),
        n_sessions=len(sessions),
        household_size_accuracy=metrics.household_size_accuracy,
        person_assignment_accuracy=metrics.person_assignment_accuracy,
        co_viewing_detection_rate=co_viewing_detected / max(1, len(ground_truth.co_viewing_sessions)),
        accuracy_lift=lift_metrics.accuracy_lift,
        revenue_lift=lift_metrics.revenue_lift,
        processing_time_seconds=processing_time,
        co_viewing_sessions_total=len(ground_truth.co_viewing_sessions),
        co_viewing_detected=co_viewing_detected,
        cross_device_links_found=len(resolution_result.cross_device_links),
    )
    
    # Determine pass/fail
    result.passed = (
        metrics.person_assignment_accuracy >= 0.70 and  # At least 70% accuracy
        lift_metrics.accuracy_lift >= 0.15 and          # 15% lift over baseline
        processing_time < 60                            # Under 60 seconds
    )
    
    if verbose:
        print(f"\n{result.get_summary()}")
        print(f"\nDetailed Resolution Metrics:")
        print(metrics.get_summary())
        print(f"\n{lift_metrics.get_summary()}")
        print(f"\n{fairness.get_summary()}")
    
    return result


def _analyze_co_viewing_detection(
    sessions: List[Session],
    ground_truth: EnhancedGroundTruth,
    households: List[HouseholdProfile]
) -> int:
    """
    Analyze how well co-viewing was detected.
    
    In co-viewing, the same TV session should be attributed to multiple people.
    We check if the resolution correctly identifies these multi-person sessions.
    """
    detected = 0
    
    for session in sessions:
        # Check if this was actually a co-viewing session
        if session.session_id in ground_truth.co_viewing_sessions:
            # Get true persons
            true_persons = ground_truth.get_persons_for_session(session.session_id)
            
            if len(true_persons) > 1:
                # This was co-viewing - check if we detected multiple people
                # (By looking at probabilities or session metadata)
                if session.person_probabilities and len(session.person_probabilities) > 1:
                    # Check if top probabilities indicate co-viewing
                    sorted_probs = sorted(
                        session.person_probabilities.items(),
                        key=lambda x: x[1],
                        reverse=True
                    )
                    
                    # If second person has >20% probability, consider it detected
                    if len(sorted_probs) >= 2 and sorted_probs[1][1] > 0.2:
                        detected += 1
    
    return detected


def run_full_integration_test_suite() -> Dict[str, Any]:
    """
    Run comprehensive integration test suite.
    
    Returns summary of all test results.
    """
    results = []
    
    # Test 1: Small scale (10 households)
    print("\n" + "="*70)
    print("TEST 1: Small Scale (10 households)")
    print("="*70)
    result1 = run_wwe_raw_integration_test(n_households=10, verbose=True)
    results.append(result1)
    
    # Test 2: Medium scale (50 households)
    print("\n" + "="*70)
    print("TEST 2: Medium Scale (50 households)")
    print("="*70)
    result2 = run_wwe_raw_integration_test(n_households=50, verbose=True)
    results.append(result2)
    
    # Test 3: Without co-viewing (baseline)
    print("\n" + "="*70)
    print("TEST 3: Baseline (no co-viewing)")
    print("="*70)
    result3 = run_wwe_raw_baseline_test(n_households=20, verbose=True)
    results.append(result3)
    
    # Summary
    summary = {
        "total_tests": len(results),
        "passed": sum(1 for r in results if r.passed),
        "failed": sum(1 for r in results if not r.passed),
        "average_accuracy": sum(r.person_assignment_accuracy for r in results) / len(results),
        "average_lift": sum(r.accuracy_lift for r in results) / len(results),
        "results": [r.to_dict() for r in results],
    }
    
    print("\n" + "="*70)
    print("INTEGRATION TEST SUITE SUMMARY")
    print("="*70)
    print(f"Total Tests: {summary['total_tests']}")
    print(f"Passed: {summary['passed']}")
    print(f"Failed: {summary['failed']}")
    print(f"Average Accuracy: {summary['average_accuracy']:.1%}")
    print(f"Average Lift: {summary['average_lift']:+.1%}")
    print("="*70)
    
    return summary


def run_wwe_raw_baseline_test(
    n_households: int = 20,
    verbose: bool = True
) -> WWETestResult:
    """Run baseline test without co-viewing for comparison."""
    from validation.enhanced_synthetic import generate_enhanced_synthetic_data, EnhancedSyntheticConfig, ViewingPattern
    
    if verbose:
        print(f"\nGenerating baseline data (no co-viewing)...")
    
    config = EnhancedSyntheticConfig(
        n_households=n_households,
        co_viewing_rate=0.0,  # No co-viewing
        viewing_patterns=[ViewingPattern.REGULAR, ViewingPattern.BINGE],
        seed=42,
    )
    
    events, ground_truth = generate_enhanced_synthetic_data(config)
    sessions = group_events_into_sessions(events, session_gap_minutes=30)
    
    # Run resolution
    resolver = ProbabilisticIdentityResolver(ResolverConfig())
    resolution_result = resolver.resolve(sessions, events)
    
    # Evaluate
    metrics = evaluate_resolution(
        households=resolution_result.households,
        sessions=sessions,
        ground_truth=ground_truth,
    )
    
    lift_metrics = calculate_attribution_lift(
        households=resolution_result.households,
        sessions=sessions,
        ground_truth=ground_truth,
        use_enhanced=True
    )
    
    result = WWETestResult(
        test_name="WWE Raw Baseline (no co-viewing)",
        n_households=n_households,
        n_events=len(events),
        n_sessions=len(sessions),
        household_size_accuracy=metrics.household_size_accuracy,
        person_assignment_accuracy=metrics.person_assignment_accuracy,
        co_viewing_detection_rate=0.0,
        accuracy_lift=lift_metrics.accuracy_lift,
        revenue_lift=lift_metrics.revenue_lift,
        processing_time_seconds=0.0,
        co_viewing_sessions_total=0,
        co_viewing_detected=0,
        cross_device_links_found=len(resolution_result.cross_device_links),
        passed=metrics.person_assignment_accuracy >= 0.70,
    )
    
    if verbose:
        print(f"\n{result.get_summary()}")
    
    return result


# Main execution
if __name__ == "__main__":
    print("Running WWE Raw Integration Test Suite...")
    results = run_full_integration_test_suite()
    
    # Exit with appropriate code
    import sys
    sys.exit(0 if results["failed"] == 0 else 1)
