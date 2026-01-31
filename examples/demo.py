#!/usr/bin/env python3
"""
Probabilistic Identity Resolution - Demo Script

Demonstrates the full identity resolution pipeline:
1. Generate synthetic household data with known ground truth
2. Run identity resolution
3. Evaluate accuracy against ground truth
4. Show attribution impact

Usage:
    python demo.py
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from validation.synthetic_households import (
    generate_synthetic_household_data,
    SyntheticConfig,
)
from validation.resolution_metrics import (
    evaluate_resolution,
    compare_to_baseline,
)
from core.probabilistic_resolver import ProbabilisticIdentityResolver, ResolverConfig
from models.streaming_event import group_events_into_sessions
from adapters.attribution_adapter import AttributionAdapter


def main():
    print("=" * 70)
    print("PROBABILISTIC IDENTITY RESOLUTION - DEMO")
    print("Solving the Netflix Co-Viewing Problem")
    print("=" * 70)
    print()

    # Step 1: Generate synthetic data
    print("Step 1: Generating synthetic household data...")
    print("-" * 50)

    config = SyntheticConfig(
        n_households=30,
        persons_per_household_range=(1, 4),
        devices_per_person_range=(1, 2),
        sessions_per_person_range=(20, 50),
        noise_level=0.15,
        seed=42,
    )

    events, ground_truth = generate_synthetic_household_data(config)

    print(f"  Generated {len(events):,} events")
    print(f"  Across {config.n_households} households")
    print(f"  With {len(ground_truth.session_to_person):,} sessions")
    print(f"  True household sizes: {dict(list({hh: len(members) for hh, members in ground_truth.household_members.items()}.items())[:5])}...")
    print()

    # Step 2: Run identity resolution
    print("Step 2: Running identity resolution...")
    print("-" * 50)

    resolver = ProbabilisticIdentityResolver(ResolverConfig(
        session_gap_minutes=30,
    ))

    result = resolver.resolve(events)

    print(result.get_summary())
    print()

    # Step 3: Evaluate accuracy
    print("Step 3: Evaluating resolution accuracy...")
    print("-" * 50)

    sessions = group_events_into_sessions(events, session_gap_minutes=30)

    # Get predicted device links
    predicted_links = [(l.device_a, l.device_b) for l in result.device_links]

    metrics = evaluate_resolution(
        households=result.households,
        sessions=sessions,
        ground_truth=ground_truth,
        predicted_device_links=predicted_links,
    )

    print(metrics.get_summary())
    print()

    # Compare to baseline
    print(compare_to_baseline(metrics, "Random Assignment"))
    print()

    # Step 4: Show attribution impact
    print("Step 4: Attribution Impact Analysis")
    print("-" * 50)

    adapter = AttributionAdapter(result)
    household_summaries = adapter.get_household_attribution_summary()

    # Show first 3 households
    for summary in household_summaries[:3]:
        print(f"\nHousehold: {summary['account_id']} ({summary['household_type']})")
        print(f"  Total Value: ${summary['total_conversion_value']:.2f}")
        print(f"  Members: {summary['estimated_size']}")

        print("\n  Traditional Attribution:")
        print(f"    Account: 100% = ${summary['total_conversion_value']:.2f}")

        print("\n  Identity-Resolved Attribution:")
        for member in summary['members']:
            share = member['attribution_share']
            value = member['attributed_value']
            print(f"    {member['label']} ({member['persona_type']}): "
                  f"{share:.0%} = ${value:.2f}")
            print(f"      Device: {member['primary_device']}, "
                  f"Genres: {', '.join(member['top_genres'])}")

    print()
    print("=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)
    print()
    print("Key Takeaways:")
    print(f"  1. Household size accuracy: {metrics.household_size_accuracy:.0%}")
    print(f"  2. Person assignment accuracy: {metrics.person_assignment_accuracy:.0%}")
    print(f"  3. Cross-device linking F1: {metrics.device_linking_f1:.2f}")
    print(f"  4. Overall score: {metrics.overall_score:.0%}")
    print()
    print("This demonstrates how identity resolution transforms:")
    print("  'Account X converted' -> 'Person A (primary adult) converted'")
    print()
    print("For Netflix, this means understanding WHO in a household")
    print("drives subscription decisions, not just which account.")


if __name__ == "__main__":
    main()
