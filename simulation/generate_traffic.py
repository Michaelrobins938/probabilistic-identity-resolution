"""
Generate Synthetic Traffic

Creates synthetic streaming data for stress testing.
Generates realistic household profiles with:
- Multiple persons per household (1-6)
- Behavioral patterns (time, device, genre)
- Cross-device usage
- Co-viewing scenarios

Usage:
    python simulation/generate_traffic.py --households 50000 --output data/synthetic.parquet
"""

import argparse
import json
from datetime import datetime
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from validation.synthetic_households import generate_synthetic_household_data, SyntheticConfig
from validation.enhanced_synthetic import generate_enhanced_synthetic_data, EnhancedSyntheticConfig


def generate_traffic(
    n_households: int = 50000,
    use_enhanced: bool = True,
    seed: int = 42,
    output_path: str = None
):
    """Generate synthetic traffic data."""
    
    print(f"🎲 Generating {n_households:,} synthetic households...")
    print(f"   Mode: {'Enhanced (realistic)' if use_enhanced else 'Basic'}")
    print()
    
    if use_enhanced:
        config = EnhancedSyntheticConfig(
            n_households=n_households,
            seed=seed,
            viewing_patterns=['regular', 'binge', 'occasional']
        )
        events, ground_truth = generate_enhanced_synthetic_data(config)
    else:
        config = SyntheticConfig(
            n_households=n_households,
            seed=seed
        )
        events, ground_truth = generate_synthetic_household_data(config)
    
    # Calculate statistics
    n_events = len(events)
    n_persons = sum(ground_truth.get_household_size(hh) for hh in ground_truth.household_members)
    
    stats = {
        'n_households': n_households,
        'n_events': n_events,
        'n_persons': n_persons,
        'avg_events_per_household': n_events / n_households,
        'avg_persons_per_household': n_persons / n_households,
        'generation_timestamp': datetime.now().isoformat(),
        'config': {
            'seed': seed,
            'use_enhanced': use_enhanced
        }
    }
    
    print(f"✅ Generated:")
    print(f"   {n_households:,} households")
    print(f"   {n_persons:,} persons (avg {stats['avg_persons_per_household']:.1f}/household)")
    print(f"   {n_events:,} events (avg {stats['avg_events_per_household']:.0f}/household)")
    print()
    
    # Save to file if requested
    if output_path:
        print(f"💾 Saving to {output_path}...")
        # Would use pandas to save parquet in real implementation
        with open(output_path.replace('.parquet', '.json'), 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"   Saved stats to {output_path.replace('.parquet', '.json')}")
    
    return events, ground_truth, stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate synthetic traffic')
    parser.add_argument('--households', type=int, default=50000, help='Number of households')
    parser.add_argument('--output', type=str, default='data/synthetic.parquet', help='Output path')
    parser.add_argument('--basic', action='store_true', help='Use basic mode (not enhanced)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    events, ground_truth, stats = generate_traffic(
        n_households=args.households,
        use_enhanced=not args.basic,
        seed=args.seed,
        output_path=args.output
    )
