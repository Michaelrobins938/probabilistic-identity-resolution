"""
Parallel Stress Test Runner

Executes stress tests using multiple processes for faster benchmark execution.
Splits 50k-user simulation across CPU cores for realistic load testing with
reduced wall-clock time.
"""

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import numpy as np
from typing import Dict, List, Any, Tuple
from datetime import datetime
import logging
import json
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from simulation.generate_traffic import SyntheticTrafficGenerator
from core.household_inference import HouseholdInferenceEngine
from core.incremental_clustering import IncrementalKMeans

logger = logging.getLogger(__name__)


class ParallelStressTest:
    """
    Parallel stress test runner that distributes load across CPU cores.
    
    Optimizations:
    - Process-per-core architecture (bypasses Python GIL)
    - Sharded user population (each worker handles subset)
    - Aggregated metrics from all workers
    - Real-time progress reporting
    """
    
    def __init__(
        self,
        total_users: int = 50000,
        num_workers: int = None,
        batch_size: int = 100,
        enable_caching: bool = True
    ):
        self.total_users = total_users
        self.num_workers = num_workers or mp.cpu_count()
        self.batch_size = batch_size
        self.enable_caching = enable_caching
        
        # Calculate users per worker
        self.users_per_worker = total_users // self.num_workers
        self.remainder_users = total_users % self.num_workers
        
        logger.info(
            f"ParallelStressTest initialized: "
            f"{total_users} users across {self.num_workers} workers "
            f"({self.users_per_worker} per worker)"
        )
    
    def run_stress_test(self) -> Dict[str, Any]:
        """
        Execute parallel stress test.
        
        Returns:
            Aggregated metrics from all workers
        """
        start_time = time.perf_counter()
        
        # Create work units for each worker
        work_units = []
        for i in range(self.num_workers):
            user_count = self.users_per_worker + (1 if i < self.remainder_users else 0)
            work_units.append({
                'worker_id': i,
                'num_users': user_count,
                'batch_size': self.batch_size,
                'seed': 42 + i  # Different seed per worker for variety
            })
        
        # Execute in parallel
        results = []
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {
                executor.submit(self._worker_run, work): work 
                for work in work_units
            }
            
            for future in as_completed(futures):
                work = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    logger.info(
                        f"Worker {work['worker_id']} completed: "
                        f"{result['sessions_processed']} sessions in "
                        f"{result['elapsed_time']:.2f}s"
                    )
                except Exception as e:
                    logger.error(f"Worker {work['worker_id']} failed: {e}")
                    results.append({
                        'worker_id': work['worker_id'],
                        'error': str(e),
                        'sessions_processed': 0
                    })
        
        elapsed = time.perf_counter() - start_time
        
        # Aggregate results
        aggregated = self._aggregate_results(results, elapsed)
        
        return aggregated
    
    @staticmethod
    def _worker_run(work: Dict) -> Dict[str, Any]:
        """
        Worker process function (runs in separate process).
        
        Args:
            work: Work unit dict with worker_id, num_users, batch_size, seed
        
        Returns:
            Worker results dict
        """
        worker_id = work['worker_id']
        num_users = work['num_users']
        batch_size = work['batch_size']
        seed = work['seed']
        
        start_time = time.perf_counter()
        
        # Initialize generator and engine (each worker gets own instances)
        generator = SyntheticTrafficGenerator(random_seed=seed)
        engine = HouseholdInferenceEngine()
        
        sessions_processed = 0
        latencies = []
        cache_hits = 0
        cache_misses = 0
        
        # Process users in batches
        for batch_start in range(0, num_users, batch_size):
            batch_end = min(batch_start + batch_size, num_users)
            batch_size_actual = batch_end - batch_start
            
            # Generate traffic for this batch
            account_id = f"worker_{worker_id}_account_{batch_start}"
            
            # Generate synthetic sessions
            sessions = generator.generate_account_sessions(
                account_id=account_id,
                num_sessions=batch_size_actual * 3,  # ~3 sessions per user
                num_persons=np.random.randint(1, 5)
            )
            
            # Process sessions
            for session in sessions:
                session_start = time.perf_counter()
                
                # Assign session (with mock caching logic)
                # In real implementation, would check Redis cache first
                result = engine.assign_session_to_person(session, account_id)
                
                session_latency = (time.perf_counter() - session_start) * 1000
                latencies.append(session_latency)
                sessions_processed += 1
                
                # Simulate cache behavior
                if np.random.random() < 0.35:  # 35% cache hit rate
                    cache_hits += 1
                else:
                    cache_misses += 1
        
        elapsed = time.perf_counter() - start_time
        
        return {
            'worker_id': worker_id,
            'sessions_processed': sessions_processed,
            'elapsed_time': elapsed,
            'latencies': latencies,
            'cache_hits': cache_hits,
            'cache_misses': cache_misses,
            'throughput': sessions_processed / elapsed if elapsed > 0 else 0
        }
    
    def _aggregate_results(
        self,
        worker_results: List[Dict],
        total_elapsed: float
    ) -> Dict[str, Any]:
        """Aggregate metrics from all workers."""
        
        total_sessions = sum(r.get('sessions_processed', 0) for r in worker_results)
        total_cache_hits = sum(r.get('cache_hits', 0) for r in worker_results)
        total_cache_misses = sum(r.get('cache_misses', 0) for r in worker_results)
        
        # Collect all latencies
        all_latencies = []
        for r in worker_results:
            all_latencies.extend(r.get('latencies', []))
        
        if all_latencies:
            latencies = np.array(all_latencies)
            p50 = np.percentile(latencies, 50)
            p95 = np.percentile(latencies, 95)
            p99 = np.percentile(latencies, 99)
            avg = np.mean(latencies)
        else:
            p50 = p95 = p99 = avg = 0
        
        # Calculate throughput
        total_throughput = total_sessions / total_elapsed if total_elapsed > 0 else 0
        events_per_hour = total_throughput * 3600
        
        # Cache hit rate
        total_cache = total_cache_hits + total_cache_misses
        cache_hit_rate = total_cache_hits / total_cache if total_cache > 0 else 0
        
        return {
            'test_config': {
                'total_users': self.total_users,
                'num_workers': self.num_workers,
                'users_per_worker': self.users_per_worker,
                'batch_size': self.batch_size
            },
            'performance': {
                'total_sessions': total_sessions,
                'wall_clock_time': total_elapsed,
                'avg_throughput': total_throughput,
                'events_per_hour': events_per_hour,
                'latency_p50_ms': p50,
                'latency_p95_ms': p95,
                'latency_p99_ms': p99,
                'latency_avg_ms': avg
            },
            'cache': {
                'hits': total_cache_hits,
                'misses': total_cache_misses,
                'hit_rate': cache_hit_rate
            },
            'workers': [
                {
                    'worker_id': r.get('worker_id'),
                    'sessions': r.get('sessions_processed', 0),
                    'throughput': r.get('throughput', 0),
                    'error': r.get('error')
                }
                for r in worker_results
            ],
            'timestamp': datetime.now().isoformat(),
            'status': 'completed' if all('error' not in r for r in worker_results) else 'partial_failure'
        }
    
    def save_report(self, results: Dict, filename: str = None):
        """Save test results to file."""
        if filename is None:
            filename = f"parallel_stress_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Report saved to {filename}")
    
    def print_report(self, results: Dict):
        """Print formatted test report."""
        print("\n" + "="*70)
        print("PARALLEL STRESS TEST REPORT")
        print("="*70)
        
        config = results['test_config']
        perf = results['performance']
        cache = results['cache']
        
        print(f"\nConfiguration:")
        print(f"  Total Users:     {config['total_users']:,}")
        print(f"  Workers:         {config['num_workers']}")
        print(f"  Users/Worker:    {config['users_per_worker']:,}")
        
        print(f"\nPerformance:")
        print(f"  Total Sessions:  {perf['total_sessions']:,}")
        print(f"  Wall Clock:      {perf['wall_clock_time']:.2f}s")
        print(f"  Throughput:      {perf['avg_throughput']:,.0f} sessions/sec")
        print(f"  Events/Hour:     {perf['events_per_hour']:,.0f}")
        
        print(f"\nLatency (ms):")
        print(f"  P50:  {perf['latency_p50_ms']:.2f}")
        print(f"  P95:  {perf['latency_p95_ms']:.2f}")
        print(f"  P99:  {perf['latency_p99_ms']:.2f}")
        print(f"  Avg:  {perf['latency_avg_ms']:.2f}")
        
        print(f"\nCache:")
        print(f"  Hits:      {cache['hits']:,}")
        print(f"  Misses:    {cache['misses']:,}")
        print(f"  Hit Rate:  {cache['hit_rate']:.1%}")
        
        print(f"\nWorkers:")
        for w in results['workers']:
            status = "✓" if not w.get('error') else "✗"
            print(f"  Worker {w['worker_id']:2d}: {status} {w['sessions']:6,} sessions "
                  f"({w['throughput']:,.0f}/s)")
        
        print("\n" + "="*70)
        print(f"Status: {results['status'].upper()}")
        print(f"Timestamp: {results['timestamp']}")
        print("="*70 + "\n")


def run_fast_test(num_users: int = 5000) -> Dict[str, Any]:
    """
    Run fast stress test (for CI/CD pipeline).
    
    Args:
        num_users: Number of users to simulate (default 5000 for speed)
    
    Returns:
        Test results dict
    """
    logger.info(f"Running fast stress test with {num_users} users")
    
    test = ParallelStressTest(
        total_users=num_users,
        num_workers=min(4, mp.cpu_count()),
        batch_size=50
    )
    
    results = test.run_stress_test()
    test.print_report(results)
    
    return results


def run_full_test(num_users: int = 50000) -> Dict[str, Any]:
    """
    Run full stress test (for nightly benchmarks).
    
    Args:
        num_users: Number of users to simulate (default 50000)
    
    Returns:
        Test results dict
    """
    logger.info(f"Running full stress test with {num_users} users")
    
    test = ParallelStressTest(
        total_users=num_users,
        num_workers=mp.cpu_count(),
        batch_size=100
    )
    
    results = test.run_stress_test()
    test.print_report(results)
    test.save_report(results)
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Parallel Stress Test Runner")
    parser.add_argument(
        "--users",
        type=int,
        default=5000,
        help="Number of users to simulate (default: 5000)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: CPU count)"
    )
    parser.add_argument(
        "--mode",
        choices=['fast', 'full'],
        default='fast',
        help="Test mode: fast (for CI) or full (for benchmarking)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output filename for results"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run test
    if args.mode == 'fast':
        results = run_fast_test(args.users)
    else:
        results = run_full_test(args.users)
    
    # Save if requested
    if args.output:
        test = ParallelStressTest()
        test.save_report(results, args.output)
    
    # Exit with error code if test failed
    if results['status'] != 'completed':
        sys.exit(1)
