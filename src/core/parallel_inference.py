"""
Parallel Household Inference Engine

Adds multiprocessing support for large-scale household inference.
Scales to millions of accounts using process pools.
"""

from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
from datetime import datetime
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import logging

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from models.streaming_event import Session
from models.household_profile import HouseholdProfile
from core.household_inference import HouseholdInferenceEngine, ClusteringConfig

logger = logging.getLogger(__name__)


@dataclass
class ParallelConfig:
    """Configuration for parallel processing."""
    max_workers: int = 4
    chunk_size: int = 100  # Accounts per batch
    use_processes: bool = True  # True = processes, False = threads
    timeout_seconds: int = 300
    

class ParallelHouseholdInference:
    """
    Parallel household inference for large-scale processing.
    
    Distributes account processing across multiple workers for speed.
    """
    
    def __init__(
        self,
        clustering_config: Optional[ClusteringConfig] = None,
        parallel_config: Optional[ParallelConfig] = None
    ):
        self.clustering_config = clustering_config or ClusteringConfig()
        self.parallel_config = parallel_config or ParallelConfig()
    
    def analyze_households_parallel(
        self,
        account_sessions: Dict[str, List[Session]]
    ) -> Dict[str, HouseholdProfile]:
        """
        Analyze multiple households in parallel.
        
        Parameters
        ----------
        account_sessions : Dict[str, List[Session]]
            Mapping of account_id -> list of sessions
        
        Returns
        -------
        Dict[str, HouseholdProfile]
            Mapping of account_id -> household profile
        """
        if not account_sessions:
            return {}
        
        # If small number of accounts, process sequentially
        if len(account_sessions) <= 4:
            return self._analyze_sequential(account_sessions)
        
        # Chunk accounts for efficient processing
        chunks = self._chunk_accounts(account_sessions, self.parallel_config.chunk_size)
        
        results = {}
        
        if self.parallel_config.use_processes:
            # Use process pool
            with ProcessPoolExecutor(max_workers=self.parallel_config.max_workers) as executor:
                # Submit all chunks
                future_to_chunk = {
                    executor.submit(self._process_chunk, chunk): chunk_id
                    for chunk_id, chunk in enumerate(chunks)
                }
                
                # Collect results as they complete
                for future in as_completed(future_to_chunk):
                    chunk_id = future_to_chunk[future]
                    try:
                        chunk_results = future.result(timeout=self.parallel_config.timeout_seconds)
                        results.update(chunk_results)
                        logger.info(f"Completed chunk {chunk_id + 1}/{len(chunks)}")
                    except Exception as e:
                        logger.error(f"Chunk {chunk_id} failed: {e}")
        else:
            # Use thread pool (for I/O bound scenarios)
            from concurrent.futures import ThreadPoolExecutor
            
            with ThreadPoolExecutor(max_workers=self.parallel_config.max_workers) as executor:
                futures = [
                    executor.submit(self._process_chunk, chunk)
                    for chunk in chunks
                ]
                
                for future in as_completed(futures):
                    try:
                        chunk_results = future.result(timeout=self.parallel_config.timeout_seconds)
                        results.update(chunk_results)
                    except Exception as e:
                        logger.error(f"Chunk failed: {e}")
        
        return results
    
    def _analyze_sequential(
        self,
        account_sessions: Dict[str, List[Session]]
    ) -> Dict[str, HouseholdProfile]:
        """Process accounts sequentially (for small batches)."""
        engine = HouseholdInferenceEngine(self.clustering_config)
        results = {}
        
        for account_id, sessions in account_sessions.items():
            try:
                household = engine.analyze_household(sessions, account_id)
                results[account_id] = household
            except Exception as e:
                logger.error(f"Failed to analyze household {account_id}: {e}")
        
        return results
    
    def _chunk_accounts(
        self,
        account_sessions: Dict[str, List[Session]],
        chunk_size: int
    ) -> List[Dict[str, List[Session]]]:
        """Split accounts into chunks for parallel processing."""
        items = list(account_sessions.items())
        chunks = []
        
        for i in range(0, len(items), chunk_size):
            chunk = dict(items[i:i + chunk_size])
            chunks.append(chunk)
        
        return chunks
    
    @staticmethod
    def _process_chunk(
        account_sessions: Dict[str, List[Session]]
    ) -> Dict[str, HouseholdProfile]:
        """Process a chunk of accounts (static method for pickling)."""
        # Create new engine instance (can't pickle the engine)
        config = ClusteringConfig()
        engine = HouseholdInferenceEngine(config)
        
        results = {}
        for account_id, sessions in account_sessions.items():
            try:
                household = engine.analyze_household(sessions, account_id)
                results[account_id] = household
            except Exception as e:
                # Log error but continue with other accounts
                print(f"Error processing {account_id}: {e}")
        
        return results


# Convenience functions

def analyze_households_parallel(
    sessions: List[Session],
    max_workers: int = 4,
    clustering_config: Optional[ClusteringConfig] = None
) -> Dict[str, HouseholdProfile]:
    """
    Analyze households from sessions using parallel processing.
    
    Parameters
    ----------
    sessions : List[Session]
        All sessions (will be grouped by account)
    max_workers : int
        Number of parallel workers
    clustering_config : ClusteringConfig, optional
        Clustering configuration
    
    Returns
    -------
    Dict[str, HouseholdProfile]
        Mapping of account_id -> household profile
    """
    # Group sessions by account
    account_sessions = {}
    for session in sessions:
        account_id = session.account_id
        if account_id not in account_sessions:
            account_sessions[account_id] = []
        account_sessions[account_id].append(session)
    
    # Run parallel analysis
    parallel_config = ParallelConfig(max_workers=max_workers)
    parallel_inference = ParallelHouseholdInference(clustering_config, parallel_config)
    
    return parallel_inference.analyze_households_parallel(account_sessions)


def benchmark_parallel_inference(
    n_households: int = 100,
    sessions_per_household: int = 50
) -> Dict[str, Any]:
    """
    Benchmark parallel vs sequential inference.
    
    Returns performance comparison.
    """
    import time
    from validation.synthetic_households import generate_synthetic_household_data
    
    # Generate test data
    from validation.synthetic_households import SyntheticConfig
    config = SyntheticConfig(n_households=n_households, sessions_per_person_range=(20, sessions_per_household))
    events, _ = generate_synthetic_household_data(config)
    
    from models.streaming_event import group_events_into_sessions
    sessions = group_events_into_sessions(events)
    
    # Group by account
    account_sessions = {}
    for session in sessions:
        if session.account_id not in account_sessions:
            account_sessions[session.account_id] = []
        account_sessions[session.account_id].append(session)
    
    results = {}
    
    # Sequential benchmark
    print(f"Benchmarking sequential processing ({n_households} households)...")
    start = time.time()
    engine = HouseholdInferenceEngine()
    sequential_results = {}
    for account_id, sessions in account_sessions.items():
        sequential_results[account_id] = engine.analyze_household(sessions, account_id)
    sequential_time = time.time() - start
    
    results['sequential'] = {
        'time_seconds': sequential_time,
        'households_processed': len(sequential_results),
        'throughput': len(sequential_results) / sequential_time,
    }
    
    # Parallel benchmarks with different worker counts
    for workers in [2, 4, 8]:
        if workers > mp.cpu_count():
            continue
        
        print(f"Benchmarking with {workers} workers...")
        start = time.time()
        parallel_config = ParallelConfig(max_workers=workers)
        parallel_inference = ParallelHouseholdInference(parallel_config=parallel_config)
        parallel_results = parallel_inference.analyze_households_parallel(account_sessions)
        parallel_time = time.time() - start
        
        speedup = sequential_time / parallel_time if parallel_time > 0 else 0
        efficiency = speedup / workers if workers > 0 else 0
        
        results[f'parallel_{workers}'] = {
            'time_seconds': parallel_time,
            'households_processed': len(parallel_results),
            'throughput': len(parallel_results) / parallel_time,
            'speedup': speedup,
            'efficiency': efficiency,
        }
    
    return results
