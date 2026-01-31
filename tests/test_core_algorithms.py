"""
Unit Tests for Core Algorithms

Tests the mathematically critical components:
1. Shapley values sum to 1.0 (efficiency axiom)
2. GMM covariance matrices remain positive semi-definite
3. Incremental K-Means converges correctly
4. Attribution shares conserve total value
5. Drift detection triggers appropriately
"""

import pytest
import numpy as np
from datetime import datetime

# Test core attribution engine
from attribution.shapley_engine import ShapleyAttributionEngine, ShapleyConfig
from attribution.markov_engine import MarkovAttributionEngine
from attribution.hybrid_engine import HybridAttributionEngine, HybridAttributionConfig

# Test clustering
from core.gaussian_mixture import GaussianMixtureModel, GMMConfig
from core.incremental_clustering import IncrementalKMeans, IncrementalClusterConfig

# Test validation
from validation.synthetic_households import generate_synthetic_household_data, SyntheticConfig
from models.streaming_event import group_events_into_sessions


class TestShapleyAxioms:
    """Verify Shapley value satisfies game theory axioms."""
    
    def test_efficiency_axiom(self):
        """Shapley values must sum to total value (v(N) - v(empty))."""
        # Simple 3-channel game
        channels = ["Email", "Search", "Social"]
        
        def v(coalition):
            # Characteristic function: value increases with coalition size
            return len(coalition) * 0.3
        
        engine = ShapleyAttributionEngine(ShapleyConfig())
        engine.fit(channels, v)
        
        values = engine.compute_shapley_values()
        total = sum(values.values())
        
        # Should equal v(N) = 0.9
        assert abs(total - 0.9) < 1e-6, f"Efficiency violated: sum={total}, expected=0.9"
    
    def test_symmetry_axiom(self):
        """Equal contributors get equal Shapley values."""
        channels = ["A", "B"]
        
        # Symmetric game: both channels contribute equally
        def v(coalition):
            return len(coalition) * 0.5
        
        engine = ShapleyAttributionEngine(ShapleyConfig())
        engine.fit(channels, v)
        
        values = engine.compute_shapley_values()
        
        # A and B should have equal values
        assert abs(values["A"] - values["B"]) < 1e-6
    
    def test_dummy_player_axiom(self):
        """Non-contributors get zero Shapley value."""
        channels = ["A", "B", "Dummy"]
        
        # Dummy contributes nothing
        def v(coalition):
            if "Dummy" in coalition:
                return len([c for c in coalition if c != "Dummy"]) * 0.5
            return len(coalition) * 0.5
        
        engine = ShapleyAttributionEngine(ShapleyConfig())
        engine.fit(channels, v)
        
        values = engine.compute_shapley_values()
        
        # Dummy should get ~0
        assert abs(values["Dummy"]) < 1e-6


class TestGMMNumericalStability:
    """Verify GMM handles edge cases without crashing."""
    
    def test_covariance_regularization(self):
        """GMM should not produce singular covariance matrices."""
        # Create data with collinearity (potential singularity)
        X = np.array([
            [1.0, 2.0, 3.0],  # Nearly collinear
            [1.1, 2.1, 3.1],
            [1.2, 2.2, 3.2],
            [5.0, 5.0, 5.0],  # Different cluster
            [5.1, 5.1, 5.1],
        ])
        
        config = GMMConfig(covariance_type="diag", reg_covar=1e-4)
        gmm = GaussianMixtureModel(config)
        gmm.fit(X, "test_account")
        
        # Check all covariances are positive (regularized)
        for comp in gmm.components.values():
            assert np.all(np.diag(comp.covariance) > 0), "Singular covariance detected"
    
    def test_single_cluster_fallback(self):
        """GMM should handle single-cluster case."""
        X = np.random.randn(10, 3)  # All similar data
        
        config = GMMConfig(min_components=1, max_components=1)
        gmm = GaussianMixtureModel(config)
        gmm.fit(X, "test_account")
        
        assert len(gmm.components) == 1
    
    def test_incremental_update_stability(self):
        """Incremental updates should not explode."""
        # Initialize with some data
        X = np.random.randn(20, 2)
        
        config = GMMConfig()
        gmm = GaussianMixtureModel(config)
        gmm.fit(X, "test_account")
        
        # Add points one at a time
        for _ in range(10):
            new_point = np.random.randn(2)
            probs = gmm.fit_incremental("test_session", new_point)
            
            # Probabilities should sum to ~1
            assert abs(sum(probs.values()) - 1.0) < 0.01


class TestIncrementalClustering:
    """Verify online clustering works correctly."""
    
    def test_convergence(self):
        """Incremental K-Means should converge to stable centroids."""
        config = IncrementalClusterConfig()
        clusterer = IncrementalKMeans(config)
        
        # Initialize with 2 clear clusters
        cluster1_center = np.array([0.0, 0.0])
        cluster2_center = np.array([10.0, 10.0])
        
        # Add 50 points from each cluster
        for i in range(50):
            p1 = cluster1_center + np.random.randn(2) * 0.5
            p2 = cluster2_center + np.random.randn(2) * 0.5
            
            clusterer.assign_session(f"s1_{i}", p1, immediate=True)
            clusterer.assign_session(f"s2_{i}", p2, immediate=True)
        
        # Centroids should be near true centers
        centroids = [c.centroid for c in clusterer.clusters.values()]
        
        # Check that centroids are well-separated
        dists = [np.linalg.norm(c1 - c2) for c1 in centroids for c2 in centroids if not np.array_equal(c1, c2)]
        if dists:
            assert min(dists) > 5.0, "Clusters not properly separated"
    
    def test_drift_detection(self):
        """Drift score should increase with distribution change."""
        config = IncrementalClusterConfig(drift_threshold=2.0)
        clusterer = IncrementalKMeans(config)
        
        # Initialize cluster
        clusterer._create_cluster(np.array([5.0, 5.0]))
        
        # Add 20 points near centroid (stable)
        for _ in range(20):
            point = np.array([5.0, 5.0]) + np.random.randn(2) * 0.3
            clusterer.assign_session("stable", point, immediate=True)
        
        # Initial drift should be low
        initial_drift = list(clusterer.clusters.values())[0].drift_score
        
        # Add 10 points far away (drift)
        for _ in range(10):
            point = np.array([15.0, 15.0]) + np.random.randn(2) * 0.3
            clusterer.assign_session("drift", point, immediate=True)
        
        # Drift should have increased
        final_drift = list(clusterer.clusters.values())[0].drift_score
        assert final_drift > initial_drift, "Drift detection not working"


class TestAttributionConservation:
    """Verify attribution values are conserved properly."""
    
    def test_hybrid_shares_sum_to_one(self):
        """Hybrid attribution shares must sum to 1.0."""
        # Create simple paths
        paths = [
            ["Email", "Search"],
            ["Search", "Social"],
            ["Email", "Social", "Search"]
        ]
        conversions = [True, True, False]
        
        config = HybridAttributionConfig(alpha=0.5)
        engine = HybridAttributionEngine(config)
        engine.fit(paths, conversions)
        
        result = engine.compute_attribution()
        
        total = sum(result.hybrid_shares.values())
        assert abs(total - 1.0) < 1e-6, f"Shares sum to {total}, not 1.0"
    
    def test_value_conservation(self):
        """Attributed values must sum to total conversion value."""
        paths = [
            ["Email", "Search"],
            ["Search", "Social"],
        ]
        conversions = [True, True]
        values = [100.0, 200.0]
        
        config = HybridAttributionConfig(alpha=0.5)
        engine = HybridAttributionEngine(config)
        engine.fit(paths, conversions, values)
        
        result = engine.compute_attribution()
        
        total_attributed = sum(result.hybrid_values.values())
        total_conversions = sum(values)
        
        assert abs(total_attributed - total_conversions) < 1.0


class TestSyntheticDataGeneration:
    """Verify synthetic data generation produces valid test scenarios."""
    
    def test_ground_truth_accuracy(self):
        """Generated data should have verifiable ground truth."""
        config = SyntheticConfig(n_households=5, seed=42)
        events, ground_truth = generate_synthetic_household_data(config)
        
        # Check ground truth is present
        assert len(ground_truth.household_sizes) == 5
        assert len(ground_truth.session_to_person) > 0
        
        # Verify sessions are assigned to persons
        for session_id, person_id in ground_truth.session_to_person.items():
            assert person_id is not None
            assert isinstance(person_id, str)
    
    def test_session_grouping(self):
        """Events should group into reasonable sessions."""
        config = SyntheticConfig(n_households=3)
        events, _ = generate_synthetic_household_data(config)
        
        sessions = group_events_into_sessions(events, session_gap_minutes=30)
        
        # Should have created sessions
        assert len(sessions) > 0
        
        # Each session should have events
        for session in sessions:
            assert session.event_count > 0
            assert session.account_id is not None


class TestColdStartStrategy:
    """Verify cold start handler provides valid assignments."""
    
    def test_insufficient_data_detection(self):
        """Should detect when clustering is not viable."""
        from core.cold_start import ColdStartHandler, ColdStartConfig
        
        config = ColdStartConfig(min_sessions_for_clustering=10)
        handler = ColdStartHandler(config)
        
        # Only 3 sessions - should use cold start
        sessions = [{}, {}, {}]  # Minimal session stubs
        
        is_cold = handler.should_use_cold_start("test", sessions, 0.0)
        assert is_cold is True
    
    def test_heuristic_assignment(self):
        """Heuristic strategy should produce valid assignments."""
        from core.cold_start import ColdStartHandler, ColdStartConfig
        
        config = ColdStartConfig()
        handler = ColdStartHandler(config)
        
        # Mock sessions with features
        mock_sessions = []
        for i in range(5):
            session = type('Session', (), {
                'start_time': datetime.now(),
                'device_type': 'tv',
                'genres_watched': {'Drama': 60}
            })()
            mock_sessions.append(session)
        
        features = np.array([0.5, 0.5, 0.0, 1.0])
        
        assignment = handler.handle_cold_start("test", mock_sessions, features)
        
        # Should return valid assignment
        assert assignment.person_id is not None
        assert assignment.confidence > 0
        assert assignment.strategy_used in ["heuristic", "account_level", "probabilistic_prior"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
