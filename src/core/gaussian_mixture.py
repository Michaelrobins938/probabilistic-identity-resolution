"""
Gaussian Mixture Model (GMM) for Non-Spherical Clusters

Addresses the "non-spherical problem" in K-means:
- K-means assumes spherical clusters (equal variance in all directions)
- Reality: Behavioral clusters are elliptical and can overlap
- GMM allows for covariance modeling, handling:
  * Nested personas (parent → adult → primary_adult)
  * Time-shifted patterns (binge watchers with C-shaped clusters)
  * Overlapping behaviors (weekend viewing from different people)

Uses Expectation-Maximization (EM) algorithm with incremental updates.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from scipy.stats import multivariate_normal
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class GMMConfig:
    """Configuration for Gaussian Mixture Model."""
    max_components: int = 6  # Maximum people per household
    min_components: int = 1
    
    # EM algorithm parameters
    max_iterations: int = 100
    tolerance: float = 1e-3
    
    # Covariance type
    covariance_type: str = "diag"  # "full", "diag", "spherical", "tied"
    
    # Regularization
    reg_covar: float = 1e-6  # Prevent singular covariance matrices
    
    # Online learning
    forgetting_factor: float = 0.95  # For incremental updates (0-1)
    min_samples_for_update: int = 5


@dataclass
class GaussianComponent:
    """Single Gaussian component in the mixture."""
    component_id: str
    mean: np.ndarray
    covariance: np.ndarray
    weight: float = 1.0
    
    # Statistics for online updates
    n_samples: int = 0
    sum_resp: float = 0.0  # Sum of responsibilities
    sum_resp_x: np.ndarray = field(default=None)
    sum_resp_x_outer: np.ndarray = field(default=None)
    
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        if self.sum_resp_x is None:
            self.sum_resp_x = np.zeros_like(self.mean)
        if self.sum_resp_x_outer is None:
            self.sum_resp_x_outer = np.zeros((len(self.mean), len(self.mean)))
    
    def pdf(self, x: np.ndarray) -> float:
        """Probability density function."""
        try:
            return multivariate_normal.pdf(x, mean=self.mean, cov=self.covariance)
        except:
            return 1e-10  # Return small probability if numerical issues
    
    def sample(self, n_samples: int = 1) -> np.ndarray:
        """Sample from this Gaussian."""
        return np.random.multivariate_normal(self.mean, self.covariance, n_samples)


class GaussianMixtureModel:
    """
    Gaussian Mixture Model for flexible clustering.
    
    Advantages over K-means:
    1. Soft assignments (probabilistic)
    2. Elliptical clusters (covariance modeling)
    3. Overlapping cluster support
    4. Captures hierarchical/nested structures
    
    Supports both batch and incremental (online) learning.
    """
    
    def __init__(self, config: Optional[GMMConfig] = None):
        self.config = config or GMMConfig()
        self.components: Dict[str, GaussianComponent] = {}
        self.n_features: int = 0
        self.account_id: Optional[str] = None
        
        # For incremental learning
        self.session_buffer: List[Tuple[str, np.ndarray]] = []
        self.total_samples = 0
        
        logger.info(f"GMM initialized (covariance_type={self.config.covariance_type})")
    
    def fit(self, X: np.ndarray, account_id: str) -> None:
        """
        Fit GMM using EM algorithm (batch mode).
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix (n_samples, n_features)
        account_id : str
            Account identifier
        """
        self.account_id = account_id
        self.n_features = X.shape[1]
        
        # Determine number of components using BIC
        best_k, best_bic = self._select_n_components(X)
        
        logger.info(f"Fitting GMM with k={best_k} components")
        
        # Initialize with K-means++ style
        self._initialize_components(X, best_k)
        
        # EM algorithm
        prev_log_likelihood = -np.inf
        
        for iteration in range(self.config.max_iterations):
            # E-step: Compute responsibilities
            resp = self._e_step(X)
            
            # M-step: Update parameters
            self._m_step(X, resp)
            
            # Compute log-likelihood
            log_likelihood = self._compute_log_likelihood(X)
            
            # Check convergence
            change = abs(log_likelihood - prev_log_likelihood)
            if change < self.config.tolerance:
                logger.info(f"EM converged at iteration {iteration}")
                break
            
            prev_log_likelihood = log_likelihood
        
        self.total_samples = len(X)
        logger.info(f"GMM fitted: {len(self.components)} components")
    
    def fit_incremental(self, session_id: str, features: np.ndarray) -> Dict[str, float]:
        """
        Incrementally update GMM with new session.
        
        Uses stochastic EM for online updates.
        
        Returns
        -------
        Dict[str, float]
            Component probabilities for this session
        """
        if len(self.components) == 0:
            # First session - create initial component
            self._create_component(features)
            return {list(self.components.keys())[0]: 1.0}
        
        # Compute current responsibilities
        x = features.reshape(1, -1)
        resp = self._e_step(x)[0]  # Responsibilities for this sample
        
        # Update component statistics incrementally
        forgetting_factor = self.config.forgetting_factor
        
        for i, (comp_id, component) in enumerate(self.components.items()):
            r = resp[i]
            
            # Update with forgetting factor (exponential decay)
            component.sum_resp = forgetting_factor * component.sum_resp + r
            component.sum_resp_x = forgetting_factor * component.sum_resp_x + r * features
            
            # Update mean
            if component.sum_resp > 0:
                component.mean = component.sum_resp_x / component.sum_resp
            
            # Update covariance (simplified diagonal for speed)
            if self.config.covariance_type == "diag":
                diff = features - component.mean
                component.covariance = (
                    forgetting_factor * component.covariance + 
                    r * np.diag(np.outer(diff, diff))
                )
            
            component.n_samples += 1
            component.last_updated = datetime.now()
        
        # Update weights
        total_resp = sum(resp)
        if total_resp > 0:
            for i, component in enumerate(self.components.values()):
                component.weight = forgetting_factor * component.weight + resp[i]
        
        self.total_samples += 1
        
        # Return probabilities
        return {comp_id: resp[i] for i, comp_id in enumerate(self.components.keys())}
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict posterior probabilities for each component.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix (n_samples, n_features)
        
        Returns
        -------
        np.ndarray
            Probability matrix (n_samples, n_components)
        """
        if len(self.components) == 0:
            return np.zeros((len(X), 1))
        
        n_samples = X.shape[0]
        n_components = len(self.components)
        
        # Compute weighted probabilities for each component
        weighted_probs = np.zeros((n_samples, n_components))
        
        for i, component in enumerate(self.components.values()):
            for j, x in enumerate(X):
                weighted_probs[j, i] = component.weight * component.pdf(x)
        
        # Normalize
        sum_probs = weighted_probs.sum(axis=1, keepdims=True)
        sum_probs[sum_probs == 0] = 1  # Avoid division by zero
        
        return weighted_probs / sum_probs
    
    def predict(self, X: np.ndarray) -> Tuple[List[str], List[float]]:
        """
        Predict component assignments.
        
        Returns
        -------
        Tuple[List[str], List[float]]
            (component_ids, confidence_scores)
        """
        proba = self.predict_proba(X)
        
        component_list = list(self.components.keys())
        predictions = []
        confidences = []
        
        for p in proba:
            idx = np.argmax(p)
            predictions.append(component_list[idx])
            confidences.append(p[idx])
        
        return predictions, confidences
    
    def _select_n_components(self, X: np.ndarray) -> Tuple[int, float]:
        """Select optimal number of components using BIC."""
        best_k = 1
        best_bic = np.inf
        
        for k in range(self.config.min_components, 
                       min(self.config.max_components + 1, len(X))):
            if k >= len(X):
                break
            
            # Quick fit to compute BIC
            bic = self._compute_bic_quick(X, k)
            
            if bic < best_bic:
                best_bic = bic
                best_k = k
        
        return best_k, best_bic
    
    def _compute_bic_quick(self, X: np.ndarray, k: int) -> float:
        """Quick BIC computation for model selection."""
        n_samples, n_features = X.shape
        
        # Fit simple model
        temp_gmm = GaussianMixtureModel(GMMConfig(max_components=k))
        temp_gmm._initialize_components(X, k)
        
        # Run few EM iterations
        for _ in range(10):
            resp = temp_gmm._e_step(X)
            temp_gmm._m_step(X, resp)
        
        log_likelihood = temp_gmm._compute_log_likelihood(X)
        
        # BIC = -2 * log_likelihood + k * log(n)
        n_params = k * (n_features + n_features * (n_features + 1) / 2 + 1)
        bic = -2 * log_likelihood + n_params * np.log(n_samples)
        
        return bic
    
    def _initialize_components(self, X: np.ndarray, k: int) -> None:
        """Initialize Gaussian components using K-means++ style."""
        n_samples, n_features = X.shape
        
        # K-means++ initialization
        indices = self._kmeans_plus_plus(X, k)
        
        self.components = {}
        for i, idx in enumerate(indices):
            comp_id = f"{self.account_id}_person_{i}"
            
            mean = X[idx]
            
            # Initialize covariance
            if self.config.covariance_type == "diag":
                covariance = np.eye(n_features) * 0.1
            elif self.config.covariance_type == "spherical":
                covariance = np.eye(n_features) * 0.1
            else:  # full or tied
                covariance = np.eye(n_features) * 0.1
            
            self.components[comp_id] = GaussianComponent(
                component_id=comp_id,
                mean=mean,
                covariance=covariance,
                weight=1.0 / k
            )
    
    def _kmeans_plus_plus(self, X: np.ndarray, k: int) -> List[int]:
        """K-means++ initialization."""
        n_samples = len(X)
        indices = []
        
        # First center: random
        indices.append(np.random.randint(0, n_samples))
        
        for _ in range(1, k):
            # Compute distances to nearest center
            distances = np.zeros(n_samples)
            for i, x in enumerate(X):
                min_dist = min(
                    np.linalg.norm(x - X[idx]) ** 2
                    for idx in indices
                )
                distances[i] = min_dist
            
            # Choose next center with probability proportional to distance²
            probs = distances / distances.sum()
            next_idx = np.random.choice(n_samples, p=probs)
            indices.append(next_idx)
        
        return indices
    
    def _e_step(self, X: np.ndarray) -> np.ndarray:
        """Expectation step: compute responsibilities."""
        n_samples = X.shape[0]
        n_components = len(self.components)
        
        resp = np.zeros((n_samples, n_components))
        
        for i, component in enumerate(self.components.values()):
            for j, x in enumerate(X):
                resp[j, i] = component.weight * component.pdf(x)
        
        # Normalize
        resp_sum = resp.sum(axis=1, keepdims=True)
        resp_sum[resp_sum == 0] = 1
        resp = resp / resp_sum
        
        return resp
    
    def _m_step(self, X: np.ndarray, resp: np.ndarray) -> None:
        """Maximization step: update parameters."""
        n_samples = X.shape[0]
        
        for i, component in enumerate(self.components.values()):
            resp_i = resp[:, i]
            
            # Update weight
            component.weight = resp_i.sum() / n_samples
            
            # Update mean
            resp_sum = resp_i.sum()
            if resp_sum > 0:
                component.mean = (resp_i[:, np.newaxis] * X).sum(axis=0) / resp_sum
            
            # Update covariance
            diff = X - component.mean
            if self.config.covariance_type == "diag":
                weighted_diff_sq = resp_i[:, np.newaxis] * (diff ** 2)
                component.covariance = np.diag(
                    weighted_diff_sq.sum(axis=0) / resp_sum + self.config.reg_covar
                )
            elif self.config.covariance_type == "spherical":
                weighted_diff_sq = resp_i[:, np.newaxis] * (diff ** 2)
                avg_var = weighted_diff_sq.sum() / (resp_sum * X.shape[1])
                component.covariance = np.eye(X.shape[1]) * (avg_var + self.config.reg_covar)
    
    def _compute_log_likelihood(self, X: np.ndarray) -> float:
        """Compute total log-likelihood."""
        log_likelihood = 0.0
        
        for x in X:
            prob = sum(
                comp.weight * comp.pdf(x)
                for comp in self.components.values()
            )
            log_likelihood += np.log(max(prob, 1e-10))
        
        return log_likelihood
    
    def _create_component(self, features: np.ndarray) -> str:
        """Create new Gaussian component."""
        comp_id = f"{self.account_id}_person_{len(self.components)}"
        
        n_features = len(features)
        covariance = np.eye(n_features) * 0.1
        
        self.components[comp_id] = GaussianComponent(
            component_id=comp_id,
            mean=features.copy(),
            covariance=covariance,
            weight=1.0 / max(1, len(self.components) + 1)
        )
        
        # Renormalize weights
        total_weight = sum(comp.weight for comp in self.components.values())
        for comp in self.components.values():
            comp.weight /= total_weight
        
        return comp_id
    
    def get_bic(self, X: np.ndarray) -> float:
        """Get Bayesian Information Criterion."""
        log_likelihood = self._compute_log_likelihood(X)
        n_samples, n_features = X.shape
        k = len(self.components)
        
        n_params = k * (n_features + n_features * (n_features + 1) / 2 + 1)
        bic = -2 * log_likelihood + n_params * np.log(n_samples)
        
        return bic
    
    def compare_to_kmeans(self, X: np.ndarray, kmeans_labels: np.ndarray) -> Dict[str, float]:
        """Compare GMM to K-means on same data."""
        # Predict with GMM
        gmm_labels, _ = self.predict(X)
        
        # Compute silhouette for both (simplified)
        from sklearn.metrics import silhouette_score
        
        try:
            gmm_silhouette = silhouette_score(X, [list(self.components.keys()).index(l) for l in gmm_labels])
        except:
            gmm_silhouette = 0.0
        
        try:
            kmeans_silhouette = silhouette_score(X, kmeans_labels)
        except:
            kmeans_silhouette = 0.0
        
        gmm_bic = self.get_bic(X)
        
        return {
            "gmm_silhouette": gmm_silhouette,
            "kmeans_silhouette": kmeans_silhouette,
            "gmm_bic": gmm_bic,
            "gmm_wins": gmm_silhouette > kmeans_silhouette
        }
