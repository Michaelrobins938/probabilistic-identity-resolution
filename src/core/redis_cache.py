"""
Redis Caching Layer for Feature Storage and Assignment Caching

Provides high-speed caching for:
- Recent session feature vectors (avoid recomputation)
- Person assignments (reduce clustering calls)
- Cluster model state (fast recovery)

Enables 10-100x speedup for repeat lookups.
"""

from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import json
import hashlib
import logging
from datetime import datetime, timedelta

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

logger = logging.getLogger(__name__)


class RedisFeatureCache:
    """
    High-performance Redis cache for feature vectors and assignments.
    
    Caching Strategy:
    - Feature vectors: Cache by (account_id, device_id) for 5 minutes
    - Assignments: Cache by (account_id, session_id) for 1 hour
    - Cluster state: Cache full model for fast recovery
    """
    
    def __init__(self, host='localhost', port=6379, db=0, enabled=True):
        self.enabled = enabled and REDIS_AVAILABLE
        self.client = None
        
        if self.enabled:
            try:
                self.client = redis.Redis(
                    host=host,
                    port=port,
                    db=db,
                    decode_responses=False,  # For binary numpy data
                    socket_connect_timeout=5,
                    socket_timeout=5,
                    health_check_interval=30
                )
                self.client.ping()
                logger.info(f"Connected to Redis at {host}:{port}")
            except Exception as e:
                logger.warning(f"Redis connection failed: {e}. Caching disabled.")
                self.enabled = False
                self.client = None
    
    def _make_key(self, prefix: str, *parts) -> str:
        """Create cache key with prefix."""
        key_parts = [prefix] + [str(p) for p in parts]
        return ":".join(key_parts)
    
    def cache_features(
        self,
        account_id: str,
        device_id: str,
        features: np.ndarray,
        ttl_seconds: int = 300
    ) -> bool:
        """
        Cache feature vector for device.
        
        Args:
            account_id: Account identifier
            device_id: Device fingerprint
            features: NumPy feature vector
            ttl_seconds: Time-to-live (default 5 minutes)
        
        Returns:
            True if cached successfully
        """
        if not self.enabled or self.client is None:
            return False
        
        key = self._make_key("feat", account_id, device_id)
        
        try:
            # Serialize numpy array to bytes
            data = features.tobytes()
            dtype = str(features.dtype)
            shape = json.dumps(features.shape)
            
            # Store with metadata
            pipe = self.client.pipeline()
            pipe.hset(key, "data", data)
            pipe.hset(key, "dtype", dtype)
            pipe.hset(key, "shape", shape)
            pipe.hset(key, "timestamp", datetime.now().isoformat())
            pipe.expire(key, ttl_seconds)
            pipe.execute()
            
            return True
        except Exception as e:
            logger.warning(f"Failed to cache features: {e}")
            return False
    
    def get_cached_features(
        self,
        account_id: str,
        device_id: str
    ) -> Optional[np.ndarray]:
        """
        Retrieve cached feature vector.
        
        Args:
            account_id: Account identifier
            device_id: Device fingerprint
        
        Returns:
            NumPy array if cached, None otherwise
        """
        if not self.enabled or self.client is None:
            return None
        
        key = self._make_key("feat", account_id, device_id)
        
        try:
            data = self.client.hget(key, "data")
            if data is None:
                return None
            
            dtype = self.client.hget(key, "dtype").decode('utf-8')
            shape = json.loads(self.client.hget(key, "shape").decode('utf-8'))
            
            # Deserialize
            features = np.frombuffer(data, dtype=dtype).reshape(shape)
            return features
            
        except Exception as e:
            logger.warning(f"Failed to retrieve cached features: {e}")
            return None
    
    def cache_assignment(
        self,
        account_id: str,
        session_id: str,
        person_id: str,
        confidence: float,
        probabilities: Dict[str, float],
        ttl_seconds: int = 3600
    ) -> bool:
        """
        Cache person assignment result.
        
        Args:
            account_id: Account identifier
            session_id: Session identifier
            person_id: Assigned person ID
            confidence: Assignment confidence
            probabilities: Full probability distribution
            ttl_seconds: Time-to-live (default 1 hour)
        
        Returns:
            True if cached successfully
        """
        if not self.enabled or self.client is None:
            return False
        
        key = self._make_key("assign", account_id, session_id)
        
        try:
            result = {
                "person_id": person_id,
                "confidence": confidence,
                "probabilities": probabilities,
                "cached_at": datetime.now().isoformat()
            }
            
            self.client.setex(key, ttl_seconds, json.dumps(result))
            return True
            
        except Exception as e:
            logger.warning(f"Failed to cache assignment: {e}")
            return False
    
    def get_cached_assignment(
        self,
        account_id: str,
        session_id: str
    ) -> Optional[Dict]:
        """
        Retrieve cached assignment.
        
        Args:
            account_id: Account identifier
            session_id: Session identifier
        
        Returns:
            Assignment dict if cached, None otherwise
        """
        if not self.enabled or self.client is None:
            return None
        
        key = self._make_key("assign", account_id, session_id)
        
        try:
            data = self.client.get(key)
            if data is None:
                return None
            
            return json.loads(data.decode('utf-8'))
            
        except Exception as e:
            logger.warning(f"Failed to retrieve cached assignment: {e}")
            return None
    
    def cache_cluster_model(
        self,
        account_id: str,
        centroids: np.ndarray,
        cluster_ids: List[str],
        feature_names: List[str],
        ttl_seconds: int = 86400
    ) -> bool:
        """
        Cache cluster model for fast recovery.
        
        Args:
            account_id: Account identifier
            centroids: Cluster centroid matrix
            cluster_ids: List of cluster IDs
            feature_names: List of feature names
            ttl_seconds: Time-to-live (default 24 hours)
        
        Returns:
            True if cached successfully
        """
        if not self.enabled or self.client is None:
            return False
        
        key = self._make_key("model", account_id)
        
        try:
            model_data = {
                "centroids": centroids.tobytes(),
                "centroids_dtype": str(centroids.dtype),
                "centroids_shape": centroids.shape,
                "cluster_ids": cluster_ids,
                "feature_names": feature_names,
                "cached_at": datetime.now().isoformat()
            }
            
            self.client.setex(key, ttl_seconds, json.dumps(model_data, default=str))
            return True
            
        except Exception as e:
            logger.warning(f"Failed to cache cluster model: {e}")
            return False
    
    def get_cached_cluster_model(
        self,
        account_id: str
    ) -> Optional[Tuple[np.ndarray, List[str], List[str]]]:
        """
        Retrieve cached cluster model.
        
        Args:
            account_id: Account identifier
        
        Returns:
            Tuple of (centroids, cluster_ids, feature_names) if cached
        """
        if not self.enabled or self.client is None:
            return None
        
        key = self._make_key("model", account_id)
        
        try:
            data = self.client.get(key)
            if data is None:
                return None
            
            model_data = json.loads(data.decode('utf-8'))
            
            # Deserialize centroids
            centroids = np.frombuffer(
                model_data["centroids"],
                dtype=model_data["centroids_dtype"]
            ).reshape(model_data["centroids_shape"])
            
            return (
                centroids,
                model_data["cluster_ids"],
                model_data["feature_names"]
            )
            
        except Exception as e:
            logger.warning(f"Failed to retrieve cached model: {e}")
            return None
    
    def invalidate_account_cache(self, account_id: str) -> int:
        """
        Invalidate all cached data for an account.
        
        Args:
            account_id: Account identifier
        
        Returns:
            Number of keys deleted
        """
        if not self.enabled or self.client is None:
            return 0
        
        try:
            # Find all keys for this account
            pattern = f"*:{account_id}:*"
            keys = self.client.keys(pattern)
            
            if keys:
                return self.client.delete(*keys)
            return 0
            
        except Exception as e:
            logger.warning(f"Failed to invalidate cache: {e}")
            return 0
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if not self.enabled or self.client is None:
            return {"enabled": False}
        
        try:
            info = self.client.info()
            return {
                "enabled": True,
                "used_memory_mb": info.get("used_memory", 0) / (1024 * 1024),
                "connected_clients": info.get("connected_clients", 0),
                "total_keys": self.client.dbsize(),
                "hit_rate": info.get("keyspace_hits", 0) / max(1, info.get("keyspace_hits", 0) + info.get("keyspace_misses", 0))
            }
        except Exception as e:
            logger.warning(f"Failed to get cache stats: {e}")
            return {"enabled": True, "error": str(e)}


# Convenience function for quick cache access
def get_feature_cache(host='localhost', port=6379) -> RedisFeatureCache:
    """Get or create feature cache instance."""
    return RedisFeatureCache(host=host, port=port)
