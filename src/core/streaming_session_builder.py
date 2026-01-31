"""
Streaming Session Builder with Redis Backing

Handles real-time session building from streaming events using Redis for persistence.
Supports Netflix-scale event volumes with micro-batching.
"""

from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import json
import hashlib
import logging
from collections import defaultdict

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from models.streaming_event import StreamingEvent, Session
from validation.input_validator import validate_events, ValidationError

logger = logging.getLogger(__name__)


@dataclass
class StreamingConfig:
    """Configuration for streaming session builder."""
    # Session gap threshold (minutes)
    session_gap_minutes: int = 30
    
    # Redis settings
    redis_host: str = 'localhost'
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None
    
    # TTL for session data (hours)
    session_ttl_hours: int = 24
    
    # Micro-batch settings
    batch_size: int = 1000
    flush_interval_seconds: int = 5
    
    # Performance
    max_in_memory_sessions: int = 10000
    enable_compression: bool = True
    
    # Validation
    strict_validation: bool = True


class RedisSessionStore:
    """Redis-backed storage for active sessions."""
    
    def __init__(self, config: StreamingConfig):
        self.config = config
        self._redis: Optional[redis.Redis] = None
        self._local_cache: Dict[str, Dict] = {}
        
        if REDIS_AVAILABLE:
            try:
                self._redis = redis.Redis(
                    host=config.redis_host,
                    port=config.redis_port,
                    db=config.redis_db,
                    password=config.redis_password,
                    decode_responses=True
                )
                self._redis.ping()
                logger.info("Redis connection established")
            except Exception as e:
                logger.warning(f"Redis unavailable, using in-memory fallback: {e}")
                self._redis = None
    
    def _make_key(self, account_id: str, device_id: str) -> str:
        """Generate Redis key for session."""
        return f"session:{account_id}:{device_id}"
    
    def get_active_session(self, account_id: str, device_id: str) -> Optional[Dict]:
        """Get active session from Redis or local cache."""
        key = self._make_key(account_id, device_id)
        
        # Try local cache first
        if key in self._local_cache:
            return self._local_cache[key]
        
        # Try Redis
        if self._redis:
            try:
                data = self._redis.get(key)
                if data:
                    session_data = json.loads(data)
                    self._local_cache[key] = session_data
                    return session_data
            except Exception as e:
                logger.error(f"Redis get error: {e}")
        
        return None
    
    def save_session(self, account_id: str, device_id: str, session_data: Dict) -> None:
        """Save session to Redis and local cache."""
        key = self._make_key(account_id, device_id)
        self._local_cache[key] = session_data
        
        if self._redis:
            try:
                ttl = self.config.session_ttl_hours * 3600
                self._redis.setex(key, ttl, json.dumps(session_data))
            except Exception as e:
                logger.error(f"Redis save error: {e}")
    
    def delete_session(self, account_id: str, device_id: str) -> None:
        """Delete session from Redis and local cache."""
        key = self._make_key(account_id, device_id)
        self._local_cache.pop(key, None)
        
        if self._redis:
            try:
                self._redis.delete(key)
            except Exception as e:
                logger.error(f"Redis delete error: {e}")
    
    def get_all_active_sessions(self, account_id: str) -> List[Dict]:
        """Get all active sessions for an account."""
        pattern = f"session:{account_id}:*"
        sessions = []
        
        # Get from local cache
        for key, data in self._local_cache.items():
            if key.startswith(f"session:{account_id}:"):
                sessions.append(data)
        
        # Get from Redis
        if self._redis:
            try:
                keys = self._redis.keys(pattern)
                for key in keys:
                    if key not in self._local_cache:
                        data = self._redis.get(key)
                        if data:
                            sessions.append(json.loads(data))
            except Exception as e:
                logger.error(f"Redis scan error: {e}")
        
        return sessions
    
    def flush_local_cache(self) -> None:
        """Flush local cache to Redis."""
        if not self._redis:
            return
        
        for key, data in self._local_cache.items():
            try:
                ttl = self.config.session_ttl_hours * 3600
                self._redis.setex(key, ttl, json.dumps(data))
            except Exception as e:
                logger.error(f"Redis flush error: {e}")
        
        self._local_cache.clear()


class StreamingSessionBuilder:
    """
    Builds sessions from streaming events in real-time.
    
    Features:
    - Micro-batching for efficiency
    - Redis persistence for durability
    - Automatic session timeout detection
    - High-throughput processing
    """
    
    def __init__(self, config: Optional[StreamingConfig] = None):
        self.config = config or StreamingConfig()
        self.store = RedisSessionStore(self.config)
        self._pending_events: List[StreamingEvent] = []
        self._last_flush = datetime.now()
        
        # Metrics
        self.events_processed = 0
        self.sessions_created = 0
        self.sessions_closed = 0
    
    def process_event(self, event: StreamingEvent) -> Optional[Session]:
        """
        Process a single streaming event.
        
        Returns completed session if this event closed a session.
        """
        # Validate
        if self.config.strict_validation:
            from validation.input_validator import StreamingEventValidator
            result = StreamingEventValidator.validate(event)
            if not result.is_valid:
                logger.error(f"Invalid event: {result.errors}")
                return None
        
        self._pending_events.append(event)
        self.events_processed += 1
        
        # Check if we should flush
        if self._should_flush():
            return self.flush()
        
        return None
    
    def process_batch(self, events: List[StreamingEvent]) -> List[Session]:
        """
        Process a batch of events efficiently.
        
        Returns list of completed sessions.
        """
        if self.config.strict_validation:
            result = validate_events(events, raise_on_error=False)
            if not result.is_valid:
                logger.error(f"Batch validation failed: {result.errors}")
                # Continue with valid events only
                events = [e for i, e in enumerate(events) 
                         if i not in [int(err.split()[1]) for err in result.errors[:10]]]
        
        completed_sessions = []
        
        # Group by account and device for efficient processing
        grouped = self._group_events_by_session(events)
        
        for (account_id, device_id), event_list in grouped.items():
            session = self._process_grouped_events(account_id, device_id, event_list)
            if session:
                completed_sessions.append(session)
        
        self.events_processed += len(events)
        return completed_sessions
    
    def _group_events_by_session(
        self, 
        events: List[StreamingEvent]
    ) -> Dict[Tuple[str, str], List[StreamingEvent]]:
        """Group events by account and device fingerprint."""
        grouped = defaultdict(list)
        for event in events:
            key = (event.account_id, event.device_fingerprint)
            grouped[key].append(event)
        
        # Sort by timestamp within each group
        for key in grouped:
            grouped[key].sort(key=lambda e: e.timestamp or datetime.min)
        
        return dict(grouped)
    
    def _process_grouped_events(
        self,
        account_id: str,
        device_id: str,
        events: List[StreamingEvent]
    ) -> Optional[Session]:
        """Process events for a single account-device pair."""
        # Get active session
        active_session = self.store.get_active_session(account_id, device_id)
        
        completed_session = None
        
        for event in events:
            gap_threshold = timedelta(minutes=self.config.session_gap_minutes)
            
            if active_session is None:
                # Start new session
                active_session = self._create_session_data(event)
            elif event.timestamp and active_session.get('last_event_time'):
                last_time = datetime.fromisoformat(active_session['last_event_time'])
                if event.timestamp - last_time > gap_threshold:
                    # Gap too large - close current session and start new
                    completed_session = self._finalize_session(active_session)
                    self.sessions_closed += 1
                    active_session = self._create_session_data(event)
                else:
                    # Add to current session
                    self._add_event_to_session(active_session, event)
            else:
                # Add to current session (no timestamp check possible)
                self._add_event_to_session(active_session, event)
        
        # Save active session back
        if active_session:
            self.store.save_session(account_id, device_id, active_session)
        
        return completed_session
    
    def _create_session_data(self, event: StreamingEvent) -> Dict:
        """Create new session data structure from event."""
        self.sessions_created += 1
        return {
            'session_id': self._generate_session_id(event),
            'account_id': event.account_id,
            'device_fingerprint': event.device_fingerprint,
            'device_type': event.device_type,
            'start_time': event.timestamp.isoformat() if event.timestamp else None,
            'last_event_time': event.timestamp.isoformat() if event.timestamp else None,
            'events': [self._event_to_dict(event)],
            'event_count': 1,
            'total_duration': event.duration_seconds or 0,
            'genres_watched': {event.content_genre: event.duration_seconds or 0} 
                            if event.content_genre else {},
            'has_conversion': event.event_type == 'conversion',
            'conversion_value': event.metadata.get('conversion_value', 0) 
                              if event.metadata and event.event_type == 'conversion' else 0,
        }
    
    def _add_event_to_session(self, session_data: Dict, event: StreamingEvent) -> None:
        """Add event to existing session."""
        session_data['events'].append(self._event_to_dict(event))
        session_data['event_count'] += 1
        session_data['last_event_time'] = event.timestamp.isoformat() if event.timestamp else None
        
        if event.duration_seconds:
            session_data['total_duration'] += event.duration_seconds
        
        if event.content_genre:
            current = session_data['genres_watched'].get(event.content_genre, 0)
            session_data['genres_watched'][event.content_genre] = current + (event.duration_seconds or 0)
        
        if event.event_type == 'conversion':
            session_data['has_conversion'] = True
            value = event.metadata.get('conversion_value', 0) if event.metadata else 0
            session_data['conversion_value'] = session_data.get('conversion_value', 0) + value
    
    def _event_to_dict(self, event: StreamingEvent) -> Dict:
        """Convert event to dictionary for storage."""
        return {
            'event_id': event.event_id,
            'event_type': event.event_type,
            'timestamp': event.timestamp.isoformat() if event.timestamp else None,
            'content_id': event.content_id,
            'content_genre': event.content_genre,
            'duration_seconds': event.duration_seconds,
        }
    
    def _finalize_session(self, session_data: Dict) -> Session:
        """Convert session data to Session object."""
        return Session(
            session_id=session_data['session_id'],
            account_id=session_data['account_id'],
            device_fingerprint=session_data['device_fingerprint'],
            device_type=session_data['device_type'],
            start_time=datetime.fromisoformat(session_data['start_time']) 
                      if session_data.get('start_time') else None,
            end_time=datetime.fromisoformat(session_data['last_event_time']) 
                    if session_data.get('last_event_time') else None,
            event_count=session_data['event_count'],
            total_duration=session_data['total_duration'],
            genres_watched=session_data['genres_watched'],
            has_conversion=session_data['has_conversion'],
            conversion_value=session_data.get('conversion_value', 0),
        )
    
    def _generate_session_id(self, event: StreamingEvent) -> str:
        """Generate unique session ID."""
        timestamp = event.timestamp or datetime.now()
        unique_str = f"{event.account_id}:{event.device_fingerprint}:{timestamp.isoformat()}"
        return hashlib.sha256(unique_str.encode()).hexdigest()[:16]
    
    def _should_flush(self) -> bool:
        """Check if pending events should be flushed."""
        if len(self._pending_events) >= self.config.batch_size:
            return True
        
        if (datetime.now() - self._last_flush).total_seconds() >= self.config.flush_interval_seconds:
            return True
        
        return False
    
    def flush(self) -> Optional[Session]:
        """Flush pending events and return any completed sessions."""
        if not self._pending_events:
            return None
        
        events = self._pending_events
        self._pending_events = []
        self._last_flush = datetime.now()
        
        completed = self.process_batch(events)
        
        # Return first completed session (if any)
        return completed[0] if completed else None
    
    def close_all_sessions(self) -> List[Session]:
        """Force close all active sessions."""
        # This is typically called at system shutdown
        completed = []
        
        # Flush any pending events
        self.flush()
        
        # Note: In a real implementation, we'd scan Redis for all active sessions
        # and close them. For now, this is a placeholder.
        
        return completed
    
    def get_metrics(self) -> Dict[str, int]:
        """Get processing metrics."""
        return {
            'events_processed': self.events_processed,
            'sessions_created': self.sessions_created,
            'sessions_closed': self.sessions_closed,
            'pending_events': len(self._pending_events),
        }


# Convenience functions

def create_streaming_builder(config: Optional[StreamingConfig] = None) -> StreamingSessionBuilder:
    """Create a streaming session builder with given configuration."""
    return StreamingSessionBuilder(config)


def process_event_stream(
    events: List[StreamingEvent],
    config: Optional[StreamingConfig] = None
) -> List[Session]:
    """
    Process a stream of events and return completed sessions.
    
    Convenience function for batch processing.
    """
    builder = StreamingSessionBuilder(config)
    return builder.process_batch(events)
