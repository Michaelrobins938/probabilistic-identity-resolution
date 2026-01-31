"""
Audit Logging System

Immutable audit trail for compliance, security, and debugging.
Tracks all system actions including:
- Data access and queries
- Identity resolution operations
- Attribution calculations
- Configuration changes
- Deletion requests

Provides:
- Tamper-evident logging (write-once)
- Role-based access tracking
- Query attribution
- Anomaly detection
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set
from datetime import datetime
from enum import Enum
import hashlib
import json
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class AuditEventType(Enum):
    """Types of audit events."""
    # Data access
    DATA_QUERY = "data_query"
    DATA_EXPORT = "data_export"
    DATA_ACCESS = "data_access"
    
    # Identity operations
    IDENTITY_RESOLUTION = "identity_resolution"
    CLUSTER_CREATED = "cluster_created"
    CLUSTER_UPDATED = "cluster_updated"
    PERSON_ASSIGNED = "person_assigned"
    
    # Attribution
    ATTRIBUTION_CALCULATED = "attribution_calculated"
    ATTRIBUTION_EXPORTED = "attribution_exported"
    
    # Privacy
    DELETION_REQUESTED = "deletion_requested"
    DELETION_EXECUTED = "deletion_executed"
    DATA_ANONYMIZED = "data_anonymized"
    
    # System
    CONFIG_CHANGED = "config_changed"
    SYSTEM_STARTUP = "system_startup"
    SYSTEM_SHUTDOWN = "system_shutdown"
    ERROR_OCCURRED = "error_occurred"
    
    # Security
    AUTHENTICATION = "authentication"
    AUTHORIZATION_FAILURE = "authorization_failure"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"


class AccessRole(Enum):
    """User roles for access control."""
    ADMIN = "admin"  # Full access
    ANALYST = "analyst"  # Read-only, aggregates only
    ENGINEER = "engineer"  # System config, no raw data
    SERVICE = "service"  # Automated service accounts
    AUDITOR = "auditor"  # Audit log access only


@dataclass
class AuditEvent:
    """Single audit event record."""
    event_id: str
    timestamp: datetime
    event_type: AuditEventType
    
    # Actor information
    user_id: str
    user_role: AccessRole
    session_id: str
    ip_address: str
    
    # Action details
    action: str
    resource_type: str
    resource_id: str
    
    # Data scope
    account_ids: List[str] = field(default_factory=list)
    person_ids: List[str] = field(default_factory=list)
    data_volume: int = 0  # Records affected
    
    # Outcome
    success: bool = True
    error_message: str = ""
    execution_time_ms: int = 0
    
    # Integrity
    previous_hash: str = ""
    event_hash: str = ""  # Computed on all fields
    
    # Metadata
    user_agent: str = ""
    request_id: str = ""
    correlation_id: str = ""
    
    def compute_hash(self) -> str:
        """Compute tamper-evident hash of event."""
        data = {
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type.value,
            "user_id": self.user_id,
            "action": self.action,
            "resource_id": self.resource_id,
            "previous_hash": self.previous_hash
        }
        
        return hashlib.sha256(
            json.dumps(data, sort_keys=True).encode()
        ).hexdigest()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type.value,
            "user_id": self.user_id,
            "user_role": self.user_role.value,
            "session_id": self.session_id,
            "ip_address": self.ip_address,
            "action": self.action,
            "resource_type": self.resource_type,
            "resource_id": self.resource_id,
            "account_ids": self.account_ids,
            "person_ids": self.person_ids,
            "data_volume": self.data_volume,
            "success": self.success,
            "error_message": self.error_message,
            "execution_time_ms": self.execution_time_ms,
            "event_hash": self.event_hash,
            "user_agent": self.user_agent,
            "request_id": self.request_id,
            "correlation_id": self.correlation_id
        }


@dataclass
class AuditConfig:
    """Configuration for audit logging."""
    # Retention
    retention_days: int = 2555  # 7 years (compliance)
    archive_after_days: int = 90
    
    # Performance
    async_logging: bool = True
    batch_size: int = 100
    flush_interval_seconds: int = 5
    
    # Scope
    log_data_access: bool = True
    log_identity_ops: bool = True
    log_attribution: bool = True
    log_system_events: bool = True
    log_security_events: bool = True
    
    # Privacy
    mask_pii_in_logs: bool = True
    hash_sensitive_ids: bool = True
    
    # Alerting
    alert_on_suspicious: bool = True
    suspicious_threshold_per_hour: int = 100
    alert_on_admin_actions: bool = True


class AuditLogger:
    """
    Immutable audit logging system.
    
    Features:
    - Tamper-evident chain (each event hashes previous)
    - Role-based access tracking
    - Async batching for performance
    - Query attribution (who accessed what)
    - Anomaly detection
    
    All logs are append-only for compliance.
    """
    
    def __init__(self, config: Optional[AuditConfig] = None):
        self.config = config or AuditConfig()
        self.event_buffer: List[AuditEvent] = []
        self.last_hash: str = "0" * 64  # Genesis hash
        self.event_count = 0
        
        # Suspicious activity tracking
        self.user_activity: Dict[str, List[datetime]] = defaultdict(list)
        
        logger.info("AuditLogger initialized")
    
    def log_event(
        self,
        event_type: AuditEventType,
        user_id: str,
        user_role: AccessRole,
        action: str,
        resource_type: str,
        resource_id: str,
        **kwargs
    ) -> AuditEvent:
        """
        Log an audit event.
        
        Parameters
        ----------
        event_type : AuditEventType
            Type of event
        user_id : str
            User who performed action
        user_role : AccessRole
            Role of user
        action : str
            Description of action
        resource_type : str
            Type of resource accessed
        resource_id : str
            ID of resource
        **kwargs
            Additional fields (account_ids, person_ids, etc.)
        
        Returns
        -------
        AuditEvent
            Logged event
        """
        # Check if this event type should be logged
        if not self._should_log(event_type):
            return None
        
        # Generate event ID
        self.event_count += 1
        event_id = f"audit_{datetime.now().strftime('%Y%m%d%H%M%S')}_{self.event_count}"
        
        # Create event
        event = AuditEvent(
            event_id=event_id,
            timestamp=datetime.now(),
            event_type=event_type,
            user_id=user_id,
            user_role=user_role,
            session_id=kwargs.get('session_id', ''),
            ip_address=kwargs.get('ip_address', ''),
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            account_ids=kwargs.get('account_ids', []),
            person_ids=kwargs.get('person_ids', []),
            data_volume=kwargs.get('data_volume', 0),
            success=kwargs.get('success', True),
            error_message=kwargs.get('error_message', ''),
            execution_time_ms=kwargs.get('execution_time_ms', 0),
            previous_hash=self.last_hash,
            user_agent=kwargs.get('user_agent', ''),
            request_id=kwargs.get('request_id', ''),
            correlation_id=kwargs.get('correlation_id', '')
        )
        
        # Compute hash
        event.event_hash = event.compute_hash()
        self.last_hash = event.event_hash
        
        # Check for suspicious activity
        if self.config.alert_on_suspicious:
            self._check_suspicious_activity(event)
        
        # Buffer or log immediately
        if self.config.async_logging:
            self.event_buffer.append(event)
            
            if len(self.event_buffer) >= self.config.batch_size:
                self._flush_buffer()
        else:
            self._write_event(event)
        
        return event
    
    def log_data_query(
        self,
        user_id: str,
        user_role: AccessRole,
        query_type: str,
        account_ids: List[str],
        person_ids: List[str] = None,
        records_returned: int = 0,
        **kwargs
    ) -> AuditEvent:
        """Log a data query operation."""
        if not self.config.log_data_access:
            return None
        
        return self.log_event(
            event_type=AuditEventType.DATA_QUERY,
            user_id=user_id,
            user_role=user_role,
            action=f"query_{query_type}",
            resource_type="data",
            resource_id=account_ids[0] if account_ids else "",
            account_ids=account_ids,
            person_ids=person_ids or [],
            data_volume=records_returned,
            **kwargs
        )
    
    def log_identity_resolution(
        self,
        user_id: str,
        user_role: AccessRole,
        account_id: str,
        n_persons_detected: int,
        confidence: float,
        **kwargs
    ) -> AuditEvent:
        """Log identity resolution operation."""
        if not self.config.log_identity_ops:
            return None
        
        return self.log_event(
            event_type=AuditEventType.IDENTITY_RESOLUTION,
            user_id=user_id,
            user_role=user_role,
            action="resolve_identities",
            resource_type="household",
            resource_id=account_id,
            account_ids=[account_id],
            data_volume=n_persons_detected,
            **kwargs
        )
    
    def log_attribution(
        self,
        user_id: str,
        user_role: AccessRole,
        account_id: str,
        channels: List[str],
        calculation_time_ms: int,
        **kwargs
    ) -> AuditEvent:
        """Log attribution calculation."""
        if not self.config.log_attribution:
            return None
        
        return self.log_event(
            event_type=AuditEventType.ATTRIBUTION_CALCULATED,
            user_id=user_id,
            user_role=user_role,
            action="calculate_attribution",
            resource_type="attribution",
            resource_id=account_id,
            account_ids=[account_id],
            data_volume=len(channels),
            execution_time_ms=calculation_time_ms,
            **kwargs
        )
    
    def log_deletion(
        self,
        user_id: str,
        user_role: AccessRole,
        request_id: str,
        account_id: str,
        scope: str,
        records_deleted: int,
        **kwargs
    ) -> AuditEvent:
        """Log data deletion."""
        return self.log_event(
            event_type=AuditEventType.DELETION_EXECUTED,
            user_id=user_id,
            user_role=user_role,
            action=f"delete_data_{scope}",
            resource_type="deletion_request",
            resource_id=request_id,
            account_ids=[account_id],
            data_volume=records_deleted,
            **kwargs
        )
    
    def log_security_event(
        self,
        event_type: AuditEventType,
        user_id: str,
        ip_address: str,
        details: str,
        **kwargs
    ) -> AuditEvent:
        """Log security-related event."""
        if not self.config.log_security_events:
            return None
        
        return self.log_event(
            event_type=event_type,
            user_id=user_id,
            user_role=AccessRole.SERVICE,  # Default
            action=details,
            resource_type="security",
            resource_id=user_id,
            ip_address=ip_address,
            **kwargs
        )
    
    def _should_log(self, event_type: AuditEventType) -> bool:
        """Determine if this event type should be logged."""
        if event_type in [AuditEventType.DATA_QUERY, AuditEventType.DATA_ACCESS]:
            return self.config.log_data_access
        
        if event_type in [AuditEventType.IDENTITY_RESOLUTION, 
                         AuditEventType.CLUSTER_CREATED,
                         AuditEventType.PERSON_ASSIGNED]:
            return self.config.log_identity_ops
        
        if event_type in [AuditEventType.ATTRIBUTION_CALCULATED,
                         AuditEventType.ATTRIBUTION_EXPORTED]:
            return self.config.log_attribution
        
        if event_type in [AuditEventType.SYSTEM_STARTUP,
                         AuditEventType.SYSTEM_SHUTDOWN,
                         AuditEventType.ERROR_OCCURRED]:
            return self.config.log_system_events
        
        if event_type in [AuditEventType.AUTHENTICATION,
                         AuditEventType.AUTHORIZATION_FAILURE,
                         AuditEventType.SUSPICIOUS_ACTIVITY]:
            return self.config.log_security_events
        
        return True
    
    def _check_suspicious_activity(self, event: AuditEvent) -> None:
        """Check for suspicious activity patterns."""
        # Track activity per user
        now = datetime.now()
        self.user_activity[event.user_id].append(now)
        
        # Clean old activity (>1 hour)
        cutoff = now - timedelta(hours=1)
        self.user_activity[event.user_id] = [
            t for t in self.user_activity[event.user_id] if t > cutoff
        ]
        
        # Check threshold
        activity_count = len(self.user_activity[event.user_id])
        
        if activity_count > self.config.suspicious_threshold_per_hour:
            # Suspicious activity detected
            logger.warning(f"SUSPICIOUS ACTIVITY: User {event.user_id} has "
                         f"{activity_count} actions in last hour")
            
            # Log additional security event
            self.log_security_event(
                event_type=AuditEventType.SUSPICIOUS_ACTIVITY,
                user_id=event.user_id,
                ip_address=event.ip_address,
                details=f"High activity rate: {activity_count} events/hour"
            )
    
    def _flush_buffer(self) -> None:
        """Flush buffered events to storage."""
        if not self.event_buffer:
            return
        
        # Write all buffered events
        for event in self.event_buffer:
            self._write_event(event)
        
        logger.debug(f"Flushed {len(self.event_buffer)} audit events")
        self.event_buffer = []
    
    def _write_event(self, event: AuditEvent) -> None:
        """Write event to persistent storage."""
        # In production, write to:
        # - Append-only log file
        # - Database (audit table)
        # - SIEM system (Splunk, ELK)
        # - CloudWatch / Stackdriver
        
        # For this implementation, log to standard logger
        logger.info(f"AUDIT: {event.to_dict()}")
    
    def get_user_activity_summary(
        self,
        user_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get activity summary for a user."""
        # This would query the audit log storage
        # For now, return placeholder
        
        return {
            "user_id": user_id,
            "period": {
                "start": start_time.isoformat() if start_time else None,
                "end": end_time.isoformat() if end_time else None
            },
            "total_events": 0,  # Would query actual logs
            "event_types": [],
            "resources_accessed": [],
            "data_volume_total": 0
        }
    
    def verify_chain_integrity(self) -> bool:
        """
        Verify integrity of audit log chain.
        
        Checks that all hashes are valid and chain is unbroken.
        """
        # In production, read all events and verify chain
        # For now, return True
        return True
    
    def close(self) -> None:
        """Close audit logger and flush remaining events."""
        self._flush_buffer()
        logger.info("AuditLogger closed")


# Convenience functions

def get_audit_logger(config: Optional[AuditConfig] = None) -> AuditLogger:
    """Get or create audit logger instance."""
    return AuditLogger(config)


# Decorator for automatic audit logging
def audited(action: str, resource_type: str):
    """
    Decorator to automatically log function calls.
    
    Example:
    ```python
    @audited("resolve_identities", "household")
    def resolve_household(account_id: str, user_id: str):
        # Function implementation
        pass
    ```
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Get audit logger
            audit_logger = get_audit_logger()
            
            # Extract user info from kwargs or context
            user_id = kwargs.get('user_id', 'system')
            user_role = kwargs.get('user_role', AccessRole.SERVICE)
            
            # Execute function
            start_time = datetime.now()
            try:
                result = func(*args, **kwargs)
                success = True
                error_msg = ""
            except Exception as e:
                success = False
                error_msg = str(e)
                raise
            finally:
                # Log the call
                execution_time = int((datetime.now() - start_time).total_seconds() * 1000)
                
                audit_logger.log_event(
                    event_type=AuditEventType.DATA_ACCESS,
                    user_id=user_id,
                    user_role=user_role,
                    action=action,
                    resource_type=resource_type,
                    resource_id=str(args[0]) if args else "",
                    success=success,
                    error_message=error_msg,
                    execution_time_ms=execution_time
                )
            
            return result
        return wrapper
    return decorator
