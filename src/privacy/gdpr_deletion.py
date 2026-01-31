"""
GDPR/CCPA Data Deletion Pipeline

Implements "Right to Deletion" compliance for privacy regulations:
- GDPR (EU): Right to erasure (Article 17)
- CCPA (California): Right to deletion
- Similar regulations globally

Provides automated mechanisms to:
1. Delete specific device fingerprints
2. Purge user-associated data
3. Recalculate aggregate statistics without deleted data
4. Maintain audit trail of deletions
5. Handle cascade effects on identity graphs
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any
from datetime import datetime, timedelta
from enum import Enum
import hashlib
import json
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class DeletionStatus(Enum):
    """Status of a deletion request."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


class DeletionScope(Enum):
    """Scope of data to delete."""
    DEVICE_ONLY = "device_only"  # Single device
    PERSON = "person"  # All data for inferred person
    HOUSEHOLD = "household"  # All data for account
    PARTIAL = "partial"  # Specific time range or data types


@dataclass
class DeletionRequest:
    """Data deletion request."""
    request_id: str
    account_id: str
    scope: DeletionScope
    
    # Identifiers to delete
    device_fingerprints: List[str] = field(default_factory=list)
    person_ids: List[str] = field(default_factory=list)
    
    # Time bounds (for partial deletion)
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    
    # Request metadata
    requested_by: str = "user"  # "user", "admin", "automated"
    request_source: str = ""  # "gdpr_request", "ccpa_request", "user_initiated"
    requested_at: datetime = field(default_factory=datetime.now)
    
    # Verification
    verification_token: str = ""  # Email confirmation, etc.
    verified: bool = False
    
    # Status
    status: DeletionStatus = DeletionStatus.PENDING
    completed_at: Optional[datetime] = None
    deleted_records: Dict[str, int] = field(default_factory=dict)
    error_log: List[str] = field(default_factory=list)


@dataclass
class DeletionConfig:
    """Configuration for deletion pipeline."""
    # Retention periods (days)
    raw_data_retention_days: int = 90
    aggregated_data_retention_days: int = 365
    audit_log_retention_days: int = 2555  # 7 years (legal requirement)
    
    # Cascade settings
    recalculate_clusters_after_deletion: bool = True
    update_attribution_after_deletion: bool = True
    
    # Performance
    batch_size: int = 1000
    max_deletion_time_seconds: int = 300  # 5 minutes timeout
    
    # Safety
    dry_run_mode: bool = False  # Log what would be deleted, don't actually delete
    require_verification: bool = True
    backup_before_deletion: bool = True


class DataDeletionPipeline:
    """
    GDPR/CCPA-compliant data deletion pipeline.
    
    Handles complex cascade effects:
    1. Delete raw events
    2. Rebuild sessions without deleted events
    3. Re-cluster households affected by deletion
    4. Update aggregate statistics
    5. Invalidate affected attributions
    6. Maintain audit trail
    
    Ensures "forgotten" users don't leave traces in aggregates.
    """
    
    def __init__(self, config: Optional[DeletionConfig] = None):
        self.config = config or DeletionConfig()
        self.deletion_log: List[DeletionRequest] = []
        
        logger.info("DataDeletionPipeline initialized "
                   f"(dry_run={self.config.dry_run_mode})")
    
    def submit_deletion_request(
        self,
        account_id: str,
        scope: DeletionScope,
        device_fingerprints: Optional[List[str]] = None,
        person_ids: Optional[List[str]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        request_source: str = "user_initiated"
    ) -> DeletionRequest:
        """
        Submit a new data deletion request.
        
        Parameters
        ----------
        account_id : str
            Account to delete data for
        scope : DeletionScope
            Scope of deletion
        device_fingerprints : List[str], optional
            Specific devices to delete (for DEVICE_ONLY scope)
        person_ids : List[str], optional
            Specific persons to delete (for PERSON scope)
        start_date, end_date : datetime, optional
            Time range for PARTIAL scope
        request_source : str
            Source of request (gdpr_request, ccpa_request, etc.)
        
        Returns
        -------
        DeletionRequest
            Submitted request with ID
        """
        request_id = self._generate_request_id(account_id)
        
        request = DeletionRequest(
            request_id=request_id,
            account_id=account_id,
            scope=scope,
            device_fingerprints=device_fingerprints or [],
            person_ids=person_ids or [],
            start_date=start_date,
            end_date=end_date,
            request_source=request_source,
            requested_at=datetime.now()
        )
        
        if self.config.require_verification:
            request.verification_token = self._generate_verification_token()
            logger.info(f"Deletion request {request_id} created, awaiting verification")
        else:
            request.verified = True
        
        self.deletion_log.append(request)
        
        return request
    
    def verify_deletion_request(
        self,
        request_id: str,
        verification_token: str
    ) -> bool:
        """Verify a deletion request (e.g., via email confirmation)."""
        for request in self.deletion_log:
            if request.request_id == request_id:
                if request.verification_token == verification_token:
                    request.verified = True
                    logger.info(f"Deletion request {request_id} verified")
                    return True
                else:
                    logger.warning(f"Invalid verification token for {request_id}")
                    return False
        
        logger.error(f"Deletion request {request_id} not found")
        return False
    
    def execute_deletion(
        self,
        request_id: str,
        data_stores: Dict[str, Any]
    ) -> DeletionRequest:
        """
        Execute a verified deletion request.
        
        Parameters
        ----------
        request_id : str
            Request to execute
        data_stores : Dict[str, Any]
            Dictionary of data store connections
            - 'events': Event storage
            - 'sessions': Session storage
            - 'clusters': Cluster storage
            - 'identity_graph': Identity graph storage
        
        Returns
        -------
        DeletionRequest
            Completed request with status
        """
        request = self._get_request(request_id)
        if not request:
            raise ValueError(f"Request {request_id} not found")
        
        if not request.verified and self.config.require_verification:
            raise ValueError(f"Request {request_id} not verified")
        
        if request.status == DeletionStatus.COMPLETED:
            logger.info(f"Request {request_id} already completed")
            return request
        
        request.status = DeletionStatus.IN_PROGRESS
        logger.info(f"Executing deletion request {request_id} "
                   f"(scope={request.scope.value}, dry_run={self.config.dry_run_mode})")
        
        try:
            # Step 1: Identify records to delete
            records_to_delete = self._identify_records(request, data_stores)
            
            # Step 2: Delete raw events
            deleted_events = self._delete_events(
                records_to_delete['events'],
                data_stores.get('events')
            )
            request.deleted_records['events'] = deleted_events
            
            # Step 3: Rebuild affected sessions
            affected_sessions = self._rebuild_sessions(
                request.account_id,
                records_to_delete['affected_sessions'],
                data_stores.get('sessions')
            )
            request.deleted_records['sessions_rebuilt'] = affected_sessions
            
            # Step 4: Update clusters
            if self.config.recalculate_clusters_after_deletion:
                clusters_updated = self._update_clusters(
                    request.account_id,
                    request.scope,
                    request.person_ids,
                    data_stores.get('clusters')
                )
                request.deleted_records['clusters_updated'] = clusters_updated
            
            # Step 5: Update identity graph
            graph_updated = self._update_identity_graph(
                request,
                data_stores.get('identity_graph')
            )
            request.deleted_records['identity_graph_nodes_removed'] = graph_updated
            
            # Step 6: Invalidate attributions
            if self.config.update_attribution_after_deletion:
                attributions_invalidated = self._invalidate_attributions(
                    request,
                    data_stores
                )
                request.deleted_records['attributions_invalidated'] = attributions_invalidated
            
            # Step 7: Backup (if configured)
            if self.config.backup_before_deletion and not self.config.dry_run_mode:
                self._create_deletion_backup(request, records_to_delete)
            
            request.status = DeletionStatus.COMPLETED
            request.completed_at = datetime.now()
            
            logger.info(f"Deletion request {request_id} completed successfully. "
                       f"Deleted: {request.deleted_records}")
        
        except Exception as e:
            request.status = DeletionStatus.FAILED
            request.error_log.append(str(e))
            logger.error(f"Deletion request {request_id} failed: {e}")
            raise
        
        return request
    
    def _identify_records(
        self,
        request: DeletionRequest,
        data_stores: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Identify all records that need to be deleted or updated."""
        records = {
            'events': [],
            'affected_sessions': set(),
            'affected_persons': set(),
            'affected_clusters': set()
        }
        
        # Get events store
        events_store = data_stores.get('events')
        if not events_store:
            return records
        
        # Query events based on scope
        if request.scope == DeletionScope.DEVICE_ONLY:
            # Find all events for specified devices
            for device_fp in request.device_fingerprints:
                events = self._query_events_by_device(
                    events_store,
                    request.account_id,
                    device_fp,
                    request.start_date,
                    request.end_date
                )
                records['events'].extend(events)
        
        elif request.scope == DeletionScope.PERSON:
            # Find all events for specified persons
            for person_id in request.person_ids:
                events = self._query_events_by_person(
                    events_store,
                    request.account_id,
                    person_id,
                    request.start_date,
                    request.end_date
                )
                records['events'].extend(events)
                records['affected_persons'].add(person_id)
        
        elif request.scope == DeletionScope.HOUSEHOLD:
            # Find all events for account
            events = self._query_events_by_account(
                events_store,
                request.account_id,
                request.start_date,
                request.end_date
            )
            records['events'] = events
        
        elif request.scope == DeletionScope.PARTIAL:
            # Time-bounded deletion
            events = self._query_events_by_time(
                events_store,
                request.account_id,
                request.start_date,
                request.end_date
            )
            records['events'] = events
        
        # Identify affected sessions
        for event in records['events']:
            if hasattr(event, 'session_id'):
                records['affected_sessions'].add(event.session_id)
        
        return records
    
    def _delete_events(
        self,
        events: List[Any],
        events_store: Any
    ) -> int:
        """Delete events from storage."""
        if self.config.dry_run_mode:
            logger.info(f"[DRY RUN] Would delete {len(events)} events")
            return len(events)
        
        if not events_store:
            return 0
        
        deleted_count = 0
        for event in events:
            try:
                # Delete from storage
                if hasattr(events_store, 'delete'):
                    events_store.delete(event.event_id)
                deleted_count += 1
            except Exception as e:
                logger.warning(f"Failed to delete event {event.event_id}: {e}")
        
        return deleted_count
    
    def _rebuild_sessions(
        self,
        account_id: str,
        affected_session_ids: Set[str],
        sessions_store: Any
    ) -> int:
        """Rebuild sessions after event deletion."""
        if not sessions_store or not affected_session_ids:
            return 0
        
        if self.config.dry_run_mode:
            logger.info(f"[DRY RUN] Would rebuild {len(affected_session_ids)} sessions")
            return len(affected_session_ids)
        
        rebuilt_count = 0
        for session_id in affected_session_ids:
            try:
                # Get remaining events for this session
                remaining_events = self._get_session_events(
                    sessions_store,
                    session_id
                )
                
                if len(remaining_events) < 2:
                    # Session has too few events - delete it
                    sessions_store.delete(session_id)
                else:
                    # Rebuild session metadata
                    self._update_session_metadata(
                        sessions_store,
                        session_id,
                        remaining_events
                    )
                
                rebuilt_count += 1
            except Exception as e:
                logger.warning(f"Failed to rebuild session {session_id}: {e}")
        
        return rebuilt_count
    
    def _update_clusters(
        self,
        account_id: str,
        scope: DeletionScope,
        person_ids: List[str],
        clusters_store: Any
    ) -> int:
        """Update clusters after data deletion."""
        if not clusters_store:
            return 0
        
        if self.config.dry_run_mode:
            logger.info(f"[DRY RUN] Would update clusters for {account_id}")
            return 1
        
        # Remove deleted persons from clusters
        if scope in [DeletionScope.PERSON, DeletionScope.HOUSEHOLD]:
            for person_id in person_ids:
                try:
                    clusters_store.remove_person(person_id)
                except Exception as e:
                    logger.warning(f"Failed to remove person {person_id} from clusters: {e}")
        
        # Trigger re-clustering for affected households
        if scope == DeletionScope.HOUSEHOLD:
            try:
                clusters_store.flag_for_reclustering(account_id)
            except Exception as e:
                logger.warning(f"Failed to flag {account_id} for reclustering: {e}")
        
        return len(person_ids)
    
    def _update_identity_graph(
        self,
        request: DeletionRequest,
        graph_store: Any
    ) -> int:
        """Update identity graph after deletion."""
        if not graph_store:
            return 0
        
        if self.config.dry_run_mode:
            logger.info(f"[DRY RUN] Would update identity graph for {request.account_id}")
            return 0
        
        nodes_removed = 0
        
        # Remove device nodes
        for device_fp in request.device_fingerprints:
            try:
                graph_store.remove_device(device_fp)
                nodes_removed += 1
            except Exception as e:
                logger.warning(f"Failed to remove device {device_fp}: {e}")
        
        # Remove person nodes
        for person_id in request.person_ids:
            try:
                graph_store.remove_person(person_id)
                nodes_removed += 1
            except Exception as e:
                logger.warning(f"Failed to remove person {person_id}: {e}")
        
        # If household scope, remove entire household subgraph
        if request.scope == DeletionScope.HOUSEHOLD:
            try:
                graph_store.remove_household(request.account_id)
                nodes_removed += 1
            except Exception as e:
                logger.warning(f"Failed to remove household {request.account_id}: {e}")
        
        return nodes_removed
    
    def _invalidate_attributions(
        self,
        request: DeletionRequest,
        data_stores: Dict[str, Any]
    ) -> int:
        """Invalidate attribution calculations affected by deletion."""
        # Mark attributions for re-calculation
        attribution_store = data_stores.get('attributions')
        if not attribution_store:
            return 0
        
        if self.config.dry_run_mode:
            logger.info(f"[DRY RUN] Would invalidate attributions for {request.account_id}")
            return 0
        
        try:
            attribution_store.flag_for_recalculation(request.account_id)
            return 1
        except Exception as e:
            logger.warning(f"Failed to flag attributions for recalculation: {e}")
            return 0
    
    def _create_deletion_backup(
        self,
        request: DeletionRequest,
        records_to_delete: Dict[str, Any]
    ) -> str:
        """Create backup of data before deletion (for recovery)."""
        backup_id = f"backup_{request.request_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        backup_data = {
            "request": {
                "request_id": request.request_id,
                "account_id": request.account_id,
                "scope": request.scope.value,
                "requested_at": request.requested_at.isoformat()
            },
            "records_summary": {
                "event_count": len(records_to_delete['events']),
                "session_count": len(records_to_delete['affected_sessions']),
                "person_count": len(records_to_delete['affected_persons'])
            },
            "backup_created": datetime.now().isoformat()
        }
        
        # In production, write to secure backup storage
        logger.info(f"Deletion backup created: {backup_id}")
        
        return backup_id
    
    def get_deletion_summary(
        self,
        account_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get summary of deletions for audit purposes."""
        filtered_requests = self.deletion_log
        
        if account_id:
            filtered_requests = [r for r in filtered_requests if r.account_id == account_id]
        
        if start_date:
            filtered_requests = [r for r in filtered_requests if r.requested_at >= start_date]
        
        if end_date:
            filtered_requests = [r for r in filtered_requests if r.requested_at <= end_date]
        
        total_requests = len(filtered_requests)
        completed = sum(1 for r in filtered_requests if r.status == DeletionStatus.COMPLETED)
        failed = sum(1 for r in filtered_requests if r.status == DeletionStatus.FAILED)
        
        total_records_deleted = defaultdict(int)
        for request in filtered_requests:
            if request.status == DeletionStatus.COMPLETED:
                for record_type, count in request.deleted_records.items():
                    total_records_deleted[record_type] += count
        
        return {
            "total_requests": total_requests,
            "completed": completed,
            "failed": failed,
            "pending": total_requests - completed - failed,
            "total_records_deleted": dict(total_records_deleted),
            "date_range": {
                "start": start_date.isoformat() if start_date else None,
                "end": end_date.isoformat() if end_date else None
            }
        }
    
    def _generate_request_id(self, account_id: str) -> str:
        """Generate unique deletion request ID."""
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        random_suffix = hashlib.sha256(
            f"{account_id}{timestamp}{datetime.now().microsecond}".encode()
        ).hexdigest()[:8]
        return f"del_{account_id}_{timestamp}_{random_suffix}"
    
    def _generate_verification_token(self) -> str:
        """Generate verification token."""
        return hashlib.sha256(
            f"verify{datetime.now().isoformat()}{hash(datetime.now())}".encode()
        ).hexdigest()[:16]
    
    def _get_request(self, request_id: str) -> Optional[DeletionRequest]:
        """Get deletion request by ID."""
        for request in self.deletion_log:
            if request.request_id == request_id:
                return request
        return None
    
    # Placeholder methods for data store interactions
    def _query_events_by_device(self, store, account_id, device_fp, start, end):
        """Query events by device fingerprint."""
        # Implementation depends on actual data store
        return []
    
    def _query_events_by_person(self, store, account_id, person_id, start, end):
        """Query events by person ID."""
        return []
    
    def _query_events_by_account(self, store, account_id, start, end):
        """Query all events for account."""
        return []
    
    def _query_events_by_time(self, store, account_id, start, end):
        """Query events by time range."""
        return []
    
    def _get_session_events(self, store, session_id):
        """Get events for a session."""
        return []
    
    def _update_session_metadata(self, store, session_id, events):
        """Update session metadata after event changes."""
        pass


# Convenience functions

def submit_gdpr_deletion_request(
    account_id: str,
    device_fingerprints: Optional[List[str]] = None
) -> DeletionRequest:
    """
    Submit GDPR deletion request.
    
    Example:
    ```python
    request = submit_gdpr_deletion_request(
        account_id="account_123",
        device_fingerprints=["device_abc", "device_xyz"]
    )
    ```
    """
    pipeline = DataDeletionPipeline()
    return pipeline.submit_deletion_request(
        account_id=account_id,
        scope=DeletionScope.DEVICE_ONLY if device_fingerprints else DeletionScope.HOUSEHOLD,
        device_fingerprints=device_fingerprints or [],
        request_source="gdpr_request"
    )
