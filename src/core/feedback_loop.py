"""
Feedback Loop for Continuous Validation

Production feedback mechanism that:
1. Compares predicted assignments to actual outcomes
2. Detects model degradation over time
3. Triggers retraining or alerts when accuracy drops
4. Validates attribution against ground truth (A/B tests)

Implements:
- Prediction storage with actual outcome tracking
- Accuracy monitoring with drift detection
- A/B test integration
- Automated retraining triggers
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Set
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class FeedbackType(Enum):
    """Type of feedback received."""
    EXPLICIT = "explicit"  # User confirmed assignment
    IMPLICIT = "implicit"  # Inferred from behavior
    AB_TEST = "ab_test"  # Ground truth from experiment
    SURVEY = "survey"  # User survey response
    MANUAL = "manual"  # Admin/manual verification


class AccuracyTrend(Enum):
    """Trend in accuracy over time."""
    IMPROVING = "improving"
    STABLE = "stable"
    DEGRADING = "degrading"
    CRITICAL = "critical"


@dataclass
class PredictionRecord:
    """Stored prediction waiting for feedback."""
    prediction_id: str
    account_id: str
    session_id: str
    
    # Prediction details
    predicted_person_id: str
    confidence: float
    timestamp: datetime
    
    # Features (for retraining)
    session_features: Dict[str, float] = field(default_factory=dict)
    
    # Actual outcome (filled later)
    actual_person_id: Optional[str] = None
    feedback_type: Optional[FeedbackType] = None
    feedback_timestamp: Optional[datetime] = None
    correct: Optional[bool] = None
    
    # Metadata
    model_version: str = "1.0.0"


@dataclass
class FeedbackLoopConfig:
    """Configuration for feedback loop."""
    # Storage
    retention_days: int = 90
    max_pending_predictions: int = 100000
    
    # Feedback collection
    explicit_feedback_ttl_hours: int = 72  # Time to collect explicit feedback
    implicit_feedback_delay_hours: int = 24  # Delay before checking implicit
    
    # Monitoring
    accuracy_window_days: int = 7
    min_samples_for_stats: int = 100
    
    # Thresholds
    accuracy_warning_threshold: float = 0.70  # Below this = warning
    accuracy_critical_threshold: float = 0.60  # Below this = critical
    confidence_calibration_threshold: float = 0.10  # Brier score diff
    
    # Auto-retraining
    enable_auto_retraining: bool = True
    retraining_trigger: str = "degradation"  # "degradation", "schedule", "manual"
    min_retraining_interval_days: int = 7


@dataclass
class AccuracyMetrics:
    """Accuracy metrics over a time window."""
    window_start: datetime
    window_end: datetime
    
    # Sample counts
    total_predictions: int
    feedback_received: int
    feedback_rate: float
    
    # Accuracy
    overall_accuracy: float
    accuracy_by_persona: Dict[str, float]
    accuracy_by_confidence: Dict[str, float]  # binned by confidence
    
    # Calibration
    brier_score: float
    calibration_error: float  # ECE
    
    # Trends
    trend: AccuracyTrend
    change_from_previous: float  # Percentage point change


class FeedbackLoop:
    """
    Continuous validation and feedback system.
    
    Workflow:
    1. Store predictions with unique IDs
    2. Collect feedback (explicit, implicit, A/B test)
    3. Compute accuracy metrics
    4. Detect degradation
    5. Trigger retraining or alerts
    
    Supports:
    - Real-time accuracy monitoring
    - A/B test validation
    - Confidence calibration tracking
    - Automated retraining triggers
    """
    
    def __init__(self, config: Optional[FeedbackLoopConfig] = None):
        self.config = config or FeedbackLoopConfig()
        
        # Storage
        self.pending_predictions: Dict[str, PredictionRecord] = {}
        self.resolved_predictions: List[PredictionRecord] = []
        
        # Metrics history
        self.accuracy_history: List[AccuracyMetrics] = []
        
        # A/B test tracking
        self.ab_tests: Dict[str, Dict] = {}
        
        # Alert tracking
        self.last_alert_time: Optional[datetime] = None
        self.last_retraining_time: Optional[datetime] = None
        
        logger.info("FeedbackLoop initialized")
    
    def store_prediction(
        self,
        account_id: str,
        session_id: str,
        predicted_person_id: str,
        confidence: float,
        session_features: Dict[str, float],
        model_version: str = "1.0.0"
    ) -> str:
        """
        Store a prediction for future feedback.
        
        Parameters
        ----------
        account_id : str
            Account identifier
        session_id : str
            Session identifier
        predicted_person_id : str
            Predicted person assignment
        confidence : float
            Prediction confidence
        session_features : Dict[str, float]
            Features used for prediction (for retraining)
        model_version : str
            Model version that made prediction
        
        Returns
        -------
        str
            Prediction ID for feedback reference
        """
        prediction_id = f"pred_{account_id}_{session_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        record = PredictionRecord(
            prediction_id=prediction_id,
            account_id=account_id,
            session_id=session_id,
            predicted_person_id=predicted_person_id,
            confidence=confidence,
            timestamp=datetime.now(),
            session_features=session_features,
            model_version=model_version
        )
        
        self.pending_predictions[prediction_id] = record
        
        # Cleanup old pending predictions
        self._cleanup_pending()
        
        return prediction_id
    
    def submit_feedback(
        self,
        prediction_id: str,
        actual_person_id: str,
        feedback_type: FeedbackType = FeedbackType.EXPLICIT
    ) -> bool:
        """
        Submit feedback for a stored prediction.
        
        Parameters
        ----------
        prediction_id : str
            ID from store_prediction
        actual_person_id : str
            Actual/ground truth person ID
        feedback_type : FeedbackType
            Type of feedback
        
        Returns
        -------
        bool
            True if feedback accepted, False if prediction not found
        """
        if prediction_id not in self.pending_predictions:
            logger.warning(f"Prediction {prediction_id} not found for feedback")
            return False
        
        record = self.pending_predictions[prediction_id]
        record.actual_person_id = actual_person_id
        record.feedback_type = feedback_type
        record.feedback_timestamp = datetime.now()
        record.correct = (record.predicted_person_id == actual_person_id)
        
        # Move to resolved
        self.resolved_predictions.append(record)
        del self.pending_predictions[prediction_id]
        
        logger.info(f"Feedback received for {prediction_id}: "
                   f"correct={record.correct}, type={feedback_type.value}")
        
        return True
    
    def submit_implicit_feedback(
        self,
        account_id: str,
        session_id: str,
        subsequent_sessions: List[Any]
    ) -> Optional[str]:
        """
        Infer feedback from subsequent behavior.
        
        If user continues on same device/pattern, prediction was likely correct.
        If user switches immediately or shows different pattern, likely wrong.
        
        Returns prediction_id if feedback submitted, None otherwise.
        """
        # Find matching prediction
        matching_pred = None
        for pred_id, record in self.pending_predictions.items():
            if record.account_id == account_id and record.session_id == session_id:
                matching_pred = record
                break
        
        if not matching_pred:
            return None
        
        # Check if enough time has passed
        elapsed = datetime.now() - matching_pred.timestamp
        if elapsed < timedelta(hours=self.config.implicit_feedback_delay_hours):
            return None
        
        # Infer from subsequent sessions
        if subsequent_sessions:
            # Check if subsequent sessions match predicted persona
            subsequent_devices = [s.device_type for s in subsequent_sessions]
            subsequent_hours = [s.start_time.hour for s in subsequent_sessions 
                              if hasattr(s, 'start_time')]
            
            # Simple heuristic: if subsequent 3+ sessions same device = correct
            if len(subsequent_devices) >= 3:
                device_consistency = len(set(subsequent_devices)) == 1
                
                if device_consistency:
                    # Likely correct
                    self.submit_feedback(
                        matching_pred.prediction_id,
                        matching_pred.predicted_person_id,
                        FeedbackType.IMPLICIT
                    )
                    return matching_pred.prediction_id
        
        return None
    
    def start_ab_test(
        self,
        test_id: str,
        account_ids: List[str],
        test_type: str = "ground_truth"  # "ground_truth", "model_comparison"
    ) -> None:
        """
        Start A/B test to collect ground truth data.
        
        For ground truth tests:
        - Ask users to confirm identity
        - Track explicit feedback
        - Measure model accuracy against confirmed labels
        """
        self.ab_tests[test_id] = {
            "test_id": test_id,
            "account_ids": set(account_ids),
            "test_type": test_type,
            "start_time": datetime.now(),
            "predictions": [],
            "feedback": []
        }
        
        logger.info(f"A/B test {test_id} started with {len(account_ids)} accounts")
    
    def end_ab_test(self, test_id: str) -> Dict[str, Any]:
        """End A/B test and compute results."""
        if test_id not in self.ab_tests:
            return {"error": "Test not found"}
        
        test = self.ab_tests[test_id]
        test["end_time"] = datetime.now()
        
        # Compute accuracy for test predictions
        test_predictions = [
            p for p in self.resolved_predictions
            if p.account_id in test["account_ids"] and 
            p.timestamp >= test["start_time"] and
            (p.timestamp <= test["end_time"] if test.get("end_time") else True)
        ]
        
        if not test_predictions:
            return {"error": "No predictions in test period"}
        
        correct = sum(1 for p in test_predictions if p.correct)
        total = len(test_predictions)
        accuracy = correct / total if total > 0 else 0
        
        results = {
            "test_id": test_id,
            "duration_days": (test["end_time"] - test["start_time"]).days,
            "total_predictions": total,
            "correct_predictions": correct,
            "accuracy": accuracy,
            "confidence_avg": np.mean([p.confidence for p in test_predictions]),
            "feedback_breakdown": self._feedback_breakdown(test_predictions)
        }
        
        logger.info(f"A/B test {test_id} completed: accuracy={accuracy:.2%}")
        
        return results
    
    def compute_accuracy_metrics(
        self,
        window_days: Optional[int] = None
    ) -> Optional[AccuracyMetrics]:
        """
        Compute accuracy metrics over time window.
        
        Returns None if insufficient data.
        """
        window = window_days or self.config.accuracy_window_days
        cutoff = datetime.now() - timedelta(days=window)
        
        # Get predictions in window
        predictions = [
            p for p in self.resolved_predictions
            if p.feedback_timestamp and p.feedback_timestamp >= cutoff
        ]
        
        if len(predictions) < self.config.min_samples_for_stats:
            logger.warning(f"Insufficient samples for accuracy: {len(predictions)}")
            return None
        
        # Compute metrics
        total = len(predictions)
        correct = sum(1 for p in predictions if p.correct)
        accuracy = correct / total
        
        # By persona (if we can infer from person_id)
        accuracy_by_persona = self._accuracy_by_persona(predictions)
        
        # By confidence bin
        accuracy_by_confidence = self._accuracy_by_confidence(predictions)
        
        # Brier score (calibration)
        brier = self._compute_brier_score(predictions)
        
        # Trend
        trend = self._compute_trend(accuracy)
        
        # Change from previous window
        change = 0.0
        if self.accuracy_history:
            prev_accuracy = self.accuracy_history[-1].overall_accuracy
            change = accuracy - prev_accuracy
        
        metrics = AccuracyMetrics(
            window_start=cutoff,
            window_end=datetime.now(),
            total_predictions=total,
            feedback_received=total,
            feedback_rate=total / max(1, total + len(self.pending_predictions)),
            overall_accuracy=accuracy,
            accuracy_by_persona=accuracy_by_persona,
            accuracy_by_confidence=accuracy_by_confidence,
            brier_score=brier,
            calibration_error=abs(accuracy - np.mean([p.confidence for p in predictions])),
            trend=trend,
            change_from_previous=change
        )
        
        self.accuracy_history.append(metrics)
        
        # Keep only last 12 windows
        if len(self.accuracy_history) > 12:
            self.accuracy_history = self.accuracy_history[-12:]
        
        logger.info(f"Accuracy metrics: {accuracy:.2%} ({trend.value})")
        
        return metrics
    
    def check_and_trigger_actions(self) -> List[str]:
        """
        Check metrics and trigger retraining or alerts.
        
        Returns list of actions taken.
        """
        actions = []
        
        metrics = self.compute_accuracy_metrics()
        if not metrics:
            return actions
        
        # Check accuracy degradation
        if metrics.overall_accuracy < self.config.accuracy_critical_threshold:
            actions.append("ALERT_CRITICAL: Accuracy below 60%")
            self._send_alert("critical", f"Accuracy degraded to {metrics.overall_accuracy:.2%}")
        
        elif metrics.overall_accuracy < self.config.accuracy_warning_threshold:
            actions.append("ALERT_WARNING: Accuracy below 70%")
            self._send_alert("warning", f"Accuracy at {metrics.overall_accuracy:.2%}")
        
        # Check trend
        if metrics.trend == AccuracyTrend.DEGRADING and metrics.change_from_previous < -0.05:
            actions.append("ALERT_DEGRADING: Accuracy declining")
        
        # Auto-retraining
        if self.config.enable_auto_retraining:
            should_retrain = False
            
            if self.config.retraining_trigger == "degradation":
                if metrics.trend == AccuracyTrend.DEGRADING or \
                   metrics.overall_accuracy < self.config.accuracy_warning_threshold:
                    should_retrain = True
            
            if should_retrain:
                # Check min interval
                if self.last_retraining_time:
                    days_since = (datetime.now() - self.last_retraining_time).days
                    if days_since < self.config.min_retraining_interval_days:
                        should_retrain = False
                
                if should_retrain:
                    actions.append("ACTION_RETRAINING: Triggered auto-retraining")
                    self._trigger_retraining()
        
        return actions
    
    def get_calibration_report(self) -> Dict[str, Any]:
        """Get confidence calibration report."""
        predictions = self.resolved_predictions
        
        if not predictions:
            return {"error": "No predictions"}
        
        # Bin by predicted confidence
        bins = defaultdict(list)
        for p in predictions:
            # 10% bins
            bin_key = int(p.confidence * 10) / 10
            bins[bin_key].append(p)
        
        calibration_data = []
        for conf_level in sorted(bins.keys()):
            preds = bins[conf_level]
            actual_accuracy = sum(1 for p in preds if p.correct) / len(preds)
            
            calibration_data.append({
                "predicted_confidence": conf_level,
                "actual_accuracy": actual_accuracy,
                "sample_count": len(preds),
                "calibration_error": abs(conf_level - actual_accuracy)
            })
        
        return {
            "brier_score": self._compute_brier_score(predictions),
            "expected_calibration_error": np.mean([d["calibration_error"] for d in calibration_data]),
            "bins": calibration_data
        }
    
    def _cleanup_pending(self):
        """Remove old pending predictions."""
        cutoff = datetime.now() - timedelta(days=7)
        expired = [
            pid for pid, record in self.pending_predictions.items()
            if record.timestamp < cutoff
        ]
        
        for pid in expired:
            del self.pending_predictions[pid]
    
    def _accuracy_by_persona(self, predictions: List[PredictionRecord]) -> Dict[str, float]:
        """Compute accuracy by inferred persona."""
        # Group by predicted person_id pattern
        by_person = defaultdict(list)
        for p in predictions:
            by_person[p.predicted_person_id].append(p)
        
        return {
            person: sum(1 for p in preds if p.correct) / len(preds)
            for person, preds in by_person.items()
            if len(preds) >= 10  # Min sample size
        }
    
    def _accuracy_by_confidence(self, predictions: List[PredictionRecord]) -> Dict[str, float]:
        """Compute accuracy by confidence bin."""
        bins = {
            "low": [],      # 0-0.5
            "medium": [],   # 0.5-0.8
            "high": []      # 0.8-1.0
        }
        
        for p in predictions:
            if p.confidence < 0.5:
                bins["low"].append(p)
            elif p.confidence < 0.8:
                bins["medium"].append(p)
            else:
                bins["high"].append(p)
        
        return {
            bin_name: sum(1 for p in preds if p.correct) / len(preds)
            for bin_name, preds in bins.items()
            if len(preds) >= 10
        }
    
    def _compute_brier_score(self, predictions: List[PredictionRecord]) -> float:
        """Compute Brier score for calibration."""
        scores = []
        for p in predictions:
            outcome = 1.0 if p.correct else 0.0
            scores.append((p.confidence - outcome) ** 2)
        
        return np.mean(scores) if scores else 0.0
    
    def _compute_trend(self, current_accuracy: float) -> AccuracyTrend:
        """Compute accuracy trend from history."""
        if len(self.accuracy_history) < 3:
            return AccuracyTrend.STABLE
        
        # Get last 3 windows
        recent = self.accuracy_history[-3:]
        accuracies = [m.overall_accuracy for m in recent]
        
        # Simple trend detection
        if accuracies[-1] > accuracies[0] + 0.05:
            return AccuracyTrend.IMPROVING
        elif accuracies[-1] < accuracies[0] - 0.05:
            if accuracies[-1] < self.config.accuracy_critical_threshold:
                return AccuracyTrend.CRITICAL
            return AccuracyTrend.DEGRADING
        
        return AccuracyTrend.STABLE
    
    def _feedback_breakdown(self, predictions: List[PredictionRecord]) -> Dict[str, int]:
        """Count feedback by type."""
        counts = defaultdict(int)
        for p in predictions:
            if p.feedback_type:
                counts[p.feedback_type.value] += 1
        return dict(counts)
    
    def _send_alert(self, severity: str, message: str):
        """Send alert (placeholder)."""
        logger.warning(f"ALERT [{severity.upper()}]: {message}")
        self.last_alert_time = datetime.now()
    
    def _trigger_retraining(self):
        """Trigger model retraining (placeholder)."""
        logger.info("ACTION: Triggering model retraining")
        self.last_retraining_time = datetime.now()
        
        # In production:
        # 1. Export training data from resolved_predictions
        # 2. Submit retraining job to ML pipeline
        # 3. Deploy new model version
        # 4. A/B test new vs old


# Convenience functions

def create_feedback_loop(config: Optional[FeedbackLoopConfig] = None) -> FeedbackLoop:
    """Create feedback loop instance."""
    return FeedbackLoop(config)
