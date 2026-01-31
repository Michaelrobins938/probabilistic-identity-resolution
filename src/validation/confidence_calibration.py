"""
Confidence Calibration with Brier Score

Measures how well probability estimates match actual outcomes.
Critical for trustworthy attribution in high-stakes scenarios.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
import math
from collections import defaultdict

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from models.streaming_event import Session
from validation.synthetic_households import GroundTruth
from validation.enhanced_synthetic import EnhancedGroundTruth


@dataclass
class CalibrationMetrics:
    """Confidence calibration metrics."""
    
    # Brier score (lower is better, 0 = perfect)
    brier_score: float = 0.0
    
    # Brier skill score (vs baseline, 1 = perfect, 0 = no skill, <0 = worse than baseline)
    brier_skill_score: float = 0.0
    
    # Reliability diagram data
    reliability_bins: Dict[float, Dict[str, float]] = field(default_factory=dict)
    
    # Calibration error (ECE - Expected Calibration Error)
    expected_calibration_error: float = 0.0
    
    # Maximum calibration error
    max_calibration_error: float = 0.0
    
    # Per-confidence bin accuracy
    bin_accuracies: Dict[str, float] = field(default_factory=dict)
    
    def get_summary(self) -> str:
        """Get human-readable summary."""
        lines = [
            "=" * 70,
            "CONFIDENCE CALIBRATION METRICS",
            "=" * 70,
            "",
            "Brier Score (lower is better)",
            f"  Score:                   {self.brier_score:.4f}",
            f"  Skill Score:             {self.brier_skill_score:.4f}",
            "  (1.0 = perfect calibration, 0.0 = no skill, negative = worse than baseline)",
            "",
            "Calibration Error",
            f"  Expected (ECE):          {self.expected_calibration_error:.4f}",
            f"  Maximum (MCE):           {self.max_calibration_error:.4f}",
            "",
            "Reliability by Confidence Bin",
        ]
        
        for bin_label, accuracy in sorted(self.bin_accuracies.items()):
            lines.append(f"  {bin_label:20s}  {accuracy:>8.1%}")
        
        lines.extend([
            "",
            "=" * 70,
        ])
        
        return "\n".join(lines)


def calculate_brier_score(
    sessions: List[Session],
    ground_truth: Any,
    use_enhanced: bool = False
) -> CalibrationMetrics:
    """
    Calculate Brier score for confidence calibration.
    
    Brier Score = mean((predicted_probability - actual_outcome)^2)
    - 0 = perfect calibration
    - 0.25 = random guessing at 50%
    - 1.0 = always wrong with 100% confidence
    
    Parameters
    ----------
    sessions : List[Session]
        Sessions with person assignments and confidences
    ground_truth : GroundTruth or EnhancedGroundTruth
        Ground truth data
    use_enhanced : bool
        Whether using enhanced ground truth
    
    Returns
    -------
    CalibrationMetrics
        Calibration metrics including Brier score
    """
    metrics = CalibrationMetrics()
    
    # Collect predictions and outcomes
    predictions = []  # List of (predicted_prob, actual_outcome)
    
    for session in sessions:
        if not session.assigned_person_id or session.assignment_confidence is None:
            continue
        
        # Get ground truth
        if use_enhanced and hasattr(ground_truth, 'get_persons_for_session'):
            true_persons = ground_truth.get_persons_for_session(session.session_id)
            is_correct = session.assigned_person_id in true_persons if true_persons else False
        else:
            true_person = ground_truth.get_person_for_session(session.session_id)
            is_correct = (session.assigned_person_id == true_person)
        
        actual = 1.0 if is_correct else 0.0
        predicted = session.assignment_confidence
        
        predictions.append((predicted, actual))
    
    if not predictions:
        return metrics
    
    # Calculate Brier score
    brier = sum((p - a) ** 2 for p, a in predictions) / len(predictions)
    metrics.brier_score = brier
    
    # Calculate Brier skill score (vs baseline of always predicting base rate)
    base_rate = sum(a for _, a in predictions) / len(predictions)
    baseline_brier = sum((base_rate - a) ** 2 for _, a in predictions) / len(predictions)
    
    if baseline_brier > 0:
        skill_score = 1 - (brier / baseline_brier)
    else:
        skill_score = 0.0
    
    metrics.brier_skill_score = skill_score
    
    # Build reliability diagram (bin by predicted probability)
    bins = defaultdict(lambda: {'predicted_sum': 0, 'actual_sum': 0, 'count': 0})
    
    for predicted, actual in predictions:
        # Bin by predicted probability (0.1 increments)
        bin_key = round(predicted, 1)
        bins[bin_key]['predicted_sum'] += predicted
        bins[bin_key]['actual_sum'] += actual
        bins[bin_key]['count'] += 1
    
    # Calculate calibration metrics
    calibration_errors = []
    
    for bin_center, data in sorted(bins.items()):
        if data['count'] > 0:
            avg_predicted = data['predicted_sum'] / data['count']
            avg_actual = data['actual_sum'] / data['count']
            
            error = abs(avg_predicted - avg_actual)
            calibration_errors.append((error, data['count']))
            
            # Store for reliability diagram
            metrics.reliability_bins[bin_center] = {
                'predicted': avg_predicted,
                'actual': avg_actual,
                'count': data['count'],
            }
            
            # Label for summary
            bin_label = f"{bin_center*100:.0f}% confidence"
            metrics.bin_accuracies[bin_label] = avg_actual
    
    # Expected Calibration Error (weighted average of bin errors)
    total_samples = sum(count for _, count in calibration_errors)
    if total_samples > 0:
        ece = sum(error * count for error, count in calibration_errors) / total_samples
        metrics.expected_calibration_error = ece
    
    # Maximum Calibration Error
    if calibration_errors:
        metrics.max_calibration_error = max(error for error, _ in calibration_errors)
    
    return metrics


def calibrate_probabilities(
    sessions: List[Session],
    method: str = "platt_scaling"
) -> List[float]:
    """
    Calibrate probability estimates using Platt scaling or isotonic regression.
    
    Parameters
    ----------
    sessions : List[Session]
        Sessions with raw confidence scores
    method : str
        Calibration method ('platt_scaling', 'isotonic', 'temperature')
    
    Returns
    -------
    List[float]
        Calibrated probabilities
    """
    # Extract confidence scores
    confidences = []
    for session in sessions:
        if session.assignment_confidence is not None:
            confidences.append(session.assignment_confidence)
    
    if not confidences:
        return []
    
    if method == "temperature":
        # Temperature scaling: p' = sigmoid(logit(p) / T)
        # Simple heuristic: T = mean(confidence) / 0.5
        mean_conf = sum(confidences) / len(confidences)
        temperature = mean_conf / 0.5 if mean_conf > 0 else 1.0
        
        calibrated = []
        for conf in confidences:
            # Convert to logit, scale, convert back
            if conf <= 0:
                logit = -10
            elif conf >= 1:
                logit = 10
            else:
                logit = math.log(conf / (1 - conf))
            
            scaled_logit = logit / temperature
            calibrated_conf = 1 / (1 + math.exp(-scaled_logit))
            calibrated.append(calibrated_conf)
        
        return calibrated
    
    elif method == "platt_scaling":
        # Platt scaling: learn A, B such that p' = sigmoid(A * logit(p) + B)
        # For now, use heuristic calibration
        return _heuristic_platt_scaling(confidences)
    
    else:
        # No calibration
        return confidences


def _heuristic_platt_scaling(confidences: List[float]) -> List[float]:
    """Apply heuristic Platt scaling."""
    # Heuristic: if average confidence > average accuracy, scale down
    # If average confidence < average accuracy, scale up
    
    mean_conf = sum(confidences) / len(confidences)
    
    # Assume we want to map mean_conf -> 0.7 (typical accuracy)
    target = 0.7
    
    if mean_conf >= target:
        # Scale down
        scale = target / mean_conf
        return [min(1.0, c * scale) for c in confidences]
    else:
        # Scale up (more complex - use sigmoid transformation)
        calibrated = []
        for c in confidences:
            if c < 0.5:
                # Boost low confidences less
                calibrated.append(c * 1.1)
            else:
                # Boost high confidences more
                calibrated.append(min(1.0, c * 1.2))
        return calibrated


def evaluate_calibration_quality(
    sessions: List[Session],
    ground_truth: Any,
    use_enhanced: bool = False
) -> Dict[str, Any]:
    """
    Comprehensive calibration evaluation with recommendations.
    
    Returns detailed analysis of calibration quality.
    """
    metrics = calculate_brier_score(sessions, ground_truth, use_enhanced)
    
    quality = {
        "brier_score": metrics.brier_score,
        "skill_score": metrics.brier_skill_score,
        "ece": metrics.expected_calibration_error,
        "mce": metrics.max_calibration_error,
        "is_well_calibrated": metrics.brier_skill_score > 0.5,
        "confidence": "",
        "recommendations": [],
    }
    
    # Determine calibration quality
    if metrics.brier_skill_score > 0.7:
        quality["confidence"] = "Excellent"
        quality["recommendations"].append("Model is well-calibrated. Continue current approach.")
    elif metrics.brier_skill_score > 0.5:
        quality["confidence"] = "Good"
        quality["recommendations"].append("Calibration is acceptable. Minor improvements possible.")
    elif metrics.brier_skill_score > 0.2:
        quality["confidence"] = "Fair"
        quality["recommendations"].append("Consider implementing Platt scaling or temperature scaling.")
    else:
        quality["confidence"] = "Poor"
        quality["recommendations"].extend([
            "Calibration needs significant improvement.",
            "Consider retraining with calibration-aware loss function.",
            "Implement post-hoc calibration (Platt scaling or isotonic regression).",
        ])
    
    # Check for overconfidence
    if metrics.expected_calibration_error > 0.1:
        quality["recommendations"].append("Model shows signs of overconfidence. Consider temperature scaling.")
    
    return quality


# Convenience functions

def is_well_calibrated(
    sessions: List[Session],
    ground_truth: Any,
    threshold: float = 0.5,
    use_enhanced: bool = False
) -> bool:
    """Quick check if model is well-calibrated."""
    metrics = calculate_brier_score(sessions, ground_truth, use_enhanced)
    return metrics.brier_skill_score >= threshold


def get_calibration_report(
    sessions: List[Session],
    ground_truth: Any,
    use_enhanced: bool = False
) -> str:
    """Get full calibration report as string."""
    metrics = calculate_brier_score(sessions, ground_truth, use_enhanced)
    quality = evaluate_calibration_quality(sessions, ground_truth, use_enhanced)
    
    report = metrics.get_summary()
    report += "\n\n"
    report += f"Calibration Quality: {quality['confidence']}\n"
    report += "\nRecommendations:\n"
    for rec in quality['recommendations']:
        report += f"  • {rec}\n"
    
    return report
