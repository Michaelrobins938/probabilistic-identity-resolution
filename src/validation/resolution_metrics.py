"""
Resolution Metrics

Measures accuracy of identity resolution against ground truth.

Key metrics:
- Household size accuracy: Did we estimate the right number of people?
- Person assignment accuracy: Did we assign sessions to the correct person?
- Cross-device linking F1: Did we correctly link devices to persons?
- Confidence calibration: Do our probability estimates match reality?
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
import math

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from models.streaming_event import Session
from models.household_profile import HouseholdProfile
from validation.synthetic_households import GroundTruth


@dataclass
class ResolutionMetrics:
    """Comprehensive metrics for identity resolution evaluation."""

    # Household size estimation
    household_size_mae: float = 0.0          # Mean absolute error
    household_size_accuracy: float = 0.0      # Exact match rate

    # Person assignment (session -> person)
    person_assignment_accuracy: float = 0.0   # % sessions assigned correctly
    person_assignment_f1: float = 0.0         # F1 score for person assignment

    # Cross-device linking
    device_linking_precision: float = 0.0     # Correct links / predicted links
    device_linking_recall: float = 0.0        # Correct links / actual links
    device_linking_f1: float = 0.0

    # Confidence calibration
    confidence_calibration: float = 0.0       # How well probabilities match reality
    confidence_mae: float = 0.0               # Mean absolute error of confidence

    # Per-persona metrics
    persona_accuracy: Dict[str, float] = field(default_factory=dict)

    # Overall
    overall_score: float = 0.0

    def get_summary(self) -> str:
        """Get human-readable summary."""
        lines = [
            "=" * 60,
            "IDENTITY RESOLUTION METRICS",
            "=" * 60,
            "",
            "HOUSEHOLD SIZE ESTIMATION",
            f"  Mean Absolute Error:     {self.household_size_mae:.2f}",
            f"  Exact Match Accuracy:    {self.household_size_accuracy:.1%}",
            "",
            "PERSON ASSIGNMENT",
            f"  Accuracy:                {self.person_assignment_accuracy:.1%}",
            f"  F1 Score:                {self.person_assignment_f1:.3f}",
            "",
            "CROSS-DEVICE LINKING",
            f"  Precision:               {self.device_linking_precision:.1%}",
            f"  Recall:                  {self.device_linking_recall:.1%}",
            f"  F1 Score:                {self.device_linking_f1:.3f}",
            "",
            "CONFIDENCE CALIBRATION",
            f"  Calibration Score:       {self.confidence_calibration:.3f}",
            f"  Confidence MAE:          {self.confidence_mae:.3f}",
            "",
        ]

        if self.persona_accuracy:
            lines.append("PER-PERSONA ACCURACY")
            for persona, acc in sorted(self.persona_accuracy.items()):
                lines.append(f"  {persona:20s}  {acc:.1%}")
            lines.append("")

        lines.append(f"OVERALL SCORE:             {self.overall_score:.1%}")
        lines.append("=" * 60)

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "household_size_mae": self.household_size_mae,
            "household_size_accuracy": self.household_size_accuracy,
            "person_assignment_accuracy": self.person_assignment_accuracy,
            "person_assignment_f1": self.person_assignment_f1,
            "device_linking_precision": self.device_linking_precision,
            "device_linking_recall": self.device_linking_recall,
            "device_linking_f1": self.device_linking_f1,
            "confidence_calibration": self.confidence_calibration,
            "confidence_mae": self.confidence_mae,
            "persona_accuracy": self.persona_accuracy,
            "overall_score": self.overall_score,
        }


def evaluate_resolution(
    households: List[HouseholdProfile],
    sessions: List[Session],
    ground_truth: GroundTruth,
    predicted_device_links: Optional[List[Tuple[str, str]]] = None
) -> ResolutionMetrics:
    """
    Evaluate identity resolution against ground truth.

    Parameters
    ----------
    households : List[HouseholdProfile]
        Inferred households
    sessions : List[Session]
        Sessions with person assignments
    ground_truth : GroundTruth
        Known ground truth
    predicted_device_links : List[Tuple[str, str]], optional
        Predicted cross-device links

    Returns
    -------
    ResolutionMetrics
        Comprehensive evaluation metrics
    """
    metrics = ResolutionMetrics()

    # 1. Evaluate household size estimation
    metrics.household_size_mae, metrics.household_size_accuracy = \
        _evaluate_household_size(households, ground_truth)

    # 2. Evaluate person assignment
    metrics.person_assignment_accuracy, metrics.person_assignment_f1, metrics.persona_accuracy = \
        _evaluate_person_assignment(sessions, ground_truth)

    # 3. Evaluate cross-device linking
    if predicted_device_links:
        metrics.device_linking_precision, metrics.device_linking_recall, metrics.device_linking_f1 = \
            _evaluate_device_linking(predicted_device_links, ground_truth)

    # 4. Evaluate confidence calibration
    metrics.confidence_calibration, metrics.confidence_mae = \
        _evaluate_confidence_calibration(sessions, ground_truth)

    # Compute overall score (weighted average)
    weights = {
        "household_size": 0.15,
        "person_assignment": 0.35,
        "device_linking": 0.25,
        "calibration": 0.25,
    }

    overall = (
        weights["household_size"] * metrics.household_size_accuracy +
        weights["person_assignment"] * metrics.person_assignment_accuracy +
        weights["device_linking"] * metrics.device_linking_f1 +
        weights["calibration"] * metrics.confidence_calibration
    )
    metrics.overall_score = overall

    return metrics


def _evaluate_household_size(
    households: List[HouseholdProfile],
    ground_truth: GroundTruth
) -> Tuple[float, float]:
    """
    Evaluate household size estimation accuracy.

    Returns
    -------
    Tuple[float, float]
        (mean_absolute_error, exact_match_accuracy)
    """
    if not households:
        return 0.0, 0.0

    errors = []
    exact_matches = 0

    for household in households:
        # Find corresponding ground truth
        # Match by account_id -> household_id
        true_size = None
        for hh_id, members in ground_truth.household_members.items():
            # Check if this household matches
            if household.account_id.replace("account_", "household_") == hh_id:
                true_size = len(members)
                break

        if true_size is None:
            continue

        predicted_size = household.estimated_size
        error = abs(predicted_size - true_size)
        errors.append(error)

        if predicted_size == true_size:
            exact_matches += 1

    if not errors:
        return 0.0, 0.0

    mae = sum(errors) / len(errors)
    accuracy = exact_matches / len(errors)

    return mae, accuracy


def _evaluate_person_assignment(
    sessions: List[Session],
    ground_truth: GroundTruth
) -> Tuple[float, float, Dict[str, float]]:
    """
    Evaluate session-to-person assignment accuracy.

    Returns
    -------
    Tuple[float, float, Dict[str, float]]
        (accuracy, f1_score, per_persona_accuracy)
    """
    if not sessions:
        return 0.0, 0.0, {}

    correct = 0
    total = 0
    persona_correct: Dict[str, int] = {}
    persona_total: Dict[str, int] = {}

    # For F1 calculation
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for session in sessions:
        session_id = session.session_id

        # Get ground truth person
        true_person = None
        for sid, pid in ground_truth.session_to_person.items():
            if sid in session_id or session_id in sid:
                true_person = pid
                break

        if true_person is None:
            continue

        # Get predicted person
        predicted_person = session.assigned_person_id

        # Get persona for this person
        persona = ground_truth.person_personas.get(true_person, "unknown")

        # Update persona counts
        if persona not in persona_total:
            persona_total[persona] = 0
            persona_correct[persona] = 0
        persona_total[persona] += 1

        total += 1

        if predicted_person is None:
            false_negatives += 1
            continue

        # Check if correct (fuzzy match on person index within household)
        # Since we don't have direct ID match, compare cluster indices
        true_idx = _extract_person_index(true_person)
        pred_idx = _extract_person_index(predicted_person)

        if true_idx == pred_idx:
            correct += 1
            true_positives += 1
            persona_correct[persona] += 1
        else:
            false_positives += 1

    if total == 0:
        return 0.0, 0.0, {}

    accuracy = correct / total

    # F1 score
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # Per-persona accuracy
    persona_accuracy = {}
    for persona in persona_total:
        if persona_total[persona] > 0:
            persona_accuracy[persona] = persona_correct[persona] / persona_total[persona]

    return accuracy, f1, persona_accuracy


def _extract_person_index(person_id: str) -> int:
    """Extract person index from ID like 'household_0001_person_2'."""
    if person_id is None:
        return -1
    try:
        # Try to extract last number
        parts = person_id.split("_")
        for part in reversed(parts):
            if part.isdigit():
                return int(part)
        return hash(person_id) % 10  # Fallback
    except:
        return -1


def _evaluate_device_linking(
    predicted_links: List[Tuple[str, str]],
    ground_truth: GroundTruth
) -> Tuple[float, float, float]:
    """
    Evaluate cross-device linking accuracy.

    Returns
    -------
    Tuple[float, float, float]
        (precision, recall, f1)
    """
    # Build ground truth link set
    true_links = set()
    for person_id, devices in ground_truth.person_devices.items():
        # All pairs of devices for this person are true links
        for i in range(len(devices)):
            for j in range(i + 1, len(devices)):
                link = tuple(sorted([devices[i], devices[j]]))
                true_links.add(link)

    # Build predicted link set
    pred_links = set()
    for d1, d2 in predicted_links:
        link = tuple(sorted([d1, d2]))
        pred_links.add(link)

    if not pred_links and not true_links:
        return 1.0, 1.0, 1.0

    if not pred_links:
        return 0.0, 0.0, 0.0

    if not true_links:
        return 0.0, 1.0, 0.0

    # Compute metrics
    true_positives = len(pred_links & true_links)
    false_positives = len(pred_links - true_links)
    false_negatives = len(true_links - pred_links)

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1


def _evaluate_confidence_calibration(
    sessions: List[Session],
    ground_truth: GroundTruth
) -> Tuple[float, float]:
    """
    Evaluate confidence calibration.

    Good calibration: When we say 80% confident, we should be right ~80% of the time.

    Returns
    -------
    Tuple[float, float]
        (calibration_score, mae)
    """
    # Bin confidences
    bins = {i / 10: {"correct": 0, "total": 0} for i in range(11)}

    confidence_errors = []

    for session in sessions:
        if session.assignment_confidence == 0:
            continue

        confidence = session.assignment_confidence

        # Find bin
        bin_key = round(confidence, 1)
        if bin_key not in bins:
            bin_key = min(bins.keys(), key=lambda x: abs(x - confidence))

        # Check if assignment was correct
        session_id = session.session_id
        true_person = None
        for sid, pid in ground_truth.session_to_person.items():
            if sid in session_id or session_id in sid:
                true_person = pid
                break

        if true_person is None:
            continue

        true_idx = _extract_person_index(true_person)
        pred_idx = _extract_person_index(session.assigned_person_id)
        is_correct = (true_idx == pred_idx)

        bins[bin_key]["total"] += 1
        if is_correct:
            bins[bin_key]["correct"] += 1

        # Track error
        actual = 1.0 if is_correct else 0.0
        confidence_errors.append(abs(confidence - actual))

    # Compute calibration score
    calibration_errors = []
    for bin_conf, data in bins.items():
        if data["total"] > 0:
            actual_accuracy = data["correct"] / data["total"]
            error = abs(bin_conf - actual_accuracy)
            calibration_errors.append(error)

    if not calibration_errors:
        return 0.5, 0.5

    # Calibration score: 1 - average calibration error
    calibration_score = 1.0 - (sum(calibration_errors) / len(calibration_errors))
    calibration_score = max(0.0, min(1.0, calibration_score))

    # MAE
    mae = sum(confidence_errors) / len(confidence_errors) if confidence_errors else 0.5

    return calibration_score, mae


def compare_to_baseline(
    our_metrics: ResolutionMetrics,
    baseline_name: str = "Random Assignment"
) -> str:
    """
    Compare our metrics to baseline performance.

    Returns formatted comparison string.
    """
    # Baseline expectations
    baselines = {
        "Random Assignment": {
            "household_size_accuracy": 0.25,  # Random guess 1-4
            "person_assignment_accuracy": 0.33,  # Random among ~3 people
            "device_linking_f1": 0.10,
            "confidence_calibration": 0.50,
        },
        "Single Person": {
            "household_size_accuracy": 0.30,  # Many households are single
            "person_assignment_accuracy": 0.50,  # Trivially correct for singles
            "device_linking_f1": 0.00,
            "confidence_calibration": 0.50,
        },
    }

    baseline = baselines.get(baseline_name, baselines["Random Assignment"])

    lines = [
        "",
        f"COMPARISON VS BASELINE ({baseline_name})",
        "-" * 50,
        "",
        f"{'Metric':<30} {'Ours':>10} {'Baseline':>10} {'Lift':>10}",
        "-" * 50,
    ]

    metrics_to_compare = [
        ("Household Size Acc.", our_metrics.household_size_accuracy, baseline["household_size_accuracy"]),
        ("Person Assignment Acc.", our_metrics.person_assignment_accuracy, baseline["person_assignment_accuracy"]),
        ("Device Linking F1", our_metrics.device_linking_f1, baseline["device_linking_f1"]),
        ("Confidence Calibration", our_metrics.confidence_calibration, baseline["confidence_calibration"]),
    ]

    for name, ours, base in metrics_to_compare:
        if base > 0:
            lift = (ours - base) / base
            lift_str = f"{lift:+.0%}"
        else:
            lift_str = "N/A"

        lines.append(f"{name:<30} {ours:>9.1%} {base:>9.1%} {lift_str:>10}")

    lines.append("-" * 50)

    return "\n".join(lines)
