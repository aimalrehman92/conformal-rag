import json
import csv
import numpy as np
from math import ceil
from collections import defaultdict

CORRECT_ANNOTATIONS = ["Y", "S"]

NO_FAILURE_R_SCORE = -100000.0


def load_subclaim_data(file_path):
    """Load calibration data from a JSON file"""
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def append_result_to_csv(csv_filename, label, y, yerr):
    """Append calibration results to CSV file"""
    formatted_results = [f"{y:.4f} ± {yerr:.4f}" for y, yerr in zip(y, yerr)]
    formatted_results.reverse()
    row = [label] + formatted_results
    with open(csv_filename, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(row)


def _get_accepted_subclaims(entry, threshold, confidence_method):
    """Helper function to get accepted subclaims based on threshold"""
    return [
        subclaim
        for subclaim in entry["subclaims"]
        if subclaim["scores"][confidence_method] + subclaim["scores"]["noise"]
        >= threshold
    ]


def _calculate_entailed_fraction(subclaims):
    """Helper function to calculate fraction of entailed/correct subclaims"""
    if not subclaims:
        return 1.0
    return np.mean(
        [
            subclaim["annotations"]["gpt"] in CORRECT_ANNOTATIONS
            for subclaim in subclaims
        ]
    )


def get_r_score(entry: dict, confidence_method: str, a: float):
    """
    Compute the critical r_a score for one data entry.

    Subclaims are progressively added from highest to lowest confidence.
    The returned score is the first threshold at which the retained
    subclaims fail to achieve the required factuality level ``a``.

    If factuality never falls below ``a``, a sentinel value smaller than
    every implemented confidence score is returned.

    Results are cached in the entry so repeated calibration runs reuse
    exactly the same value.
    """
    r_score_key = f"r_score_{a}_{confidence_method}"

    if r_score_key in entry:
        return entry[r_score_key]

    scores = [
        subclaim["scores"][confidence_method] + subclaim["scores"]["noise"]
        for subclaim in entry["subclaims"]
    ]
    threshold_set = sorted(scores, reverse=True)

    for threshold in threshold_set:
        accepted_subclaims = _get_accepted_subclaims(
            entry, threshold, confidence_method
        )
        entailed_fraction = _calculate_entailed_fraction(accepted_subclaims)

        if entailed_fraction < a:
            entry[r_score_key] = threshold
            return threshold

    entry[r_score_key] = NO_FAILURE_R_SCORE
    return NO_FAILURE_R_SCORE


def compute_threshold(alpha, calibration_data, a, confidence_method):
    """
    Compute the finite-sample split-conformal threshold.

    When the requested quantile corresponds to the (n + 1)-th order
    statistic, return +infinity. This is the conservative conformal
    solution when the calibration sample is too small for the requested
    error level.
    """
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    if not calibration_data:
        raise ValueError("calibration_data must contain at least one entry")

    r_scores = [
        get_r_score(entry, confidence_method, a)
        for entry in calibration_data
    ]

    n = len(r_scores)
    quantile_target_index = ceil((n + 1) * (1 - alpha))

    if quantile_target_index > n:
        return np.inf

    return sorted(r_scores)[quantile_target_index - 1]

    
# Make sure the split calibrate_range ratio are all same not just in overall level but in group level
# not return data in list but in a map with each group name as key
def split_group(data, calibrate_range=0.5):
    group_data = defaultdict(list)
    calibration_data = defaultdict(list)
    test_data = []

    for entry in data:
        group = entry["groups"][0]  # Use first group as default
        group_data[group].append(entry)

    for group, group_entries in group_data.items():
        split_index = ceil(len(group_entries) * calibrate_range)
        calibration_data[group].extend(group_entries[:split_index])
        test_data.extend(group_entries[split_index:])

    return calibration_data, test_data

# Analyze Functions #

def percentage_highest_not_S(data, key="relavance"):
    count_total = 0
    count_not_S = 0

    for item in data:
        subclaims = item.get("subclaims", [])
        if not subclaims:
            continue

        # Sort subclaims by (score[key] + score[noise]), descending
        subclaims_sorted = sorted(
            subclaims,
            key=lambda sc: sc["scores"].get(key, 0) + sc["scores"].get("noise", 0),
            reverse=True
        )

        top_annotation = subclaims_sorted[0].get("annotations", {}).get("gpt", None)

        count_total += 1
        if top_annotation != "S":
            count_not_S += 1

    if count_total == 0:
        return 0.0  # Avoid division by zero

    return (count_not_S / count_total) * 100