"""
Utilities for the authors' original conformal implementation.

This module intentionally preserves behavior from the public repository,
including known edge cases and inconsistencies. Do not apply correctness
fixes here; corrected behavior belongs in src.calibration.utils.
"""

import csv
from math import ceil
from collections import defaultdict

import numpy as np

CORRECT_ANNOTATIONS = ["Y", "S"]


def append_result_to_csv(csv_filename, label, y, yerr):
    """Append calibration results to CSV file."""
    formatted_results = [
        f"{y_value:.4f} ± {yerr_value:.4f}" for y_value, yerr_value in zip(y, yerr)
    ]
    formatted_results.reverse()
    row = [label] + formatted_results

    with open(csv_filename, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(row)


def _get_accepted_subclaims(entry, threshold, confidence_method):
    """Get accepted subclaims using the authors' threshold rule."""
    return [
        subclaim
        for subclaim in entry["subclaims"]
        if subclaim["scores"][confidence_method] + subclaim["scores"]["noise"]
        >= threshold
    ]


def _calculate_entailed_fraction(subclaims):
    """Calculate the fraction of accepted subclaims marked correct."""
    if not subclaims:
        return 1.0

    return np.mean(
        [
            subclaim["annotations"]["gpt"] in CORRECT_ANNOTATIONS
            for subclaim in subclaims
        ]
    )


def get_r_score(entry: list, confidence_method: str, a: float):
    """
    Preserve the authors' original r-score behavior.

    In particular, when factuality never falls below ``a``, the public
    implementation caches -1 but returns -100000 on the first call.
    Subsequent calls therefore return the cached -1.
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
            entry,
            threshold,
            confidence_method,
        )
        entailed_fraction = _calculate_entailed_fraction(accepted_subclaims)

        if entailed_fraction < a:
            entry[r_score_key] = threshold
            return threshold

    # Intentional reproduction of the public implementation.
    entry[r_score_key] = -1
    return -100000


def compute_threshold(alpha, calibration_data, a, confidence_method):
    """
    Preserve the authors' original conformal threshold computation.

    There is intentionally no guard for an unavailable (n + 1)-th order
    statistic. In that situation the public implementation raises
    IndexError while indexing the sorted calibration scores.
    """
    r_scores = [get_r_score(entry, confidence_method, a) for entry in calibration_data]

    quantile_target_index = ceil((len(r_scores) + 1) * (1 - alpha))

    return sorted(r_scores)[quantile_target_index - 1]


def split_group(data, calibrate_range=0.5):
    """
    Preserve the authors' original group split helper.

    The caller is responsible for shuffling ``data`` before this function
    is called.
    """
    group_data = defaultdict(list)
    calibration_data = defaultdict(list)
    test_data = []

    for entry in data:
        group = entry["groups"][0]
        group_data[group].append(entry)

    for group, group_entries in group_data.items():
        split_index = ceil(len(group_entries) * calibrate_range)
        calibration_data[group].extend(group_entries[:split_index])
        test_data.extend(group_entries[split_index:])

    return calibration_data, test_data
