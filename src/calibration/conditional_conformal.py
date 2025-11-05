import os
import csv
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict


from src.calibration.conformal import SplitConformalCalibration
from src.calibration.utils import compute_threshold
from src.calibration.utils import append_result_to_csv
from src.calibration.utils import split_group


CORRECT_ANNOTATIONS = ["S"]


class GroupConditionalConformal(SplitConformalCalibration):
    def __init__(self, dataset_name: str, result_dir: str, runs: int = 1000):
        super().__init__(dataset_name, runs)
        self.result_dir = result_dir


    def compute_conformal_results(
        self, data: list, alphas: np.ndarray, a: float, plot_group_results: bool = False
    ):

        results = {}
        for confidence_method in self.confidence_method:
            results[confidence_method] = {}
            for alpha in tqdm(
                alphas, desc=f"Computing conformal results for {confidence_method}"
            ):
                # TODO add grouping
                groups = None
                

                thresholds_result = []
                correctness_list = []
                fraction_removed_list = []
                test_data = []
                for _ in range(self.runs):
                    random.shuffle(data)
                    calibration_data, test_data = split_group(data)
                    groups = list(calibration_data.keys())

                    assert (
                        len(calibration_data) != 0
                    ), "Calibration data should not be empty"
                    assert len(test_data) != 0, "Test data should not be empty"
                    thresholds = self._compute_threshold_by_group(
                        alpha, calibration_data, a, confidence_method, groups=groups
                    )

                    correctness, fraction_removed = (
                        self._evaluate_conformal_correctness(
                            test_data, thresholds, a, confidence_method
                        )
                    )
                    thresholds_result.append(thresholds)
                    correctness_list.append(correctness)
                    fraction_removed_list.append(fraction_removed)

                results[confidence_method][alpha] = {
                    "threshold": thresholds_result,
                    "correctness": correctness_list,
                    "fraction_removed": fraction_removed_list,
                }

        return results
    
    def _compute_threshold_by_group(
        self,
        alpha: float,
        calibration_data: list,
        a: float,
        confidence_method: str,
        groups: list | None = None,
    ):
        thresholds = {}
        for group in groups:
            group_data = calibration_data[group]
            thresholds[group] = compute_threshold(
                alpha, group_data, a, confidence_method
            )
        return thresholds
        
    def _evaluate_conformal_correctness(
        self, data: list, thresholds: dict, a: float, confidence_method: str
    ):
        """
        Evaluates the performance of a conformal prediction model on test data.
        Parameters:
        data (list): A list of dictionaries, where each dictionary represents an entry with subclaims.
        threshold (float): The similarity score threshold to determine if a subclaim is correctly retained.
        a (float): The threshold for the correctly retained percentage to consider an entry as correctly retained.
        Returns:
        tuple: A tuple containing two lists:
            - correctly_retained (float): Percentage of data that are correctly retained.
            - fraction_removed (float): Percentage of subclaims removed for each entry.
        """

        correctly_retained = []
        
        fraction_removed = []

        for entry in data:
            removal_count = 0
            retained_cnt = 0
            correctly_retained_count = 0
            threshold = thresholds[entry["groups"][0]] # Get threshold for the group
            if threshold is None:
                raise ValueError(
                    f"Threshold for group {entry['groups'][0]} is None. Check your calibration data."
                )
            for subclaim in entry["subclaims"]:
                # Find similarity score
                score = subclaim["scores"][confidence_method]
                noise = subclaim["scores"]["noise"]
                if score + noise >= threshold:
                    retained_cnt += 1
                    if (
                        subclaim.get("annotations", {}).get("gpt", "")
                        in CORRECT_ANNOTATIONS
                    ):
                        correctly_retained_count += 1

                else:
                    removal_count += 1

            total_subclaims = len(entry["subclaims"])

            # Calculate fraction of removed subclaims
            entry_removal_rate = (
                0 if total_subclaims == 0 else removal_count / total_subclaims
            )
            fraction_removed.append(
                entry_removal_rate
            )  # e.g. fraction_removed = [0.2, 0.5, 0.6, 0.2, 0.7] - one element per data entry

            # Calculate correctly retained rate
            correctly_retained_percentage = (
                correctly_retained_count / retained_cnt if retained_cnt > 0 else 1
            )
            correctly_retained.append(correctly_retained_percentage >= a)

        return np.mean(correctly_retained), np.mean(fraction_removed)
    
    def compute_factual_results(self, data, alphas, a, calibrate_range=0.5):
        overall_results = {}
        per_group_results  = defaultdict(lambda: defaultdict(dict))

        for method in self.confidence_method:
            overall_results[method] = {}
            for alpha in tqdm(
                alphas, desc=f"Computing factual results for {method}"
            ):
                # trackers for this (method, alpha)
                overall_correctness = []
                thresholds_per_group = defaultdict(list)
                correctness_per_group = defaultdict(list)

                for _ in range(self.runs):
                    random.shuffle(data)
                    calibration_data, test_data = split_group(data, calibrate_range)
                    groups = list(calibration_data.keys())

                    # assert on nonempty
                    assert calibration_data, "No calibration groups"
                    assert len(test_data) != 0, "Test data should not be empty"

                    # compute all thresholds at once
                    thresholds = self._compute_threshold_by_group(
                        alpha, calibration_data, a, method, groups=groups
                    )

                    fraction_correct = self._evaluate_factual_correctness(
                        test_data, thresholds, a, method
                    )
                    for group in groups:
                        thresholds_per_group[group].append(thresholds[group])
                        correctness_per_group[group].append(
                            fraction_correct[group]
                        )
                    overall_correctness.append(fraction_correct["overall"])

                # package overall
                overall_results[method][alpha] = {
                    "threshold": thresholds_per_group,
                    "correctness": overall_correctness,
                    "factuality": 1 - alpha,
                }

                # package per‐group
                for grp in thresholds_per_group:
                    per_group_results[grp][method][alpha] = {
                        "threshold": thresholds_per_group[grp],
                        "correctness": correctness_per_group[grp],
                        "factuality": 1 - alpha,
                    }

        # now write your CSVs
        for grp, grp_results in per_group_results.items():
            csv_name = os.path.join(self.result_dir, f"{self.dataset_name}_{grp}_factual_correctness.csv")
            self._write_csv_header(csv_name, alphas)
            for method, res in grp_results.items():
                lvl, corr, err = self.process_factual_correctness_results(res)
                append_result_to_csv(
                    csv_filename=csv_name,
                    label=f"{method}_factual_correctness",
                    y=corr,
                    yerr=err,
                )

        return overall_results
        

    def _evaluate_factual_correctness(
        self,
        data: list,
        thresholds: dict,
        a: float,
        confidence_method: str,
    ):
        """
        Evaluates the factual correctness of subclaims within the provided data,
        using a per‑group threshold but computing overall accuracy over all entries.

        Args:
            data (list): A list of dicts, each with "groups" and "subclaims".
            thresholds (dict): Mapping from group_name -> threshold float.
            a (float): The accuracy level to compare the correctly retained percentage against.
            confidence_method (str): Which score key to use for similarity.

        Returns:
            dict: { "overall": float, "<group1>": float, "<group2>": float, … }
        """
        per_group_percentages = defaultdict(list)
        total_pass = 0
        total_entries = 0

        for entry in data:
            group_name = entry["groups"][0]
            retained_cnt = 0
            correctly_retained_cnt = 0

            # count retained & correctly retained
            for sub in entry["subclaims"]:
                score = sub["scores"][confidence_method]
                noise = sub["scores"]["noise"]
                if score + noise >= thresholds[group_name]:
                    retained_cnt += 1
                    if sub.get("annotations", {}).get("gpt", "") in CORRECT_ANNOTATIONS:
                        correctly_retained_cnt += 1

            # pct for this entry (1.0 if nothing retained)
            pct = (correctly_retained_cnt / retained_cnt) if retained_cnt > 0 else 1.0
            per_group_percentages[group_name].append(pct)

            # update global pass/fail
            total_entries += 1
            if pct >= a:
                total_pass += 1

        # compute per‑group correctness
        per_group_correctness = {
            grp: sum(1 for pct in pct_list if pct >= a) / len(pct_list)
            for grp, pct_list in per_group_percentages.items()
        }

        # compute overall exactly as “fraction of all entries passing”
        overall = total_pass / total_entries if total_entries > 0 else 0.0

        # assemble result
        result = {"overall": overall}
        result.update(per_group_correctness)
        return result