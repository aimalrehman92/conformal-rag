"""
Authors' original split conformal implementation.

This module intentionally preserves the behavior of the public repository,
including global RNG usage, in-place data shuffling, and dependence on the
authors' original threshold utilities.
"""

import csv
import os
import random

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from src.calibration.author_utils import append_result_to_csv
from src.calibration.author_utils import compute_threshold
from src.calibration.base_calibration import ICalibration

CORRECT_ANNOTATIONS = ["S"]


class AuthorSplitConformalCalibration(ICalibration):
    """
    Faithful preservation of the public repository's split conformal behavior.
    """

    def __init__(self, dataset_name: str, runs: int = 1000):
        self.dataset_name = dataset_name
        self.confidence_method = [
            "relavance",
            "frequency",
            "query_claim_cosine_similarity",
            "doc_claim_cosine_similarity",
            "min_log_prob",
            "random",
            "ordinal",
        ]
        self.runs = runs

    def plot_conformal_removal(
        self,
        data,
        alphas,
        a,
        fig_filename,
        csv_filename,
    ):
        cache_filename = (
            f"{os.path.splitext(os.path.abspath(csv_filename))[0]}"
            "_conformal_removal_cache.npy"
        )

        if not os.path.exists(cache_filename):
            results = self.compute_conformal_results(
                data,
                alphas,
                a,
            )
            print(f"Caching results to {cache_filename}")
            np.save(cache_filename, results)
        else:
            print(f"Loading cached results from {cache_filename}")
            results = np.load(
                cache_filename,
                allow_pickle=True,
            ).item()

        ax = None

        for confidence_method, result in results.items():
            correctness, fraction_removed, yerr = (
                self.process_conformal_removal_results(result)
            )

            self._write_csv_header(csv_filename, alphas)

            append_result_to_csv(
                csv_filename=csv_filename,
                label=f"{confidence_method}_conformal_removal_rate",
                y=fraction_removed,
                yerr=yerr,
            )

            print(f"Producing conformal plot for {confidence_method}")

            ax = self.plot_conformal_removal_rate_by_alpha(
                correctness,
                fraction_removed,
                yerr,
                a,
                confidence_method,
                fig_filename,
                ax,
            )

            print(f"Conformal plot saved to {fig_filename}")

    def compute_conformal_results(
        self,
        data: list,
        alphas: np.ndarray,
        a: float,
    ):
        results = {}

        for confidence_method in self.confidence_method:
            results[confidence_method] = {}

            for alpha in tqdm(
                alphas,
                desc=("Computing conformal results for " f"{confidence_method}"),
            ):
                thresholds = []
                correctness_list = []
                fraction_removed_list = []

                for _ in range(self.runs):
                    # Intentional author behavior:
                    # mutate the shared input list using global RNG state.
                    random.shuffle(data)

                    split_index = len(data) // 2
                    calibration_data = data[:split_index]
                    test_data = data[split_index:]

                    assert (
                        len(calibration_data) != 0
                    ), "Calibration data should not be empty"

                    assert len(test_data) != 0, "Test data should not be empty"

                    threshold = compute_threshold(
                        alpha,
                        calibration_data,
                        a,
                        confidence_method,
                    )

                    correctness, fraction_removed = (
                        self._evaluate_conformal_correctness(
                            test_data,
                            threshold,
                            a,
                            confidence_method,
                        )
                    )

                    thresholds.append(threshold)
                    correctness_list.append(correctness)
                    fraction_removed_list.append(fraction_removed)

                results[confidence_method][alpha] = {
                    "threshold": thresholds,
                    "correctness": correctness_list,
                    "fraction_removed": fraction_removed_list,
                }

        return results

    def process_conformal_removal_results(self, results: dict):
        x = []
        y = []
        yerr = []

        for _, results_for_alpha in results.items():
            x_per_alpha = np.mean(results_for_alpha["correctness"])
            y_per_alpha = np.mean(results_for_alpha["fraction_removed"])

            x.append(x_per_alpha)
            y.append(y_per_alpha)
            yerr.append(
                np.std(results_for_alpha["fraction_removed"])
                * 1.96
                / np.sqrt(len(results_for_alpha["fraction_removed"]))
            )

        return x, y, yerr

    def plot_conformal_removal_rate_by_alpha(
        self,
        x,
        y,
        yerr,
        a,
        confidence_method,
        fig_filename,
        ax=None,
    ):
        if not ax:
            fig, ax = plt.subplots(
                figsize=(8, 6),
                dpi=800,
            )

            ax.set_title(
                ("Conformal Plots for " f"{self.dataset_name} " f"Datasets (a={a})"),
                fontsize=20,
            )

            x_label = (
                f"Fraction achieving avg factuality >= {a}"
                if a != 1
                else "Fraction of factual outputs"
            )

            ax.set_xlabel(
                x_label,
                fontsize=16,
            )
            ax.set_ylabel(
                "Average percent removed",
                fontsize=16,
            )
        else:
            fig = ax.figure

        ax.errorbar(
            x,
            y,
            yerr=yerr,
            label=confidence_method,
            linewidth=2,
        )

        ax.legend(
            loc="upper left",
            bbox_to_anchor=(0.02, 0.98),
            fontsize=10,
        )

        fig.savefig(
            fig_filename,
            bbox_inches="tight",
        )

        return ax

    def _write_csv_header(
        self,
        csv_filename,
        alphas,
    ):
        target_factuality = [f"{(1 - x):.2f}" for x in alphas][::-1]

        header = ["target_factuality"] + target_factuality

        os.makedirs(
            os.path.dirname(csv_filename),
            exist_ok=True,
        )

        if not os.path.exists(csv_filename):
            with open(
                csv_filename,
                mode="w",
                newline="",
            ) as file:
                csv.writer(file).writerow(header)

    def _evaluate_conformal_correctness(
        self,
        data: list,
        threshold: float,
        a: float,
        confidence_method: str,
    ):
        correctly_retained = []
        fraction_removed = []

        for entry in data:
            removal_count = 0
            retained_cnt = 0
            correctly_retained_count = 0

            for subclaim in entry["subclaims"]:
                score = subclaim["scores"][confidence_method]
                noise = subclaim["scores"]["noise"]

                if score + noise >= threshold:
                    retained_cnt += 1

                    if (
                        subclaim.get(
                            "annotations",
                            {},
                        ).get("gpt", "")
                        in CORRECT_ANNOTATIONS
                    ):
                        correctly_retained_count += 1
                else:
                    removal_count += 1

            total_subclaims = len(entry["subclaims"])

            entry_removal_rate = (
                0 if total_subclaims == 0 else removal_count / total_subclaims
            )

            fraction_removed.append(entry_removal_rate)

            correctly_retained_percentage = (
                correctly_retained_count / retained_cnt if retained_cnt > 0 else 1
            )

            correctly_retained.append(correctly_retained_percentage >= a)

        return (
            np.mean(correctly_retained),
            np.mean(fraction_removed),
        )

    def plot_factual_removal(
        self,
        data,
        alphas,
        a,
        fig_filename,
        csv_filename,
        plot_group_results=False,
    ):
        x_values = np.linspace(
            1 - alphas[-1] - 0.05,
            1 - alphas[0] + 0.03,
            100,
        )

        fig, ax = plt.subplots(
            figsize=(8, 6),
            dpi=800,
        )

        ax.plot(
            x_values,
            x_values,
            "--",
            color="gray",
            linewidth=2,
            label="Conformal guarantee lower bounds",
        )

        cache_filename = (
            f"{os.path.splitext(os.path.abspath(csv_filename))[0]}"
            "_factual_correctness_cache.npy"
        )

        if not os.path.exists(cache_filename):
            results = self.compute_factual_results(
                data,
                alphas,
                a,
            )

            print(f"Caching results to {cache_filename}")

            np.save(
                cache_filename,
                results,
            )
        else:
            print(f"Loading cached results from {cache_filename}")

            results = np.load(
                cache_filename,
                allow_pickle=True,
            ).item()

        for confidence_method, result in results.items():
            conf_level, correctness, yerr = self.process_factual_correctness_results(
                result
            )

            append_result_to_csv(
                csv_filename=csv_filename,
                label=(f"{confidence_method}" "_factual_correctness"),
                y=correctness,
                yerr=yerr,
            )

            print(
                "Producing factual removal plot for "
                f"{confidence_method}: "
                f"{fig_filename}"
            )

            ax = self.plot_factual_removal_rate_by_alpha(
                conf_level,
                correctness,
                a,
                confidence_method,
                fig_filename,
                ax,
            )

            print(f"Conformal plot saved to {fig_filename}")

            if plot_group_results:
                raise NotImplementedError("Not implemented")

    def compute_factual_results(
        self,
        data,
        alphas,
        a,
    ):
        results = {}

        for confidence_method in self.confidence_method:
            results[confidence_method] = {}

            for alpha in tqdm(
                alphas,
                desc=("Computing factual results for " f"{confidence_method}"),
            ):
                thresholds = []
                correctness = []

                for _ in range(self.runs):
                    # Intentional author behavior.
                    random.shuffle(data)

                    split_index = len(data) // 2
                    calibration_data = data[:split_index]
                    test_data = data[split_index:]

                    assert (
                        len(calibration_data) != 0
                    ), "Calibration data should not be empty"

                    assert len(test_data) != 0, "Test data should not be empty"

                    threshold = compute_threshold(
                        alpha,
                        calibration_data,
                        a,
                        confidence_method,
                    )

                    fraction_correct = self._evaluate_factual_correctness(
                        test_data,
                        threshold,
                        a,
                        confidence_method,
                    )

                    thresholds.append(threshold)
                    correctness.append(fraction_correct)

                results[confidence_method][alpha] = {
                    "threshold": thresholds,
                    "correctness": correctness,
                    "factuality": 1 - alpha,
                }

        return results

    def process_factual_correctness_results(
        self,
        results: dict,
    ):
        x = []
        y = []
        yerr = []

        for alpha, results_for_alpha in results.items():
            x.append(1 - alpha)

            y.append(np.mean(results_for_alpha["correctness"]))

            yerr.append(
                np.std(results_for_alpha["correctness"])
                * 1.96
                / np.sqrt(len(results_for_alpha["correctness"]))
            )

        return x, y, yerr

    def plot_factual_removal_rate_by_alpha(
        self,
        x,
        y,
        a,
        confidence_method,
        fig_filename,
        ax=None,
    ):
        if not ax:
            fig, ax = plt.subplots(
                figsize=(8, 6),
                dpi=800,
            )
        else:
            fig = ax.figure

        ax.set_xlabel(
            f"Target factuality (1 - {chr(945)})",
            fontsize=16,
        )

        ax.set_ylabel(
            "Empirical factuality",
            fontsize=16,
        )

        ax.set_title(
            ("Factual correctness for " f"{self.dataset_name} " f"Datasets (a={a})"),
            fontsize=20,
        )

        ax.plot(
            x,
            y,
            label=confidence_method,
            linewidth=2,
        )

        ax.legend(
            loc="upper left",
            bbox_to_anchor=(0.02, 0.98),
            fontsize=10,
        )

        fig.savefig(
            fig_filename,
            bbox_inches="tight",
            dpi=800,
        )

        return ax
