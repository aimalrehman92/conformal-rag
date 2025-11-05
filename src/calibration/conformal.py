import os
import csv
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.calibration.base_calibration import ICalibration
from src.calibration.utils import compute_threshold
from src.calibration.utils import append_result_to_csv


CORRECT_ANNOTATIONS = ["S"]


class SplitConformalCalibration(ICalibration):
    """
    Implementation of standard conformal calibration.
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
        self, data, alphas, a, fig_filename, csv_filename
    ):

        # compute the correctness and fraction removed for each alpha

        cache_filename = f"{os.path.splitext(os.path.abspath(csv_filename))[0]}_conformal_removal_cache.npy"
        if not os.path.exists(cache_filename):
            results = self.compute_conformal_results(
                data, alphas, a
            )
            print(f"Caching results to {cache_filename}")
            np.save(cache_filename, results)

        else:
            print(f"Loading cached results from {cache_filename}")
            results = np.load(cache_filename, allow_pickle=True).item()

        ax = None
        for confidence_method, result in results.items():
            correctness, fraction_removed, yerr = (
                self.process_conformal_removal_results(result)
            )

            # write the results to csv file
            self._write_csv_header(csv_filename, alphas)
            append_result_to_csv(
                csv_filename=csv_filename,
                label=f"{confidence_method}_conformal_removal_rate",
                y=fraction_removed,
                yerr=yerr,
            )

            # plot the results
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
        self, data: list, alphas: np.ndarray, a: float
    ):

        results = {}
        for confidence_method in self.confidence_method:
            results[confidence_method] = {}
            for alpha in tqdm(
                alphas, desc=f"Computing conformal results for {confidence_method}"
            ):

                thresholds = []
                correctness_list = []
                fraction_removed_list = []
                for _ in range(self.runs):
                    random.shuffle(data)
                    split_index = len(data) // 2
                    calibration_data = data[:split_index]
                    test_data = data[split_index:]

                    assert (
                        len(calibration_data) != 0
                    ), "Calibration data should not be empty"
                    assert len(test_data) != 0, "Test data should not be empty"

                    threshold = compute_threshold(alpha, calibration_data, a, confidence_method)

                    correctness, fraction_removed = (
                        self._evaluate_conformal_correctness(
                            test_data, threshold, a, confidence_method
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
        """
        x: list of average correctness
        y: list of average fraction removed
        yerr: list of standard error of fraction removed
        """
        x, y, yerr = [], [], []
        for alpha, results_for_alpha in results.items():
            x_per_alpha = np.mean(
                results_for_alpha["correctness"]
            )  # correct retainment percentage at a specific alpha value, averaging over 1000 times of shuffled data
            y_per_alpha = np.mean(
                results_for_alpha["fraction_removed"]
            )  # removal percentage at a specific alpha value, averaging, averaging over 1000 times of shuffled data
            x.append(x_per_alpha)
            y.append(y_per_alpha)
            yerr.append(
                (
                    np.std(results_for_alpha["fraction_removed"])
                    * 1.96
                    / np.sqrt(len(results_for_alpha["fraction_removed"]))
                )
            )

        return x, y, yerr

    def plot_conformal_removal_rate_by_alpha(
        self, x, y, yerr, a, confidence_method, fig_filename, ax=None
    ):
        if not ax:
            fig, ax = plt.subplots(figsize=(8, 6), dpi=800)
            ax.set_title(
                f"Conformal Plots for {self.dataset_name} Datasets (a={a})", fontsize=20
            )
            x_label = (
                f"Fraction achieving avg factuality >= {a}"
                if a != 1
                else "Fraction of factual outputs"
            )
            ax.set_xlabel(x_label, fontsize=16)
            ax.set_ylabel("Average percent removed", fontsize=16)
        else:
            fig = ax.figure

        # Plot the data
        ax.errorbar(x, y, yerr=yerr, label=confidence_method, linewidth=2)

        # set the legend
        ax.legend(loc="upper left", bbox_to_anchor=(0.02, 0.98), fontsize=10)

        # Save the figure
        fig.savefig(fig_filename, bbox_inches="tight")

        return ax  # Return the ax for further modifications if needed

    def _write_csv_header(self, csv_filename, alphas):
        target_factuality = [f"{(1-x):.2f}" for x in alphas][::-1]
        header = ["target_factuality"] + target_factuality

        # Ensure the directory exists
        os.makedirs(os.path.dirname(csv_filename), exist_ok=True)

        if not os.path.exists(csv_filename):
            with open(csv_filename, mode="w", newline="") as file:
                csv.writer(file).writerow(header)

    def _evaluate_conformal_correctness(
        self, data: list, threshold: float, a: float, confidence_method: str
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

    def plot_factual_removal(
        self, data, alphas, a, fig_filename, csv_filename, plot_group_results=False
    ):
        x_values = np.linspace(1 - alphas[-1] - 0.05, 1 - alphas[0] + 0.03, 100)
        fig, ax = plt.subplots(figsize=(8, 6), dpi=800)
        ax.plot(
            x_values,
            x_values,
            "--",
            color="gray",
            linewidth=2,
            label="Conformal guarantee lower bounds",
        )

        cache_filename = f"{os.path.splitext(os.path.abspath(csv_filename))[0]}_factual_correctness_cache.npy"
        if not os.path.exists(cache_filename):
            results = self.compute_factual_results(data, alphas, a)
            print(f"Caching results to {cache_filename}")
            np.save(cache_filename, results)

        else:
            print(f"Loading cached results from {cache_filename}")
            results = np.load(cache_filename, allow_pickle=True).item()

        for confidence_method, result in results.items():
            conf_level, corretness, yerr = self.process_factual_correctness_results(
                result
            )

            append_result_to_csv(
                csv_filename=csv_filename,
                label=f"{confidence_method}_factual_correctness",
                y=corretness,
                yerr=yerr,
            )

            print(
                f"Producing factual removal plot for {confidence_method}: {fig_filename}"
            )
            ax = self.plot_factual_removal_rate_by_alpha(
                conf_level, corretness, a, confidence_method, fig_filename, ax
            )
            print(f"Conformal plot saved to {fig_filename}")

            if plot_group_results:
                # self.plot_factual_group_results(results, csv_filename, x)
                raise NotImplementedError("Not implemented")

    def compute_factual_results(self, data, alphas, a):
        results = {}
        for confidence_method in self.confidence_method:
            results[confidence_method] = {}
            for alpha in tqdm(
                alphas, desc=f"Computing factual results for {confidence_method}"
            ):
                thresholds = []
                correctness = []
                for _ in range(self.runs):
                    random.shuffle(data)
                    split_index = len(data) // 2
                    calibration_data = data[:split_index]
                    test_data = data[split_index:]

                    assert (
                        len(calibration_data) != 0
                    ), "Calibration data should not be empty"
                    assert len(test_data) != 0, "Test data should not be empty"

                    threshold = compute_threshold(alpha, calibration_data, a, confidence_method)
                    fraction_correct = self._evaluate_factual_correctness(
                        test_data, threshold, a, confidence_method
                    )
                    thresholds.append(threshold)
                    correctness.append(fraction_correct)

                results[confidence_method][alpha] = {
                    "threshold": thresholds,
                    "correctness": correctness,
                    "factuality": 1 - alpha,
                }
        return results

    def process_factual_correctness_results(self, results: dict):
        """
        x: confidence level
        y: list of average factual correctness
        yerr: list of standard error of factual correctness
        """
        x, y, yerr = [], [], []
        for alpha, results_for_alpha in results.items():

            x.append(1 - alpha)
            y.append(np.mean(results_for_alpha["correctness"]))
            yerr.append(
                (
                    np.std(results_for_alpha["correctness"])
                    * 1.96
                    / np.sqrt(len(results_for_alpha["correctness"]))
                )
            )

        return x, y, yerr

    def plot_factual_removal_rate_by_alpha(
        self, x, y, a, confidence_method, fig_filename, ax=None
    ):
        if not ax:
            fig, ax = plt.subplots(figsize=(8, 6), dpi=800)
        else:
            fig = ax.figure  # Get the figure from the provided ax

        ax.set_xlabel(f"Target factuality (1 - {chr(945)})", fontsize=16)
        ax.set_ylabel("Empirical factuality", fontsize=16)
        ax.set_title(
            f"Factual correctness for {self.dataset_name} Datasets (a={a})", fontsize=20
        )

        # Plot the data
        ax.plot(x, y, label=confidence_method, linewidth=2)

        # Set legend
        ax.legend(loc="upper left", bbox_to_anchor=(0.02, 0.98), fontsize=10)

        # Save the figure
        fig.savefig(fig_filename, bbox_inches="tight", dpi=800)

        return ax  # Return the ax for further modifications if needed

    def _evaluate_factual_correctness(
        self, data: list, threshold: float, a: float, confidence_method: str
    ):
        """
        Evaluates the factual correctness of subclaims within the provided data.
        This function processes a list of data entries, each containing subclaims with similarity scores.
        It calculates the percentage of correctly retained subclaims based on a given threshold and
        compares it to a specified accuracy level `a`.
        Args:
            data (list): A list of dictionaries, where each dictionary represents an entry containing subclaims.
            threshold (float): The similarity score threshold above which subclaims are considered retained.
            a (float): The accuracy level to compare the correctly retained percentage against.
        Returns:
            float: The percentage of entries in the data that satisfy the correct level of accuracy `a`.
        """

        correctly_retained = []
        # Process each item in the list
        for entry in data:
            # Extract subclaims from each item
            retained_cnt = 0
            correctly_retained_count = 0
            for subclaim in entry["subclaims"]:

                # Extract the score and noise
                score = subclaim["scores"][confidence_method]
                noise = subclaim["scores"]["noise"]

                # Add the subclaim to the collection if similarity score is above threshold
                if score + noise >= threshold:
                    retained_cnt += 1
                    if (
                        subclaim.get("annotations", {}).get("gpt", "")
                        in CORRECT_ANNOTATIONS
                    ):
                        correctly_retained_count += 1

            # Calculate correctly retained rate
            correctly_retained_percentage = (
                correctly_retained_count / retained_cnt if retained_cnt > 0 else 1
            )
            correctly_retained.append(correctly_retained_percentage)

        correctness_list = [
            correctly_retained_percentage >= a
            for correctly_retained_percentage in correctly_retained
        ]
        # percentage of test data satisfying correct level of a
        return sum(correctness_list) / len(correctness_list)
