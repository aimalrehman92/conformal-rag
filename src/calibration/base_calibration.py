from abc import ABC, abstractmethod


class ICalibration(ABC):
    """
    Interface for calibration methods.
    """

    @abstractmethod
    def plot_conformal_removal(
        self, data, alphas, a, fig_filename, csv_filename, plot_group_results=False
    ):
        pass

    @abstractmethod
    def plot_factual_removal(
        self, data, alphas, a, fig_filename, csv_filename, plot_group_results=False
    ):
        pass
