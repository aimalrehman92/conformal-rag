import os
import yaml
import logging
import datetime
import json
import shutil
from pathlib import Path


class ConfigManager:
    """Utility class to manage configuration loading, saving and logging"""

    def __init__(
        self,
        config_path=None,
        path_config_path=None,
        dataset_config_path=None,
        run_id=None,
    ):
        """
        Initialize the ConfigManager with a config file path

        Args:
            config_path (str): Path to the YAML config file
            run_id (str): Optional identifier for the run
        """
        self.config = {}
        self.normalized_config = {}
        self.run_id = run_id or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = f"logs/{self.run_id}"

        if config_path:
            self.config = self.load_config(config_path)
            self.normalized_config = self._normalize_config(self.config)
        if path_config_path:
            self.path_config = self.load_config(path_config_path)
        if dataset_config_path:
            self.dataset_config = self.load_config(dataset_config_path)

    def load_config(self, config_path):
        """
        Load configuration from a YAML file

        Args:
            config_path (str): Path to the YAML config file

        Returns:
            dict: The loaded configuration
        """
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def _infer_provider(self, model_name):
        """
        Infer the provider for models used by the legacy configuration.

        Future research configurations should specify the provider
        explicitly; this inference exists only for backward compatibility.
        """
        if not model_name:
            return None

        model_name = model_name.lower()

        if model_name.startswith("gpt-") or model_name.startswith("text-embedding-"):
            return "openai"

        if "llama" in model_name:
            return "huggingface"

        return "unknown"

    def _normalize_config(self, config):
        """
        Convert either the legacy repository configuration or the future
        research configuration into one canonical internal structure.

        The original config is left unchanged so legacy code continues
        to work while modules are migrated incrementally.
        """
        experiment_config = config.get("experiment", {})
        dataset_config = config.get("dataset", {})
        index_config = config.get("index", {})

        legacy_rag_config = config.get("rag", {})
        legacy_conformal_config = config.get("conformal_prediction", {})

        models_config = config.get("models", {})
        retrieval_config = config.get("retrieval", {})
        conformal_config = config.get("conformal", {})

        generator_config = models_config.get("generator", {})
        decomposer_config = models_config.get("claim_decomposer", {})
        verifier_config = models_config.get("claim_verifier", {})
        frequency_config = models_config.get("frequency_scorer", {})
        embedding_config = models_config.get("embedding", {})

        generator_model = generator_config.get(
            "name",
            legacy_rag_config.get("response_model"),
        )

        decomposer_model = decomposer_config.get(
            "name",
            legacy_rag_config.get("fact_generation_model"),
        )

        verifier_model = verifier_config.get(
            "name",
            legacy_conformal_config.get("claim_verification_model"),
        )

        frequency_model = frequency_config.get(
            "name",
            "gpt-4o-mini",
        )

        embedding_model = embedding_config.get(
            "name",
            index_config.get("embedding_model"),
        )

        return {
            "experiment": {
                "seed": experiment_config.get(
                    "seed",
                    config.get("seed", 42),
                ),
                "runs": experiment_config.get("runs", 1000),
            },
            "dataset": {
                "name": dataset_config.get("name"),
                "type": dataset_config.get("type", "builtin"),
                "query_size": dataset_config.get("query_size"),
                "wiki_db_file": dataset_config.get("wiki_db_file"),
                "custom": dataset_config.get("custom", {}),
            },
            "models": {
                "generator": {
                    "provider": generator_config.get(
                        "provider",
                        self._infer_provider(generator_model),
                    ),
                    "name": generator_model,
                    "temperature": generator_config.get(
                        "temperature",
                        legacy_rag_config.get("response_temperature", 0.7),
                    ),
                },
                "claim_decomposer": {
                    "provider": decomposer_config.get(
                        "provider",
                        self._infer_provider(decomposer_model),
                    ),
                    "name": decomposer_model,
                },
                "claim_verifier": {
                    "provider": verifier_config.get(
                        "provider",
                        self._infer_provider(verifier_model),
                    ),
                    "name": verifier_model,
                },
                "frequency_scorer": {
                    "provider": frequency_config.get(
                        "provider",
                        self._infer_provider(frequency_model),
                    ),
                    "name": frequency_model,
                },
                "embedding": {
                    "provider": embedding_config.get(
                        "provider",
                        self._infer_provider(embedding_model),
                    ),
                    "name": embedding_model,
                },
            },
            "retrieval": {
                "strategy": retrieval_config.get(
                    "strategy",
                    "single_hop",
                ),
                "top_k": retrieval_config.get(
                    "top_k",
                    legacy_rag_config.get("retrival_topk", 10),
                ),
                "threshold": retrieval_config.get(
                    "threshold",
                    legacy_rag_config.get("retrival_threshold", 0.3),
                ),
                "multi_hop": {
                    "max_hops": retrieval_config.get("multi_hop", {}).get(
                        "max_hops",
                        3,
                    ),
                    "accumulate_documents": retrieval_config.get(
                        "multi_hop",
                        {},
                    ).get(
                        "accumulate_documents",
                        True,
                    ),
                    "stop_if_no_new_documents": retrieval_config.get(
                        "multi_hop",
                        {},
                    ).get(
                        "stop_if_no_new_documents",
                        True,
                    ),
                    "query_rewriter": {
                        "provider": retrieval_config.get(
                            "multi_hop",
                            {},
                        )
                        .get("query_rewriter", {})
                        .get(
                            "provider",
                            "openai",
                        ),
                        "model": retrieval_config.get(
                            "multi_hop",
                            {},
                        )
                        .get("query_rewriter", {})
                        .get(
                            "model",
                            "gpt-4o-mini",
                        ),
                        "temperature": retrieval_config.get(
                            "multi_hop",
                            {},
                        )
                        .get("query_rewriter", {})
                        .get(
                            "temperature",
                            0.0,
                        ),
                        "max_documents": retrieval_config.get(
                            "multi_hop",
                            {},
                        )
                        .get("query_rewriter", {})
                        .get(
                            "max_documents",
                            5,
                        ),
                        "max_chars_per_document": retrieval_config.get(
                            "multi_hop",
                            {},
                        )
                        .get("query_rewriter", {})
                        .get(
                            "max_chars_per_document",
                            1200,
                        ),
                    },
                },
            },
            "index": {
                "delete_existing": index_config.get(
                    "delete_existing",
                    False,
                ),
                "embedding_model": embedding_model,
                "truncation_config": index_config.get(
                    "truncation_config",
                    {},
                ),
            },
            "conformal": {
                "aggregation_strategy": conformal_config.get(
                    "aggregation_strategy",
                    legacy_conformal_config.get("aggregation_strategy"),
                ),
                "scoring_strategy": conformal_config.get(
                    "scoring_strategy",
                    legacy_conformal_config.get("scoring_strategy"),
                ),
                "split_conformal": conformal_config.get(
                    "split_conformal",
                    legacy_conformal_config.get("split_conformal", True),
                ),
                "alphas": conformal_config.get(
                    "alphas",
                    legacy_conformal_config.get("conformal_alphas", {}),
                ),
                "a_value": conformal_config.get(
                    "a_value",
                    legacy_conformal_config.get("a_value", 1.0),
                ),
            },
        }

    def save_config(self, output_path=None):
        """
        Save the current configuration to a YAML file

        Args:
            output_path (str): Path to save the config file, defaults to log directory

        Returns:
            str: Path to the saved config file
        """
        if output_path is None:
            os.makedirs(self.log_dir, exist_ok=True)
            output_path = os.path.join(self.log_dir, f"config_{self.run_id}.yaml")

        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, "w") as f:
            yaml.dump(self.config, f, default_flow_style=False)

        return output_path

    def setup_logging(self, log_level=logging.INFO):
        """
        Setup logging configuration

        Args:
            log_level: Logging level

        Returns:
            str: Path to the log file
        """
        os.makedirs(self.log_dir, exist_ok=True)
        log_file = os.path.join(self.log_dir, f"run_{self.run_id}.log")

        logging.basicConfig(
            level=log_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        )

        # Disable httpx logs
        logging.getLogger("httpx").setLevel(logging.WARNING)

        # Log some initial information
        logging.info(f"Starting run with ID: {self.run_id}")
        logging.info(f"Log file: {log_file}")

        return log_file, self.run_id

    def log_config(self):
        """Log the important parts of the configuration"""
        if not self.config:
            logging.warning("No configuration loaded to log")
            return

        logging.info("=== Run Configuration ===")

        # Log dataset info
        if "dataset" in self.config:
            logging.info(f"Dataset: {self.config['dataset']['name']}")
            logging.info(f"Query size: {self.config['dataset']['query_size']}")

        # Log index info
        if "index" in self.config:
            logging.info(f"Embedding model: {self.config['index']['embedding_model']}")
            logging.info(
                f"Delete existing index: {self.config['index']['delete_existing']}"
            )

        logging.info("========================")

    def update_config(self, updates):
        """
        Update the configuration with new values

        Args:
            updates (dict): Dictionary containing updates to apply

        Returns:
            dict: The updated configuration
        """
        # This is a simple implementation that only handles top-level keys
        for key, value in updates.items():
            if (
                isinstance(value, dict)
                and key in self.config
                and isinstance(self.config[key], dict)
            ):
                self.config[key].update(value)
            else:
                self.config[key] = value

        self.normalized_config = self._normalize_config(self.config)

        return self.config

    def copy_run_artifacts(self, result_dir):
        """
        Copy config and logs to a results directory for reproducibility

        Args:
            result_dir (str): Path to the results directory

        Returns:
            str: Path to the result run directory
        """
        result_run_dir = os.path.join(result_dir, "config")
        os.makedirs(result_run_dir, exist_ok=True)

        # Get the latest config and log files
        config_files = sorted(Path(self.log_dir).glob("config_*.yaml"))
        # log_files = sorted(Path(self.log_dir).glob("run_*.log"))

        if config_files:
            latest_config = str(config_files[-1])
            shutil.copy2(latest_config, os.path.join(result_run_dir, "config.yaml"))

        # if log_files:
        #     latest_log = str(log_files[-1])
        #     shutil.copy2(latest_log, os.path.join(result_run_dir, "run.log"))

        return result_run_dir
