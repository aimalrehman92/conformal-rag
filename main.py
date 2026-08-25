import os
import argparse
import numpy as np
import logging
import sqlite3
import yaml
from pathlib import Path

from src.common.config_manager import ConfigManager
from src.dataloader.dataloader import DataLoader
from src.data_processor.query_processor import QueryProcessor
from src.common.file_manager import FileManager
from src.common.faiss_manager import FAISSIndexManager
from src.common.retriever_factory import create_retriever
from src.subclaim_processor.scorer.subclaim_scorer import SubclaimScorer
from src.subclaim_processor.subclaim_processor import process_subclaims
from src.calibration.conformal import SplitConformalCalibration
from src.calibration.conditional_conformal import GroupConditionalConformal


def parse_args(dataset_aliases):
    """Parse command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="conf/config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Override dataset name from config",
        choices=dataset_aliases,
    )
    parser.add_argument(
        "--query_size", type=int, default=None, help="Override query size from config"
    )
    parser.add_argument("--run_id", type=str, help="Custom run identifier")
    return parser.parse_args()


def validate_wiki_db_schema(wiki_db_path: str) -> None:
    """
    Validate that the configured WikiDB is a readable SQLite database
    with the documents(title, text) schema expected by the pipeline.
    """
    required_columns = {"title", "text"}
    database_uri = Path(wiki_db_path).resolve().as_uri() + "?mode=ro"

    try:
        connection = sqlite3.connect(database_uri, uri=True)

        try:
            table_info = connection.execute("PRAGMA table_info(documents)").fetchall()
        finally:
            connection.close()

    except sqlite3.DatabaseError as exc:
        raise ValueError(
            f"Configured WikiDB at '{wiki_db_path}' is not a valid "
            "readable SQLite database."
        ) from exc

    if not table_info:
        raise ValueError(
            f"Configured WikiDB at '{wiki_db_path}' does not contain "
            "the required 'documents' table."
        )

    columns = {row[1] for row in table_info}
    missing_columns = required_columns - columns

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(
            f"Configured WikiDB at '{wiki_db_path}' has an invalid "
            f"'documents' schema. Missing column(s): {missing}."
        )


def main():
    avaliable_datasets = []
    with open("conf/dataset_config.yaml", "r") as f:
        dataset_config = yaml.safe_load(f)
        avaliable_datasets = list(dataset_config["datasets"].keys())
    # Parse arguments
    args = parse_args(avaliable_datasets)

    # Initialize config manager
    config_manager = ConfigManager(
        config_path=args.config,
        path_config_path="conf/path_config.yaml",
        dataset_config_path="conf/dataset_config.yaml",
        run_id=args.run_id,
    )

    dataset_aliases = list(dataset_config["datasets"].keys())

    # Setup logging
    log_file, run_id = config_manager.setup_logging()

    # Update config with command line arguments if provided
    if args.dataset is not None or args.query_size is not None:
        updates = {"dataset": {}}
        if args.dataset is not None:
            updates["dataset"]["name"] = args.dataset
        if args.query_size is not None:
            updates["dataset"]["query_size"] = args.query_size
        config_manager.update_config(updates)

    # Save updated config
    config_file = config_manager.save_config()
    logging.info(f"Configuration saved to: {config_file}")

    # Log important config values
    config_manager.log_config()

    # Get the config
    config = config_manager.config
    research_config = config_manager.normalized_config
    path_config = config_manager.path_config
    dataset_config = config_manager.dataset_config

    ####################################### Data and Folder Set up ############################################

    experiment_config = research_config["experiment"]
    dataset_runtime_config = research_config["dataset"]
    models_config = research_config["models"]
    index_config = research_config["index"]
    conformal_config = research_config["conformal"]
    retrieval_config = research_config["retrieval"]

    seed = experiment_config["seed"]
    runs = experiment_config["runs"]

    dataset_name = dataset_runtime_config["name"]
    dataset_type = dataset_runtime_config["type"]
    query_size = dataset_runtime_config["query_size"]
    wiki_db_file = dataset_runtime_config["wiki_db_file"]

    delete_existing_index = index_config["delete_existing"]
    embedding_model = models_config["embedding"]["name"]
    frequency_model = models_config["frequency_scorer"]["name"]
    index_truncation_config = index_config["truncation_config"]

    truncation_strategy = index_truncation_config["strategy"]
    truncate_by = index_truncation_config["truncate_by"]

    response_model = models_config["generator"]["name"]

    alpha_config = conformal_config["alphas"]
    conformal_alphas = np.arange(
        alpha_config["start"],
        alpha_config["end"],
        alpha_config["step"],
    )

    a_value = conformal_config["a_value"]
    split_conformal = conformal_config["split_conformal"]

    logging.info(f"Experiment seed: {seed}")
    logging.info(f"Calibration runs: {runs}")
    logging.info(f"Dataset type: {dataset_type}")

    dataset_custom_config = dataset_config["datasets"].get(dataset_name)
    if not dataset_custom_config:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    full_dataset_name = dataset_custom_config["name"]
    index_store_dir = dataset_custom_config["index_store"]
    group_conditional_conformal = dataset_custom_config.get("is_grouped", False)

    raw_data_dir = os.path.join(path_config["paths"]["raw_data_dir"], full_dataset_name)
    processed_data_dir = os.path.join(
        path_config["paths"]["processed_data_dir"], full_dataset_name
    )
    response_dir = os.path.join(path_config["paths"]["response_dir"], full_dataset_name)
    wiki_db_path = os.path.join(path_config["paths"]["wiki_db_dir"], wiki_db_file)
    result_dir = os.path.join(
        path_config["paths"]["result_dir"], full_dataset_name, run_id
    )

    # set up directories
    for dir_path in [raw_data_dir, processed_data_dir, response_dir, result_dir]:
        os.makedirs(dir_path, exist_ok=True)
        logging.info(f"Directory ensured: {dir_path}")

    # Determine raw data file path
    if dataset_name == "medlf_qa":
        raw_data_path = os.path.join(
            path_config["paths"]["raw_data_dir"],
            "MedLFQA",
        )

    elif dataset_name == "fact_score":
        raw_data_path = os.path.join(
            raw_data_dir,
            "factscore_names.txt",
        )

    else:
        raw_data_file = f"raw_{dataset_name}.json"
        raw_data_path = os.path.join(raw_data_dir, raw_data_file)

    logging.info(f"Raw data path: {raw_data_path}")

    # Load data if needed
    if not os.path.exists(raw_data_path):
        logging.info(f"Raw data not found. Loading data for {dataset_name}")
        data_loader = DataLoader(dataset_name)
        data_loader.load_qa_data(output_path=raw_data_path)
        logging.info(f"Data loaded and saved to {raw_data_path}")

    # Require the explicitly configured WikiDB.
    #
    # Do not silently build a database from a hard-coded Wikipedia snapshot:
    # the corpus version is part of the experimental configuration and must be
    # supplied explicitly for reproducible runs.
    if not os.path.isfile(wiki_db_path):
        raise FileNotFoundError(
            "Configured WikiDB not found at "
            f"'{wiki_db_path}'. "
            "Provide the intended SQLite WikiDB before running the experiment. "
            "The database must use the repository's documents(title, text) schema."
        )

    validate_wiki_db_schema(wiki_db_path)
    logging.info(f"Using validated WikiDB: {wiki_db_path}")

    # Process queries and documents
    input_file = raw_data_path
    if dataset_name == "medlf_qa":
        input_file = os.path.join(path_config["paths"]["raw_data_dir"], "MedLFQA")

    query_output_file = f"{dataset_name}_queries.json"
    document_output_file = f"{dataset_name}_documents.txt"

    CP_result_fig_path = os.path.join(
        result_dir, f"{dataset_name}_{query_size}_a={a_value:.2f}_CP_removal.png"
    )
    GCP_result_fig_path = os.path.join(
        result_dir, f"{dataset_name}_{query_size}_a={a_value:.2f}_GCP_removal.png"
    )
    factual_result_fig_path = os.path.join(
        result_dir,
        f"{dataset_name}_{query_size}_a={a_value:.2f}_factual_correctness.png",
    )
    group_factual_result_fig_path = os.path.join(
        result_dir,
        f"group_{dataset_name}_{query_size}_a={a_value:.2f}_factual_correctness.png",
    )
    result_path = os.path.join(
        result_dir, f"{dataset_name}_{query_size}_a={a_value:.2f}.csv"
    )
    group_result_path = os.path.join(
        result_dir, f"group_{dataset_name}_{query_size}_a={a_value:.2f}.csv"
    )
    ####################################### End of Data and Folder Set up ######################################

    # Create QueryProcessor
    logging.info("Initializing QueryProcessor")
    query_processor = QueryProcessor(db_path=wiki_db_path, query_size=query_size)

    # Create queries data
    logging.info("Processing queries")
    queries, query_path = query_processor.get_queries(
        dataset=dataset_name,
        input_file=input_file,
        output_dir=processed_data_dir,
        output_file=query_output_file,
        seed=seed,
    )
    logging.info(f"Query size: {len(queries)}")

    # Create documents data
    logging.info("Processing documents")
    document_path = query_processor.get_documents(
        query_dir=query_path,
        output_dir=processed_data_dir,
        output_file=document_output_file,
    )
    logging.info(f"Documents saved to {document_path}")

    index_cache_config = {
        "dataset": dataset_runtime_config,
        "seed": seed,
        "embedding": models_config["embedding"],
        "truncation_config": index_truncation_config,
    }

    index_fingerprint = ConfigManager.fingerprint(index_cache_config)

    logging.info(f"Index configuration fingerprint: {index_fingerprint}")

    subclaim_cache_config = {
        "cache_schema_version": 2,
        "dataset": dataset_runtime_config,
        "seed": seed,
        "index_fingerprint": index_fingerprint,
        "retrieval": retrieval_config,
        "models": {
            "generator": models_config["generator"],
            "claim_decomposer": models_config["claim_decomposer"],
            "claim_verifier": models_config["claim_verifier"],
            "frequency_scorer": models_config["frequency_scorer"],
        },
        "scoring_strategy": conformal_config["scoring_strategy"],
    }

    subclaim_fingerprint = ConfigManager.fingerprint(subclaim_cache_config)

    logging.info(f"Subclaim configuration fingerprint: {subclaim_fingerprint}")

    subclaims_path = os.path.join(
        response_dir,
        (
            f"{dataset_name}_{query_size}_"
            f"subclaims_with_scores_{subclaim_fingerprint}.json"
        ),
    )

    # Index creation and retrieval
    os.makedirs(index_store_dir, exist_ok=True)
    index_file_path = os.path.join(
        index_store_dir,
        f"index_{index_fingerprint}.faiss",
    )
    indice2fm_path = os.path.join(
        index_store_dir,
        f"indice2fm_{index_fingerprint}.json",
    )

    logging.info(f"Setting up FAISS index manager")
    faiss_manager = FAISSIndexManager(
        index_truncation_config=index_truncation_config,
        embedding_model=embedding_model,
        index_path=index_file_path,
        indice2fm_path=indice2fm_path,
    )

    if delete_existing_index:
        logging.info("Deleting existing index as requested")
        faiss_manager.delete_index()

    # Create index if it does not exist
    document_file = FileManager(
        document_path, index_truncation_config=index_truncation_config
    )

    logging.info(
        f"Using truncation strategy: {truncation_strategy}, truncate_by: {truncate_by}"
    )

    # If Index doesn't exist yet
    if not os.path.exists(index_file_path):
        try:
            logging.info(f"Creating new index with document '{document_path}'")
            faiss_manager.upsert_file_to_faiss(
                document_file,
                truncation_strategy=truncation_strategy,
                truncate_by=truncate_by,
            )
            logging.info("Index created successfully")
        except Exception as e:
            error_msg = f"Failed to create new index: {str(e)}"
            logging.error(error_msg)
            raise RuntimeError(error_msg)

    # If Index exists but current document isn't indexed
    elif document_path not in faiss_manager.indice2fm:
        # Verify index integrity
        logging.info("Checking index integrity")
        if not faiss_manager.is_indice_align():
            error_msg = "Index corruption detected: index and indice2fm are not aligned"
            logging.error(error_msg)
            raise ValueError(error_msg)

        try:
            logging.info(f"Adding document '{document_path}' to existing index")
            faiss_manager.upsert_file_to_faiss(
                document_file,
                truncation_strategy=truncation_strategy,
                truncate_by=truncate_by,
            )
            logging.info("Document added to index successfully")
        except Exception as e:
            error_msg = f"Failed to add document to index: {str(e)}"
            logging.error(error_msg)
            raise RuntimeError(error_msg)

    # Case 3: Document is already indexed
    else:
        logging.info(f"Document '{document_path}' is already indexed")

    logging.info(f"Initializing retrieval strategy: {retrieval_config['strategy']}")

    retriever = create_retriever(
        strategy=retrieval_config["strategy"],
        faiss_manager=faiss_manager,
        truncation_strategy=truncation_strategy,
        truncate_by=truncate_by,
        multi_hop_config=retrieval_config["multi_hop"],
    )

    # generate subclaims with scores
    logging.info(f"Initializing SubclaimScorer with embedding model {embedding_model}")
    scorer = SubclaimScorer(
        index_truncation_config=index_truncation_config,
        embedding_model=embedding_model,
        frequency_model=frequency_model,
        index_path=index_file_path,
        indice2fm_path=indice2fm_path,
    )

    logging.info(f"Using frequency scoring model: {frequency_model}")
    logging.info(f"Processing subclaims and generating scores")

    subclaim_with_annotation_data = process_subclaims(
        query_path=query_path,
        subclaims_path=subclaims_path,
        scorer=scorer,
        config=research_config,
        retriever=retriever,
    )

    logging.info(f"Subclaims processed and saved to {subclaims_path}")

    # calibration and conformal prediction results
    if split_conformal:
        logging.info("Running split conformal prediction")
        conformal = SplitConformalCalibration(
            dataset_name=dataset_name,
            runs=runs,
            seed=seed,
        )
        logging.info(
            f"Plotting conformal removal with alphas: {conformal_alphas}, a={a_value}"
        )
        conformal.plot_conformal_removal(
            data=subclaim_with_annotation_data,
            alphas=conformal_alphas,
            a=a_value,
            fig_filename=CP_result_fig_path,
            csv_filename=result_path,
        )
        logging.info(f"CP removal plot saved to {CP_result_fig_path}")

        logging.info("Plotting factual removal")
        conformal.plot_factual_removal(
            data=subclaim_with_annotation_data,
            alphas=conformal_alphas,
            a=a_value,
            fig_filename=factual_result_fig_path,
            csv_filename=result_path,
        )
        logging.info(f"Factual removal plot saved to {factual_result_fig_path}")
        logging.info(f"Results saved to {result_path}")

    if group_conditional_conformal:
        logging.info("Running group conditional conformal prediction")
        conformal = GroupConditionalConformal(
            dataset_name=dataset_name,
            result_dir=result_dir,
            runs=runs,
            seed=seed,
        )
        logging.info(
            f"Plotting conformal removal with alphas: {conformal_alphas}, a={a_value}"
        )
        conformal.plot_conformal_removal(
            data=subclaim_with_annotation_data,
            alphas=conformal_alphas,
            a=a_value,
            fig_filename=GCP_result_fig_path,
            csv_filename=group_result_path,
        )
        logging.info(f"CP removal plot saved to {GCP_result_fig_path}")

        logging.info("Plotting factual removal")
        conformal.plot_factual_removal(
            data=subclaim_with_annotation_data,
            alphas=conformal_alphas,
            a=a_value,
            fig_filename=group_factual_result_fig_path,
            csv_filename=group_result_path,
        )
        logging.info(f"Factual removal plot saved to {factual_result_fig_path}")
        logging.info(f"Results saved to {result_path}")

    # Copy config and log files to result directory for reproducibility
    result_run_dir = config_manager.copy_run_artifacts(result_dir)
    logging.info(
        f"Run completed successfully. Results and logs saved to {result_run_dir}"
    )


if __name__ == "__main__":
    main()
