import os
import argparse
import numpy as np
import logging
import yaml
from pathlib import Path

from src.common.config_manager import ConfigManager
from src.dataloader.dataloader import DataLoader
from src.data_processor.query_processor import QueryProcessor
from src.common.file_manager import FileManager
from src.common.faiss_manager import FAISSIndexManager
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
        "--query_size", type=int, default=500, help="Override query size from config"
    )
    parser.add_argument("--run_id", type=str, help="Custom run identifier")
    return parser.parse_args()


def main():
    avaliable_datasets = []
    with open("conf/dataset_config.yaml", 'r') as f:
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
    if args.dataset or args.query_size:
        updates = {"dataset": {}}
        if args.dataset:
            updates["dataset"]["name"] = args.dataset
        if args.query_size:
            updates["dataset"]["query_size"] = args.query_size
        config_manager.update_config(updates)

    # Save updated config
    config_file = config_manager.save_config()
    logging.info(f"Configuration saved to: {config_file}")

    # Log important config values
    config_manager.log_config()

    # Get the config
    config = config_manager.config
    path_config = config_manager.path_config
    dataset_config = config_manager.dataset_config

    ####################################### Data and Folder Set up ############################################
    dataset_name = config["dataset"]["name"]
    query_size = config["dataset"]["query_size"]
    wiki_db_file = config["dataset"]["wiki_db_file"]

    delete_existing_index = config["index"]["delete_existing"]
    embedding_model = config["index"]["embedding_model"]
    index_truncation_config = config["index"]["truncation_config"]
    truncation_strategy = index_truncation_config["strategy"]
    truncate_by = index_truncation_config["truncate_by"]

    response_model = config["rag"]["response_model"]

    alpha_config = config["conformal_prediction"]["conformal_alphas"]
    conformal_alphas = np.arange(
        alpha_config["start"], alpha_config["end"], alpha_config["step"]
    )
    a_value = config["conformal_prediction"]["a_value"]

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
        input_file = os.path.join(path_config["paths"]["raw_data_dir"], "MedLFQA")
        raw_data_path = input_file
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

    # create wiki db if needed
    if not os.path.exists(wiki_db_path) or not os.path.isfile(wiki_db_path):
        wiki_source = os.path.join(
            path_config["paths"]["wiki_db_dir"],
            "enwiki-20171001-pages-meta-current-withlinks-abstracts",
        )
        if not os.path.exists(wiki_source):
            raise FileNotFoundError(f"Wiki source data not found at {wiki_source}")
        logging.info(f"Wiki DB not found. Creating from source {wiki_source}")
        data_loader = DataLoader(dataset_name)
        data_loader.create_wiki_db(source_path=wiki_source, output_path=wiki_db_path)
        logging.info(f"Wiki DB created at {wiki_db_path}")

    # Process queries and documents
    input_file = raw_data_path
    if dataset_name == "medlf_qa":
        input_file = os.path.join(path_config["paths"]["raw_data_dir"], "MedLFQA")

    query_output_file = f"{dataset_name}_queries.json"
    document_output_file = f"{dataset_name}_documents.txt"

    subclaims_path = os.path.join(
        response_dir,
        f"{dataset_name}_{query_size}_subclaims_with_scores_{response_model}.json",
    )
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

    # Index creation and retrieval
    os.makedirs(index_store_dir, exist_ok=True)
    index_file_path = os.path.join(index_store_dir, f"index_{query_size}.faiss")
    indice2fm_path = os.path.join(index_store_dir, f"indice2fm_{query_size}.json")

    logging.info(f"Setting up FAISS index manager")
    faiss_manager = FAISSIndexManager(
        index_truncation_config=index_truncation_config,
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

    # generate subclaims with scores
    logging.info(f"Initializing SubclaimScorer with embedding model {embedding_model}")
    scorer = SubclaimScorer(
        index_truncation_config=index_truncation_config,
        embedding_model=embedding_model,
        index_path=index_file_path,
        indice2fm_path=indice2fm_path,
    )

    logging.info(f"Processing subclaims and generating scores")
    subclaim_with_annotation_data = process_subclaims(
        query_path=query_path,
        subclaims_path=subclaims_path,
        faiss_manager=faiss_manager,
        scorer=scorer,
        config=config,
    )
    logging.info(f"Subclaims processed and saved to {subclaims_path}")

    # calibration and conformal prediction results
    if config["conformal_prediction"]["split_conformal"]:
        logging.info("Running split conformal prediction")
        conformal = SplitConformalCalibration(dataset_name=dataset_name)
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
        conformal = GroupConditionalConformal(dataset_name=dataset_name, result_dir=result_dir)
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
