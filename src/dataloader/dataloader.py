import os
import re
import bz2
import sqlite3
import json
from collections import defaultdict
from datasets import load_dataset
from transformers import RobertaTokenizer
from src.rag.retrieval import MAX_LENGTH, SPECIAL_SEPARATOR


class DataLoader:
    def __init__(self, dataset: str):
        self.dataset = dataset

    def load_qa_data(self, output_path: str):
        if os.path.exists(output_path):
            print(f"Dataset already exists at {output_path}.")
        else:
            print(f"Loading {self.dataset} dataset.")
            if self.dataset == "fact_score":
                load_fact_score_data(output_path)
            elif self.dataset == "hotpot_qa":
                load_hotpot_qa_data(output_path)
            elif self.dataset == "pop_qa":
                load_pop_qa_data(output_path)
            elif self.dataset == "medlfqa":
                output_path = load_medlfqa_data("data/.source_data/MedLFQA")
                clean_medlfqa_data(data_path=output_path, output_path=output_path)

    def create_wiki_db(
        self,
        source_path: str,
        output_path: str,
    ):
        """
        Create the canonical WikiDB expected by DocDB.

        The source must be an explicit directory containing the compressed
        Wikipedia JSONL files. The output uses the repository's canonical
        documents(title, text) SQLite schema.

        Wikipedia text is tokenized and chunked using the same RoBERTa-based
        representation as DocDB.build_db(), including SPECIAL_SEPARATOR between
        passages.
        """

        if os.path.exists(output_path):
            raise FileExistsError(
                f"Refusing to overwrite existing database at '{output_path}'."
            )

        if not os.path.isdir(source_path):
            raise FileNotFoundError(
                f"Wiki source directory not found at '{source_path}'."
            )

        output_directory = os.path.dirname(output_path)
        if output_directory:
            os.makedirs(output_directory, exist_ok=True)

        print(f"Building WikiDB from {source_path}")
        print(f"Output database: {output_path}")

        tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
        connection = sqlite3.connect(output_path)

        try:
            cursor = connection.cursor()
            cursor.execute("CREATE TABLE documents (title PRIMARY KEY, text)")

            seen_titles = set()
            pending_rows = []
            document_count = 0
            batch_size = 10000

            for folder_name in sorted(os.listdir(source_path)):
                folder_path = os.path.join(source_path, folder_name)

                if not os.path.isdir(folder_path):
                    continue

                for file_name in sorted(os.listdir(folder_path)):
                    if not file_name.endswith(".bz2"):
                        continue

                    file_path = os.path.join(folder_path, file_name)
                    print(f"Reading {file_path}")

                    with bz2.open(file_path, "rt", encoding="utf-8") as source_file:
                        for line in source_file:
                            if not line.strip():
                                continue

                            data = json.loads(line)
                            title = data["title"]

                            if title in seen_titles:
                                continue

                            seen_titles.add(title)

                            text = data.get("text", "")

                            if isinstance(text, str):
                                text = [text]

                            passages = [[]]

                            for sentence in text:
                                sentence = str(sentence)

                                if not sentence.strip():
                                    continue

                                tokens = tokenizer(sentence)["input_ids"]
                                max_length = MAX_LENGTH - len(passages[-1])

                                if len(tokens) <= max_length:
                                    passages[-1].extend(tokens)
                                else:
                                    passages[-1].extend(tokens[:max_length])
                                    offset = max_length

                                    while offset < len(tokens):
                                        passages.append(
                                            tokens[offset : offset + MAX_LENGTH]
                                        )
                                        offset += MAX_LENGTH

                            decoded_passages = [
                                tokenizer.decode(tokens)
                                for tokens in passages
                                if any(token not in {0, 2} for token in tokens)
                            ]

                            if not decoded_passages:
                                continue

                            encoded_text = SPECIAL_SEPARATOR.join(decoded_passages)
                            pending_rows.append((title, encoded_text))
                            document_count += 1

                            if len(pending_rows) >= batch_size:
                                cursor.executemany(
                                    "INSERT INTO documents VALUES (?, ?)",
                                    pending_rows,
                                )
                                connection.commit()
                                pending_rows = []

                                print(f"Saved {document_count} Wikipedia documents.")

            if pending_rows:
                cursor.executemany(
                    "INSERT INTO documents VALUES (?, ?)",
                    pending_rows,
                )

            connection.commit()

        except Exception:
            connection.rollback()
            raise

        finally:
            connection.close()

        print(f"Created WikiDB at {output_path} " f"with {document_count} documents.")


def load_fact_score_data(output_path: str):
    # raise NotImplementedError
    pass


def load_hotpot_qa_data(output_path: str):
    """Load HotpotQA dataset and save validation set to json file."""

    dataset = load_dataset("kilt_tasks", "hotpotqa")
    dataset["validation"].to_json(output_path, orient="records", lines=True)
    print("HotpotQA validation set saved to", output_path)

    return


def load_pop_qa_data(output_path: str):
    """Load PopQA dataset and save test set to json file."""

    dataset = load_dataset("akariasai/popQA")
    dataset["test"].to_json(output_path, orient="records", lines=True)
    print("PopQA test set saved to", output_path)

    return


def load_medlfqa_data(output_path: str = "data/.source_data/MedLFQA"):
    """Load MedLFQA dataset and save to json file."""

    if not os.path.exists(f"{output_path}"):
        os.system(f"mkdir -p {output_path}")
    dataset_names = [
        "healthsearch_qa",
        "kqa_golden",
        "kqa_silver_wogold",
        "live_qa",
        "medication_qa",
    ]
    for fname in dataset_names:
        if f"{fname}.jsonl" in os.listdir(output_path):
            print(f"Dataset {fname} already exists.")
            continue
        else:
            os.system(
                f"wget -O {output_path}/{fname}.jsonl https://raw.githubusercontent.com/jjcherian/conformal-safety/refs/heads/main/data/MedLFQAv2/{fname}.jsonl"
            )

    print(f"MedLFQA dataset saved to {output_path}")

    return output_path


def remove_specific_leading_chars(input_string):
    # Remove leading commas
    input_string = re.sub(r"^,+", "", input_string)
    # Remove numbers followed by a comma
    return re.sub(r"^\d+,+", "", input_string)


def clean_medlfqa_data(data_path: str, output_path: str):
    """Clean the MedLFQA dataset to remove unwanted characters and fields."""
    suffix = ".jsonl"
    datasets = {}

    # Load datasets
    for fname in os.listdir(data_path):
        if fname.endswith(suffix):
            dataset_name = fname[: -len(suffix)]
            with open(os.path.join(data_path, fname), "r") as fp:
                datasets[dataset_name] = [json.loads(line) for line in fp]

    # Clean questions and filter duplicates
    filtered_datasets = {}
    redundant_prompts = defaultdict(int)

    for name, dataset in datasets.items():
        seen_questions = set()
        filtered_dataset = []

        for pt in dataset:
            pt["Question"] = remove_specific_leading_chars(pt["Question"]).strip()
            if pt["Question"] not in seen_questions:
                seen_questions.add(pt["Question"])
                filtered_dataset.append(pt)
                redundant_prompts[pt["Question"]] += 1

        filtered_datasets[name] = filtered_dataset

    # Filter out questions that are redundant across datasets
    for name, dataset in filtered_datasets.items():
        if name not in {"kqa_golden", "live_qa"}:
            filtered_datasets[name] = [
                pt for pt in dataset if redundant_prompts[pt["Question"]] == 1
            ]

    if not os.path.exists(output_path):
        os.system(f"mkdir -p {output_path}")

    # Save cleaned datasets
    for name, dataset in filtered_datasets.items():
        filepath = os.path.join(output_path, f"{name}.json")
        json_objects = []
        for pt in dataset:
            json_objects.append(pt)
        with open(filepath, "w") as outfile:
            json.dump(json_objects, outfile, indent=4)
            # for pt in dataset:
            #     json.dump(pt, outfile)
            #     outfile.write('\n')
            print(f"Saved {name} dataset to {filepath}")


# WikiDB creation is intentionally not run automatically from this module.
# The Wikipedia source snapshot and output database path must both be
# supplied explicitly by experiment setup code.
