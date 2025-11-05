import os
import re
import ast
import bz2
import sqlite3
import json
from collections import defaultdict
from datasets import load_dataset


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
        source_path: str = "data/raw/WikiDB/enwiki-20171001-pages-meta-current-withlinks-abstracts",
        output_path: str = "data/raw/WikiDB/enwiki_20190113.db",
    ):
        "Create a SQLite database from the Wikipedia dump data."

        if os.path.exists(output_path):
            print(f"Database already exists at {output_path}.")
            return

        if not os.path.exists(source_path):
            raise FileNotFoundError(f"Source path {source_path} not found.")
        else:
            print(f"Reading data from {source_path}")
            # Create a connection to the SQLite database
            conn = sqlite3.connect(output_path)
            cursor = conn.cursor()

            # Create a table to store the content
            cursor.execute("""DROP TABLE IF EXISTS wiki_content""")
            cursor.execute(
                """
            CREATE TABLE IF NOT EXISTS wiki_content (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT,
                url TEXT,
                text TEXT
            )
            """
            )

            # Iterate through each bz2 file in the folder
            for folder in os.listdir(source_path):
                folder_path = f"{source_path}/{folder}"
                for file_name in os.listdir(folder_path):
                    if file_name.endswith(".bz2"):
                        file_path = os.path.join(folder_path, file_name)
                        with bz2.open(file_path, "rt") as file:
                            content = file.read()
                            lines = content.split("\n")
                            for line in lines:
                                if line.strip():
                                    data = json.loads(line)
                                    line = ast.literal_eval(line)
                                    id = line.get("id", "")
                                    title = line["title"]
                                    url = line.get("url", "")
                                    text = str(line.get("text", ""))
                                    cursor.execute(
                                        """
                                    INSERT INTO wiki_content (id, title, url, text)
                                    VALUES (?, ?, ?, ?)
                                    """,
                                        (id, title, url, text),
                                    )
                                    # print(f'Inserted {title} into the database')

            # Commit the changes and close the connection
            conn.commit()
            conn.close()
            print(f"Created database at {output_path}")


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


# example code
if __name__ == "__main__":
    # loader = DataLoader("fact_score")
    # loader.load_qa_data("data/raw/FactScore/factscore_names.txt")

    # loader = DataLoader("hotpot_qa")
    # loader.load_qa_data("data/raw/HotpotQA/hotpotqa_validation_set.jsonl")

    # loader = DataLoader("pop_qa")
    # loader.load_qa_data("data/raw/PopQA/popQA_test.json")

    loader = DataLoader("medlfqa")
    loader.load_qa_data("data/raw/MedLFQA/")

    loader.create_wiki_db(output_path="data/raw/WikiDB/enwiki-20230401.db")
