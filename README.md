<p align="center">
<a href="https://layer6.ai/"><img src="https://github.com/layer6ai-labs/DropoutNet/blob/master/logs/logobox.jpg" width="180"></a>
</p>

# Response Quality Assessment for Retrieval-Augmented Generation via Conditional Conformal Factuality

This repository contains code and resources for our paper "Response Quality Assessment for Retrieval-Augmented
Generation via Conditional Conformal Factuality" published at SIGIR 2025. [Link to Paper](https://dl.acm.org/doi/10.1145/3726302.3730244).


## Structure

```bash
.
├── conf/                  # Configuration file location
├── data/
│   ├── out/               # Final subclaims with scores (follows `subclaims_schema`)
│   ├── processed/         # Standardized test data (follows `base_schema`)
│   └── raw/               # Original raw data from source (unstructured)
├── index_store/           # Chunked documents and embeddings
├── logs/                  # Config and logs in format `run_{data}_{run_id}`
├── src/
│   ├── calibration/       # Conformal prediction calibration logic
│   ├── common/            # Reusable components (e.g., config manager, FAISS vector DB manager)
│   ├── data_processor/    # Converts raw QA data to standardized format (see `data/processed`)
│   ├── dataloader/        # Loads data from source datasets (e.g., AkariASAI/PopQA, KILT benchmark)
│   ├── rag/               # RAG system components for document retrieval
│   ├── subclaim_processor/# Generates, scores, and annotates subclaims for different datasets
│   └── utils/             # Miscellaneous utilities
```


## Data
### Query Data
This repository includes the following query datasets:
- [FactScore](https://github.com/shmsw25/FActScore)
- [PopQA](https://huggingface.co/datasets/akariasai/PopQA)
- [HotpotQA](https://huggingface.co/datasets/hotpotqa/hotpot_qa)
- [MedLFQA](https://github.com/dmis-lab/OLAPH/tree/main/MedLFQA)
- [MedLFQAv2](https://github.com/jjcherian/conformal-safety/tree/main/data/MedLFQAv2)

### Wikipedia Extraction
We utilize Wikipedia dumps for knowledge retrieval (source: https://github.com/shmsw25/FActScore). This following data file is not included in this repo. Please download it manually and put it under
`\data\raw` folder in order to generate reference doucuments for wiki-based queries (PopQA and HotpotQA)
- [enwiki-20230401.db](https://drive.google.com/file/d/1mekls6OGOKLmt7gYtHs0WGf5oTamTNat/view?usp=drive_link)

## Usage
This project is built on python3.11
First, build a python environment using [requirements.txt](requirements.txt).
Then run the pipeline:
```python
python main.py --config conf/config.yaml --dataset fact_score --query_size 500
```
The code uses only one dataset at a time in one thread.
Avaliable datasets currently are:
["fact_score", "hotpot_qa", "pop_qa", "medlf_qa"]

## Conditional Conformal
This repo only supports conditional conformal for the `medlf_qa` dataset. By the default config is in `/conf/dataset_config.yaml`.
Conditioning is activated in the code with `medlf_qa.is_grouped = true` while other datasets have this set to false.
The factuality results will be written to different csv files under the `result/${datetime}_${run_id}` folder, with files named by each different pre-defined group.

## Pregenerated Metadata
To avoid API-based LLM costs, one can choose to use pre-generated metadata for verifying the conformal prediction part. 
You can get required metadata here: https://drive.google.com/drive/folders/1aLbHxS6V1ipMH8FpVCxKmr8oMYfqmRgb?usp=drive_link


## Citing

If you use any part of this repository in your research, please cite the associated paper with the following bibtex entry:


```
@inproceedings{feng2025response,
  author = {Feng, Naihe and Sui, Yi and Hou, Shiyi and Cresswell, Jesse C. and Wu, Ga},
  title = {Response Quality Assessment for Retrieval-Augmented Generation via Conditional Conformal Factuality},
  year = {2025},
  isbn = {9798400715921},
  doi = {10.1145/3726302.3730244},
  booktitle = {Proceedings of the 48th International ACM SIGIR Conference on Research and Development in Information Retrieval},
  series = {SIGIR '25}
}
```

## License

This project is licensed under the [MIT License](https://opensource.org/license/mit).
