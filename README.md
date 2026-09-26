# Conformal Factuality Control for Multi-Hop Retrieval-Augmented Generation

This repository contains the implementation used for the paper:

**Conformal Factuality Control for Multi-Hop Retrieval-Augmented Generation**

**Muhammad Aimal Rehman** and **Chi-Kuang Yeh**  
Department of Mathematics & Statistics, Georgia State University

The project studies claim-level conformal factuality filtering for retrieval-augmented generation (RAG), with a particular focus on **multi-hop RAG**. The goal is to improve response-level factual reliability by selectively retaining generated claims that satisfy a split-conformal threshold calibrated from factuality-labeled examples.

> **Paper:** arXiv link will be added after submission.

---

## Overview

The pipeline supports:

- **Plain / single-hop RAG** for reference validation
- **Multi-hop RAG** with iterative query rewriting and evidence accumulation
- Atomic-claim decomposition of generated responses
- Claim-level factuality verification
- Retrieval-based claim scoring
- Split-conformal calibration
- Claim filtering at user-specified reliability targets
- Reliability–selectivity evaluation, including:
  - response-level full support
  - non-empty response rate
  - claim retention rate
  - supported-claim rate
  - conditional full-support rate

The main paper evaluates the framework on:

- **HotpotQA**
- **Natural Questions (NQ)**
- **TriviaQA**

using:

- **Meta Llama 3.1 8B Instruct**
- **OpenAI GPT-4o-mini**

The public code in this repository corresponds to the Plain RAG and Multi-Hop RAG experiments reported in the paper. Agentic-RAG experiments are outside the scope of this release and are not part of the paper.

---

## Method at a Glance

For a question \(x\), the Multi-Hop RAG pipeline performs iterative retrieval and query rewriting for up to a fixed number of hops. Retrieved evidence is accumulated across hops and used to generate a response.

The generated response is then decomposed into atomic claims:

\[
\hat{y} \rightarrow \{c_1, c_2, \ldots, c_p\}.
\]

Each claim receives a retrieval-based relevance score. A split-conformal calibration set with factuality annotations determines a threshold \(\hat{q}_\alpha\). At test time, claims that do not satisfy the calibrated criterion are filtered from the response.

This produces a reliability–informativeness trade-off: stricter targets generally increase response-level factual support while decreasing claim retention and the fraction of non-empty responses.

---

## Main Experimental Configuration

The paper uses the following core settings across the controlled experiments:

| Component | Setting |
|---|---|
| Retrieval | FAISS-based dense retrieval |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` |
| Retrieved documents per hop | 10 |
| Retrieval similarity threshold | 0.3 |
| Maximum multi-hop depth | 3 |
| Evidence handling | Accumulate documents across hops |
| Early stopping | Stop when a hop retrieves no new evidence |
| Query rewriting temperature | 0 |
| Conformal method | Split conformal |
| Claim-score aggregation | Maximum |
| Scoring combination | Product |
| Conformal parameter | \(a=1\) |
| Evaluation subset | 60 questions per benchmark |
| Repeated conformal splits | 100 seeded runs |

The experiments evaluate multiple reliability targets, including 80%, 85%, 90%, 92.5%, and 95%.

---

## Key Result

Across the six main Multi-Hop RAG configurations in the paper, the unfiltered baseline response-level full-support rate ranges from **55.60% to 76.03%**.

At the **95% conformal target**, response-level full support increases to **95.80%–97.20%**, with the expected cost in selectivity:

- claim retention: **4.41%–31.09%**
- non-empty response rate: **9.70%–51.40%**

These results are intended to be interpreted jointly: conformal filtering improves factual reliability by selectively retaining supported content rather than by correcting unsupported generations.

---

## Repository Structure

```text
.
├── conf/                       # Experiment and dataset configuration
├── data/                       # Dataset inputs and processed artifacts
├── src/
│   ├── calibration/            # Split-conformal calibration logic
│   ├── common/                 # Shared utilities, embeddings, FAISS, configuration
│   ├── data_processor/         # Query/document preprocessing
│   ├── dataloader/             # Dataset loading utilities
│   ├── rag/                    # RAG and retrieval components
│   ├── subclaim_processor/     # Claim generation, scoring, and verification
│   └── utils/                  # Supporting utilities
├── main.py                     # Main pipeline entry point
├── requirement.txt             # Base Python dependencies
├── requirements-colab.txt      # Google Colab environment dependencies
├── requirements-macos.txt      # macOS-specific environment dependencies
├── LICENSE
└── README.md
```

The repository also includes local Hugging Face embedding support for experiments that do not rely on API-hosted embedding models.

---

## Installation

The project was developed with Python 3.11-compatible environments.

Clone the repository:

```bash
git clone https://github.com/aimalrehman92/conformal-rag.git
cd conformal-rag
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

For a standard local installation:

```bash
pip install -r requirement.txt
```

For the macOS research environment:

```bash
pip install -r requirements-macos.txt
```

For Google Colab:

```bash
pip install -r requirements-colab.txt
```

---

## Configuration

Experiment settings are controlled through files under `conf/`.

Before running an experiment, verify the relevant configuration for:

- dataset
- generation model
- embedding model
- retrieval settings
- conformal parameters
- input/output paths
- API-backed model credentials, when applicable

API keys or other secrets should be supplied through environment variables or a local `.env` file and should **not** be committed to the repository.

---

## Running the Pipeline

The main entry point is:

```bash
python main.py --config conf/config.yaml --dataset <dataset_name> --query_size <N>
```

For example:

```bash
python main.py \
  --config conf/config.yaml \
  --dataset hotpot_qa \
  --query_size 60
```

Experiment-specific configuration files under `conf/` should be used to reproduce the Plain RAG or Multi-Hop RAG settings for the desired model and dataset.

Because some stages use API-hosted language models, exact reproduction may require the corresponding provider credentials.

---

## Experimental Protocol

For the paper experiments:

1. A fixed 60-question subset is used for each benchmark.
2. The RAG system generates responses from retrieved evidence.
3. Responses are decomposed into atomic claims.
4. Claims are assigned retrieval-based scores.
5. Claim factuality is verified against the available evidence.
6. Split-conformal calibration determines the filtering threshold for each target level.
7. The same generated artifacts are reused across 100 seeded conformal splits.
8. Reliability and selectivity metrics are aggregated across runs.

The Plain RAG + Llama 3.1 8B + HotpotQA experiment is used as a reference validation setting. The principal results in the paper are the six Multi-Hop RAG configurations spanning two model families and three datasets.

---

## Experiment Notebooks

The Google Colab notebooks used to execute the paper experiments will be added to the public repository separately.

They provide the paper-specific orchestration for:

- Llama 3.1 8B
- GPT-4o-mini
- HotpotQA
- Natural Questions
- TriviaQA
- repeated split-conformal evaluation
- reliability–selectivity summaries

---

## Important Interpretation

The conformal procedure is a **selective filtering mechanism**, not a factuality-correction mechanism.

A higher target may produce a highly reliable retained response while also removing many claims or, in some cases, producing an empty response. For that reason, response-level full support should always be interpreted together with non-empty response rate and claim retention.

The formal conformal guarantee also depends on the assumptions described in the paper, including the calibration/test exchangeability conditions and the quality of the factuality annotations used during calibration.

---

## Acknowledgment of Upstream Code

This repository was originally forked from:

**layer6ai-labs/conformal-rag**  
https://github.com/layer6ai-labs/conformal-rag

The upstream repository accompanied:

> Naihe Feng, Yi Sui, Shiyi Hou, Jesse C. Cresswell, and Ga Wu.  
> *Response Quality Assessment for Retrieval-Augmented Generation via Conditional Conformal Factuality.*  
> SIGIR 2025.

The present repository extends that codebase for the research described in **Conformal Factuality Control for Multi-Hop Retrieval-Augmented Generation**, including the multi-hop retrieval framework, experiment configurations, local embedding support, and the evaluation setup used in our study.

Please cite the appropriate upstream work when using components originating from the original repository.

---

## Citation

If you use this repository, please cite our paper. The final arXiv BibTeX entry will be added after the arXiv submission is available.

For now:

```bibtex
@article{rehman2026conformal,
  title   = {Conformal Factuality Control for Multi-Hop Retrieval-Augmented Generation},
  author  = {Rehman, Muhammad Aimal and Yeh, Chi-Kuang},
  year    = {2026},
  note    = {Preprint}
}
```

For the upstream conformal-RAG implementation, please also cite:

```bibtex
@inproceedings{feng2025response,
  title     = {Response Quality Assessment for Retrieval-Augmented Generation via Conditional Conformal Factuality},
  author    = {Feng, Naihe and Sui, Yi and Hou, Shiyi and Cresswell, Jesse C. and Wu, Ga},
  booktitle = {Proceedings of the 48th International ACM SIGIR Conference on Research and Development in Information Retrieval},
  pages     = {2832--2836},
  year      = {2025},
  series    = {SIGIR '25},
  doi       = {10.1145/3726302.3730244}
}
```

---

## License

This repository retains the MIT License from the upstream project. See [`LICENSE`](LICENSE) for details.
