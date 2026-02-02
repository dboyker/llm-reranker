# Listwise LLM Reranker

A Python project that trains a Large Language Model (LLM) to perform **listwise reranking** for information retrieval.

Usually, reranking tasks are performed by cross-encoder, so why use an LLM?

Compared to traditional cross-encoder rerankers, LLMs offer several advantages:

- **Listwise reasoning**: LLMs can consider the entire candidate list jointly, enabling true listwise ranking instead of scoring documents independently.
- **Explainability**: LLMs can justify their reranking.
- **Task adaptability**: The same LLM can be adapted to different domains or ranking criteria (e.g. factuality, diversity, recency).

## Overview
The pipeline:
1. Build a reranking dataset using **BM25** as the first-stage retriever  
2. **Fine-tune** an LLM to rerank candidate documents listwise  
3. **Evaluate** reranking performance with standard IR metrics  

## Setup

### Requirements

- Python 3.12+
- The project can be installed and run using [uv](https://github.com/astral-sh/uv).
- Other dependancies are specified in `pyproject.toml`

### Installation
Dependancies can be installed with:
```
uv sync
```

## Usage

### Configuration

### Build dataset (with BM25)
```
cd src
uv run dataset.py
```

### Fine-tune the LLM
```
cd src
uv run train.py
```

### Evaluation
```
cd src
uv run eval.py
```

## Notes & Next steps

- The queries from the original MSMarco dataset that are used in this project are the one which have only one corresponding document. This could change in the future.
- Implement logging system.