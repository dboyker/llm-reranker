"""Module to evaluate the reranking."""
import ast
from pathlib import Path

import numpy as np
import yaml

import data_utils
import metrics


def score(name:str, relevant_docs: np.ndarray, preds: np.ndarray) -> None:
    score = metrics.mrr(relevant_docs=relevant_docs, top_k_docs=preds)
    print(f"MRR ({name}): {score:.2f}")


def evaluate(config):
    
    # Load data
    path =  Path(config["data_path"]) / "dataset_dev_judged.jsonl"
    dataset = data_utils.load_dataset(path)  # @TODO: use HF dataset
    relevant_docs = np.array([d["relevant_id"] for d in dataset]).reshape(-1, 1)
    print(f"Top k used: {len(dataset[0]["top_k"])}")

    # Initial score: bm25
    top_k_docs = np.array([d["top_k"] for d in dataset])
    score(name="bm25", relevant_docs=relevant_docs, preds=top_k_docs)

    # Maximum possible score -> we use the ground truth to evaluate
    best_preds = np.array([ast.literal_eval(x["completion"]) for x in dataset])
    score(name="max", relevant_docs=relevant_docs, preds=best_preds)

    # Score with LLMs
    for path in Path(config["data_path"]).iterdir():
        if path.is_file() and path.stem.startswith("pred"):
            preds = np.load(path)
            
            # Evaluation
            score(name=f"LLM {path.stem}", relevant_docs=relevant_docs, preds=preds)

            # Formatting errors
            formatting_error_count = 0
            for i in range(preds.shape[0]):
                if np.any(preds[i, :] == -9):
                    formatting_error_count += 1
            print(f"LLM formatting errors: {formatting_error_count / len(preds):.0%}")


if __name__ == "__main__":
    # Config & setup
    with open("../config/config.yml") as f:
        config = yaml.safe_load(f)
    evaluate(config)
