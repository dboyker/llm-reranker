"""Module to evaluate the reranking."""
from pathlib import Path

import numpy as np
import yaml
from datasets import load_dataset

import metrics


def evaluate(data_path: str, dataset_name: str, eval_ks: list[int]) -> None:
    """Evaluation function.
    
    Scenarios:
    - BM25
    - Reranking
    - Max possible
    - LLM (base)
    - LLM (fine-tuned)
    """
    # Load data
    dataset = load_dataset(data_path)
    dataset = dataset[dataset_name]
    relevant_docs = np.array([d["relevant_doc_id"] for d in dataset]).reshape(-1, 1)
    print(f"Top k used: {len(dataset[0]["top_k_bm25"])}")

    # Initial score: bm25
    top_k_docs = np.array([np.array(d["top_k_bm25"]) for d in dataset])
    print(metrics.score(name="", relevant_docs=relevant_docs, preds=top_k_docs, at_k=eval_ks))

    # Initial score: BGE
    top_k_docs = np.array([np.array(d["top_k_bge"]) for d in dataset])
    print(metrics.score(name="", relevant_docs=relevant_docs, preds=top_k_docs, at_k=eval_ks))

    # Maximum possible score -> we use the ground truth to evaluate, if the ground truth is present in the initial retrieval (bm25)
    best_preds = [d["relevant_doc_id"] if d["relevant_doc_id"] in d["top_k_bm25"] else "" for d in dataset]
    best_preds = np.array(best_preds).reshape(-1, 1)
    print(metrics.score(name="", relevant_docs=relevant_docs, preds=best_preds, at_k=eval_ks))
  
    # Score with LLMs
    for path in Path(data_path).iterdir():
        if path.is_file() and path.stem.startswith("pred"):
            print(f"Evaluation predictions from {path}")
            
            # Evaluation
            preds = np.load(path, allow_pickle=True)
            print(metrics.score(name="", relevant_docs=relevant_docs, preds=preds, at_k=eval_ks))


if __name__ == "__main__":
    # Config & setup
    with open("../config/config.yml") as f:
        config = yaml.safe_load(f)
    evaluate(config["data_path"], "validation", eval_ks=config["eval_ks"])
