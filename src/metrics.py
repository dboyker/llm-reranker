import numpy as np


def mrr(relevant_docs: np.array, top_k_docs: np.array) -> float:
    matches = relevant_docs == top_k_docs
    first_relevant_idx = np.argmax(matches == 1, axis=1)
    has_relevant = np.any(matches == 1, axis=1)
    rr = np.zeros(len(matches))
    rr[has_relevant] = 1.0 / (first_relevant_idx[has_relevant] + 1)
    return rr.mean()


def accuracy(relevant_docs: np.array, top_k_docs: np.array) -> float:
    return (relevant_docs == top_k_docs).any(axis=1).mean()


def score(name:str, relevant_docs: np.ndarray, preds: np.ndarray, at_k: list[int]) -> dict:
    """Scoring function.

    Implemented:
    - MRR@k
    - Accuracy@k
    """
    res = {}
    for k in at_k:
        res[f"{name}_mrr@{k}"] = mrr(relevant_docs=relevant_docs, top_k_docs=preds[:, :k])
        res[f"{name}_accuracy@{k}"] = accuracy(relevant_docs=relevant_docs, top_k_docs=preds[:, :k])
    return res