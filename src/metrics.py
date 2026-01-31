import numpy as np


def mrr(relevant_docs: np.array, top_k_docs: np.array):
    matches = relevant_docs == top_k_docs
    first_relevant_idx = np.argmax(matches == 1, axis=1)
    has_relevant = np.any(matches == 1, axis=1)
    rr = np.zeros(len(matches))
    rr[has_relevant] = 1.0 / (first_relevant_idx[has_relevant] + 1)
    return rr.mean()
