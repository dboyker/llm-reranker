"""Module to build a reranking dataset."""
import json
import random
from collections import defaultdict
from pathlib import Path

import bm25s
import ir_datasets
import numpy as np
import Stemmer
import yaml

DATASET = "msmarco-passage"
SPLITS = ["train/judged", "dev/judged"]
STEMMER = Stemmer.Stemmer("english")
PROMPT_TEMPLATE = (
    "### Instructions ###\n"
    "You are an expert ranking assistant. Your task is to rerank the following "
    "list of documents according to their relevance to the given query."
    "Rank documents higher if they directly answer the query with clear, factual information."
    "Consider all items together (listwise) and return only the ids of the items, from most "
    "relevant to least relevant.\n"
    "- Do not include the item text or scores, only the IDs.\n"
    "- Return each ID exactly once\n" 
    "- Do not add or remove IDs\n\n"
    "### Query ###\n"
    "{query}\n\n"
    "### Documents ###\n"
    "{items}\n\n"
    "### Output format ###\n"
    "most_relevant_id, second_relevant_id, third_relevant_id, fourth_relevant_id, fifth_relevant_id\n\n"
)
RNG = random.Random(42)


def get_queries_and_qrels(dataset: ir_datasets.datasets.base.Dataset) -> tuple[dict, dict]:
    """From a ir_dataset, create and retur 2 dicts: one for queries, one for the qrels.
    
    Keep only the queries that have only one relevant document.
    """
    qrels = defaultdict(list)
    for q in dataset.qrels_iter():
        qrels[q.query_id].append(q.doc_id)
    qrels = {k: v[0] for k, v in qrels.items() if len(v) == 1}
    queries = {q.query_id: q.text for q in dataset.queries_iter() if q.query_id in qrels.keys()}
    return queries, qrels


def index_and_save_bm25_retriever(document_corpus: list[str], bm25_retriever_name: str) -> None:
    """Create a bm25 index based on a document corpus. Save the index locally."""
    corpus_tokens = bm25s.tokenize(document_corpus, stopwords="en", stemmer=STEMMER)
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens)
    retriever.save(bm25_retriever_name)


def compute_top_k(retriever: bm25s.BM25, queries: dict, doc_ids, top_k: int) -> dict:
    """Compute the top k closest documents from a series of queries.
    
    The documents are indexed in the bm25s retriever.
    
    The document ids (doc_ids) must be provided in the indexing order, so the function can return them, reranked.
    """
    query_tokens = bm25s.tokenize([v for _, v in queries.items()], stemmer=STEMMER)
    results, _ = retriever.retrieve(query_tokens, k=top_k)
    query_keys = list(queries.keys())
    top_k = {query_keys[qi]: [doc_ids[di] for di in r] for qi, r in enumerate(results)}
    return top_k


def get_prompt(query: str, top_k_ids: list[str], top_k_texts: list[str]):
    """Create a reranking prompt for a given query and its top k corresponding documents (as given by BM25)."""
    ordered_idx = np.argsort(top_k_ids)
    items = list(zip(top_k_ids, top_k_texts))
    ordered_items = [items[i] for i in ordered_idx]
    formatted_items = "\n".join(f"{i}. {text}" for i, text in ordered_items)
    prompt = PROMPT_TEMPLATE.format(query=query, items=formatted_items)
    return prompt


def build_list_of_prompts(queries: dict, docs: dict, qrels: dict, top_k: dict) -> list[dict]:
    """Build and return a dataset (list of dict) containing the reranking prompts."""
    dataset = []
    for q_id, query in queries.items():
        top_k_ids = top_k[q_id]  # Corresponding top_k
        top_k_texts = [docs[i] for i in top_k_ids]
        relevant_id = qrels[q_id]
        
        # Permutation to avoid "memorizing" a position pattern
        permutation_ids = np.random.permutation(len(top_k_ids)).tolist() 
        top_k_ids = [top_k_ids[i] for i in permutation_ids]
        top_k_texts = [top_k_texts[i] for i in permutation_ids]

        # Mapping + replace
        id_mapping = {k: v for k, v in zip(permutation_ids, top_k_ids)}
        if relevant_id not in top_k_ids:
            id_mapping[-1] = relevant_id
            relevant_id = -1
        else:
            relevant_id = permutation_ids[top_k_ids.index(relevant_id)]
        top_k_ids = permutation_ids
        
        # Build prompt and completion target for LLM
        prompt = get_prompt(query, top_k_ids, top_k_texts)
        completion = top_k_ids.copy()
        if relevant_id in completion:  # @TODO: only works if one best id
            completion.remove(relevant_id)
            completion.insert(0, relevant_id)
        completion = str(completion).replace("[", "").replace("]", "")
        entry = {"prompt": prompt, "completion": completion, "top_k": top_k_ids, "relevant_id": relevant_id, "id_mapping": id_mapping}
        print(prompt)
        print(completion)
        dataset.append(entry)
        print("-")
    return dataset


def build_dataset(doc_limit: int, bm25_retriever_name: str, top_k: int, split: str) -> list[dict]:
    """Build and return a reranking dataset for a given MSMarco split."""
    # Fetch data
    print("Fetch data")
    ds = ir_datasets.load(f"{DATASET}/{split}")
    queries, qrels = get_queries_and_qrels(dataset=ds)
    docs = {x.doc_id: x.text for x in ds.docs_iter()[:doc_limit]}

    # Freeze doc ordering
    doc_ids = list(docs.keys())
    doc_texts = [docs[d] for d in doc_ids]

    # Apply doc limit
    doc_ids_set = set(docs.keys())
    qrels = {k: v for k, v in qrels.items() if v in doc_ids_set}
    queries = {k: v for k, v in queries.items() if k in qrels.keys()}
    
    # Index bm25 retriever
    print("Index and save BM25")
    index_and_save_bm25_retriever(document_corpus=doc_texts, bm25_retriever_name=bm25_retriever_name)
    retriever = bm25s.BM25.load("index_bm25", load_corpus=False)

    print("Retrieve top_k with BM25")
    top_k = compute_top_k(retriever, queries=queries, doc_ids=doc_ids, top_k=top_k)

    # Build dataset
    print("Build dataset")
    dataset = build_list_of_prompts(queries, docs, qrels, top_k)
    return dataset


def main():
    """Fetch MSMarco data -> BM25 → sanity check → build listwise reranking dataset."""
    # Open Config
    with open("../config/config.yml") as f:
        config = yaml.safe_load(f)

    for split in SPLITS:
        print(f"Processing split {split}")

        dataset = build_dataset(
            doc_limit=config["doc_limit"],
            bm25_retriever_name=config["bm25_retriever_name"],
            top_k=config["top_k"],
            split=split
            )
        
        # Save results
        print("Save dataset")
        data_path = Path(config["data_path"])
        data_path.mkdir(exist_ok=True)
        with open(data_path / f"dataset_{split.replace("/", "_")}.jsonl", "w") as f:
            for entry in dataset:
                json.dump(entry, f)
                f.write("\n")


if __name__ == "__main__":
    main()
