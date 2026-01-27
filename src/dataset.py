"""Module to build a reranking dataset."""
import json
import random
from collections import defaultdict
from pathlib import Path

import bm25s
import ir_datasets
import Stemmer
import yaml

DATASET = "msmarco-passage"
SPLITS = ["train/judged", "dev/judged"]
STEMMER = Stemmer.Stemmer("english")
PROMPT_TEMPLATE = (
    "Instruction:\n"
    "You are an expert ranking assistant. Your task is to rerank the following "
    "list of items according to their relevance to the given query. Consider all "
    "items together (listwise) and return only the ids of the items, from most "
    "relevant to least relevant."
    "- Do not include the item text or scores, only the IDs."
    "- Return each ID exactly once" 
    "- Do not add or remove IDs\n\n"
    "Query:\n"
    "{query}\n\n"
    "Items to rank:\n"
    "{items}\n\n"
    "Output Format:\n"
    "[id1, id2, id3, ..., idN]\n\n"
    "Example:\n"
    "If item with id 24 is most relevant, item with id 13 second, and item with id 31 least, return:\n"
    "[24, 13, 31]"
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
    items = list(zip(top_k_ids, top_k_texts))
    RNG.shuffle(items)  # We shuffle the candidates so the model does not memorize order
    formatted_items = "\n\n".join(f"ID: {i}. {text}" for i, text in items)
    prompt = PROMPT_TEMPLATE.format(query=query, items=formatted_items)
    return prompt


def build_list_of_prompts(queries: dict, docs: dict, qrels: dict, top_k: dict) -> list[dict]:
    """Build and return a dataset (list of dict) containing the reranking prompts."""
    dataset = []
    for q_id, query in queries.items():
        top_k_ids = top_k[q_id]  # Corresponding top_k
        top_k_texts = [docs[i] for i in top_k_ids]
        prompt = get_prompt(query, top_k_ids, top_k_texts)
        relevant_id = qrels[q_id]
        best_ranking_ids = top_k_ids.copy()
        if relevant_id in best_ranking_ids:  # @TODO: only works if one best id
            best_ranking_ids.remove(relevant_id)
            best_ranking_ids.insert(0, relevant_id)
        dataset.append({"prompt": prompt, "completion": str(best_ranking_ids), "top_k": top_k_ids, "relevant_id": relevant_id})
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