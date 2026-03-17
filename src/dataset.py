"""Module to build a reranking dataset."""
import json
import random
from collections import defaultdict
from pathlib import Path

import ir_datasets
import torch
import yaml
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

DATASET = "wikir/en1k"
SPLITS = ["training", "validation"]
PROMPT_TEMPLATE = (
    "### Instructions ###\n"
    "You are an expert ranking assistant. Your task is to find the most relevant document given a query.\n"
    "- Before selecting the best document, rate each document’s relevance on a scale from 0 to 10, then choose the most relevant document.\n"
    "- return only the ID of the most relevant item.\n"
    "- Do not include the item text or scores, only the ID.\n\n"
    "### Query ###\n"
    "{query}\n\n"
    "### Documents ###\n"
    "{items}\n\n"
    )
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RERANKING_MODEL_ID = "BAAI/bge-reranker-base"
rng = random.Random(42)


def get_prompt(query: str, top_k_ids: list[str], top_k_texts: list[str]):
    """Create a reranking prompt for a given query and its top k corresponding documents (as given by BM25)."""
    formatted_items = "\n\n".join(f"[{i}] {text}" for i, text in list(zip(top_k_ids, top_k_texts)))
    prompt = PROMPT_TEMPLATE.format(query=query, items=formatted_items)
    return prompt


def build_list_of_prompts(queries: dict, docs: dict, qrels: dict, scoreddocs: dict, reranked_scoreddocs: dict, reranking_top_k: int) -> list[dict]:
    """Build and return a dataset (list of dict) containing the reranking prompts."""
    dataset = []
    for q_id, doc_ids in qrels.items():
        q_text = queries[q_id]
        d_id = doc_ids[0]
        if q_id not in reranked_scoreddocs:  # Should be removed
            continue
        reranked_ids = reranked_scoreddocs[q_id][:reranking_top_k]
        rng.shuffle(reranked_ids)  # We shuffle the list to avoid the model to learn the order
        reranked_docs = [docs[i] for i in reranked_ids]
        id_mapping = {str(i): idx for i, idx in enumerate(reranked_ids)}
        if d_id not in reranked_ids:
            id_mapping["-"] = d_id
        inv_id_mapping = {v: k for k, v in id_mapping.items()}
        completion = inv_id_mapping[d_id]  # ID to be returned by LLM
        prompt = get_prompt(q_text, id_mapping.keys(), reranked_docs)
        entry = {
            "prompt": prompt,
            "completion": completion,
            "id_mapping": id_mapping,
            "relevant_doc_id": d_id,
            "top_k_bm25": scoreddocs[q_id],
            "top_k_bge": reranked_scoreddocs[q_id],
            }
        dataset.append(entry)
    return dataset


def load_ir_dataset(split):
    """Load a dataset using ir_datasets. Return the different components of the dataset."""
    ds = ir_datasets.load(f"{DATASET}/{split}")
    qrels = defaultdict(list)
    for q in ds.qrels_iter():
        if q.relevance != 2:
            continue
        qrels[q.query_id].append(q.doc_id)
    queries = {q.query_id: q.text for q in ds.queries_iter()}
    docs = {x.doc_id: x.text for x in ds.docs_iter()}
    scoreddocs = defaultdict(list)
    for p in ds.scoreddocs_iter():
        scoreddocs[p.query_id].append(p.doc_id)
    return queries, docs, qrels, scoreddocs


def rerank_documents(queries: dict, docs: dict, scoreddocs: dict) -> dict:
    """Rerank the initial ranking provided in scoreddocs.
    
    The scoreddocs results are the one from BM25.
    """
    print("Reranking")
    tokenizer = AutoTokenizer.from_pretrained(RERANKING_MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(RERANKING_MODEL_ID).to(DEVICE)
    model.eval()
    reranked_scoreddocs = {}
    for query_id, doc_ids in tqdm(list(scoreddocs.items())[:1000]):
        pairs = [[queries[query_id], docs[d_id]] for d_id in doc_ids]
        with torch.no_grad():
            inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512).to(DEVICE)
            scores = model(**inputs, return_dict=True).logits.view(-1, ).float()
            sorted_ids = torch.argsort(scores, descending=True).cpu().numpy()
            bm25_results = scoreddocs[query_id]
            reranked_scoreddocs[query_id] = [bm25_results[i] for i in sorted_ids]
    return reranked_scoreddocs


def build_dataset(split: str, reranking_top_k: int) -> list[dict]:
    """Build and return a reranking dataset for a given MSMarco split."""
    # Fetch data
    print("Fetch data")
    queries, docs, qrels, scoreddocs = load_ir_dataset(split)

    # Apply reranking
    reranked_scoreddocs = rerank_documents(queries, docs, scoreddocs)

    # Build dataset
    print("Build dataset")
    dataset = build_list_of_prompts(queries, docs, qrels, scoreddocs, reranked_scoreddocs, reranking_top_k)
    return dataset


def main():
    # Open Config
    with open("../config/config.yml") as f:
        config = yaml.safe_load(f)

    for split in SPLITS:
        print(f"Processing split {split}")
        dataset = build_dataset(split=split, reranking_top_k=config["reranking_top_k"])
        
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
