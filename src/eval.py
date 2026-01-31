"""Module to evaluate the reranking."""
import ast
import os
import re
from pathlib import Path

import dotenv
import numpy as np
import torch
import yaml
from huggingface_hub import login
from tqdm import tqdm
from transformers import pipeline
from transformers.pipelines.text_generation import TextGenerationPipeline

import data_utils
import metrics

dotenv.load_dotenv()
DEVICE = 0 if torch.cuda.is_available() else -1


def llm_pred(pipe: TextGenerationPipeline, prompts: list[str], batch_size: int, top_k: int) -> np.ndarray:
    """Perform reranking using a LLM.

    @TODO: check that doc IDs exist.
    @TODO: parse output instead of simple format validation.
    @TODO: Use real batching function.
    """    
    messages = [[{"role": "user", "content": prompt}] for prompt in prompts]
    correct_output_pattern = fr"^\[\d+(?:, \d+){{{top_k-1}}}\]$"
    preds = []
    for i in tqdm(range(0, len(messages), batch_size)):
        batch = messages[i:i + batch_size]
        for out in pipe(batch, max_new_tokens=50):
            # Prediction
            pred_ids = out[-1]["generated_text"][-1]["content"]

            # Make sure answers are correctly formated
            if bool(re.fullmatch(correct_output_pattern, pred_ids)):
                pred_ids= [str(x) for x in ast.literal_eval(pred_ids)]
            else:
                pred_ids = [""] * top_k
            preds.append(pred_ids)
    return np.array(preds)


def score(name:str, relevant_docs: np.ndarray, preds: np.ndarray) -> None:
    score = metrics.mrr(relevant_docs=relevant_docs, top_k_docs=preds)
    print(f"MRR ({name}): {score:.2f}")


def evaluate(config):
    
    # Load data
    path =  Path(config["data_path"]) / "dataset_train_judged.jsonl"
    dataset = data_utils.load_dataset(path)  # @TODO: use HF dataset
    relevant_docs = np.array([d["relevant_id"] for d in dataset]).reshape(-1, 1)
    print(f"Top k used: {len(dataset[0]["top_k"])}")

    # Initial score: bm25
    top_k_docs = np.array([d["top_k"] for d in dataset])
    score(name="bm25", relevant_docs=relevant_docs, preds=top_k_docs)

    # Maximum possible score -> we use the ground truth to evaluate
    best_preds = np.array([ast.literal_eval(x["completion"]) for x in dataset])
    score(name="max", relevant_docs=relevant_docs, preds=best_preds)

    # Score with LLM
    pipe = pipeline("text-generation", model=config["model_id"], device=DEVICE, dtype=torch.bfloat16)
    llm_preds = llm_pred(
        pipe=pipe,
        prompts=[x["prompt"] for x in dataset],
        batch_size=config["pred_batch_size"],
        top_k=config["top_k"],
        )
    score(name="LLM", relevant_docs=relevant_docs, preds=llm_preds)

    # Score with LLM by replacing formatting errors with initial top_k
    formatting_error_count = 0
    for i in range(llm_preds.shape[0]):
        if llm_preds[i, 0] == "":
            formatting_error_count += 1
            llm_preds[i] = dataset[i]["top_k"]
    print(f"LLM formatting errors: {formatting_error_count / len(llm_preds):.0%}")
    score(name="LLM with formatting error fix", relevant_docs=relevant_docs, preds=llm_preds)


if __name__ == "__main__":
    # Config & setup
    login(token=os.getenv("HF_TOKEN"))
    with open("../config/config.yml") as f:
        config = yaml.safe_load(f)
    evaluate(config)
