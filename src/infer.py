"""Inference script."""
import argparse
import os
from pathlib import Path

import dotenv
import numpy as np
import yaml
from datasets import load_dataset
from transformers import pipeline

dotenv.load_dotenv()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-type",
        nargs="+",
        choices=["base", "fine-tuned"],
        default=["base", "fine-tuned"],
        help="Model(s) to use."
    )
    parser.add_argument(
        "--dataset",
        choices=["train", "validation"],
        default="validation",
        help="Dataset to use."
    )
    return parser.parse_args()


def parse_pred(pred: str, top_k: int) -> list[str]:
    # Delete all elements that are not numerical or ","
    pred = "".join(c for c in pred if (c.isdigit()) or (c == ","))
    pred_ids = pred.lstrip(",").split(",")
    pred_ids = [x for x in pred_ids if x != ""]
    # We make sure the size of the pred is top_k: pad or cut
    pred_ids += [-9] * top_k
    pred_ids = pred_ids[:top_k]
    pred_ids = [int(x) for x in pred_ids]
    return pred_ids


def infer(dataset, pipe) -> np.ndarray:
    # @TODO: implement batching
    preds = []
    for prompt, completion in zip(dataset["prompt"], dataset["completion"]):
        messages = [{"role": "user", "content": prompt}]
        out = pipe(messages, use_cache=False)  # @TODO: investigate why use_cache is necessary to avoid crashing here
        pred = out[-1]["generated_text"][-1]["content"]
        pred_ids = parse_pred(pred, config["top_k"])
        print(pred_ids, completion)
        preds.append(pred_ids)
    return np.array(preds)


def main(config, dataset_name, model_types) -> None:
    dataset = load_dataset(config["data_path"])[dataset_name]

    model_to_hf_id = {"base": config["model_id"], "fine-tuned": config["fine_tuned_model_path"]}
    for m in model_types:
        hf_model_id = model_to_hf_id[m]
        out_name = "pred_" + hf_model_id + ".npy"
        out_name = out_name.replace("/", "_").replace("-", "_")
    
        # Pipeline
        pipe = pipeline(
            "text-generation",
            model=hf_model_id,
            max_new_tokens=20,
            do_sample=False,
            token=os.getenv("HF_TOKEN"),
            )

        # Predictions
        preds = infer(dataset, pipe)
        
        # Save
        np.save(Path(config["data_path"]) / out_name, preds)


if __name__ == "__main__":
    args = parse_args()
    with open("../config/config.yml") as f:
        config = yaml.safe_load(f)    
    main(config, args.dataset, args.model_type)
