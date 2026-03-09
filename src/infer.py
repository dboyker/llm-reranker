"""Inference script."""
import argparse
import os
from pathlib import Path

import dotenv
import numpy as np
import yaml
from datasets import Dataset, load_dataset
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


def infer(dataset: Dataset, pipe: pipeline, top_k: int, batch_size: int) -> np.ndarray:
    """Inference function."""
    pred_ids = np.zeros((len(dataset["prompt"]), top_k))
    pred_texts = np.zeros(len(dataset["prompt"]), dtype=object)

    messages = [[{"role": "user", "content": p}] for p in dataset["prompt"]]
    outputs = pipe(messages, batch_size=batch_size)

    for idx, out in enumerate(outputs):
        out_text = out[-1]["generated_text"][-1]["content"]
        out_id = parse_pred(out_text, top_k)

        pred_ids[idx, :] = out_id
        pred_texts[idx] = out_text

    return pred_ids


def main(config: dict, dataset_name: str, model_types: list[str]) -> None:
    dataset = load_dataset(config["data_path"])[dataset_name]

    model_to_hf_id = {"base": config["model_id"], "fine-tuned": config["fine_tuned_model_path"]}
    for m in model_types:
        print(m)
        hf_model_id = model_to_hf_id[m]
        out_name = "pred_" + hf_model_id + ".npy"
        out_name = out_name.replace("/", "_").replace("-", "_").replace("__", "_")
    
        # Pipeline
        pipe = pipeline(
            "text-generation",
            model=hf_model_id,
            max_new_tokens=20,
            do_sample=False,
            token=os.getenv("HF_TOKEN"),
            )

        # Predictions
        preds = infer(dataset, pipe, config["top_k"], config["pred_batch_size"])
        
        # Save
        np.save(Path(config["data_path"]) / out_name, preds)


if __name__ == "__main__":
    args = parse_args()
    with open("../config/config.yml") as f:
        config = yaml.safe_load(f)    
    main(config, args.dataset, args.model_type)
