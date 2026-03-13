"""Inference script."""
import argparse
from pathlib import Path

import dotenv
import numpy as np
import torch
import yaml
from datasets import Dataset, load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, BitsAndBytesConfig, Gemma3ForCausalLM

dotenv.load_dotenv()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-type",
        nargs="+",
        choices=["base", "fine-tuned"],
        default=["base"],
        help="Model(s) to use."
    )
    parser.add_argument(
        "--dataset",
        choices=["train", "validation"],
        default="validation",
        help="Dataset to use."
    )
    return parser.parse_args()


def infer(dataset: Dataset, model, tokenizer, max_new_token: int):
    # Setup results
    top_k_reranking = len([k for k in dataset[0]["id_mapping"] if k != "-1"])
    pred_ids = np.zeros(shape=(len(dataset), top_k_reranking), dtype=object)

    # Inference ofor each dataset entry
    for i, entry in enumerate(tqdm(dataset)):
        messages = [[{"role": "user", "content": [{"type": "text", "text": entry["prompt"]}]}]]  # @TODO: add system prompt?
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_token,
                return_dict_in_generate=True,
                output_scores=True,
                do_sample=False)  # output_logits=True?
            
        # Extract ranking based on logits (of the digits)
        digit_ids = [tokenizer.encode(str(i), add_special_tokens=False)[0] for i in range(top_k_reranking)]
        logits = outputs.scores[0][0]
        probs = torch.softmax(logits, dim=-1)
        sorted_ids = np.argsort([probs[tid].item() for tid in digit_ids])[::-1]
        
        # Sorted ids -> actual ids
        id_mapping = entry["id_mapping"]
        ids = [id_mapping[str(i)] for i in sorted_ids]
        pred_ids[i, :] = ids
    return pred_ids


def main(config: dict, dataset_name: str, model_types: list[str]) -> None:
    dataset = load_dataset(config["data_path"])[dataset_name]
    model_to_hf_id = {"base": config["model_id"], "fine-tuned": config["fine_tuned_model_path"]}

    # Inference over all the specified model types
    for m_type in model_types:
        
        # Select model ID and prepare results file
        print(f"Inference using: {m_type}")
        hf_model_id = model_to_hf_id[m_type]
        out_name = "pred_" + hf_model_id + ".npy"
        out_name = out_name.replace("/", "_").replace("-", "_").replace("__", "_")
    
        # Inference
        quant_config = BitsAndBytesConfig(load_in_8bit=True)
        model = Gemma3ForCausalLM.from_pretrained(hf_model_id, quantization_config=quant_config)
        model = model.eval()
        tokenizer = AutoTokenizer.from_pretrained(hf_model_id)    
        preds = infer(dataset, model, tokenizer, max_new_token=config["pred_max_token"])
        
        # Save
        np.save(Path(config["data_path"]) / out_name, preds)


if __name__ == "__main__":
    args = parse_args()
    with open("../config/config.yml") as f:
        config = yaml.safe_load(f)    
    main(config, args.dataset, args.model_type)
