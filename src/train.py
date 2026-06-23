#!pip install trl
"""Training script."""
import subprocess
subprocess.check_call(["pip", "install", "--upgrade", "transformers"])

import os

import dotenv
import numpy as np
import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainerCallback
from trl import SFTConfig, SFTTrainer

import metrics
from infer import infer

dotenv.load_dotenv()
SFT_CONFIG_ARGS = dict(
    completion_only_loss=True,  # Important to train only on the completion, not the entire prompt
    auto_find_batch_size=True,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    warmup_steps=2,
    max_steps=2,
    learning_rate=2e-4,
    fp16=False,  # @TODO
    logging_steps=5,
    output_dir="outputs",
    optim="paged_adamw_8bit",
    eval_on_start=True,
    #disable_tqdm=True,
    )


class ValidationCallback(TrainerCallback):

    def __init__(self, eval_dataset, callback_steps, eval_ks: list[int]):
        self.eval_dataset = eval_dataset
        self.callback_steps = callback_steps
        self.eval_ks = eval_ks
    
    def on_evaluate(self, args, state, control, **kwargs):
        print(f"Validation callback at step  {state.global_step}")
        model = kwargs["model"]
        tokenizer = kwargs["processing_class"]
        pred_ids  = infer(dataset=self.eval_dataset, model=model, tokenizer=tokenizer)
        relevant_docs = np.array([d["relevant_doc_id"] for d in self.eval_dataset]).reshape(-1, 1)
        score = metrics.score(name="", relevant_docs=relevant_docs, preds=pred_ids, at_k=self.eval_ks)
        print(score)


def train(model, tokenizer, dataset, eval_ks):
    """Training function: PEFT."""
    lora_config = LoraConfig(
        r=8,
        target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],  # No need since we use a custom validation callback
        args=SFTConfig(**SFT_CONFIG_ARGS),
        peft_config=lora_config,
        processing_class=tokenizer,
    )
    callback = ValidationCallback(eval_dataset=dataset["validation"], callback_steps=10, eval_ks=eval_ks)
    trainer.add_callback(callback)
    trainer.train()
    return trainer


if __name__ == "__main__":
    with open("../config/config.yml") as f:
        config = yaml.safe_load(f)
    print(config)

    # Dataset
    dataset = load_dataset(config["data_path"])

    # Model & tokenizer
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        config["model_id"],
        quantization_config=bnb_config,
        device_map="auto",
        token=os.environ['HF_TOKEN']
        )
    tokenizer = AutoTokenizer.from_pretrained(config["model_id"], token=os.environ['HF_TOKEN'])

    # Token count check
    tokenized_datasets = dataset.map(lambda x: tokenizer(x["prompt"]), batched=True)
    train_token_count = [len(x["input_ids"]) for x in tokenized_datasets["train"]]
    dev_token_count = [len(x["input_ids"]) for x in tokenized_datasets["validation"]]
    print(min(train_token_count), max(train_token_count))
    print(min(dev_token_count), max(dev_token_count))

    # Training
    trainer = train(base_model, tokenizer, dataset, config["eval_ks"])

    # Save Lora
    lora_path = "/dbfs/llm_reranker_lora"
    trainer.model.save_pretrained(lora_path)
    tokenizer.save_pretrained(lora_path)

    # Save full model
    full_model_path = "/dbfs/llm_reranker"
    merged_model = trainer.model.merge_and_unload()
    merged_model.save_pretrained(full_model_path)
    tokenizer.save_pretrained(full_model_path)
