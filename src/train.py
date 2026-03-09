!pip install trl
"""Training script."""
import os

import numpy as np
import time
import metrics
from infer import infer
import torch
import dotenv
import yaml
from datasets import load_dataset
from peft import LoraConfig
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainerCallback
from trl import SFTConfig, SFTTrainer

dotenv.load_dotenv()
SFT_CONFIG_ARGS = dict(
    completion_only_loss=True,  # Important to train only on the completion, not the entire prompt
    auto_find_batch_size=True,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    warmup_steps=2,
    max_steps=20,
    learning_rate=2e-4,
    fp16=False,
    logging_steps=5,
    output_dir="outputs",
    optim="paged_adamw_8bit",
    eval_strategy="steps",
    eval_steps=5,
    eval_on_start=True,
    )


class ValidationCallback(TrainerCallback):
    def __init__(self, eval_dataset, top_k, callback_steps):
        self.eval_dataset = eval_dataset
        self.top_k = top_k
        self.callback_steps = callback_steps
    
    def on_evaluate(self, args, state, control, **kwargs):
        """
        import torch
        model.eval()
        inputs = tokenizer("Write a haiku about gradient descent.", return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        model.train()
        """
        print("Validation callback")
        model = kwargs["model"]
        tokenizer = kwargs["processing_class"]
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=20, do_sample=False)
        pred_ids  = infer(dataset=self.eval_dataset, pipe=pipe, top_k=self.top_k, batch_size=1)
        relevant_docs = np.array([d["relevant_id"] for d in self.eval_dataset]).reshape(-1, 1)
        score = metrics.mrr(relevant_docs=relevant_docs, top_k_docs=pred_ids)
        print(score)


def train(model, tokenizer, dataset, top_k):
    """Training function: PEFT."""
    lora_config = LoraConfig(
        r=8,
        target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        args=SFTConfig(**SFT_CONFIG_ARGS),
        peft_config=lora_config,
        processing_class=tokenizer,
    )
    callback = ValidationCallback(eval_dataset=dataset["validation"], top_k=top_k, callback_steps=10)
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
    trainer = train(base_model, tokenizer, dataset, config["top_k"])

    # Save Lora
    lora_path = "/dbfs/llm_reranker_lora"
    trainer.model.save_pretrained(lora_path)
    tokenizer.save_pretrained(lora_path)

    # Save full model
    full_model_path = "/dbfs/llm_reranker"
    merged_model = trainer.model.merge_and_unload()
    merged_model.save_pretrained(full_model_path)
    tokenizer.save_pretrained(full_model_path)

    #model.save_pretrained("/dbfs/llm_reranker_lora")
    #full_model = PeftModel.from_pretrained(base_model, "/dbfs/llm_reranker_lora")
    #full_model = full_model.merge_and_unload()
    #full_model.save_pretrained("/dbfs/llm_reranker")
