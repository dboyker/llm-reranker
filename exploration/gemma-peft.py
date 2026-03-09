"""gemma-3-1b-it fine-tuning using PEFT. Inspired from https://huggingface.co/blog/gemma-peft.

Deps: peft bitsandbytes trl python-dotenv
"""

import os

import dotenv
import torch
import transformers
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTTrainer

dotenv.load_dotenv("../.env")


MODEL_ID = "google/gemma-3-1b-it"


def formatting_func(example):
    text = f"Quote: {example['quote']}\nAuthor: {example['author']}<eos>"
    return text


def generate(model, tokenizer, text: str, device: str):
    inputs = tokenizer(text, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=20)
    return tokenizer.decode(outputs[0], skip_special_tokens=True) 


def train(model, tokenizer, dataset):
    lora_config = LoraConfig(
        r=8,
        target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        args=transformers.TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=2,
            max_steps=50,
            learning_rate=2e-4,
            fp16=False,
            logging_steps=10,
            output_dir="outputs",
            optim="paged_adamw_8bit"
        ),
        peft_config=lora_config,
        formatting_func=formatting_func,
        processing_class=tokenizer,
    )
    trainer.train()
    return model


if __name__ == "__main__":
    dataset = load_dataset("Abirate/english_quotes")

    # Model & tokenizer
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map={"": 0},
        token=os.environ['HF_TOKEN']
        )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=os.environ['HF_TOKEN'])

    # Dataset
    dataset = load_dataset("Abirate/english_quotes")

    # Example of generation BEFORE fine-tuning
    res = generate(model, tokenizer, "Quote: Imagination is more", device="cuda:0")
    print(res)

    # Fine-tuning
    model = train(model, tokenizer, dataset)

    # Example of generation AFTER fine-tuning
    res = generate(model, tokenizer, "Quote: Imagination is more", device="cuda:0")
    print(res)
    