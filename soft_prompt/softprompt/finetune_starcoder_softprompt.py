import torch
import sys
import os
import pickle as pkl
import torch.distributed
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    default_data_collator,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from trl import SFTTrainer
from tqdm import tqdm
import json
import argparse
from datasets import load_dataset
import numpy as np
import sys

sys.path.append("..")
from support.config import hidden_dim
from dataloaders.datadealer import DataDealer
from support.Search_with_CodeT5 import construct_faiss_index, search_with_faiss


parser = argparse.ArgumentParser()
parser.add_argument("--log", type=str, default="logs")
parser.add_argument("--wandb", action="store_true", default=False)
# Path
parser.add_argument("--root_path", type=str, default="../../train_data/")
parser.add_argument(
    "--model_path",
    type=str,
    default="/ext0/hcchai/codemate/starcoder/starcoder2-7b",
)
parser.add_argument("--model", type=str, default="starcoder")
parser.add_argument("-d", "--dataset", type=str, default="CodeContest")
parser.add_argument("-l", "--language", type=str, default="c++")
parser.add_argument("--eval_steps", type=int, default=200)
parser.add_argument("--save_steps", type=int, default=200)
parser.add_argument("--lr_scheduler_type", type=str, default="linear")
parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
parser.add_argument("--total_batch_size", type=int, default=256)
parser.add_argument("--train_size", type=int, default=10609)
parser.add_argument("--val_size", type=int, default=1000)
parser.add_argument("--resume_from_checkpoint", type=str, default=None)
parser.add_argument("--lora_remote_checkpoint", type=str, default=None)
parser.add_argument("--ignore_data_skip", type=str, default="False")
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--wd", type=float, default=0)
parser.add_argument("--use_lora", type=int, default=1)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--epochs", type=int, default=1)
parser.add_argument("--cutoff_len", type=int, default=2048)
parser.add_argument("--load_in_8bit", action="store_true", help="Load model 8 bit.")
parser.add_argument("--extracted_query", action="store_true")
parser.add_argument("--extracted_prompt", action="store_true")
parser.add_argument(
    "-r", "--retrieval_method", type=str, default="NL-CL", choices=["NL-CL", "NL-NL"]
)

args = parser.parse_args()
datadealer = DataDealer(
    dataset=args.dataset,
    split="train",
    extracted=args.extracted_query,
    softprompt=True,
    cutofflen=3000,
    retrieval_method=args.retrieval_method,
)

args.output_path = f"../trained_models/{args.dataset}/{args.language}/{args.model_path.split('/')[-1]}_graphrag{datadealer.extracted_token}_{args.extracted_prompt}_{args.retrieval_method}/"
if not os.path.exists(args.output_path):
    os.makedirs(args.output_path)
print(f"Modlel will be stored at {args.output_path}")


# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(args.model_path)
tokenizer.padding_side = "right"
tokenizer.pad_token = tokenizer.eos_token

MICRO_BATCH_SIZE = args.per_device_eval_batch_size
BATCH_SIZE = min(args.total_batch_size, args.train_size)

GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
EPOCHS = args.epochs

LEARNING_RATE = args.lr
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
VAL_SET_SIZE = args.val_size  # 2000
OUTPUT_DIR = args.output_path
TARGET_MODULES = [
    "q_proj",
    # "o_proj",
    # "k_proj",
    "v_proj",
    # "gate_proj",
    # "up_proj",
    # "down_proj",
]
world_size = int(os.environ.get("WORLD_SIZE", 1))
ddp = world_size != 1
if ddp:
    device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
    GRADIENT_ACCUMULATION_STEPS = GRADIENT_ACCUMULATION_STEPS // world_size
# Model
from model.starcoder4code import starcoder4Code

model = starcoder4Code(
    input_dim=256,
    output_dim=hidden_dim[args.model],
    load_in_8bit=args.load_in_8bit,
    use_lora=args.use_lora,
    lora_r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    lora_target_modules=TARGET_MODULES,
    model_path=args.model_path,
)
model.train()

CUTOFF_LEN = datadealer.cutofflen  # cutoff of input
LANG = args.language


def generate_and_tokenize_prompt(data_point):
    # This function masks out the labels for the input,
    # so that our loss is computed only on the response.
    query = data_point["query"] if args.extracted_query else data_point["input"]
    task = data_point["query"] if args.extracted_prompt else data_point["input"]
    graph_emb = search_with_faiss(
        query, datadealer.graph_emb_list, datadealer.index, datadealer.pca, 1
    )
    user_prompt = f"Question: Please use {LANG} to write a correct solution to a programming problem. We also have the syntax graph embedding of a similary problem encoded in <GraphEmb> for you to refer to. You should give executable completed code and nothing else. The problem:\n{task}\n\nAnswer:"
    output_prompt = f"{data_point['output']}"
    unk_ = model.starcoder_tokenizer.unk_token
    user_prompt = user_prompt.replace("<GraphEmb>", unk_)
    len_user_prompt_tokens = len(
        tokenizer(
            user_prompt,
            truncation=True,
            max_length=CUTOFF_LEN,
        )["input_ids"]
    )
    try:
        assert len_user_prompt_tokens < CUTOFF_LEN
    except:
        print("Wtmd too long!")
    full_tokens = tokenizer(
        user_prompt + output_prompt,
        truncation=True,
        max_length=CUTOFF_LEN,
    )["input_ids"]

    return {
        "input_ids": full_tokens,
        "labels": [-100] * len_user_prompt_tokens
        + full_tokens[len_user_prompt_tokens:],
        "attention_mask": [1] * (len(full_tokens)),
        "encoded_emb": torch.tensor(graph_emb),
    }


# Dataset
dataset = datadealer.give_train_dataset_for_transformers()
print("Data loaded")
train_dataset = dataset["train"].map(generate_and_tokenize_prompt)
train_dataset = train_dataset.remove_columns("output")
train_dataset = train_dataset.remove_columns("input")
if args.extracted_query:
    train_dataset = train_dataset.remove_columns("query")
MAX_STEPS = max((len(dataset["train"])) // BATCH_SIZE * EPOCHS, EPOCHS)


config = {
    "learning_rate": LEARNING_RATE,
    "num_train_epochs": 1,
    "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
    "per_device_train_batch_size": MICRO_BATCH_SIZE,
    "gradient_checkpointing": False,
}

# Define training args
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    # bf16=True,
    fp16=False,
    logging_strategy="steps",
    logging_steps=1,
    evaluation_strategy="no",
    save_strategy="epoch",
    max_steps=MAX_STEPS,
    logging_dir=args.log,
    load_best_model_at_end=False,
    save_safetensors=False,
    optim="adamw_torch",
    ddp_find_unused_parameters=False if ddp else None,
    remove_unused_columns=False,  # Set to False when debugging
    label_names=["labels"],
    **{k: v for k, v in config.items() if k != "lora_config"},
)

# Define trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=DataCollatorForSeq2Seq(
        tokenizer, return_tensors="pt", padding="longest"
    ),
    # dataset_text_field=["input_ids", "labels", "attention_mask"],
    # callbacks=EarlyStoppingCallback(2),
)
model.starcoder_model.config.use_cache = False


print("Start training...")
trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

model.starcoder_model.save_pretrained(OUTPUT_DIR)
