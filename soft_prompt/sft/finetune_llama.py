from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
    prepare_model_for_int8_training,
    get_peft_model_state_dict,
)
import torch
import sys
import transformers
import os
from transformers import (
    AutoModel,
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

sys.path.append("../")
from dataloaders.datadealer import DataDealer

CUTOFF_LEN = 4096
parser = argparse.ArgumentParser()
parser.add_argument("--log", type=str, default="logs")
parser.add_argument("--wandb", action="store_true", default=False)
parser.add_argument(
    "--model_path", type=str, default="/ext0/hcchai/codemate/llama2/Llama-2-13b-chat-hf"
)
parser.add_argument("--model", type=str, default="llamawoextracted")
parser.add_argument("-d", "--dataset", type=str, default="CodeContest")
parser.add_argument("-l", "--language", type=str, default="c++")
parser.add_argument("--eval_steps", type=int, default=200)
parser.add_argument("--save_steps", type=int, default=200)
parser.add_argument("--lr_scheduler_type", type=str, default="linear")
parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
parser.add_argument("--total_batch_size", type=int, default=256)
parser.add_argument("--train_size", type=int, default=65536)
parser.add_argument("--val_size", type=int, default=1000)
parser.add_argument("--resume_from_checkpoint", type=str, default=None)
parser.add_argument("--lora_remote_checkpoint", type=str, default=None)
parser.add_argument("--ignore_data_skip", type=str, default="False")
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--wd", type=float, default=0)
parser.add_argument("--use_lora", type=int, default=1)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--epochs", type=int, default=1)
parser.add_argument("--load_in_8bit", action="store_true", help="Load model 8 bit.")

args = parser.parse_args()
datadealer = DataDealer(
    dataset=args.dataset, split="train", extracted=False, softprompt=False
)
args.output_path = f"../trained_models/{args.dataset}/{args.language}/{args.model_path.split('/')[-1]}_sft/"
if not os.path.exists(args.output_path):
    os.makedirs(args.output_path)
print(f"Model will be stored at{args.output_path}")
LANG = args.language


def generate_and_tokenize_prompt(data_point):
    # This function masks out the labels for the input,
    # so that our loss is computed only on the response.
    user_prompt = f"<s>[INST] <<SYS>> You are a helpful coding assistant, and you should help to write programm correctly and efficiently. <</SYS>>. Please use {LANG} to write a correct solution to a programming problem. You should give executable completed code and nothing else. The problem:\n{data_point['input']} [/INST]"
    output_prompt = f"{data_point['output']}</s>"
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
    }


transformers.set_seed(args.seed)

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(args.model_path)
tokenizer.padding_side = "right"
tokenizer.pad_token = tokenizer.eos_token

# Dataset
dataset = datadealer.give_train_dataset_for_transformers()
print("Data loaded")
train_dataset = dataset["train"].map(generate_and_tokenize_prompt)
print("Data processed")


MICRO_BATCH_SIZE = args.per_device_eval_batch_size
BATCH_SIZE = min(args.total_batch_size, args.train_size)

GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
EPOCHS = args.epochs
MAX_STEPS = max((len(dataset["train"])) // BATCH_SIZE * EPOCHS, EPOCHS)
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
model = AutoModelForCausalLM.from_pretrained(
    args.model_path,
    load_in_8bit=args.load_in_8bit,
    device_map="auto",
)
model.train()

peft_config = LoraConfig(
    # task_type=TaskType.CAUSAL_LM,
    # inference_mode=False,
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=TARGET_MODULES,
)

if args.load_in_8bit:
    model = prepare_model_for_int8_training(model)

model = get_peft_model(model, peft_config)


config = {
    "lora_config": peft_config,
    "learning_rate": LEARNING_RATE,
    "num_train_epochs": 1,
    "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
    "per_device_train_batch_size": MICRO_BATCH_SIZE,
    "gradient_checkpointing": False,
}

# Define training args
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    fp16=False,
    logging_strategy="steps",
    logging_steps=1,
    evaluation_strategy="no",
    save_strategy="epoch",
    max_steps=MAX_STEPS,
    eval_steps=args.eval_steps,
    save_steps=1,
    logging_dir=args.log,
    load_best_model_at_end=False,
    optim="adamw_torch",
    ddp_find_unused_parameters=False if ddp else None,
    save_safetensors=False,
    **{k: v for k, v in config.items() if k != "lora_config"},
)

# Define trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=train_dataset,
    data_collator=DataCollatorForSeq2Seq(
        tokenizer, return_tensors="pt", padding="longest"
    ),
    # dataset_text_field=["input_ids", "labels", "attention_mask"],
    # callbacks=EarlyStoppingCallback(2),
)
model.config.use_cache = False

# old_state_dict = model.state_dict
# model.state_dict = (
#     lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
# ).__get__(model, type(model))
# if torch.__version__ >= "2" and sys.platform != "win32":
#     print("compiling the model")
#     model = torch.compile(model)

print("Start training...")
trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

model.save_pretrained(OUTPUT_DIR)
