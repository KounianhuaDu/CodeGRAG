from peft import (
    get_peft_model,
    set_peft_model_state_dict,
    LoraConfig,
    TaskType,
    prepare_model_for_int8_training,
    get_peft_model_state_dict,
)
import torch
import safetensors
import sys
import transformers
import os
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
import re
import sys

sys.path.append("..")
from dataloaders.datadealer import DataDealer
from support.Search_with_CodeT5 import construct_faiss_index, search_with_faiss

transformers.set_seed(42)
parser = argparse.ArgumentParser()
parser.add_argument("--log", type=str, default="logs")
parser.add_argument("--wandb", action="store_true", default=False)
parser.add_argument(
    "--model_path", type=str, default="/ext0/hcchai/codemate/llama2/Llama-2-13b-chat-hf"
)
parser.add_argument("--model", type=str, default="Llama2")
parser.add_argument("-d", "--dataset", type=str, default="CodeForce")
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
parser.add_argument("--load_in_8bit", action="store_true", help="Load model 8 bit.")
parser.add_argument("--load_trained_model", action="store_true")
parser.add_argument(
    "-t",
    "--trained_model_path",
    type=str,
    default="/home/jzchen/ML/Code/trained_models/CodeContest/c++/llama/checkpoint-41",
)

args = parser.parse_args()
datadealer = DataDealer(
    dataset=args.dataset, split="test", extracted=True, softprompt=False
)

assert os.path.exists(args.trained_model_path)
print(f"Model will be loaded from {args.trained_model_path}")

transformers.set_seed(args.seed)

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
    use_safetensors=False,
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

if args.load_trained_model:
    print("Load lora weights")
    adapters_weights = torch.load(
        os.path.join(args.trained_model_path, "adapter_model.bin"),
    )
    # **** Important!! Must load in this way!!!****
    state_dict = {
        k: v.cuda()
        for k, v in adapters_weights.items()
        if "lora" in k or "score" in k or "embedding_proj" in k
    }
    set_peft_model_state_dict(model, state_dict)

print("Model loaded")
model.eval()

LANG = args.language


def generate_and_tokenize_prompt(task):
    # user_prompt = f"### Instruction:\n Use the Task below and the Input given to write the Response, which is a c++ programming code that can solve the following Task: \n \n### Task: \nPlease use c++ to write a correct solution to a programming problem. We also have the syntax graph embedding of a similary problem encoded in <GraphEmb> for you to refer to. You should just give executable completed code to the input problem and no other explanation.\n \n### Input:\n{task}\n\n### Response:\n"
    user_prompt = f"<s>[INST] <<SYS>> You are a helpful coding assistant, and you should help to write programm correctly and efficiently. <</SYS>>. Please use {LANG} to write a correct solution to a programming problem. You should give executable completed code and nothing else. The problem:\n{task} [/INST]"
    output_prompt = ""
    len_user_prompt_tokens = len(
        tokenizer(
            user_prompt,
            truncation=True,
            max_length=datadealer.cutofflen,
        )["input_ids"]
    )
    try:
        assert len_user_prompt_tokens < datadealer.cutofflen
    except:
        print("Wtmd too long!")
    full_tokens = tokenizer(
        user_prompt + output_prompt,
        truncation=True,
        max_length=datadealer.cutofflen,
    )["input_ids"]
    return {
        "input_ids": full_tokens,
        "attention_mask": [1] * (len(full_tokens)),
        "labels": [-100] * len_user_prompt_tokens
        + full_tokens[len_user_prompt_tokens:],
    }


samples = []
start_task_id = len(samples)
failed_num = 0
problem_file = json.load(
    open(
        f"/home/jzchen/ML/Code/data/train/{args.dataset}/{args.language}_Gemma_test.json",
        "r",
    )
)
idx = 0
for task_id, problem in datadealer.iter_test_data():
    task = problem_file[idx]
    input_dict = generate_and_tokenize_prompt(task)
    try:
        generate_ids = model.generate(
            tokenizer(
                tokenizer.decode(list(torch.tensor(input_dict["input_ids"]))),
                return_tensors="pt",
            ).input_ids.to("cuda"),
            max_length=2048,
        )

        output = tokenizer.batch_decode(
            generate_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        failed, completion = datadealer.output_filter(output, args.model)
    except KeyboardInterrupt as e:
        assert 0
    except:
        failed = 1
        completion = ""
    failed_num += failed
    print("*" * 10)
    print(completion)
    print("*" * 10)
    temp_dict = datadealer.formatted_return(completion, problem, task_id)
    samples.append(temp_dict)
    idx += 1


print(f"Final failed num {failed_num}")
datadealer.save_results(
    samples,
    f"/home/jzchen/ML/AfterCodegrag/output/{args.model_path.split('/')[-1]}_{args.dataset}{args.load_trained_model}.jsonl",
)
