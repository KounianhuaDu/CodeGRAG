import pandas as pd
import re
import os
import copy
import torch
import numpy as np
import pickle as pkl
import argparse
import json
from fastchat.model import load_model, get_conversation_template, add_model_args
from sentence_transformers import SentenceTransformer
from tqdm import tqdm, trange
import sys

sys.path.append("../")
from dataloaders.datadealer import DataDealer

parser = argparse.ArgumentParser()
parser.add_argument("--chunk_interval", type=str, default="0:-1")
parser.add_argument("--dataset", type=str, default="appsnew")
parser.add_argument("--language", type=str, default="python")
parser.add_argument("--split", type=str, default="train")
parser.add_argument(
    "--model_path",
    type=str,
    default="/ext0/hcchai/codemate/gemma/gemma-7b-it",
)
parser.add_argument("--temperature", type=float, default=0.01)
args = parser.parse_args()

datadealer = DataDealer(
    dataset=args.dataset,
    split=args.split,
    extracted=False,
    softprompt=False,
)

descriptions = datadealer.get_prompt_description(chunk_interval=args.chunk_interval)
# Load model
import transformers
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    AutoModelForCausalLM,
    AutoTokenizer,
)
from peft import (
    prepare_model_for_int8_training,
)

model = AutoModelForCausalLM.from_pretrained(
    args.model_path, load_in_8bit=True, device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(args.model_path, add_eos_token=True)
model = prepare_model_for_int8_training(model)

tokenizer.pad_token_id = 0


def get_template(description):
    inputs = f"Below is an OJ programming problem, which is too long, I want you to simplify and summarize the problem to make the description shoter while contains necessary information, the problem:\n {description}. ONLY give me the simplified problem along with input and output format."
    # user_prompt = (
    #     f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. "
    #     f"USER: {inputs} ASSISTANT: "
    # )
    return inputs


def question2answer(model, tokenizer, txt):
    conv = get_conversation_template(args.model_path)
    conv.append_message(conv.roles[0], txt)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer([prompt]).input_ids
    output_ids = model.generate(
        torch.as_tensor(input_ids).to(torch.device("cuda:0")),
        do_sample=True,
        temperature=args.temperature,
        repetition_penalty=1.0,
        max_new_tokens=500,
    )
    return answer_filter(tokenizer.decode(output_ids[0], skip_special_tokens=True))


def answer_filter(ans):
    idx = ans.rfind("### Assistant:")
    if idx != -1:
        return ans[idx + len("### Assistant:") :].strip()
    else:
        idx = ans.rfind("**Problem:**")
        if idx != -1:
            return ans[idx + len("**Problem:**") :].strip()
        else:
            return ans.strip()


answers = []
for question in tqdm(descriptions):
    try:
        cur = question2answer(model, tokenizer, get_template(question))
        print("*" * 10)
        print(cur)
        print("*" * 10)
        answers.append(cur)
    except KeyboardInterrupt as e:
        assert 0
    except:
        answers.append(descriptions)
datadealer.save_as_json(
    datadealer.extraction_fileter(descriptions, answers),
    os.path.join(
        f"../data/train/{args.dataset}",
        f"{args.language}_{args.chunk_interval}_gemma.json",
    ),
)
