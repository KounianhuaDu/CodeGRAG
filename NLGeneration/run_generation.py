import os 
import numpy as np
import argparse
from tqdm import tqdm
import gzip
from collections import defaultdict
import pickle as pkl

from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

import os
import sys
sys.path.append("..")
from utils.utils import *

def generate_prompt_for_wizard(input):
    INSTRUCTION = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.
    ### Instruction:
    {input}
    ### Response:"""
    return INSTRUCTION

def build_instruction(src_language: str, question: str):
    return '''
Please summarize the function of the given {} code in natural language in two sentences using the least words. Please do not output any redundant analysis. Just output the brief summarization of the function.\n
The given problem:\n
{}
'''.strip().format(src_language, question.strip())

def generate_one_completion(src_language, problem, model, tokenizer, device):    
    sys_msg = "You are a professional programmer that analyzes the function of code blocks and translates the corresponding function to natural language."
    
    prompt = build_instruction(src_language, problem)

    generation_config = GenerationConfig(
        pad_token_id=tokenizer.pad_token_id,
        do_sample=True,
        temperature=0.01,
        max_new_tokens=512,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        # top_p=1
    )
    prompt_batch = [generate_prompt_for_wizard(prompt)]
    encoding = tokenizer(prompt_batch, return_tensors="pt", truncation=True, max_length=2048).to(device)
    with torch.no_grad():
        gen_tokens = model.generate(
            **encoding,
            generation_config=generation_config
        )
    # logging.info(gen_tokens)
    if gen_tokens is not None:
        gen_seqs = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
    else:
        gen_seqs = None
    completion_seq = None
    if gen_seqs is not None:
        for seq_idx, gen_seq in enumerate(gen_seqs):
            completion_seq = gen_seq.split("### Response:")[1]
            completion_seq = completion_seq.replace('\t', '    ')
    return completion_seq

def main(args):
    code_path = os.path.join(args.root, 'code')
    code_files = os.listdir(code_path)
    
    if args.gpu >= 0 and torch.cuda.is_available():
        device = 'cuda:{}'.format(args.gpu)
    else:
        device = 'cpu'
    
    tokenizer = AutoTokenizer.from_pretrained('/ext0/hcchai/codemate/wizard/WizardCoder-15B-V1.0')
    model = AutoModelForCausalLM.from_pretrained('/ext0/hcchai/codemate/wizard/WizardCoder-15B-V1.0', torch_dtype=torch.float16).half().to(device)

    nls = []
    codes = []
    for code_file in tqdm(code_files):
        with open(os.path.join(code_path, code_file), 'r') as f:
            code = f.read()
        codes.append(code)
        nl = generate_one_completion(args.lang, code, model, tokenizer, device)
        nls.append(nl)
    
    with open(os.path.join(args.output, 'nls.pkl'), 'wb') as f:
        pkl.dump(nls, f)
    with open(os.path.join(args.output, 'codes.pkl'), 'wb') as f:
        pkl.dump(codes, f)
    
    

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Using different models to generate function")
    parser.add_argument("--model_name", default="gpt-3.5-turbo", help="test model")
    parser.add_argument("--lang", default="C++")
    parser.add_argument("--output", default="./data/CPP/generation", help="output path")
    parser.add_argument("--root", default="./data/CPP")
    parser.add_argument("--gpu", type=int, default="0", help="Set GPU Ids : Eg: For CPU = -1, For Single GPU = 0")

    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    main(args)