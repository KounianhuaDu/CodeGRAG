import os 
import numpy as np
import argparse
from tqdm import tqdm
import gzip
from collections import defaultdict
import pickle as pkl
from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

import os
import sys
sys.path.append("..")
from utils.utils import *
from utils.config import *

def build_instruction(src_language: str, question: str):
    return '''
Please summarize the function of the given {} code in natural language using the least words. Please do not output any redundant analysis. Just output the brief summarization of the function.\n
The given problem:\n
{}
'''.strip().format(src_language, question.strip())

def generate_one_completion(src_language, problem):    
    sys_msg = "You are a professional programmer that analyzes the function of code blocks and translates the corresponding function to natural language."
    
    prompt = build_instruction(src_language, problem)

    response = client.chat.completions.create(
                model='gpt-3.5-turbo-0125',
                messages=[{"role": "system", "content": sys_msg}, {"role": "user", "content": prompt}],
                max_tokens=1024,  # 调整生成文本的长度
                temperature=0,
                frequency_penalty=0,
                presence_penalty=0,
                logprobs=True,
                top_logprobs=1
            )
    message = response.choices[0].message.content
    return message

def main(args):
    code_path = os.path.join(args.root, 'code')
    code_files = os.listdir(code_path)
    
    if args.gpu >= 0 and torch.cuda.is_available():
        device = 'cuda:{}'.format(args.gpu)
    else:
        device = 'cpu'
    
    nls = []
    codes = []
    for code_file in tqdm(code_files):
        with open(os.path.join(code_path, code_file), 'r') as f:
            code = f.read()
        codes.append(code)
        nl = generate_one_completion(args.lang, code)
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
    
    client = OpenAI()
    
    main(args)