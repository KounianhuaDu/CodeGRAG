import torch
import random
import numpy as np
import re
import subprocess
import gzip
import json
import os
import argparse
from codegeex.data.data_utils import write_jsonl
from tqdm import tqdm

'''def extract_generation_code(message):
    if re.findall(f'(?is)```(.*)```', message):
        raw_code = re.findall(f'(?is)```(.*)```', message)[0]
        start = raw_code.find('{')
        end = raw_code.rfind('}')
        code = raw_code[start+1:end+1]+'\n'
    else:
        code = message
    return code'''

def extract_generation_code(raw_code):
    if not raw_code:
        return " "
    start = raw_code.find('{')
    end = raw_code.rfind('}')
    code = raw_code[start+1:end+1]+'\n'

    if "int main()" in code:
        main_start = code.index("int main()")
        code = code[:main_start]

    return code

def main(path):
    passed = []
    failed = []
    with open(os.path.join(path, 'samples_results.jsonl'), 'r') as f:
        for line in tqdm(f):
            line = json.loads(line)
            state = line['result']
            task_id = line['task_id']
            if state == "passed":
                passed.append(task_id)
            else:
                failed.append(failed)
            
    

    write_jsonl(os.path.join(path, 'samples_new.jsonl'), samples)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Using different models to generate function")
    parser.add_argument("--lang", default="c++", choices=['c++','python','java'])
    parser.add_argument("--path", default="../output", help="output path")

    args = parser.parse_args()
    
    main(args.path)