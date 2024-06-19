import click
import json
import os 
import numpy as np
import openai
import logging
import argparse
from tqdm import tqdm
import re
import gzip
from collections import defaultdict

from codegeex.benchmark.utils import read_dataset, IMPORT_HELPER
from codegeex.data.data_utils import write_jsonl

import os
import sys
sys.path.append("..")
from utils.config import *
from utils.gpt4_utils import *

def build_instruction(language: str, question: str):
    return '''
Please continue to complete the {} function according to the requirements and function declarations. You are not allowed to modify the given code and do the completion only.\n
Please return all completed function in a codeblock.\n
Here is the problem:\n
{}
'''.strip().format(language, question.strip())

def generate_one_completion(problem, language='c++'):
    task = problem['prompt']
    declaration = problem['declaration']
    prompt = build_instruction(language, task)
    response = openai.ChatCompletion.create(
        model='gpt-4',
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1024,  # 调整生成文本的长度
        temperature=0.0,  # 调整生成文本的随机性
        top_p=0.0,
    )
    message = response.choices[0]["message"]["content"]

    print(message)
    
    code = extract_generation_code(problem['task_id'], message, declaration, language)
    print(code)
    exit()
    return code

def main(output_path, language):
    if language == 'python':
        shift = 7
    elif language == 'c++':
        shift = 4
    elif language == 'java':
        shift = 5
    while(True):
        check_point_path = os.path.join(output_path, 'checkpoint.npy')
        if not os.path.exists(check_point_path):
            samples = []
        else:
            samples = np.load(check_point_path, allow_pickle=True).tolist()

        if int(len(samples)) >= 164:
            break

        try:
            start_task_id = len(samples)
            for task_id in tqdm(problems):
                if int(task_id[shift:]) < int(start_task_id):
                    continue
                else:
                    completion=generate_one_completion(problems[task_id], language)
                    temp_dict = dict(task_id=task_id, generation=completion, prompt=problems[task_id]["prompt"], test=problems[task_id]["test"], declaration=problems[task_id]["declaration"])
                    samples.append(temp_dict)

            write_jsonl(os.path.join(output_path, 'samples.jsonl'), samples)

            if int(len(samples)) >= 164:
                break

        except KeyboardInterrupt:
            np.save(check_point_path, samples)

        except Exception as e:
            print(str(e))
            np.save(check_point_path, samples)
        
        return 0
    

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Using different models to generate function")
    parser.add_argument("--model_name", default="gpt-3.5-turbo", help="test model")
    parser.add_argument("--lang", default="c++", choices=['c++','python','java'])
    parser.add_argument("--output", default="../output", help="output path")

    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)

    if args.lang == 'python':
        problem_file = '../data/humaneval-x/python/data/humaneval_python.jsonl.gz'
    elif args.lang == 'c++':
        problem_file = '../data/humaneval-x/cpp/data/humaneval_cpp.jsonl.gz'
    elif args.lang == 'java':
        problem_file = '../data/humaneval-x/java/data/humaneval_java.jsonl.gz'
    
    problems = defaultdict(dict)
    with gzip.open(problem_file, 'rb') as f:
        for line in f:
            line = eval(line)
            problems[line['task_id']] = line
    
    main(args.output, args.lang)

