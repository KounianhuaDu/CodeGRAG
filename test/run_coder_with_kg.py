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
from utils.utils import *

from algo.SelfRevision import construct_faiss_index, search_with_faiss
import pickle as pkl

def build_instruction_with_g(knowledge, question, language):
    return '''
Please continue to complete the {} function. You are not allowed to modify the given code and do the completion only.\n
The syntax graph of a similar code might be:\n
{}
You can refer to the above knowledge to do the completion. The problem:
\n
{}
'''.strip().format(language, knowledge, question.strip())

def build_instruction_with_code(knowledge, question, language):
    return '''
Please continue to complete the {} function. You are not allowed to modify the given code and do the completion only.\n
A similar code might be:\n
{}
You can refer to the above knowledge to do the completion. The problem:
\n
{}
'''.strip().format(language, knowledge, question.strip())

def generate_one_completion(problem, index, data_list, pca, k, value,  language='c++'):
    task = problem["input"]
    kg = search_with_faiss(task, data_list, index, pca, k)
    
    #print(task)
    if value == 'raw_code':
        prompt = build_instruction_with_code(kg, task, language)
    elif value == 'graph':
        prompt = build_instruction_with_g(kg, task, language)
    
    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=[{"role": "user", "content": prompt}],
        max_tokens=3072,  # 调整生成文本的长度
        temperature=0.0,  # 调整生成文本的随机性
        top_p=0.0,
    )
    raw_code = response.choices[0]["message"]["content"]
    
    code_ = []
    start = False
    for line in raw_code.split("\n"):
        if line.strip().startswith('def'):
            start = True
            code_.append(line)
            continue
        if start and (len(line.strip()) > 0 and line[0] != ' ' and line[0] != '\t'):
            break
        if start:
            code_.append(line)
    code = "\n".join(code_)
    #code = extract_function_body(message, language)

    return code

def main(k, data_path, output_path, value, language):
    embeddings_path = os.path.join(data_path, 'codes_emb.npy')

    if value == 'raw_code':
        codes_path = os.path.join(data_path, 'codes.pkl')
    elif value == 'graph':
        codes_path = os.path.join(data_path, 'graphs.pkl')
    else:
        raise NotImplementedError

    embeddings = np.load(embeddings_path)
    with open(codes_path, 'rb') as f:
        data_list = pkl.load(f)
    index, pca = construct_faiss_index(embeddings)
    
    while(True):
        check_point_path = os.path.join(output_path, 'checkpoint.npy')
        if not os.path.exists(check_point_path):
            samples = []
            done_problems = []
        else:
            samples = np.load(check_point_path, allow_pickle=True).tolist()
            done_problems = [s["question_id"] for s in samples]
        
        if int(len(samples)) >= 230:
            break

        try:
            for task_id in tqdm(problems):
                if task_id in done_problems:
                    continue
                else:
                    completion=generate_one_completion(problems[task_id], index, data_list, pca, k, value, language)
                    temp_dict = dict(_id=task_id, generate_results=[completion])
                    samples.append(temp_dict)
            
            write_jsonl(os.path.join(output_path, 'samples.jsonl'), samples)

            if int(len(samples)) >= 230:
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

    parser.add_argument("--datapath", default="../data/FinalData/LongCodes", help="data path")
    parser.add_argument("--value", choices=['raw_code','graph'])
    
    parser.add_argument('--k', default=1, type=int)

    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)

    problem_file = "../data/CoderEval-Input4Models/CEPythonRaw.jsonl"
    
    problems = defaultdict(dict)
    with open(problem_file, 'r') as f:
        for line in f:
            line = json.loads(line)
            problems[line["question_id"]] = line
    #k, data_path, output_path, value, language
    main(args.k, args.datapath, args.output, args.value, args.lang)

