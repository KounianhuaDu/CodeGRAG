import click
import json
import os 
import numpy as np
import openai
import logging
import argparse
from tqdm import tqdm
import re
import pickle as pkl
from collections import defaultdict

from codegeex.benchmark.utils import read_dataset, IMPORT_HELPER
from codegeex.data.data_utils import write_jsonl

import os
import sys
sys.path.append("..")
from utils.config import *
from utils.utils import *
from algo.SelfRevision import construct_faiss_index, search_with_faiss

def first_instruction(languge, question):
    return '''
Please continue to complete the function. You are not allowed to modify the given code and do the completion only. Please return all completed function in a codeblock. Here is the given code to do completion:
```{}
{}
```
'''.strip().format(languge.lower(), question.strip())

def second_instruction(knowledge, languge, question):
    return '''
Please continue to complete the function. You are not allowed to modify the given code and do the completion only. Please return all completed function in a codeblock. \n Relative knowledge:\n {}\n Here is the given code to do completion:
```{}
{}
```
'''.strip().format(knowledge, languge.lower(), question.strip())

def generate_with_self_revision(problem, index, data_list, pca, k, value, languge='c++',):
    task = problem['prompt']
    declaration = problem['declaration']
    task_id = problem['task_id']

    func_name = first_round_res[task_id]['declaration'].split('\n')[-2]
    query = func_name + '\n' + first_round_res[task_id]['generation']
    
    revision_hints = search_with_faiss(query, data_list, index, pca, k)
    if value == 'raw_code':
        revision_hints = "```c++\n" + revision_hints + "\n```"
    
    second_prompt = second_instruction(revision_hints, languge, task)
    
    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=[{"role": "user", "content": second_prompt}],
        max_tokens=2048,  # 调整生成文本的长度
        temperature=0.01,  # 调整生成文本的随机性
        top_p=0.95,

    )
    message = response.choices[0]["message"]["content"]
    
    final_code = extract_generation_code(message, languge)

    return final_code

def main(k, data_path, output_path, value):
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
        check_point_path = os.path.join(output_path, f'checkpoint_with_{value}_{k}.npy')
        if not os.path.exists(check_point_path):
            samples = []
        else:
            samples = np.load(check_point_path,allow_pickle=True).tolist()

        if int(len(samples)) >= 164:
            break

        try:
            start_task_id = len(samples)

            for task_id in tqdm(problems):
                if int(task_id[4:]) < int(start_task_id):
                    continue
                else:
                    completion=generate_with_self_revision(problems[task_id], index, data_list, pca, k, value)
                    temp_dict = dict(task_id=task_id, generation=completion, prompt=problems[task_id]["prompt"], test=problems[task_id]["test"], declaration=problems[task_id]['declaration'])
                    samples.append(temp_dict)

            write_jsonl(os.path.join(output_path, f"samples_with_{value}_{k}.jsonl"), samples)

            if int(len(samples)) >= 164:
                break

        except KeyboardInterrupt:
            np.save(check_point_path, samples)

        except Exception as e:
            print(str(e))
            np.save(check_point_path, samples)
        
    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Using different models to generate function")
    parser.add_argument("--model_name", default="gpt-3.5-turbo", help="test model")
    
    parser.add_argument("--datapath", default="../data/Cgraphs", help="data path")
    parser.add_argument("--output", default="/home/knhdu/output/FinalVersion", help="output path")
    parser.add_argument("--first_round_res", default="/home/knhdu/1121/samples.jsonl", help="output path")
    parser.add_argument("--value", choices=['raw_code','graph'])
    
    parser.add_argument('--gpu', type=int, default=0, help='Set GPU Ids : Eg: For CPU = -1, For Single GPU = 0')
    parser.add_argument('--seed', default=1, type=int)

    parser.add_argument('--k', default=1, type=int)

    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    problem_file = "/home/rrt/codemate/CodeGeeX/codegeex/benchmark/humaneval-x/cpp/data/humaneval_cpp.jsonl.gz"
    problems = read_dataset(problem_file, dataset_type="humaneval") 
    
    first_round_res = defaultdict(dict)
    with open(args.first_round_res, 'r', encoding='utf-8') as f:
        for line in f:
            line = json.loads(line)
            task_id = line['task_id']
            first_round_res[task_id] = line

    main(args.k, args.datapath, args.output, args.value)
