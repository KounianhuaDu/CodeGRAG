import click
import json
import os 
import numpy as np
import openai
import logging
import argparse
from tqdm import tqdm
import re

from codegeex.benchmark.utils import read_dataset, IMPORT_HELPER
from codegeex.data.data_utils import write_jsonl

import os
import sys
sys.path.append("..")
from utils.config import *
from utils.utils import *

from algo.SelfRevision import construct_faiss_index, search_with_faiss_multi
import pickle as pkl

def build_instruction(knowledge, question: str):
    return '''
Please continue to complete the C++ function according to the requirements and function declarations. You are not allowed to modify the given code and do the completion only.\n
Relative Knowledge:\n
{}
\n
{}
'''.strip().format(knowledge, question.strip())

'''
Relative Knowledge:\n
{}
\n
'''


def generate_one_completion(problem, index, code_data_list, graph_data_list, pca, k):
    task = problem['prompt']
    declaration = problem['declaration']

    query = declaration
    knowledge_code, knowledge_graph = search_with_faiss_multi(query, code_data_list, graph_data_list, index, pca, k)

    prompt_code = build_instruction(knowledge_code, task)
    prompt_graph = build_instruction(knowledge_graph, task)

    code_response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=[{"role": "user", "content": prompt_code}],
        max_tokens=1024,  # 调整生成文本的长度
        temperature=0.0,  # 调整生成文本的随机性
        top_p=0.0,
    )
    code_message = code_response.choices[0]["message"]["content"]
    code_code = extract_function_body(code_message)

    graph_response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=[{"role": "user", "content": prompt_graph}],
        max_tokens=1024,  # 调整生成文本的长度
        temperature=0.0,  # 调整生成文本的随机性
        top_p=0.0,
    )
    graph_message = graph_response.choices[0]["message"]["content"]
    graph_code = extract_function_body(graph_message)
    return code_code, graph_code

def main(k, data_path, output_path, value):

    embeddings_path = os.path.join(data_path, 'codes_emb.npy')
    embeddings = np.load(embeddings_path)

    with open(os.path.join(data_path, 'codes.pkl'), 'rb') as f:
        code_data_list = pkl.load(f)
    with open(os.path.join(data_path, 'graphs.pkl'), 'rb') as f:
        graph_data_list = pkl.load(f)

    index, pca = construct_faiss_index(embeddings)

    while(True):
        check_point_path = os.path.join(output_path, 'checkpoint.npy')
        if not os.path.exists(check_point_path):
            samples = [[], []]
        else:
            samples = np.load(check_point_path, allow_pickle=True).tolist()

        if int(len(samples)) >= 164:
            break

        try:
            start_task_id = len(samples)

            for task_id in tqdm(problems):
                if int(task_id[4:]) < int(start_task_id):
                    continue
                else:
                    completion_with_code, completion_with_graph = generate_one_completion(problems[task_id], index, code_data_list, graph_data_list, pca, k)
                    
                    code_temp_dict = dict(task_id=task_id, generation=completion_with_code, prompt=problems[task_id]["prompt"], test=problems[task_id]["test"], declaration=problems[task_id]["declaration"])
                    graph_temp_dict = dict(task_id=task_id, generation=completion_with_graph, prompt=problems[task_id]["prompt"], test=problems[task_id]["test"], declaration=problems[task_id]["declaration"])
                    
                    samples[0].append(code_temp_dict)
                    samples[1].append(graph_temp_dict)

            write_jsonl(os.path.join(output_path, 'samples_with_code.jsonl'), samples[0])
            write_jsonl(os.path.join(output_path, 'samples_with_graph.jsonl'), samples[1])

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
    
    parser.add_argument("--datapath", default="../data/Cgraphs", help="data path")
    parser.add_argument("--output", default="/home/knhdu/output/FinalVersion", help="output path")
    parser.add_argument("--value", choices=['raw_code','graph'])
    
    parser.add_argument('--gpu', type=int, default=0, help='Set GPU Ids : Eg: For CPU = -1, For Single GPU = 0')
    parser.add_argument('--seed', default=1, type=int)

    parser.add_argument('--k', default=1, type=int)

    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    problem_file = "/home/rrt/codemate/CodeGeeX/codegeex/benchmark/humaneval-x/cpp/data/humaneval_cpp.jsonl.gz"
    problems = read_dataset(problem_file, dataset_type="humaneval") 
    

    main(args.k, args.datapath, args.output, args.value)

