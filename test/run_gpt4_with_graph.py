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

from algo.SelfRevision import construct_faiss_index, search_with_faiss
import pickle as pkl

def build_instruction(knowledge, question, language):
    return '''
Please continue to complete the {} function according to the requirements and function declarations. You are not allowed to modify the given code and do the completion only.\n
Please return all completed function in a codeblock.\n
The syntax graph of a relevant code might be:\n
{}
You can refer to the above knowledge to do the completion. The problem:
\n
{}
'''.strip().format(language, knowledge, question.strip())

'''
Relative Knowledge:\n
{}
\n
'''


def generate_one_completion(language, problem, index, graph_data_list, pca, k):
    task = problem['prompt']
    declaration = problem['declaration']

    query = declaration
    knowledge_graph = search_with_faiss(query, graph_data_list, index, pca, k)

    prompt_graph = build_instruction(knowledge_graph, task, language)

    graph_response = openai.ChatCompletion.create(
        model='gpt-4',
        messages=[{"role": "user", "content": prompt_graph}],
        max_tokens=1024,  # 调整生成文本的长度
        temperature=0.0,  # 调整生成文本的随机性
        top_p=0.0,
    )
    message = graph_response.choices[0]["message"]["content"]
    code = extract_generation_code(problem['task_id'], message, declaration, language)
    return code

def main(language, k, data_path, output_path):

    embeddings_path = os.path.join(data_path, 'codes_emb.npy')
    embeddings = np.load(embeddings_path)

    with open(os.path.join(data_path, 'graphs.pkl'), 'rb') as f:
        graph_data_list = pkl.load(f)

    index, pca = construct_faiss_index(embeddings)

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
                    completion_with_graph = generate_one_completion(language, problems[task_id], index, graph_data_list, pca, k)
                    graph_temp_dict = dict(task_id=task_id, generation=completion_with_graph, prompt=problems[task_id]["prompt"], test=problems[task_id]["test"], declaration=problems[task_id]["declaration"])
                    
                    samples.append(graph_temp_dict)

            write_jsonl(os.path.join(output_path, 'samples_with_graph.jsonl'), samples)

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
    parser.add_argument("--lang", choices=['c++','python','java'])

    parser.add_argument('--gpu', type=int, default=0, help='Set GPU Ids : Eg: For CPU = -1, For Single GPU = 0')
    parser.add_argument('--seed', default=1, type=int)

    parser.add_argument('--k', default=1, type=int)

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

    main(args.lang, args.k, args.datapath, args.output)

