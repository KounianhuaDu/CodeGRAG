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

from codegeex.benchmark.utils import read_dataset, IMPORT_HELPER
from codegeex.data.data_utils import write_jsonl

import os
import sys
sys.path.append("..")
from utils.config import *
from utils.codet5 import get_prompt_code
from utils.fas import get_prompt_textbook
from utils.utils import *
from algo.faissSearch import get_prompt_faiss
from algo.SelfRevision import construct_faiss_index, search_with_faiss


problem_file = "/home/rrt/codemate/CodeGeeX/codegeex/benchmark/humaneval-x/cpp/data/humaneval_cpp.jsonl.gz"
problems = read_dataset(problem_file, dataset_type="humaneval")                

def generate_with_self_revision(task, index, data_list, pca, k):
    prompt = "请根据需求和函数声明生成C++函数体代码，请不要输出函数声明，只输出可编译的函数体代码。"
    #prompt = "Here is a code generation question, please generate python function codes according to the requirements of the question. Let's think of it step by step.\n"
    first_prompt = prompt + task
    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=[{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": first_prompt}],
        max_tokens=1024,  # 调整生成文本的长度
        temperature=0.01,  # 调整生成文本的随机性
        top_p=0.75,

    )
    message = response.choices[0]["message"]["content"]
    first_code = extract_function_body(message)
    
    revision_hints = search_with_faiss(first_code, data_list, index, pca, k)
    second_prompt = prompt + '\nrelative knowledge for reference:\n' + revision_hints + '\n' + task
    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=[{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": second_prompt}],
        max_tokens=1024,  # 调整生成文本的长度
        temperature=0.01,  # 调整生成文本的随机性
        top_p=0.75,

    )
    message = response.choices[0]["message"]["content"]
    final_code = extract_function_body(message)

    return final_code

def main(k):
    data_path = '../data/Cgraphs'
    embeddings_path = os.path.join(data_path, 'codes_emb.npy')
    codes_path = os.path.join(data_path, 'codes.pkl')
    embeddings = np.load(embeddings_path)
    with open(codes_path, 'rb') as f:
        data_list = pkl.load(f)
    index, pca = construct_faiss_index(embeddings)

    while(True):
        check_point_path = f'../output/humanevalX/checkpoint_with_revision_{k}.npy'
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
                    #(task, index, data_list, k):
                    completion=generate_with_self_revision(problems[task_id]["prompt"], index, data_list, pca, k)
                    #print(completion)
                    temp_dict = dict(task_id=task_id, generation=completion, prompt=problems[task_id]["prompt"], test=problems[task_id]["test"])
                    samples.append(temp_dict)

            write_jsonl(f"../output/humanevalX/samples_with_revision_{k}.jsonl", samples)

            if int(len(samples)) >= 164:
                break

        except KeyboardInterrupt:
            np.save(check_point_path, samples)

        except Exception as e:
            print(str(e))
            np.save(check_point_path, samples)
        
        return 0
    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Using different models to generate function")
    parser.add_argument("--model_name", default="gpt-3.5-turbo", help="test model")
    
    parser.add_argument("--datapath", default="../data", help="data path")
    parser.add_argument("--output", default="../output", help="output path")
    
    parser.add_argument("--file_name", default="oj-problems-20230620-group_checked_v15.csv", help="file name of test data")
    parser.add_argument("--language", default="chinese", help="problem language")
    parser.add_argument("--version", default="v4", help="generate version")
    parser.add_argument("--qa_file", default="codes_total")
    parser.add_argument("--mulStage", default="answer", choices=["answer", "code", "concept"])
    parser.add_argument('--gpu', type=int, default=0, help='Set GPU Ids : Eg: For CPU = -1, For Single GPU = 0')
    parser.add_argument('--seed', default=1, type=int)

    parser.add_argument('--k', default=1, type=int)


    args = parser.parse_args()

    main(args.k)
