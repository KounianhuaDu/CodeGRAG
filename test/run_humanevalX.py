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
from utils.codet5 import get_prompt_code
from utils.fas import get_prompt_textbook
from utils.utils import *
from algo.faissSearch import get_prompt_faiss


problem_file = "/home/rrt/codemate/CodeGeeX/codegeex/benchmark/humaneval-x/cpp/data/humaneval_cpp.jsonl.gz"
problems = read_dataset(problem_file, dataset_type="humaneval")                

def generate_one_completion(task):
    qa_file = "None"
    if qa_file != "None":
        #retrieval_part
        data_list = []
        with open(qa_file_path[qa_file], 'r', encoding='utf-8') as f:
            for line in f:
               data = json.loads(line)
               data_list.append(data)
        embeddings = np.load(emb_path[qa_file])
    query = task  
    if qa_file in ["textbook", "textbook_total"]:
        knowledge = get_prompt_textbook(query, 1, embeddings, data_list)
    elif qa_file in ["codes", "codes_total"]:
        knowledge = get_prompt_code(query, 1, embeddings, data_list)
    elif qa_file != "None" :
        knowledge = get_prompt_faiss(query, 1, embeddings, data_list)
    
    prompt = "请根据需求和函数声明生成C++函数体代码，请不要输出函数声明，只输出可编译的函数体代码"
    #prompt = "Here is a code generation question, please generate python function codes according to the requirements of the question. Let's think of it step by step.\n"
    
    #tokens = knowledge.split()
    #truncated_tokens = tokens[:500]
    #knowledge = " ".join(truncated_tokens)
    
    if qa_file in ["textbook", "total", "textbook_total"]:
        input = prompt + '\nrelative knowledge for reference:\n' + knowledge + '\n' + task
    elif qa_file in ["codes", "codes_total"]:
        input = prompt + '\nrelative codes for reference:\n' + knowledge + '\n' + task
    elif qa_file == "None":
        input = prompt + task
    
    #print(input)
    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=[{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": input}],
        max_tokens=1024,  # 调整生成文本的长度
        temperature=0.01,  # 调整生成文本的随机性
        top_p=0.75,

    )
    #print(response)
    
    message = response.choices[0]["message"]["content"]
    #print(message)
    #print('Code')
    code = extract_function_body(message)
    #print(code)
    return code

def main():

    while(True):
        check_point_path = '../output/humanevalX/checkpoint.npy'
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
                    completion=generate_one_completion(problems[task_id]["prompt"])
                    #print(completion)
                    temp_dict = dict(task_id=task_id, generation=completion, prompt=problems[task_id]["prompt"], test=problems[task_id]["test"])
                    samples.append(temp_dict)

            write_jsonl("samples.jsonl", samples)

            if int(len(samples)) >= 164:
                break

        except KeyboardInterrupt:
            np.save(check_point_path, samples)

        except Exception as e:
            print(str(e))
            np.save(check_point_path, samples)
        
        return 0
    

def check():
    check_point_path = '/home/hcchai/codemate_code/humaneval/check_point.npy'
    samples = np.load(check_point_path,allow_pickle=True).tolist()
    print(samples)
    print(len(samples))

if __name__=="__main__":
    main()
    # check()
    pass
