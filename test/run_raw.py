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

def build_instruction(languge: str, question: str):
    return '''
Please continue to complete the function. You are not allowed to modify the given code and do the completion only. Please return all completed function in a codeblock. Please only generate the function body. Here is the given code to do completion:
```{}
{}
```
'''.strip().format(languge.lower(), question.strip())

def generate_one_completion(problem, language='cpp'):
    task = problem['prompt']
    declaration = problem['declaration']
    prompt = build_instruction(language, task)

    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2048,  # 调整生成文本的长度
        temperature=0.0,  # 调整生成文本的随机性
        top_p=0.0,
    )
    message = response.choices[0]["message"]["content"]
    print(message)
    exit()

    #problem['output'] = message
    
    #return extract_generation_code(problem, lang_code=language)

    code = extract_generation_code(message, language)
    return code

def main(output_path):

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
                if int(task_id[4:]) < int(start_task_id):
                    continue
                else:
                    completion=generate_one_completion(problems[task_id])
                    #print(completion)
                    temp_dict = dict(task_id=task_id, generation=completion, prompt=problems[task_id]["prompt"], test=problems[task_id]["test"], declaration=problems[task_id]["declaration"])
                    #temp_dict = generate_one_completion(problems[task_id])
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
    parser.add_argument("--output", default="../output", help="output path")

    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)

    problem_file = "/home/rrt/codemate/CodeGeeX/codegeex/benchmark/humaneval-x/cpp/data/humaneval_cpp.jsonl.gz"
    problems = read_dataset(problem_file, dataset_type="humaneval")                
    
    main(args.output)

