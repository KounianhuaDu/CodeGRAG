import os 
import numpy as np
from openai import OpenAI
import argparse
from tqdm import tqdm
import gzip
from collections import defaultdict

from codegeex.data.data_utils import write_jsonl
from codegeex.benchmark.utils import read_translation_dataset


import os
import sys
sys.path.append("..")
from utils.config import *
from utils.utils import *

def build_instruction(src_language: str, dst_language: str, question: str):
    return '''
Please translate the given {} code to {} code. You are not allowed to modify the function of the code and do the translation only.\n
The given problem:\n
{}
'''.strip().format(src_language, dst_language, question.strip())

def generate_one_completion(problem, trans):
    if trans == 'cpp2python':
        src_language = 'C++'
        dst_language = 'python'
    elif trans == 'python2cpp':
        src_language = 'python'
        dst_language = 'C++'
    
    sys_msg = "You are a helpful code translator that translates code in one programming language to that in another programming language."
    
    task = problem['prompt']
    prompt = build_instruction(src_language, dst_language, task)
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
    code = extract_translation(message, trans)
    return code

def main(output_path, entries, dst_problems, trans):
    if trans == 'python2cpp':
        src_shift = 7
        dst_task = 'CPP/'
    elif trans == 'cpp2python':
        src_shift = 4
        dst_task = 'Python/'
    
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
            for task_id in tqdm(entries):
                if int(task_id[src_shift:]) < int(start_task_id):
                    continue
                else:
                    completion=generate_one_completion(entries[task_id], trans)
                    dst_id = dst_task + task_id[src_shift:]
                    temp_dict = dict(task_id=dst_id, generation=completion, prompt=dst_problems[dst_id]["prompt"], test=dst_problems[dst_id]["test"], declaration=dst_problems[dst_id]["declaration"])
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
    parser.add_argument("--trans", default="cpp2python", choices=['cpp2python', 'python2cpp'])
    parser.add_argument("--output", default="../output", help="output path")

    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    client = OpenAI()
    
    python_problem_file = '../data/humaneval-x/python/data/humaneval_python.jsonl.gz'
    python_problems = defaultdict(dict)
    with gzip.open(python_problem_file, 'rb') as f:
        for line in f:
            line = eval(line)
            python_problems[line['task_id']] = line
    
    cpp_problem_file = '../data/humaneval-x/cpp/data/humaneval_cpp.jsonl.gz'
    cpp_problems = defaultdict(dict)
    with gzip.open(cpp_problem_file, 'rb') as f:
        for line in f:
            line = eval(line)
            cpp_problems[line['task_id']] = line

    if args.trans == 'cpp2python':
        entries = read_translation_dataset('../data/humaneval-x/cpp/data/humaneval_cpp.jsonl.gz',
                                            '../data/humaneval-x/python/data/humaneval_python.jsonl.gz',
                                            lang_src='cpp',
                                            lang_tgt='python',
                                            dataset_type="humaneval")
    elif args.trans == 'python2cpp':
        entries = read_translation_dataset('../data/humaneval-x/python/data/humaneval_python.jsonl.gz',
                                           '../data/humaneval-x/cpp/data/humaneval_cpp.jsonl.gz',
                                            lang_src='python',
                                            lang_tgt='cpp',
                                            dataset_type="humaneval")
    
    if args.trans == 'cpp2python':
        main(args.output, entries, python_problems, args.trans)
    elif args.trans == 'python2cpp':
        main(args.output, entries, cpp_problems, args.trans)

