import click
import json
import os 
import numpy as np
import openai
import logging
import argparse
from tqdm import tqdm
import re


API_KEY = "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"
openai.api_key = API_KEY

os.environ["http_proxy"] = "http://127.0.0.1:8888"
os.environ["https_proxy"] = "http://127.0.0.1:8888"
os.environ["all_proxy"]="socks5://127.0.0.1:8889"

from human_eval.data import write_jsonl, read_problems
from codes.codet5 import get_prompt_code
from retrieval_models.faissSearch import get_prompt_faiss
from textbook.fas import get_prompt_textbook

qa_file_path = {
        "wiki-C": '/home/share/CodeMateData/QAData/crawler/dd_wiki-C-492.jsonl',
        "wiki-Alg":'/home/share/CodeMateData/QAData/crawler/dd_wiki-Alg-489.jsonl',
        "stackoverflow-DS":'/home/share/CodeMateData/QAData/crawler/dd_stackoverflow-DS-33124.jsonl',
        "stackoverflow-Alg":'/home/share/CodeMateData/QAData/crawler/dd_stackoverflow-Alg-11561.jsonl',
        "stackoverflow-C":'/home/share/CodeMateData/QAData/crawler/dd_stackoverflow-C-38405.jsonl',
        "CSDN-DS":'/home/share/CodeMateData/QAData/crawler/dd_filteredCSDN-DS-15201.jsonl',
        "CSDN-C":'/home/share/CodeMateData/QAData/crawler/dd_filteredCSDN-C-10746.jsonl',
        "CSDN-Alg":'/home/share/CodeMateData/QAData/crawler/dd_filteredCSDN-Alg-5718.jsonl',
        "leetcode":'/home/share/CodeMateData/QAData/crawler/leetcode_2442.jsonl',
        "high-train":'/home/share/CodeMateData/QAData/crawler/trainingData/highQualityTrain.jsonl',
        "high-test":'/home/share/CodeMateData/QAData/crawler/trainingData/highQualityTest.jsonl',
        "BELLE2":'/home/share/CodeMateData/QAData/dataset_filtered/Belle2_Positive-83016.jsonl',
        "BELLE":'/home/share/CodeMateData/QAData/dataset_filtered/Belle_Positive_1809.jsonl',
        "moss":'/home/share/CodeMateData/QAData/dataset_filtered/filtered-moss-1187390.jsonl',
        "total":'retrieval_models/QA_total.json',
        "wiki":'retrieval_models/wiki.json',
        "stackoverflow":'retrieval_models/stackoverflow.json',
        "textbook":'/home/wmzhang/codemate/newtest/textbook/textbook.json',
        "filtered":'retrieval_models/filtered.json',
        "codes": '/home/wmzhang/codemate/codeT5+/output.json',
        "codes_total": "/home/rrt/codemate/CodeTest/codes/codes_total.json", 
        "python_qa": "/home/share/CodeMateData/CodeData/LeetcodeData/leetcode-with-description-python2359.jsonl",
        "codes_qa": "/home/rrt/codemate/CodeGeeX/codes/codes_qa.json"
        }

emb_path = {
        "wiki-C": '/home/wmzhang/codemate/newtest/retrieval_models/wiki-C.npy',
        "wiki-Alg":'/home/wmzhang/codemate/newtest/retrieval_models/wiki-Alg.npy',
        "stackoverflow-DS":'/home/wmzhang/codemate/newtest/retrieval_models/stackoverflow-DS.npy',
        "stackoverflow-Alg":'/home/wmzhang/codemate/newtest/retrieval_models/stackoverflow-Alg.npy',
        "stackoverflow-C":'/home/wmzhang/codemate/newtest/retrieval_models/stackoverflow-C.npy',
        "CSDN-DS":'/home/wmzhang/codemate/newtest/retrieval_models/CSDN-DS.npy',
        "CSDN-C":'/home/wmzhang/codemate/newtest/retrieval_models/CSDN-C.npy',
        "CSDN-Alg":'/home/wmzhang/codemate/newtest/retrieval_models/CSDN-Alg.npy',
        "leetcode":'/home/wmzhang/codemate/newtest/retrieval_models/leetcode.npy',
        "high-train":'/home/wmzhang/codemate/newtest/retrieval_models/high-train.npy',
        "high-test":'/home/wmzhang/codemate/newtest/retrieval_models/high-test.npy',
        "BELLE2":'/home/wmzhang/codemate/newtest/retrieval_models/BELLE2.npy',
        "BELLE":'/home/wmzhang/codemate/newtest/retrieval_models/BELLE.npy',
        "moss":'/home/wmzhang/codemate/newtest/retrieval_models/moss.npy',
        "total":'retrieval_models/total.npy',
        "wiki":'retrieval_models/wiki.npy',
        "stackoverflow":'retrieval_models/stackoverflow.npy',
        "textbook":'/home/wmzhang/codemate/newtest/textbook/textbook.npy',
        "filtered":'retrieval_models/filtered.npy',
        "codes": '/home/wmzhang/codemate/codeT5+/code.npy',
        "codes_total": "/home/rrt/codemate/CodeTest/codes/codes_total.npy",
        "python_qa": "/home/rrt/codemate/humaneval/codes/leetcode_qa.npy",
        "codes_qa": "/home/rrt/codemate/CodeGeeX/codes/codes_qa.npy"
                }
                

problems = read_problems()

def generate_one_completion(task):
    qa_file = "codes_qa"
    if qa_file != "None":
        #retrieval_part
        data_list = []
        with open(qa_file_path[qa_file], 'r', encoding='utf-8') as f:
            for line in f:
               data = json.loads(line)
               data_list.append(data)
        embeddings = np.load(emb_path[qa_file])
    query = task  
    if qa_file == "textbook":
        knowledge = get_prompt_textbook(query, 1, embeddings, data_list)
    elif qa_file in ["codes", "codes_total", "python_qa", "codes_qa"]:
        knowledge = get_prompt_code(query, 1, embeddings, data_list)
    elif qa_file != "None" :
        knowledge = get_prompt_faiss(query, 1, embeddings, data_list)
    
    prompt = "Here is a code generation question, please generate python function codes according to the requirements of the question, do not output redundant parsing, function declaration and other information and only output compilable code.\n"
    #prompt = "Here is a code generation question, please generate python function codes according to the requirements of the question. Let's think of it step by step.\n"
    
    if qa_file in ["textbook", "total"]:
        input = prompt + '\nrelative knowledge for reference:\n' + knowledge + '\n' + task
    elif qa_file in ["codes", "codes_total", "python_qa", "codes_qa"]:
        input = prompt + '\nrelative codes for reference:\n' + knowledge + '\n' + task
    elif qa_file == "None":
        input = prompt + task
    
    print(input)
    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=[{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": input}],
        max_tokens=1024,  # 调整生成文本的长度
        temperature=0.01,  # 调整生成文本的随机性
        top_p=0.75,

    )
    
    
    message = response.choices[0]["message"]["content"]

    parts = message.split('```python', 1)
    if len(parts) > 1:
        code = parts[1]
    else:
        code = message

    parts_2 = code.split('```', 1)

    if len(parts_2) > 1:
        code = parts_2[0]
    else:
        code = code
    
    print(code)
    exit()

    return code.strip()


def main():

    while(True):
        check_point_path = '/home/rrt/codemate/humaneval/check_point.npy'
        if not os.path.exists(check_point_path):
            samples = []
        else:
            samples = np.load(check_point_path,allow_pickle=True).tolist()

        if int(len(samples)) >= 164:
            break

        try:
            start_task_id = len(samples)

            for task_id in tqdm(problems):
                if int(task_id[10:]) < int(start_task_id):
                    continue
                else:
                    completion=generate_one_completion(problems[task_id]["prompt"])
                    print(completion)
                    temp_dict = dict(task_id=task_id, completion=completion)
                    samples.append(temp_dict)

            write_jsonl("python_None.jsonl", samples)

            if int(len(samples)) >= 164:
                break

        except Exception as e:
            print(str(e))
            np.save(check_point_path, samples)
        
        return 0
    

def check():
    check_point_path = '/home/hcchai/codemate_code/humaneval/check_point.npy'
    samples = np.load(check_point_path, allow_pickle=True).tolist()
    print(samples)
    print(len(samples))

if __name__=="__main__":
    main()
    # check()
    pass
