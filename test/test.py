import json
import gzip
import os
import sys
sys.path.append("..")
from tqdm import tqdm
from collections import defaultdict
from codegeex.data.data_utils import write_jsonl

problem_file = '../data/humaneval-x/java/data/humaneval_java.jsonl.gz'
    
problems = defaultdict(dict)
with gzip.open(problem_file, 'rb') as f:
    for line in f:
        line = eval(line)
        problems[line['task_id']] = line

samples = []
shift = 5
start_task_id = len(samples)
for task_id in tqdm(problems):
    if int(task_id[shift:]) < int(start_task_id):
        continue
    else:
        completion = problems[task_id]['canonical_solution']
        temp_dict = dict(task_id=task_id, generation=completion, prompt=problems[task_id]["prompt"], test=problems[task_id]["test"], declaration=problems[task_id]["declaration"])
        samples.append(temp_dict)

write_jsonl('samples.jsonl', samples)
