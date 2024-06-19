import os
from collections import defaultdict
import pickle as pkl
import json
from tqdm import tqdm

root_path = '../data/humaneval-x/'

cpp_file = os.path.join(root_path, 'cpp', 'data', 'humaneval_cpp.jsonl')
python_file = os.path.join(root_path, 'python', 'data', 'humaneval_python.jsonl')
contents = defaultdict(dict)

with open(cpp_file, 'r') as f:
    for line in tqdm(f):
        line = json.loads(line)
        task_id = line['task_id'][4:]
        prompt = line['prompt']
        dec = line['declaration']
        solution = line['canonical_solution']
        contents[task_id]['cpp'] = [prompt, dec, solution]

with open(python_file, 'r') as f:
    for line in tqdm(f):
        line = json.loads(line)
        task_id = line['task_id'][7:]
        prompt = line['prompt']
        dec = line['declaration']
        solution = line['canonical_solution']
        contents[task_id]['python'] = [prompt, dec, solution]

with open('crossLingual.pkl', 'wb') as f:
    pkl.dump(contents, f)






    