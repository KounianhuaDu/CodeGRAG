import json
import gzip
from collections import defaultdict
import os

problems = defaultdict(dict)
with gzip.open('../data/humaneval-x/python/data/humaneval_python.jsonl.gz', 'rb') as f:
    for line in f:
        line = eval(line)
        code_name = line['task_id'][7:] + '.py'
        dec = line['declaration']
        code = line['canonical_solution']
        code = dec+code
        file = open(os.path.join('../data/transcode/pyth' , code_name), "w")
        file.write(code)
        file.close()
        