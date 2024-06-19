import json
import os
from tqdm import tqdm

def transform(data_path, output_path):
    with open(data_path, 'r', encoding='utf-8') as f:
        cnt=0
        for line in tqdm(f):
            line = json.loads(line)
            line = str(line)
            code_content = line[12:-3]
            code_name = 'code_'+str(cnt)
            filename = os.path.join(output_path, code_name+'.cpp')
            file = open(filename, "w")
            file.write(code_content)
            file.close()
            cnt += 1

transform('../data/codes_onlycode.json', '../data/apex_data/')

    
