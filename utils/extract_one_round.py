import json
import os 
import argparse
from tqdm import tqdm

'''def extract_code(path, line):
    code_name = line['task_id'][4:]
    code_name = 'CPP_' + code_name
    code_content = line['declaration'] + line['generation']
    filename = os.path.join(path, code_name+'.cpp')
    file = open(filename, "w")
    file.write(code_content)
    file.close()'''

def extract_code(path, line):
    code_name = line['task_id'][7:]
    code_name = 'Python_' + code_name
    code_content = line['declaration'] + line['generation']
    filename = os.path.join(path, code_name+'.py')
    file = open(filename, "w")
    file.write(code_content)
    file.close()

def main(path, result_file, outputpath):
    with open(os.path.join(path, result_file), 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            line = json.loads(line)
            extract_code(outputpath, line)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Using different models to generate function")
    
    parser.add_argument("--path", default="../output", help="data path")
    parser.add_argument("--file", default="samples.jsonl", help="data path")
    parser.add_argument("--output", default="../output/OneRoundRes", help="data path")
    
    args = parser.parse_args()
    
    main(args.path, args.file, args.output)