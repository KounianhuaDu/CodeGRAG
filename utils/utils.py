import torch
import random
import numpy as np
import re
import subprocess
import gzip
import json

def seed_all(seed, gpu):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)

def stream_jsonl(filename: str):
    """
    Parses each jsonl line and yields it as a dictionary
    """
    if filename.endswith(".gz"):
        with open(filename, "rb") as gzfp:
            with gzip.open(gzfp, "rt") as fp:
                for line in fp:
                    if any(not x.isspace() for x in line):
                        yield json.loads(line)
    else:
        with open(filename, "r") as fp:
            for line in fp:
                if any(not x.isspace() for x in line):
                    yield json.loads(line)
    
def extract_function_body(raw_code, language):
    if language == 'python':
        code_ = []
        start = False
        for line in raw_code.split("\n"):
            if line.strip().startswith('def'):
                start = True
                continue
            if start and (len(line.strip()) > 0 and line[0] != ' ' and line[0] != '\t'):
                break
            if start:
                code_.append(line)
        code = "\n".join(code_)
        return code
    elif language == 'c++':
        pattern = r"^\s*([\w:<>,\s]+)\s+([\w]+)\s*\((.*)\)\s*\{"
        match = re.match(pattern, raw_code)
        if match:
            start = raw_code.find('{')
            end = raw_code.rfind('}')
            code = raw_code[start+1:end+1]+'\n\n'
            return code
        else:
            if '}' not in raw_code[-4:]:
                raw_code += '}\n\n' 
            return raw_code.strip()
    elif language == 'java':
        start = raw_code.find('{')
        end = raw_code.rfind('}')
        code = raw_code[start+1:end+1]+'\n}'
        return code

def extract_generation_code(message, languge):
    raw_code = re.findall(f'(?is)```(.*)```', message)[0]
    start = raw_code.find('{')
    end = raw_code.rfind('}')
    code = raw_code[start+1:end+1]+'\n\n'
    return code
    