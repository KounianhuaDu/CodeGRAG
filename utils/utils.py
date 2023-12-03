import torch
import random
import numpy as np
import re
import subprocess

def seed_all(seed, gpu):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    
def extract_function_body(cpp_code):
    pattern = r"^\s*([\w:<>,\s]+)\s+([\w]+)\s*\((.*)\)\s*\{"
    
    match = re.match(pattern, cpp_code)
    
    if match:
        start = cpp_code.find('{')
        end = cpp_code.rfind('}')
        code = cpp_code[start+1:end+1]+'\n\n'
        return code
    else:
        if '}' not in cpp_code[-4:]:
             cpp_code += '}\n\n' 
        return cpp_code.strip()

def extract_generation_code(message, languge):
    cpp_code = re.findall(f'(?is)```(.*)```', message)[0]
    start = cpp_code.find('{')
    end = cpp_code.rfind('}')
    code = cpp_code[start+1:end+1]+'\n\n'
    return code
    