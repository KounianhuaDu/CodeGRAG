import os
import pickle as pkl

def extract(rootpath, language):
    codepath = os.path.join(rootpath, 'code')
    '''nlpath = os.path.join(rootpath, 'nl')
    rawcodepath = os.path.join(rootpath, 'rawcode')
    os.makedirs(nlpath, exist_ok=True)
    os.makedirs(rawcodepath, exist_ok=True)'''
    
    if language == 'cpp':
        shift = 2
    elif language == 'python':
        shift = 1
    nl_list = []
    rawcode_list = []
    codes = os.listdir(codepath)
    for codefile in codes:
        with open(os.path.join(codepath, codefile), 'r') as f:
            code = f.read()
        code = code.split('\n')
        nl = code[0][shift:]
        rawcode = '\n'.join(code[1:])
        nl_list.append(nl)
        rawcode_list.append(rawcode)
    
    with open(os.path.join(rootpath, 'nls.pkl'), 'wb') as f:
        pkl.dump(nl_list, f)
    with open(os.path.join(rootpath, 'rawcodes.pkl'), 'wb') as f:
        pkl.dump(rawcode_list, f)

extract('./data/Python', 'python')
    
