import dgl
import networkx as nx
from matplotlib import pyplot as plt
import numpy as np 
import pickle as pkl
import argparse
import os
from collections import defaultdict
from tqdm import tqdm
import json
import re

from extract_cfg import build_cfg_from_code
from extract_dfg import build_dfg_from_code

import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap, ListedColormap


def build_graph(graph):
    ndata, edata = graph

    names = []
    kinds = []
    etypes = set()
    for node_info in ndata:
        names.append(node_info['name'])
        kinds.append(node_info['kind'])

    edges_src = defaultdict(list)
    edges_dst = defaultdict(list)
    
    name_graph = defaultdict(list)
    kind_graph = defaultdict(list)

    max_nid = 0
    for edge_info in edata:
        src, dst = edge_info.get('between', [-1, -1])
        if src == -1:
            src, dst = edge_info['betweeen']
        if src is None or dst is None:
            return None, None, None
        etype = edge_info['edgeType']
        etypes.add(etype)
        
        if src >= len(names) or dst >= len(names):
            return None, None, None
        src_type = names[src]
        dst_type = names[dst]
        name_graph[(src_type, etype, dst_type)].append((src, dst))

        src_kind = kinds[src]
        dst_kind = kinds[dst]
        kind_graph[(src_kind, etype, dst_kind)].append((src, dst))

        max_nid = max(max_nid, src, dst)
    
    meta_name = [key for key, value in name_graph.items()]
    meta_kind = [key for key, value in kind_graph.items()]

    return meta_name, meta_kind, list(etypes)

def extract_generation_code(message):
    raw_code = re.findall(f'(?is)```(.*)```', message)
    if not raw_code:
        return None
    else:
        raw_code = raw_code[0]
    raw_code = raw_code.split('\n')
    if raw_code[0] == 'python':
        raw_code = raw_code[1:]
    raw_code = '\n'.join(raw_code)
    return raw_code

def process_data(args, data_path, label_path, output_path):
    name_graphs = []
    kind_graphs = []
    etypes = []
    code_list = []
    with open(os.path.join(data_path, 'leetcode-with-description-python2359.jsonl'), 'r') as f:
        for line in tqdm(f):
            line = json.loads(line)
            problem_des = line['Question'].split('\n')
            problem_des = problem_des[0]
            raw_code = line['Answer']
            code = extract_generation_code(raw_code)
            if code:
                cfg = build_cfg_from_code(code)
                if cfg:
                    meta_name, meta_kind, etype = build_graph(cfg)
                    if meta_name:
                        code = '#'+problem_des+'\n'+code
                        code_list.append(code)
                        name_graphs.append(meta_name)
                        kind_graphs.append(meta_kind)
                        etypes.append(etype)
    print(len(name_graphs))
    with open(os.path.join(output_path, 'name_graphs.pkl'), 'wb') as f:
        pkl.dump(name_graphs, f)
    with open(os.path.join(output_path, 'kind_graphs.pkl'), 'wb') as f:
        pkl.dump(kind_graphs, f)
    with open(os.path.join(output_path, 'etype_graphs.pkl'), 'wb') as f:
        pkl.dump(etypes, f)
    with open(os.path.join(output_path, 'codes.pkl'), 'wb') as f:
        pkl.dump(code_list, f)
        

     
if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Using different models to generate function")
    
    parser.add_argument("--datapath", default="../data/CPPData/CodeT5/LongCodes", help="data path")
    parser.add_argument("--output", default="../data/CPPData/CodeT5/LongCodes", help="output path")
    parser.add_argument("--i", type=int, default=0)

    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)      

    process_data(args,args.datapath, label_path = None, output_path=args.output)





