import torch
import dgl 
import pickle as pkl
from collections import defaultdict

import os
from tqdm import tqdm
import subprocess

def build_graph(path):
    with open(path, 'rb') as f:
        ndata, edata, name = pkl.load(f)
    
    edges_src = defaultdict(list)
    edges_dst = defaultdict(list)
    etypes = set()

    for edge_info in edata:
        src, dst = edge_info['between']
        etype = edge_info['edgeType']
        etypes.add(etype)
        edges_src[etype].append(src)
        edges_dst[etype].append(dst)
    
    data_dict = {
        ('node', etype, 'node'): (edges_src[etype], edges_dst[etype])
        for etype in list(etypes)
    }
    #print(data_dict)
    if data_dict:
        g = dgl.heterograph(data_dict)
    else:
        g=None
    return g, name

def get_codes(name, root_path = '../data/rrt/source_codes'):
    file_name = os.path.join(root_path, name[:-4]+'.cpp')
    file = open(file_name, "r")
    content = file.read()
    return content

def process_data(data_path='../data/rrt/graph',  output_path='../data/rrt/processed'):
    files = [os.path.join(data_path, file) for file in os.listdir(data_path)]
    
    graph_lists = []
    code_lists = []
    for file in tqdm(files):
        g, name = build_graph(file)
        code = get_codes(name)   
        if not g:
            continue
        if not code:
            continue
        graph_lists.append(g)
        code_lists.append(code)
    
    os.makedirs(output_path, exist_ok=True)
    with open(os.path.join(output_path, 'graphs.pkl'), 'wb') as f:
        pkl.dump(graph_lists, f)
    with open(os.path.join(output_path, 'codes.pkl'), 'wb') as f:
        pkl.dump(code_lists, f)
    
process_data()




