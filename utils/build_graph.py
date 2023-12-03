import torch
import dgl 
import pickle as pkl
from collections import defaultdict

import os
from tqdm import tqdm
import subprocess

def build_graph(path, label_map):
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
    g = dgl.heterograph(data_dict)
    label = label_map[name]
    return g, label, name

def get_codes(name, root_path = '../data/Cgraphs/source_code'):
    file_name = os.path.join(root_path, name+'.cpp')
    file = open(file_name, "r")
    content = file.read()
    return content

def process_data(data_path='../data/parsed-graph/all_parsed_data', label_path = '../data/parsed-graph', output_path='../data/Cgraphs'):
    files = [os.path.join(data_path, file) for file in os.listdir(data_path)]
    with open(os.path.join(label_path, 'ymap.pkl'), 'rb') as f:
        label_map = pkl.load(f)
    
    graph_lists = []
    label_lists = []
    code_lists = []
    name_lists = []
    for file in tqdm(files):
        g, label, name = build_graph(file, label_map)
        code = get_codes(name)
        if not code:
            continue
        graph_lists.append(g)
        label_lists.append(label)
        code_lists.append(code)
        name_lists.append(name)
    
    os.makedirs(output_path, exist_ok=True)
    with open(os.path.join(output_path, 'graphs.pkl'), 'wb') as f:
        pkl.dump(graph_lists, f)
    with open(os.path.join(output_path, 'labels.pkl'), 'wb') as f:
        pkl.dump(label_lists, f)
    with open(os.path.join(output_path, 'codes.pkl'), 'wb') as f:
        pkl.dump(code_lists, f)
    with open(os.path.join(output_path, 'names.pkl'), 'wb') as f:
        pkl.dump(name_lists, f)
     
process_data()




