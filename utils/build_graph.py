import torch
import dgl 
import pickle as pkl
from collections import defaultdict
import argparse

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
    if data_dict:
        g = dgl.heterograph(data_dict)
    else:
        return None, None, None
    if label_map == None:
        label = None
    else:
        label = label_map[name]
    return g, label, name

def get_codes(name, root_path):
    if name[-4] == '.':
        name = name[:-4] 
    file_name = os.path.join(root_path, name+'.cpp')
    file = open(file_name, "r")
    content = file.read()
    return content

def process_data(data_path, label_path, output_path):
    graph_path = os.path.join(data_path, 'graph')
    graph_files = [os.path.join(graph_path, file) for file in os.listdir(graph_path)]
    
    if label_path:
        with open(os.path.join(label_path, 'ymap.pkl'), 'rb') as f:
            label_map = pkl.load(f)
    else:
        label_map = None
    
    graph_lists = []
    label_lists = []
    code_lists = []
    name_lists = []
    for file in tqdm(graph_files):
        g, label, name = build_graph(file, label_map)
        if not g:
            continue
        code = get_codes(name, os.path.join(data_path, 'code'))
        if not code:
            continue
        graph_lists.append(g)
        if label:
            label_lists.append(label)
        code_lists.append(code)
        name_lists.append(name)
    
    os.makedirs(output_path, exist_ok=True)
    with open(os.path.join(output_path, 'graphs.pkl'), 'wb') as f:
        pkl.dump(graph_lists, f)
    if label:
        with open(os.path.join(output_path, 'labels.pkl'), 'wb') as f:
            pkl.dump(label_lists, f)
    with open(os.path.join(output_path, 'codes.pkl'), 'wb') as f:
        pkl.dump(code_lists, f)
    with open(os.path.join(output_path, 'names.pkl'), 'wb') as f:
        pkl.dump(name_lists, f)
     
if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Using different models to generate function")
    
    parser.add_argument("--datapath", default="../data/FinalData/LongCodes", help="data path")
    parser.add_argument("--output", default="../output/FinalData/LongCodes", help="output path")

    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)      

    process_data(args.datapath, label_path = None, output_path=args.output)




