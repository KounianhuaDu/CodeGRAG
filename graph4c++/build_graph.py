import torch
import dgl 
import pickle as pkl
from collections import defaultdict
import argparse

import os
from tqdm import tqdm
import subprocess

from transformers import AutoModel, AutoTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"
checkpoint = "../model_weights/codet5p-110m-embedding"
device = "cuda"  # for GPU usage or "cpu" for CPU usage

tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
model = AutoModel.from_pretrained(checkpoint, trust_remote_code=True).to(device)

def build_graph(path, label_map):
    with open(path, 'rb') as f:
        ndata, edata, name = pkl.load(f)
    
    edges_src = defaultdict(list)
    edges_dst = defaultdict(list)
    etypes = set()
    max_nid = 0

    for edge_info in edata:
        src, dst = edge_info.get('between', [-1, -1])
        if src == -1:
            src, dst = edge_info['betweeen']
        if src is None or dst is None:
            return None
        etype = edge_info['edgeType']
        etypes.add(etype)
        edges_src[etype].append(src)
        edges_dst[etype].append(dst)
        max_nid = max(max_nid, src, dst)
    
    data_dict = {
        ('node', etype, 'node'): (edges_src[etype], edges_dst[etype])
        for etype in list(etypes)
    }
    #print(data_dict)
    num_nodes_dict = {
        'node': len(ndata)
    }

    if max_nid >= len(ndata):
        return None

    if data_dict:
        g = dgl.heterograph(data_dict, num_nodes_dict=num_nodes_dict)
        #g = dgl.heterograph(data_dict)
    else:   
        return None, None, None

    if label_map == None:
        label = None
    else:
        label = label_map[name]
    
    node_kinds =[]
    node_types = []
    for node_info in ndata:
        node_kinds.append(node_info['kind'])
        node_types.append(node_info['type'])
    

    kinds = tokenizer(node_kinds, padding='longest', truncation=True, return_tensors='pt').to(device)
    with torch.no_grad():
        kinds = model(**kinds)
    
    types = tokenizer(node_types, padding='longest', truncation=True, return_tensors='pt').to(device)
    with torch.no_grad():
        types = model(**types)
    
   
    kinds = kinds.cpu()
    types = types.cpu()
    g.nodes['node'].data['kind'] = kinds
    g.nodes['node'].data['type'] = types
    
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

    print(len(code_lists))
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
    
    parser.add_argument("--datapath", default="../data/CPPData/CodeT5/LongCodes", help="data path")
    parser.add_argument("--output", default="../data/CPPData/CodeT5/LongCodes", help="output path")

    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)      

    process_data(args.datapath, label_path = None, output_path=args.output)




