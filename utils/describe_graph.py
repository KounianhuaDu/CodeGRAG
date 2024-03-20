import dgl
import networkx as nx
from matplotlib import pyplot as plt
import numpy as np 
import pickle as pkl
import argparse
import os
from collections import defaultdict
from tqdm import tqdm

import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap, ListedColormap


def build_graph(path, label_map):
    with open(path, 'rb') as f:
        ndata, edata, name = pkl.load(f)

    types = []
    kinds = []
    names = []
    etypes = set()
    for node_info in ndata:
        types.append(node_info['type'])
        kinds.append(node_info['kind'])
        names.append(node_info['name'])

    
    edges_src = defaultdict(list)
    edges_dst = defaultdict(list)
    
    type_graph = defaultdict(list)
    kind_graph = defaultdict(list)
    name_graph = defaultdict(list)

    max_nid = 0
    for edge_info in edata:
        src, dst = edge_info.get('between', [-1, -1])
        if src == -1:
            src, dst = edge_info['betweeen']
        if src is None or dst is None:
            return None
        etype = edge_info['edgeType']
        etypes.add(etype)
        
        src_type = types[src]
        dst_type = types[dst]
        type_graph[(src_type, etype, dst_type)].append((src, dst))

        src_kind = kinds[src]
        dst_kind = kinds[dst]
        kind_graph[(src_kind, etype, dst_kind)].append((src, dst))

        src_name = names[src]
        dst_name = names[dst]
        name_graph[(src_name, etype, dst_name)].append((src, dst))

        max_nid = max(max_nid, src, dst)
    
    meta_type = [key for key, value in type_graph.items()]
    meta_kind = [key for key, value in kind_graph.items()]
    meta_name = [key for key, value in name_graph.items()]
    return meta_type, meta_kind, meta_name, list(etypes)

def process_data(args, data_path, label_path, output_path):
    graph_path = os.path.join(data_path, 'graph')
    graph_files = [os.path.join(graph_path, file) for file in os.listdir(graph_path)]
    
    label_map = None
    type_graphs = []
    kind_graphs = []
    name_graphs = []
    etypes = []
    for file in tqdm(graph_files):
        meta_type, meta_kind, meta_name, etype = build_graph(file, label_map)
        type_graphs.append(meta_type)
        kind_graphs.append(meta_kind)
        name_graphs.append(meta_name)
        etypes.append(etype)
    
    with open(os.path.join(output_path, 'type_graphs.pkl'), 'wb') as f:
        pkl.dump(type_graphs, f)
    with open(os.path.join(output_path, 'kind_graphs.pkl'), 'wb') as f:
        pkl.dump(kind_graphs, f)
    with open(os.path.join(output_path, 'name_graphs.pkl'), 'wb') as f:
        pkl.dump(name_graphs, f)
    with open(os.path.join(output_path, 'etype_graphs.pkl'), 'wb') as f:
        pkl.dump(etypes, f)
        

     
if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Using different models to generate function")
    
    parser.add_argument("--datapath", default="../data/CPPData/CodeT5/LongCodes", help="data path")
    parser.add_argument("--output", default="../data/CPPData/CodeT5/LongCodes", help="output path")
    parser.add_argument("--i", type=int, default=0)

    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)      

    process_data(args,args.datapath, label_path = None, output_path=args.output)





