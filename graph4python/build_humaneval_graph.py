import dgl 
import pickle as pkl
from collections import defaultdict
import argparse
from tqdm import tqdm
import os

def build_graph(path):
    with open(path, 'rb') as f:
        ndata, edata = pkl.load(f)
        
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
    num_nodes_dict = {
        'node': len(ndata)
    }

    if max_nid >= len(ndata):
        return None

    if data_dict:
        g = dgl.heterograph(data_dict, num_nodes_dict=num_nodes_dict)
    else:   
        return None
    
    node_kinds =[]
    node_types = []
    for node_info in ndata:
        node_kinds.append(node_info['kind'])
        node_types.append(node_info['name'])
    return g

def process_data(data_path, output_path):
    graph_lists = os.listdir(data_path)
    for file in tqdm(graph_lists):
        g = build_graph(os.path.join(data_path, file))
        if not g:
            continue
        with open(os.path.join(output_path, file), 'wb') as f:
            pkl.dump(g, f)

     
if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Using different models to generate function")
    
    parser.add_argument("--datapath", default="./data/humaneval_graphs/cfg", help="data path")
    parser.add_argument("--output", default="./data/humaneval_graphs/dgl", help="output path")

    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)      

    process_data(args.datapath, output_path=args.output)




