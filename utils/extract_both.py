import os
import sys
sys.path.append("..")
import argparse
import pickle as pkl
from tqdm import tqdm
import json
import re
from collections import defaultdict
import torch 
import dgl
from algo.unixcoder import UniXcoder
from extract_cfg import build_cfg_from_code
from extract_dfg import build_dfg_from_code
from transformers import AutoModel, AutoTokenizer


def extract_meta(graph):
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
            return None
        etype = edge_info['edgeType']
        etypes.add(etype)
        
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

def build_graph(args, graph):
    ndata, edata = graph
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
        #g = dgl.heterograph(data_dict)
    else:
        return None
    
    node_types =[]
    node_names = []
    for node_info in ndata:
        node_types.append(node_info['kind'])
        node_names.append(node_info['name'])

    if args.model == 'codet5':
        types = tokenizer(node_types, padding='longest', truncation=True, return_tensors='pt').to(device)
        with torch.no_grad():
            types = model(**types)
        names = tokenizer(node_names, padding='longest', truncation=True, return_tensors='pt').to(device)
        with torch.no_grad():
            names = model(**names)
    else:
        tokens_ids = model.tokenize(node_types, mode="<encoder-only>", padding=True)
        source_ids = torch.tensor(tokens_ids).to(device)
        with torch.no_grad():
            tokens_embeddings, types = model(source_ids)
        tokens_ids = model.tokenize(node_names, mode="<encoder-only>", padding=True)
        source_ids = torch.tensor(tokens_ids).to(device)
        with torch.no_grad():
            tokens_embeddings, names = model(source_ids)
        
    types = types.cpu()
    names = names.cpu()
    g.nodes['node'].data['kind'] = types
    g.nodes['node'].data['type'] = names

    return g

def transform_to_homo(args, g):
    # node feature
    kind_feat = g.nodes['node'].data['kind']
    type_feat = g.nodes['node'].data['type']
    # edge feature
    etypes = g.etypes
    if args.model == 'codet5':
        etypes = tokenizer(etypes, padding='longest', truncation=True, return_tensors='pt').to(device)
        with torch.no_grad():
            etypes = model(**etypes)
    else:
        tokens_ids = model.tokenize(etypes, mode="<encoder-only>", padding=True)
        source_ids = torch.tensor(tokens_ids).to(device)
        with torch.no_grad():
            tokens_embeddings, etypes = model(source_ids)
    etypes = etypes.cpu()

    g = dgl.to_homogeneous(g)
    g.ndata['kind'] = kind_feat
    g.ndata['type'] = type_feat
    g.edata['h'] = etypes[g.edata['_TYPE']]
    return g


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Using different models to generate function")
    parser.add_argument("--data", default="../data/PythonData/LeetcodeData", help="data path")
    parser.add_argument("--output", default="../data/PythonData/LeetcodeData", help="output path")
    parser.add_argument("--model", choices=['codet5','unixcoder'])
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.model == 'codet5':
        checkpoint = "../model_weights/codet5p-110m-embedding"
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
        model = AutoModel.from_pretrained(checkpoint, trust_remote_code=True).to(device)
    else:
        model = UniXcoder("../model_weights/unixcoder-base-nine")
        model.to(device)

    code_list = []
    cfg_list = []
    dfg_list = []
    
    kind_graphs = []
    name_graphs = []
    etypes = []
    homo_gs = []
    i = 0
    with open(os.path.join(args.data, 'leetcode-with-description-python2359.jsonl'), 'r') as f:
        for line in tqdm(f):
            line = json.loads(line)
            problem_des = line['Question'].split('\n')
            problem_des = problem_des[0]
            raw_code = line['Answer']
            code = extract_generation_code(raw_code)
            if code:
                cfg = build_cfg_from_code(code)
                if cfg:
                    meta_name, meta_kind, etype = extract_meta(cfg)
                    cfg = build_graph(args, cfg)
                    if cfg:
                        homo = transform_to_homo(args, cfg)
                        if homo:
                            code = '#'+problem_des+'\n'+code
                            code_list.append(code)
                            cfg_list.append(cfg)
                            name_graphs.append(meta_name)
                            kind_graphs.append(meta_kind)
                            etypes.append(etype)
                            homo_gs.append(homo)
                            
            '''if code:
                dfg = build_dfg_from_code(code)
                if dfg:
                    cfg = build_cfg_from_code(code)
                    if cfg:
                        dfg = build_graph(dfg)
                        cfg = build_graph(cfg)
                        if dfg and cfg:
                            code_list.append(code)
                            cfg_list.append(cfg)
                            dfg_list.append(dfg)'''
    print(len(code_list))
    with open(os.path.join(args.output, 'codes.pkl'), 'wb') as f:
        pkl.dump(code_list, f)
    with open(os.path.join(args.output, 'graphs.pkl'), 'wb') as f:
        pkl.dump(cfg_list, f)
    with open(os.path.join(args.output, 'kind_graphs.pkl'), 'wb') as f:
        pkl.dump(kind_graphs, f)
    with open(os.path.join(args.output, 'name_graphs.pkl'), 'wb') as f:
        pkl.dump(name_graphs, f)
    with open(os.path.join(args.output, 'etype_graphs.pkl'), 'wb') as f:
        pkl.dump(etypes, f)
    with open(os.path.join(args.output, 'homo_graphs.pkl'), 'wb') as f:
        pkl.dump(homo_gs, f)
    '''with open(os.path.join(args.output, 'dfgs.pkl'), 'wb') as f:
        pkl.dump(dfg_list, f)'''
