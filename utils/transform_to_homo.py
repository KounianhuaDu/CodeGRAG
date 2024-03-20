import torch
import dgl
import os
import pickle as pkl
from tqdm import tqdm, trange
import argparse
from transformers import AutoModel, AutoTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"
checkpoint = "../model_weights/codet5p-110m-embedding"
device = "cuda"  # for GPU usage or "cpu" for CPU usage
tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
model = AutoModel.from_pretrained(checkpoint, trust_remote_code=True).to(device)

def transform(path):
    with open(os.path.join(path, 'graphs.pkl'), 'rb') as f:
        graphs = pkl.load(f)

    homo_gs = []
    for g in tqdm(graphs):
        # node feature
        kind_feat = g.nodes['node'].data['kind']
        type_feat = g.nodes['node'].data['type']
        # edge feature
        etypes = g.etypes
        etypes = tokenizer(etypes, padding='longest', truncation=True, return_tensors='pt').to(device)
        with torch.no_grad():
            etypes = model(**etypes).cpu()
        g = dgl.to_homogeneous(g)
        g.ndata['kind'] = kind_feat
        g.ndata['type'] = type_feat
        g.edata['h'] = etypes[g.edata['_TYPE']]
        homo_gs.append(g)
    
    with open(os.path.join(path, 'homo_graphs.pkl'), 'wb') as f:
        pkl.dump(homo_gs, f)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Using different models to generate function")
    parser.add_argument("--data", default="../data/CPPData/CodeT5/LongCodes", help="data path")
    args = parser.parse_args()

    transform(args.data)