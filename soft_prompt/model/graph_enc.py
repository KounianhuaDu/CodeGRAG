import jieba
import json
import numpy as np
import torch
import dgl

import os
import sys

sys.path.append("..")

from gnn import CodeGNN
from tqdm import tqdm
import pickle as pkl
import argparse


def main(path):
    device = "cuda"  # for GPU usage or "cpu" for CPU usage
    model = CodeGNN(256, 256).to(device)
    model.load_state_dict(
        torch.load(
            "/home/jzchen/ML/AfterCodegrag/trained_models/appsnew/CodeGNN/CodeGNN_0.0001_1e-05_1_0.5.pth"
        )
    )
    model.eval()

    with open(os.path.join(path, "graphs.pkl"), "rb") as f:
        graphs = pkl.load(f)

    embed_list = []
    for start_idx in tqdm(range(0, len(graphs), 256)):
        batch_gs = graphs[start_idx : min(start_idx + 256, len(graphs))]
        batch_gs = list(
            map(
                lambda x: dgl.to_homogeneous(x, edata=["h"], ndata=["kind", "type"]),
                batch_gs,
            )
        )
        batch_gs = dgl.batch(batch_gs).to(device)
        outputs = model.get_graph_embedding(batch_gs)
        embed_list.append(outputs.detach().cpu())

    embeddings = torch.cat(embed_list, dim=0)
    embeddings = embeddings.numpy()

    np.save(os.path.join(path, "graphs_emb.npy"), embeddings)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Using different models to generate function"
    )

    parser.add_argument(
        "--path",
        default="/home/jzchen/ML/AfterCodegrag/raw_data/APPS/dgl",
        help="data path",
    )

    args = parser.parse_args()

    main(args.path)
