import torch
import dgl
import os
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import pickle as pkl
import random
import pandas as pd
from support import config
from tqdm import tqdm, trange
import numpy as np


from transformers import AutoModel, AutoTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"
checkpoint = "/home/jzchen/ML/Code/models/codet5p-110m-embedding"
device = "cuda"  # for GPU usage or "cpu" for CPU usage

tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
model = AutoModel.from_pretrained(checkpoint, trust_remote_code=True).to(device)


class CodeGNNDataset(Dataset):
    def __init__(self, codes, graphs):
        self.codes = torch.tensor(np.load(codes))
        with open(graphs, "rb") as f:
            self.graphs = pkl.load(f)
        print("Data Prepared")
        self.len = len(self.graphs)

    def __getitem__(self, index):
        return self.codes[index], self.graphs[index]

    def __len__(self):
        return self.len


class Collator(object):
    def __init__(self):
        pass

    def collate(self, batch):
        batch_codes, batch_graphs = map(list, zip(*batch))
        batch_graphs = list(
            map(
                lambda x: dgl.to_homogeneous(x, edata=["h"], ndata=["kind", "type"]),
                batch_graphs,
            )
        )
        # → dgl.batch(g_less_etypes, g_more_etypes) # works perfectly
        # → dgl.batch(g_more_etypes, g_less_etypes) # gives error
        # batch_graphs = sorted(batch_graphs, key=lambda x: len(x.etypes))
        batch_codes = torch.stack(batch_codes)
        batch_graphs = dgl.batch(batch_graphs)
        return batch_codes, batch_graphs


def load_data(batch_size, path="../data", num_workers=8):
    dataset = CodeGNNDataset(
        os.path.join(path, "codes_emb.npy"),
        os.path.join(path, "graphs.pkl"),
        # os.path.join(path, "codes_emb.npy"),
        # '/home/jzchen/ML/AfterCodegrag/raw_data/APPS/cfg/2909_0.pkl'
    )
    collator = Collator()
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collator.collate,
    )
    return loader
