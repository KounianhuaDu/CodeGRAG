import torch
import dgl
import os
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import pickle as pkl
import random
import pandas as pd
from utils import config
from tqdm import tqdm, trange
import numpy as np


from transformers import AutoModel, AutoTokenizer

'''os.environ["TOKENIZERS_PARALLELISM"] = "false"
checkpoint = "../model_weights/codet5p-110m-embedding"
device = "cuda"  # for GPU usage or "cpu" for CPU usage

tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
model = AutoModel.from_pretrained(checkpoint, trust_remote_code=True).to(device)
'''
'''os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UniXcoder("../model_weights/unixcoder-base-nine")
model.to(device)'''

class CodeGNNDataset(Dataset):
    def __init__(self, codes, graphs):
        #self.codes = torch.tensor(np.load(codes))
        with open(codes, 'rb') as f:
            self.codes = pkl.load(f)
        with open(graphs, 'rb') as f:
            self.graphs = pkl.load(f)
        print('Data Prepared')    

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
        batch_codes = np.stack(batch_codes)
        batch_graphs = dgl.batch(batch_graphs)
        return batch_codes, batch_graphs

def load_data(batch_size, path='../data', num_workers=8):
    dataset = CodeGNNDataset(os.path.join(path, 'codes.pkl'), os.path.join(path, 'homo_graphs.pkl'))
    collator = Collator()
    loader= DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collator.collate)
    return loader
    