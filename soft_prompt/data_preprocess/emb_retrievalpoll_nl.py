import numpy as np
from transformers import AutoModel, AutoTokenizer
import torch
import faiss
import pickle as pkl

import os
from tqdm import tqdm


os.environ["TOKENIZERS_PARALLELISM"] = "false"

checkpoint = "/home/jzchen/ML/Code/models/codet5p-110m-embedding"
device = "cuda"  # for GPU usage or "cpu" for CPU usage
tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
model = AutoModel.from_pretrained(checkpoint, trust_remote_code=True).to(device)

data_file = pkl.load(
    open("/home/jzchen/ML/Code/data/train/CodeContest/emb/nls.pkl", "rb")
)
results = []
for nl in tqdm(data_file):
    inputs = tokenizer.encode(nl, return_tensors="pt", max_length=2048).to(device)
    query_embed = model(inputs)[0]
    query_embed = query_embed.cpu().detach().numpy()
    results.append(query_embed)

results = np.stack(results)
print(results.shape)
np.save("/home/jzchen/ML/Code/data/train/CodeContest/emb/nls.npy", results)
