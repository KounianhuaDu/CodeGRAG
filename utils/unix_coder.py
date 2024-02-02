import jieba
import json
import numpy as np
import torch

import os
import sys
sys.path.append("..")

from tqdm import tqdm
import pickle as pkl
import argparse
from algo.unixcoder import UniXcoder


def main(codepath):
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = UniXcoder("../model_weights/unixcoder-base-nine")
        model.to(device)

        with open(os.path.join(codepath,'codes.pkl'), 'rb') as f:
                documents = pkl.load(f)

        embed_list = []
        for start_idx in tqdm(range(0, len(documents), 256)):
                batch_docs = documents[start_idx: min(start_idx + 256, len(documents))]
                tokens_ids = model.tokenize(batch_docs, mode="<encoder-only>", padding=True)
                source_ids = torch.tensor(tokens_ids).to(device)
                #inputs = tokenizer(batch_docs, padding='longest', truncation=True, return_tensors='pt').to(device)
                with torch.no_grad():
                        tokens_embeddings, outputs = model(source_ids)
                embed_list.append(outputs)
        
        embeddings = torch.cat(embed_list, dim=0)
        embeddings = embeddings.cpu().numpy()

        np.save(os.path.join(codepath,'codes_emb.npy'), embeddings)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Using different models to generate function")
    
    parser.add_argument("--path", default="../data", help="data path")
    
    args = parser.parse_args()
    
    main(args.path)

    
    