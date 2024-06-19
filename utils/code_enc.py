import jieba
import json
import numpy as np
import torch
# from sentence_transformers import SentenceTransformer, util
from transformers import AutoModel, AutoTokenizer
import os
from tqdm import tqdm
import pickle as pkl
import argparse

def main(codepath):
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        checkpoint = "../model_weights/codet5p-110m-embedding"
        device = "cuda"  # for GPU usage or "cpu" for CPU usage

        tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
        model = AutoModel.from_pretrained(checkpoint, trust_remote_code=True).to(device)

        with open(os.path.join(codepath,'codes.pkl'), 'rb') as f:
                documents = pkl.load(f)

        embed_list = []
        for start_idx in tqdm(range(0, len(documents), 256)):
                batch_docs = documents[start_idx: min(start_idx + 256, len(documents))]
                inputs = tokenizer(batch_docs, padding='longest', truncation=True, return_tensors='pt').to(device)
                with torch.no_grad():
                        outputs = model(**inputs)
                embed_list.append(outputs)
        
        embeddings = torch.cat(embed_list, dim=0)
        embeddings = embeddings.cpu().numpy()

        np.save(os.path.join(codepath,'codes_emb.npy'), embeddings)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Using different models to generate function")
    
    parser.add_argument("--path", default="../data", help="data path")
    
    args = parser.parse_args()
    
    main(args.path)

    
    