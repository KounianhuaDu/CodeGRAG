import json
import numpy as np
import torch

import os
import sys

sys.path.append("..")

from tqdm import tqdm
import pickle as pkl
import argparse
from unixcoder import UniXcoder
from unixcoder import UniXcoder
from transformers import AutoModel, AutoTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"
checkpoint = "/home/jzchen/ML/Code/models/codet5p-110m-embedding"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = UniXcoder("/home/jzchen/ML/codenew/Code/model_weights/unixcoder-base-nine")
# model.to(device)
tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
model = AutoModel.from_pretrained(checkpoint, trust_remote_code=True).to(device)


def main(codepath):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(os.path.join(codepath, "codes.pkl"), "rb") as f:
        documents = pkl.load(f)

    embed_list = []
    for start_idx in tqdm(range(0, len(documents), 256)):
        batch_docs = documents[start_idx : min(start_idx + 256, len(documents))]
        tokens_ids = tokenizer(
            batch_docs, padding="longest", truncation=True, return_tensors="pt"
        ).to(device)
        # inputs = tokenizer(batch_docs, padding='longest', truncation=True, return_tensors='pt').to(device)
        with torch.no_grad():
            outputs = model(**tokens_ids)
        embed_list.append(outputs)

    embeddings = torch.cat(embed_list, dim=0)
    embeddings = embeddings.cpu().numpy()
    print(embeddings.shape)
    np.save(os.path.join(codepath, "codes_emb.npy"), embeddings)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Using different models to generate function"
    )

    parser.add_argument(
        "--path",
        default="/home/jzchen/ML/AfterCodegrag/raw_data/APPS/dgl-train",
        help="data path",
    )

    args = parser.parse_args()

    main(args.path)
