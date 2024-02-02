import torch
import jieba
import json
import numpy as np
from transformers import AutoModel, AutoTokenizer
import faiss
import pickle as pkl
from sklearn import preprocessing
from algo.unixcoder import UniXcoder

import os
import sys
sys.path.append("..")

os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UniXcoder("../model_weights/unixcoder-base-nine")
model.to(device)

def construct_faiss_index(embeddings):
    #pca = decomposition.PCA(n_components=16, svd_solver='full')
    
    pca = faiss.PCAMatrix(embeddings.shape[-1], 32)
    '''
    pca.train(embeddings)
    assert pca.is_trained
    embeddings = pca.apply(embeddings)
    '''

    #embeddings = preprocessing.normalize(embeddings, 'l2')

    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embeddings)
    return index, pca

def search_with_faiss(query, data_list, index, pca, k):
    '''inputs = tokenizer.encode(query, return_tensors = "pt").to(device)
    query_embed = model(inputs)[0]
    query_embed = query_embed.cpu().detach().numpy() '''

    tokens_ids = model.tokenize([query], mode="<encoder-only>", padding=True)
    source_ids = torch.tensor(tokens_ids).to(device)
    with torch.no_grad():
        tokens_embeddings, query_embed = model(source_ids)

    query_embed = query_embed.cpu().detach().numpy() 
    #query_embed = pca.apply(query_embed)
    #query_embed = preprocessing.normalize(query_embed, 'l2')
    
    distances, indices = index.search(query_embed, k)

    prompt_str_list = [str(data_list[idx.item()]) for idx in indices[0][:k]]
    prompt = '\n'.join(prompt_str_list)
    
    return prompt


  
if __name__ == '__main__':
    data_path = '../data/Cgraphs'
    embeddings_path = os.path.join(data_path, 'codes_emb.npy')
    codes_path = os.path.join(data_path, 'codes.pkl')

    embeddings = np.load(embeddings_path)
    with open(codes_path, 'rb') as f:
        data_list = pkl.load(f)

    index, pca = construct_faiss_index(embeddings)
    print(1)
    query = '#include<iostream.h>\n#define SQR(x) x*x\nvoid main()\n{\nint a=10,k=2,m=1;\na/=SQR(k+m);cout<<a;\n}\n执行上面的C++程序后，a的值是____。'
    prompt = search_with_faiss(query, data_list, index, pca, k=5)

    print(prompt)
