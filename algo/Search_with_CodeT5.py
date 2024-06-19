import jieba
import json
import numpy as np
from transformers import AutoModel, AutoTokenizer
import faiss
import pickle as pkl
from sklearn import preprocessing

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

checkpoint = "/home/rrt/codet5p-110m-embedding/"
device = "cuda"  # for GPU usage or "cpu" for CPU usage
tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
model = AutoModel.from_pretrained(checkpoint, trust_remote_code=True).to(device)

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
    inputs = tokenizer.encode(query, return_tensors = "pt").to(device)
    query_embed = model(inputs)[0]
    query_embed = query_embed.cpu().detach().numpy() 

    query_embed = np.expand_dims(query_embed, axis=0)
    #query_embed = pca.apply(query_embed)
    #query_embed = preprocessing.normalize(query_embed, 'l2')
    
    distances, indices = index.search(query_embed, k)

    prompt_str_list = [str(data_list[idx.item()]) for idx in indices[0][:k]]
    prompt = '\n'.join(prompt_str_list)
    return prompt

def return_idx(query, index, pca, k):
    inputs = tokenizer.encode(query, return_tensors = "pt").to(device)
    query_embed = model(inputs)[0]
    query_embed = query_embed.cpu().detach().numpy() 

    query_embed = np.expand_dims(query_embed, axis=0)
    distances, indices = index.search(query_embed, k)

    return indices[0][:k]

def search_with_faiss_multi(query, code_data_list, graph_data_list, index, pca, k):
    inputs = tokenizer.encode(query, return_tensors = "pt").to(device)
    query_embed = model(inputs)[0]
    query_embed = query_embed.cpu().detach().numpy() 

    query_embed = np.expand_dims(query_embed, axis=0)
    
    distances, indices = index.search(query_embed, k)

    prompt_str_list = [str(code_data_list[idx.item()]) for idx in indices[0][:k]]
    code_prompt = '\n'.join(prompt_str_list)

    prompt_str_list = [str(graph_data_list[idx.item()]) for idx in indices[0][:k]]
    graph_prompt = '\n'.join(prompt_str_list)

    return code_prompt, graph_prompt

  
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
