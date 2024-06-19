import jieba
import json
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
import os
import sys
sys.path.append("..")
from tqdm import tqdm
import pickle as pkl
import argparse
from algo.unixcoder import UniXcoder

def normalize(data):
    """normalize matrix by rows"""
    return data/np.linalg.norm(data,axis=1,keepdims=True)

def extract_nl(prompt):
    nls = prompt.split('#')
    nl = nls[0]
    return nl

def test(code_reprs, desc_reprs, save_path):

    # calculate similarity
    sum_1, sum_5, sum_10, sum_mrr = [], [], [], []
    test_sim_result, test_rank_result = [], []
    for i in tqdm(range(0, code_reprs.shape[0])):
        desc_vec = np.expand_dims(desc_reprs[i], axis=0) # [1 x n_hidden]
        sims = np.dot(code_reprs, desc_vec.T)[:,0] # [n_processed]
        negsims = np.negative(sims)
        predict = np.argsort(negsims)
        
        # SuccessRate@k
        predict_1, predict_5, predict_10 = [int(predict[0])], [int(k) for k in predict[0:5]], [int(k) for k in predict[0:10]]
        sum_1.append(1.0) if i in predict_1 else sum_1.append(0.0)
        sum_5.append(1.0) if i in predict_5 else sum_5.append(0.0)
        sum_10.append(1.0) if i in predict_10 else sum_10.append(0.0)
        # MRR
        predict_list = predict.tolist()
        rank = predict_list.index(i)
        sum_mrr.append(1/float(rank+1))

        # results need to be saved
        predict_20 = [int(k) for k in predict[0:20]]
        sim_20 = [sims[k] for k in predict_20]
        test_sim_result.append(zip(predict_20, sim_20))
        test_rank_result.append(rank+1)

    print(f'R@1={np.mean(sum_1)}, R@5={np.mean(sum_5)}, R@10={np.mean(sum_10)}, MRR={np.mean(sum_mrr)}')
    sim_result_filename, rank_result_filename = 'sim.npy', 'rank.npy'
    np.save(os.path.join(save_path, sim_result_filename), test_sim_result)

def main(codepath):
    '''with open(os.path.join(codepath,'filtered.pkl'), 'rb') as f:
        [codes, descriptions] = pkl.load(f)'''

    with open(os.path.join(codepath, 'crossLingual.pkl'), 'rb') as f:
        contents = pkl.load(f)
    
    cpp_prompts = []
    cpp_decs = []
    cpp_solutions = []

    nls = []

    python_prompts = []
    python_decs = []
    python_solutions = []

    for key in contents.keys():
        cpp_prompt, cpp_dec, cpp_solution = contents[key]['cpp']
        nl = extract_nl(cpp_prompt)
        nls.append(nl)
        python_prompt, python_dec, python_solution = contents[key]['python']

        cpp_prompts.append(cpp_prompt)
        cpp_decs.append(cpp_dec)
        cpp_solutions.append(cpp_solution)

        python_prompts.append(python_prompt)
        python_decs.append(python_dec)
        python_solutions.append(python_solution)
    
    python_querys = []
    for nl, python_dec in zip(nls, python_decs):
        python_querys.append(nl + '\n' + python_dec)

    codes = cpp_solutions
    descriptions = python_querys

    
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    device = "cuda:4"  # for GPU usage or "cpu" for CPU usage
    model = UniXcoder("../model_weights/unixcoder-base-nine")
    model.to(device)
    
    code_embed_list = []
    for start_idx in tqdm(range(0, len(codes), 128)):
        batch_docs = codes[start_idx: min(start_idx + 128, len(codes))]
        tokens_ids = model.tokenize(batch_docs, mode="<encoder-only>", padding=True)
        source_ids = torch.tensor(tokens_ids).to(device)
        #inputs = tokenizer(batch_docs, padding='longest', truncation=True, return_tensors='pt').to(device)
        with torch.no_grad():
            tokens_embeddings, outputs = model(source_ids)
        code_embed_list.append(outputs)
    
    code_embeddings = torch.cat(code_embed_list, dim=0)
    code_embeddings = code_embeddings.cpu().numpy()
    code_embeddings = normalize(code_embeddings)
    np.save(os.path.join(codepath,'codes_emb.npy'), code_embeddings)

    des_embed_list = []
    for start_idx in tqdm(range(0, len(descriptions), 128)):
        batch_docs = descriptions[start_idx: min(start_idx + 128, len(descriptions))]
        tokens_ids = model.tokenize(batch_docs, mode="<encoder-only>", padding=True)
        source_ids = torch.tensor(tokens_ids).to(device)
        #inputs = tokenizer(batch_docs, padding='longest', truncation=True, return_tensors='pt').to(device)
        with torch.no_grad():
            tokens_embeddings, outputs = model(source_ids)
        des_embed_list.append(outputs)
    
    des_embeddings = torch.cat(des_embed_list, dim=0)
    des_embeddings = des_embeddings.cpu().numpy()
    des_embeddings = normalize(des_embeddings)
    np.save(os.path.join(codepath,'descriptions_emb.npy'), des_embeddings)

    test(code_embeddings, des_embeddings, './')

#filter('../data')
main('../data')

'''codepath = '../data'
code_reprs = np.load(os.path.join(codepath,'codes_emb.npy'))
desc_reprs = np.load(os.path.join(codepath,'descriptions_emb.npy'))
test(code_reprs, desc_reprs, codepath)'''




