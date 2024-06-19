import jieba
import json
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
import os
from tqdm import tqdm
import pickle as pkl
import argparse

def normalize(data):
    """normalize matrix by rows"""
    return data/np.linalg.norm(data,axis=1,keepdims=True)

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


def main(rootpath, model_name, checkpoint, device):
    with open(os.path.join(rootpath, 'nls.pkl'), 'rb') as f:
        descriptions = pkl.load(f)
    with open(os.path.join(rootpath, 'rawcodes.pkl'), 'rb') as f:
        codes = pkl.load(f)
    
    os.makedirs(os.path.join(rootpath, model_name), exist_ok=True)
    
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    model = AutoModel.from_pretrained(checkpoint, trust_remote_code=True).to(device)

    code_embed_list = []
    for start_idx in tqdm(range(0, len(codes), 128)):
        batch_docs = codes[start_idx: min(start_idx + 128, len(codes))]
        inputs = tokenizer(batch_docs, padding='longest', truncation=True, return_tensors='pt').to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        code_embed_list.append(outputs)
    
    code_embeddings = torch.cat(code_embed_list, dim=0)
    code_embeddings = code_embeddings.cpu().numpy()
    code_embeddings = normalize(code_embeddings)
    np.save(os.path.join(rootpath, model_name, 'codes_emb.npy'), code_embeddings)

    des_embed_list = []
    for start_idx in tqdm(range(0, len(descriptions), 128)):
        batch_docs = descriptions[start_idx: min(start_idx + 128, len(descriptions))]
        inputs = tokenizer(batch_docs, padding='longest', truncation=True, return_tensors='pt').to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        des_embed_list.append(outputs)
    
    des_embeddings = torch.cat(des_embed_list, dim=0)
    des_embeddings = des_embeddings.cpu().numpy()
    des_embeddings = normalize(des_embeddings)
    np.save(os.path.join(rootpath, model_name, 'descriptions_emb.npy'), des_embeddings)

    test(code_embeddings, des_embeddings, './')

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Using different models to generate function")
    parser.add_argument("--model_name", default="codet5", help="test model")
    parser.add_argument("--root", default="../NLGeneration/data/CPP")
    parser.add_argument("--modelpath", default="../model_weights/codet5p-110m-embedding/")
    parser.add_argument("--gpu", type=int, default="0", help="Set GPU Ids : Eg: For CPU = -1, For Single GPU = 0")
    
    args = parser.parse_args()
    
    if args.gpu >= 0 and torch.cuda.is_available():
        device = 'cuda:{}'.format(args.gpu)
    else:
        device = 'cpu'
    
    main(args.root, args.model_name, args.modelpath, device)





