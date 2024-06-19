import click
import json
import os 
import numpy as np
import openai
import logging
import argparse
from tqdm import tqdm
import re

from codegeex.benchmark.utils import read_dataset, IMPORT_HELPER
from codegeex.data.data_utils import write_jsonl

import os
import sys
sys.path.append("..")
from utils.config import *
from utils.utils import *

from algo.SelfRevision import construct_faiss_index, return_idx
import pickle as pkl

from collections import defaultdict

#return_idx(query, index, pca, k)

def build_index(problem, index, pca, k):
    task = problem['prompt']
    declaration = problem['declaration']
    query = declaration
    idxs = return_idx(query, index, pca, k)
    return idxs

def main(k, data_path, output_path):

    embeddings_path = os.path.join(data_path, 'codes_emb.npy')
    embeddings = np.load(embeddings_path)
    index, pca = construct_faiss_index(embeddings)

    index_dict = defaultdict(list)
    for task_id in tqdm(problems):
        idx = build_index(problems[task_id], index, pca, k)
        index_dict[task_id] = idx
    
    with open(os.path.join(output_path, 'retrieval_idx.pkl'), 'wb') as f:
        pkl.dump(index_dict, f)
    

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Using different models to generate function")
    parser.add_argument("--model_name", default="gpt-3.5-turbo", help="test model")
    
    parser.add_argument("--datapath", default="../data/Cgraphs", help="data path")
    parser.add_argument("--output", default="/home/knhdu/output/FinalVersion", help="output path")
    parser.add_argument("--value", choices=['raw_code','graph'])
    
    parser.add_argument('--gpu', type=int, default=0, help='Set GPU Ids : Eg: For CPU = -1, For Single GPU = 0')
    parser.add_argument('--seed', default=1, type=int)

    parser.add_argument('--k', default=1, type=int)

    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    problem_file = "/home/rrt/codemate/CodeGeeX/codegeex/benchmark/humaneval-x/cpp/data/humaneval_cpp.jsonl.gz"
    problems = read_dataset(problem_file, dataset_type="humaneval") 

    main(args.k, args.datapath, args.output)

