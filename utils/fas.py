import jieba
import json
import numpy as np
from sentence_transformers import SentenceTransformer, util
import faiss
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def get_prompt_textbook(query, top_k, embeddings, data_list):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embed = model.encode(query) # 384
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embeddings)
    k = top_k
    query_embed = np.expand_dims(query_embed, axis=0)
    distances, indices = index.search(query_embed, k)
    #print("Top-5 相似项的索引：", indices)
    #print("对应的距离：", distances)
    prompt_list = [{'question': data_list[idx.item()]['data']} for idx in indices[0][:top_k]]
    prompt_str_list = ['相关知识点:' + data_list[idx.item()]['data']  for idx in indices[0][:top_k]]
    prompt = '\n'.join(prompt_str_list)
    return prompt

    
    
    
    
    
if __name__ == '__main__':
    data_list = []
    with open('textbook.json', 'r', encoding='utf-8') as f:
        for line in f:
           data = json.loads(line)
           data_list.append(data)
    # documents = [qa['Question'] for qa in data_list]
    embeddings = np.load('textbook.npy')
    
    print("PROMPT 1")
    prompt1 = get_prompt_faiss('若森林F有15条边、25个结点，则F包含树的个数是____',2, embeddings, data_list)
    print(prompt1)
    print("PROMPT 2")
    prompt2 = get_prompt_faiss('#include<iostream.h>\n#define SQR(x) x*x\nvoid main()\n{\nint a=10,k=2,m=1;\na/=SQR(k+m);cout<<a;\n}\n执行上面的C++程序后，a的值是____。',2, embeddings, data_list)
    print(prompt2)
