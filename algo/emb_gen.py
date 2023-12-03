import jieba
import json
import numpy as np
import torch
# from sentence_transformers import SentenceTransformer, util
from transformers import AutoModel, AutoTokenizer
import os
from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "false"
checkpoint = "../model_weights/codet5p-110m-embedding"
device = "cuda"  # for GPU usage or "cpu" for CPU usage

tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
model = AutoModel.from_pretrained(checkpoint, trust_remote_code=True).to(device)
qa_file_path = "/home/rrt/codemate/CodeTest/codes/codes_total.json"

data_list = []
with open(qa_file_path, 'r', encoding='utf-8') as f:
    for line in f:
       data = json.loads(line)
       data_list.append(data)
documents = [qa['Question'] for qa in data_list]


embed_list = []
for start_idx in tqdm(range(0, len(documents), 256)):
        batch_docs = documents[start_idx: min(start_idx + 256, len(documents))]
        inputs = tokenizer(batch_docs, padding='longest', truncation=True, return_tensors='pt').to(device)
        with torch.no_grad():
             outputs = model(**inputs)
        print(outputs.shape)
        embed_list.append(outputs)

embeddings = torch.cat(embed_list, dim=0)
print(embeddings.shape)
print(type(embeddings))
embeddings = embeddings.cpu().numpy()
print(embeddings.shape)
print(type(embeddings))
np.save('./codes_total.npy', embeddings)
# 获取嵌入模型的输出
# embedding_output = embedded_output.last_hidden_state

# print("Embedding Output Shape:", embedded_output.shape)
    
    