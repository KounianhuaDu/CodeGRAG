# 模型下载
from modelscope import snapshot_download

model_dir = snapshot_download(
    "AI-ModelScope/CodeLlama-7b-Instruct-hf", cache_dir="/ext0/jzchen/model_weights"
)
