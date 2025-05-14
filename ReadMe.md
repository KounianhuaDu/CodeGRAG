# CodeGRAG

This is the repo for CodeGRAG.

## Requirements

## Data Preparation
For C++ codes:
- AST/CFG/GraphView
```bash
sh run_ast_cfg_graph.sh
```

- Build DGL Graph
```bash
python build_graph.py
```

- Code embedding
We use code-T5 to encode the codes. Under [`utils`](utils/) folder:
```bash
python code_enc.py --path [codepath]
```

## Model Weights
You should first prepare model weights under model_weights/ folder. Following weights are included:
- codet5p-110m-embedding
- unixcoder-base-nine


## Run
Under the [`test`](test/) folder:

### Meta Graph Prompt
For one round generation, please run:
```bash
python run_raw_multilingual.py --lang [programming_language] --output [your_output_path]
```
```bash
python run_with_code.py --lang [programming_language] --output [your_output_path] --ret_method [retrieval_model] --datapath [retrieval_pool]
```
```bash
python run_with_graph.py --lang [programming_language] --output [your_output_path] --ret_method [retrieval_model] --datapath [retrieval_pool]
```
Two result files are included:
- cpp_results/samples_with_graph.jsonl
- python_results/samples_with_graph.jsonl

### Soft Prompting
See under the [`soft_prompt`](soft_prompt/) folder.

### Citation
If you find this repo useful, please cite our paper:

@article{du2024codegrag,

  title={CodeGRAG: Extracting Composed Syntax Graphs for Retrieval Augmented Cross-Lingual Code Generation},
  
  author={Du, Kounianhua and Rui, Renting and Chai, Huacan and Fu, Lingyue and Xia, Wei and Wang, Yasheng and Tang, Ruiming and Yu, Yong and Zhang, Weinan},
  
  journal={arXiv preprint arXiv:2405.02355},
  
  year={2024}
}

