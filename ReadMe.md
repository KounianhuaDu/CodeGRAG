# Graph Syntax RAG

This is the repo for Graph Syntax RAG.

## Requirements

## Data Preparation
- AST/CFG/GraphView
For ast/cfg/graph_view, under the [`mvg`](mvg/) folder, run:
```bash
python getCFG.py --rootpath /home/knhdu/ext0/CodeRAG/output/OneRound/ --astpath astpath/ --cfgpath cfgpath/
```
```bash
python call_graphGen.py --writepath ./graph --astpath ./astpath/ --cfgpath ./cfgpath/ --picky 0
```
- Build DGL Graph
Under utils folder:
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

### Single Round Direct Prompt
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


### Multiple Round Generation

- Preparation Stage:
First, prepare the raw_code/ast/cfg/graph_view for the generated result.

For raw code, under the [`utils`](utils/) folder, run:
```bash
python extract_one_round.py
```

For ast/cfg/graph_view, under the [`mvg`](mvg/) folder, run:
```bash
python getCFG.py --rootpath /home/knhdu/ext0/CodeRAG/output/OneRound/ --astpath astpath/ --cfgpath cfgpath/
```
```bash
python call_graphGen.py --writepath ./graph --astpath ./astpath/ --cfgpath ./cfgpath/ --picky 0
```

- Two Round with Retrieved Raw Code Block

- Two Round with Retrieved Graph View