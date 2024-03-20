import click
import json
import os 
import numpy as np
import openai
import logging
import argparse
from tqdm import tqdm
import re
import gzip
from collections import defaultdict

from codegeex.benchmark.utils import read_dataset, IMPORT_HELPER
from codegeex.data.data_utils import write_jsonl

import os
import sys
sys.path.append("..")
from utils.config import *
from utils.utils import *

import pickle as pkl


def generate_one_completion():
    prompt = "You are a professional researcher now. Please rewrite and polish the following paragraph for me: Code correction is a primary stage in  the coding workflow. After code generation, it is necessary to revise the generated code based on feedback from compilers and the pass rate of test cases. Proficiency in code correction is fundamental to LLM self-correction, substantially improving the code capabilities of LLMs. "
    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1024,  # 调整生成文本的长度
        temperature=0.0,  # 调整生成文本的随机性
        top_p=0.0,
    )
    message = response.choices[0]["message"]["content"]
    print(message)


generate_one_completion()