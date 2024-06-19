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
    prompt = '''You are a professional researcher now. Please write  review for a paper, the abstract of which is :\n
    In this paper, we present a novel sequential recommendation (SR)
scheme called Rec2vec; it learns both the dynamic preferences
of users and the popularity of items, trends, and then provides
users with serendipitous recommendations. We address the similarities between SR and natural language processing (NLP) tasks, and
achieve surprising predictions by capturing the underlying trends
and patterns in the data. However, SR models deal with more complicated and diverse data than NLP models, and need to address
these differences and task disconnection issues. To bridge these
differences, Rec2vec focuses on a latent space that can accommodate multiple representations underlying the different models or
tasks. Towards exploring this space, Rec2vec derives an Advanced
Attention Mechanism (AAM), and a Latent Space Layer (LSL) with
two training objectives, namely Time Decay Cross Entropy (TDCE),
and Future Preference Modeling (FPM). These components enable
Rec2vec to encode both users and their historical behavior as instructions, and decode candidate items. Unlike previous SR schemes,
the novelty of Rec2vec lies in combining preference learning and
prediction tasks via AAM, unifying Transformer architectures and
VAE on LSL, and proposing a dynamic latent space to capture
trends through TDCE and FPM. Experiments on datasets show that
Rec2vec significantly advances SR by capturing different trends
and providing serendipitous recommendations.
'''

    response = openai.ChatCompletion.create(
        model='gpt-4',
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1024,  # 调整生成文本的长度
        temperature=0.0,  # 调整生成文本的随机性
        top_p=0.0,
    )
    message = response.choices[0]["message"]["content"]
    print(message)


generate_one_completion()