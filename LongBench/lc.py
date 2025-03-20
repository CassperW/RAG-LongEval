import jsonlines
from tqdm import tqdm
import numpy as np
import pandas as pd

from RAG.BGE import bge_embedder
from RAG.Mistral import Mistral

data_path = '/home/yujie/Data/LongBench/data/2wikimqa.jsonl'

data_list = []

with jsonlines.open(data_path) as reader:
    for obj in reader:
        data_list.append(obj)

embedding = bge_embedder()
llm = Mistral()

answer_list = []
ground_truths = []

for data in tqdm(data_list, desc="Processing Questions", unit="question"):
    question = data['input']
    document = data['context']
    ground_truth = data['answers']

    res = llm.generate(question=question, context=document)
    answer_list.append(res)
    ground_truths.append(ground_truth)

with open('lc_answers.txt', 'w') as f:
    for item in answer_list:
        f.write("%s\n" % item)

with open('ground_truths.txt', 'w') as f:
    for item in ground_truths:
        f.write("%s\n" % item)
        