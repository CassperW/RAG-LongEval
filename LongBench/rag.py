import jsonlines
from tqdm import tqdm
import numpy as np
import pandas as pd

from RAG.BGE import bge_embedder
from RAG.Mistral import Mistral
from langchain.text_splitter import CharacterTextSplitter
from qdrant_client import QdrantClient, models
from qdrant_client.models import Distance, VectorParams, PointStruct

data_path = '/home/yujie/Data/LongBench/data/2wikimqa.jsonl'

data_list = []

with jsonlines.open(data_path) as reader:
    for obj in reader:
        data_list.append(obj)

embedding = bge_embedder()
llm = Mistral()
client = QdrantClient(path="./vdb") 

answer_list = []

for data in tqdm(data_list, desc="Processing Questions", unit="question"):
    question = data['input']
    document = data['context']
    ground_truth = data['answers']

    text_splitter = CharacterTextSplitter(separator=".", chunk_size=512, chunk_overlap=50, length_function=len)

    texts = text_splitter.split_text(document)

    question_embedding = embedding.get_embeddings(question)
    vectors= embedding.get_embeddings(texts)

    if client.collection_exists("RAG"):
        client.delete_collection("RAG")

    client.create_collection(
        collection_name="RAG",
        vectors_config=VectorParams(size=1024, distance=Distance.EUCLID),
        hnsw_config = models.HnswConfigDiff(        
            m = 32,   
            ef_construct = 500 
        )
    )

    operation_info = client.upsert(
        collection_name="RAG",
        wait=True,
        points=[
            PointStruct(id=i, vector=vectors[i].tolist())
            for i in range(len(vectors))
        ],
    )

    search_result = client.search(
        collection_name="RAG", query_vector=question_embedding, limit=3
    )
    ids = [point.id for point in search_result]

    context = ""
    for i in range(len(ids)):
        context += ("Context " + str(i) + ": " + texts[ids[i]] + "\n\n")

    res = llm.generate(question=question, context=context)
    answer_list.append(res)

with open('rag_answers.txt', 'w') as f:
    for item in answer_list:
        cleaned_item = item.replace("\n", "").replace("\r", "")
        f.write("%s\n" % cleaned_item)