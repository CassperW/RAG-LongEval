import pickle
from qdrant_client import QdrantClient, models
from qdrant_client.models import Distance, VectorParams, PointStruct
from tqdm import tqdm

with open('text_vectors.pkl', 'rb') as f:
    text_vectors = pickle.load(f)

with open('questions_vectors.pkl', 'rb') as f:
    questions_vectors = pickle.load(f)

with open('text_chunks.pkl', 'rb') as f:
    text_chunks = pickle.load(f)

client = QdrantClient(path="./vdb") 

import json
with open('question_text.json', 'r') as f:
    question_text = json.load(f)
question_idx = question_text['idx_with_questions']

total = len(text_vectors)
total 

k = 5
id_topk = []

for i in range(total):
    chunk_num = len(text_vectors[i])
    
    if chunk_num >= k :
        id_topk.append(i)
    
with open(f'id_top{k}.json', 'w') as f:
    json.dump(id_topk, f)

ids_list = []
context_list = []
# for i in range(len(id_topk)):
for i in tqdm(range(len(id_topk))):
    idx = id_topk[i]
    vectors = text_vectors[idx]
    question_embedding = questions_vectors[idx]
    chunks = text_chunks[idx]

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
            PointStruct(id=j, vector=vectors[j].tolist())
            for j in range(len(vectors))
        ],
    )

    search_result = client.search(
        collection_name="RAG", query_vector=question_embedding, limit=k
    )
    ids = [point.id for point in search_result]
    ids_list.append(ids)
    
    formatted_contexts = []
    for j in range(len(ids)):
        formatted_contexts.append(f"Context {j+1}:\nContent: {chunks[ids[j]]}")
    context = "\n\n".join(formatted_contexts)
    context_list.append(context)

with open(f'ids_list_{k}.json', 'w') as f:
    json.dump(ids_list, f)

def get_generate_prompt(item, context):
    replace_dict = {"{question}": item['question'], "{instruction}": item['instruction']}
    prompt_template = item['prompt_template']
    for k, v in replace_dict.items():
        prompt_template = prompt_template.replace(k, v)
    doc_str = context
    prompt_template = prompt_template.replace("{docs}", doc_str)
    item['docs'] = doc_str
    item['prompt'] = prompt_template
    return item

loong_path = '/home/yujie/Data/loong/loong.jsonl'

with open(loong_path, 'r') as f:
    lines = f.readlines()

# generate prompt with question and text
prompts = []
# for i in range(len(id_topk)):
for i in tqdm(range(len(id_topk))):
    id_in_loong = question_idx[id_topk[i]]
    item = json.loads(lines[id_in_loong])
    prompt = get_generate_prompt(item, context_list[i])
    prompts.append(prompt['prompt'])

with open(f'loong_ragtop{k}_prompts.json', 'w', encoding='utf-8') as f:
    json.dump(prompts, f, ensure_ascii=False, indent=4)
