import json 
from Model.BGE import bge_embedder
from tqdm import tqdm
from langchain.text_splitter import CharacterTextSplitter
import pickle

with open('doc_strs.json', 'r') as f:
    doc_strs = json.load(f)

loong_path = '/home/yujie/Data/loong/loong.jsonl'

with open(loong_path, 'r') as f:
    lines = f.readlines()

questions_text = []
idx_with_questions = []

for i in tqdm(range(len(lines)), desc="gen_questions"):
    line = json.loads(lines[i])
    question = line['question']
    if question == '':
        continue
    questions_text.append(line['question'])
    idx_with_questions.append(i)

data_to_save = {
    "idx_with_questions": idx_with_questions,
    "questions_text": questions_text
}

with open("question_text.json", "w", encoding="utf-8") as f:
    json.dump(data_to_save, f, ensure_ascii=False, indent=4)


text_splitter = CharacterTextSplitter(separator="\n\n", chunk_size=1024, chunk_overlap=200, length_function=len)
embedding = bge_embedder()

questions_vectors = embedding.get_embeddings(questions_text)

text_vector_list = []
for document in tqdm(idx_with_questions, desc="Processing documents"): 
    document = doc_strs[i]
    
    texts = text_splitter.split_text(document)

    text_vectors= embedding.get_embeddings(texts)
    
    text_vector_list.append(text_vectors)

with open('text_vectors.pkl', 'wb') as f:
    pickle.dump(text_vector_lists, f)

with open('questions_vectors.pkl', 'wb') as f:
    pickle.dump(questions_vectors, f)