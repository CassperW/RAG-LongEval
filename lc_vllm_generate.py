from tqdm import tqdm

from vllm import LLM, SamplingParams
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

import json 
with open('lc_prompts.json', 'r') as f:
    lc_prompts = json.load(f)

# with open('question_text.json', 'r') as f:
#     question_text = json.load(f)
# question_idx = question_text['idx_with_questions']

with open('financial_top5_idx.json', 'r') as f:
    financial_top5_idx = json.load(f)

model_id = "/home/yujie/models/llama3"

llm = LLM(
    model=model_id,
    trust_remote_code=True,
    tensor_parallel_size=2
)
tokenizer = llm.get_tokenizer()
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=256
                                 ,stop_token_ids=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")])


output_filename = 'lc_answers.jsonl'

for i in tqdm(range(len(financial_top5_idx)), desc="Generating Answers"):
    idx = financial_top5_idx[i]

    prompt = lc_prompts[idx]

    conversation = [{'role': 'user', 'content': prompt}]

    outputs = llm.chat(conversation, sampling_params)

    generated_text = outputs[0].outputs[0].text
    with open(output_filename, 'a', encoding='utf-8') as f:
        json_string = json.dumps(generated_text, ensure_ascii=False)
        f.write(json_string + '\n')
    

