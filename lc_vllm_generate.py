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

start_index = 0
try:
    with open(output_filename, 'r', encoding='utf-8') as f:
        start_index = sum(1 for line in f) # 计算文件行数
    print(f"Output file found. Resuming from index {start_index}.")
except FileNotFoundError:
    print("Output file not found. Starting from index 0.")
    start_index = 0

with open(output_filename, 'a', encoding='utf-8') as f: 
    for i in tqdm(range(start_index, len(financial_top5_idx)), initial=start_index, total=len(financial_top5_idx), desc="Generating Answers"):
        idx = financial_top5_idx[i]
        prompt = lc_prompts[idx]
        conversation = [{'role': 'user', 'content': prompt}]

        try:
            outputs = llm.chat(conversation, sampling_params)
            generated_text = outputs[0].outputs[0].text
            json_string = json.dumps(generated_text, ensure_ascii=False)
            f.write(json_string + '\n')
        except ValueError as e:
            print(f"\nSkipping index {i} (idx: {idx}) due to ValueError: {e}")
            # 可以选择将错误信息记录到另一个文件
            with open('error_log.txt', 'a', encoding='utf-8') as err_f:
                err_f.write(f"Index: {i}, Idx: {idx}, Error: {e}\n")
            continue # 继续下一次循环
        except Exception as e: # 捕获其他可能的意外错误
            print(f"\nSkipping index {i} (idx: {idx}) due to unexpected error: {e}")
            with open('error_log.txt', 'a', encoding='utf-8') as err_f:
                err_f.write(f"Index: {i}, Idx: {idx}, Error: {e}\n")
            continue # 继续下一次循环

