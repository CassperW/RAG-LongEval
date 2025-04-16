from tqdm import tqdm

import transformers
import torch

import json 
with open('lc_prompts.json', 'r') as f:
    lc_prompts = json.load(f)

# with open('question_text.json', 'r') as f:
#     question_text = json.load(f)
# question_idx = question_text['idx_with_questions']

with open('financial_top5_idx.json', 'r') as f:
    financial_top5_idx = json.load(f)

model_id = "/home/yujie/models/llama3"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

output_filename = 'lc_answers.jsonl'

for i in tqdm(range(len(financial_top5_idx)), desc="Generating Answers"):
    idx = financial_top5_idx[i]

    messages = [
        {"role": "user", "content": lc_prompts[idx]},
    ]

    prompt = pipeline.tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )

    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = pipeline(
        prompt,
        max_new_tokens=256,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
        pad_token_id=pipeline.tokenizer.eos_token_id
    )

    answer = outputs[0]["generated_text"][len(prompt):]

    with open(output_filename, 'a', encoding='utf-8') as f:
        json_string = json.dumps(answer, ensure_ascii=False)
        f.write(json_string + '\n')

    torch.cuda.empty_cache()

