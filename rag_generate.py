from tqdm import tqdm

import transformers
import torch

import json 
with open('loong_ragtop5_prompts.json', 'r') as f:
    rag_prompts = json.load(f)

model_id = "/home/yujie/models/llama3"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

answer_list = []
for i in tqdm(range(len(rag_prompts))):
    messages = [
        {"role": "user", "content": rag_prompts[i]},
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
    answer_list.append(answer)

    torch.cuda.empty_cache()

# with open('rag_answers.txt', 'w') as f:
#     for item in answer_list:
#         cleaned_item = item.replace("\n", "").replace("\r", "")
#         f.write("%s\n" % cleaned_item)

# --- Save Results to JSON ---
output_filename = 'rag_answers.json' 
print(f"Saving {len(answer_list)} results to {output_filename}...")

try:
    with open(output_filename, 'w', encoding='utf-8') as f:
        # Use json.dump to write the list directly to the file
        # ensure_ascii=False allows non-ASCII characters (like Chinese) to be saved correctly
        # indent=4 makes the JSON file human-readable (optional, remove for smaller file size)
        json.dump(answer_list, f, ensure_ascii=False, indent=4) 
    print("Results saved successfully.")
except IOError as e:
    print(f"Error writing results to {output_filename}: {e}")
except Exception as e:
    print(f"An unexpected error occurred during saving: {e}")