from tqdm import tqdm
import json
from pathlib import Path
import glob

loong_path = '/home/yujie/Data/loong/loong.jsonl'

with open(loong_path, 'r') as f:
    lines = f.readlines()

file_handle_cache = {}
def close_cached_files():
    for file, handle in file_handle_cache.items():
        if isinstance(handle, io.IOBase):
            handle.close()
    file_handle_cache.clear()

def get_content(item, doc_name, idx):
    global file_handle_cache
    doc_path = '/home/yujie/Data/loong/doc'
    doc_type, doc_level = item['type'], item['level']
    docPath = Path(doc_path) / doc_type
    doc = None

    if doc_type == 'financial':
        if str(doc_level).strip() != '4':
            _file = glob.glob(f"{docPath}/*2024-{doc_name}*.txt")[0]
        else:
            _file = glob.glob(f"{docPath}/*{doc_name}*.txt")[0]
        try:
            with open(_file, 'r') as txt_file:
                _doc_name = Path(_file).stem.split('-')[-1]
                doc = f"《{_doc_name}》\n" + txt_file.read() + "\n\n"
        except IOError:
            print(f"Error: File {_file} could not be opened.")

    elif doc_type == 'paper':
        path = docPath / doc_name
        print(path)
        try:
            with open(path, 'r') as txt_file:
                content = txt_file.read()
                doc_name = content.split('\n', 1)[0].strip("#").strip()
                doc = f"{doc_name}\n" + content + "\n\n"
        except IOError:
            print(f"Error: File {path} could not be opened.")

    elif doc_type == 'legal':
        _file = docPath / "legal.json"
        if _file in file_handle_cache:
            legal_js = file_handle_cache[_file]
            # txt_file.seek(0)
        else:
            with open(_file, 'r') as txt_file:
                legal_js = json.load(txt_file)
                file_handle_cache[_file] = legal_js

        if doc_level == 4 and ('阅读以上判决文书，我将给你若干份判决结果：' in item['instruction']):
            content = legal_js[doc_name]["content"]
        else:
            content = legal_js[doc_name]["content"] + legal_js[doc_name]["result"]
        doc = f"《判决文书{idx + 1}》\n" + content + "\n\n"


    else:
        raise "doc_type not valid!"

    return doc

def get_contents(item, doc_names):
    contents = []
    for idx, doc_name in enumerate(doc_names):
        content = get_content(item, doc_name, idx)
        contents.append(content)
    return contents

def get_doc_str(item, prompt_template):
    # len_prompt_template = token_length(prompt_template) - token_length("{docs}")
    # is_shuffle = item.get("shuffle_doc", True)

    docs = item['doc'] # if not args.rag else item["recall_chunks"][:args.rag_num]
    docs_list = []

    # if args.rag:
    #     for doc in docs:
            # if len_prompt_template + sum(token_length(s) for s in docs_list) + token_length(doc) > args.max_length:
            #     continue
            # docs_list.append(doc)
    # else:
    # read content from given doc names
    contents = get_contents(item, docs)
    # shuffle
    # if is_shuffle and item['type'] == 'financial':
    #     random.shuffle(contents)
    for content in contents:
        # if len_prompt_template + sum(token_length(s) for s in docs_list) + token_length(content) > args.max_length:
        #     continue
        docs_list.append(content)

    docs_str = "".join(docs_list)
    return docs_str

def get_generate_prompt(item):
    replace_dict = {"{question}": item['question'], "{instruction}": item['instruction']}
    prompt_template = item['prompt_template']
    for k, v in replace_dict.items():
        prompt_template = prompt_template.replace(k, v)
    doc_str = get_doc_str(item, prompt_template)
    prompt_template = prompt_template.replace("{docs}", doc_str)
    item['docs'] = doc_str
    item['prompt'] = prompt_template
    return item

prompts = []
doc_strs = []
for line in tqdm(lines, desc="gen_prompts"):
    item = json.loads(line)
    doc_type, set_level, level = item['type'], item['set'], item['level']
    prompt = get_generate_prompt(item)
    prompts.append(prompt['prompt'])
    doc_strs.append(prompt['docs'])

# 保存 prompts 到 JSON 文件
with open('lc_prompts.json', 'w', encoding='utf-8') as f:
    json.dump(prompts, f, ensure_ascii=False, indent=4)

# 保存 doc_strs 到 JSON 文件
with open('doc_strs.json', 'w', encoding='utf-8') as f:
    json.dump(doc_strs, f, ensure_ascii=False, indent=4)
    
