import difflib
import re
import string
import collections
# import nltk
# from nltk.translate import meteor_score as ms
# from rouge_score import rouge_scorer
from simhash import Simhash
import math
# from bleurt import score
import torch
import json
import numpy as np

def compute_exact_match_ratio(output_text, gen_query):
    matcher = difflib.SequenceMatcher(None, output_text, gen_query)
    return matcher.ratio()

def normalize_answer(s):
  """Lower text and remove punctuation, articles and extra whitespace."""
  def remove_articles(text):
    regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
    return re.sub(regex, ' ', text)
  def white_space_fix(text):
    return ' '.join(text.split())
  def remove_punc(text):
    exclude = set(string.punctuation)
    return ''.join(ch for ch in text if ch not in exclude)
  def lower(text):
    return text.lower()
  return white_space_fix(remove_articles(remove_punc(lower(s))))

def get_tokens(s):
  if not s: return []
  return normalize_answer(s).split()

def compute_f1(a_gold, a_pred):
      gold_toks = get_tokens(a_gold)
      pred_toks = get_tokens(a_pred)
      common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
      num_same = sum(common.values())
      if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
      if num_same == 0:
        return 0
      precision = 1.0 * num_same / len(pred_toks)
      recall = 1.0 * num_same / len(gold_toks)
      f1 = (2 * precision * recall) / (precision + recall)
      return f1

# def rouge(answer, ideal_answer):
#     scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
#     # Calculate the ROUGE score
#     score = scorer.score(answer, ideal_answer)
#     # Extract the F1 score for ROUGE-1
#     rouge_score = score['rouge1'].fmeasure
#     return rouge_score


def Sim_hash(ideal_answer,generated_answer):
    return Simhash(generated_answer).distance(Simhash(ideal_answer))

def calculate_perplexity(ideal_answer,answer):
    answer_tokens = answer.strip().split()
    ideal_tokens = ideal_answer.strip().split()

    # Build a frequency distribution of ideal tokens
    token_frequency = {}
    total_tokens = 0
    for token in ideal_tokens:
        token_frequency[token] = token_frequency.get(token, 0) + 1
        total_tokens += 1

    # Calculate perplexity
    log_sum = 0
    for token in answer_tokens:
        frequency = token_frequency.get(token, 0)
        if frequency == 0:
            # Set a small probability for unseen tokens
            probability = 1 / (total_tokens + 1)
        else:
            probability = frequency / total_tokens
        log_sum += math.log2(probability)
    if len(answer_tokens) >0:
        perplexity = 2 ** (-log_sum / len(answer_tokens))
    else:
        perplexity = 0
    return perplexity

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def evaluator(length, answers, ground_truths, save_path_and_name):
    count=0
    question_count = 0
    count_value =0
    count_value_f1 =0
    blue_score_count =0
    meteor_score_count =0
    rouge_score_count =0
    sentence_similarity_score_count=0
    Sim_hash_score_count=0
    bleurt_score1_count=0
    bert_score1_count=0
    perplexity_score_count=0

    question_count = length

    for i in range(question_count):
        ans = answers[i]
        gold_answer = ground_truths[i]

        value = compute_exact_match_ratio(gold_answer,ans)
        count_value += value
        
        value_f1 = compute_f1(gold_answer,ans)
        count_value_f1 = count_value_f1+value_f1

        # blue_score =blue(gold_answer, ans)
        # blue_score_count = blue_score+blue_score_count

        # meteor_score Accuracy
        # meteor_score =meteor(gold_answer, ans)
        # meteor_score_count = meteor_score+meteor_score_count

        # rouge_score =rouge(gold_answer, ans)
        # rouge_score_count = rouge_score+rouge_score_count

        # sentence_similarity_score =sentence_similarity(gold_answer, ans)
        # sentence_similarity_score_count = sentence_similarity_score+sentence_similarity_score_count

        Sim_hash_score =Sim_hash(gold_answer, ans)
        Sim_hash_score_count += float(Sim_hash_score)

        perplexity_score =calculate_perplexity(gold_answer, ans)
        perplexity_score_count = perplexity_score+perplexity_score_count

        # bleurt_score1 =bleurt_score(gold_answer, ans)
        # bleurt_score1_count = bleurt_score1+bleurt_score1_count

        # try:
        #     bert_score1 =bert_score(gold_answer, ans)
        #     bert_score1_count = bert_score1+bert_score1_count
        # except Exception as e:
        #     print(f"Error calculating BERT score: {e}")
        #     continue

        import sys
        progress = (i + 1) / question_count * 100
        bar = '=' * int(progress) + ' ' * (100 - int(progress))
        print(f'\r[{bar}] {progress:.2f}%', end='')
        sys.stdout.flush()

    # print("Count value ----",count_value,"F1----",count_value_f1)
    # print("Avg EM Accuracy",count_value/question_count)
    # print("Avg f1 Accuracy",count_value_f1/question_count)
    # print("Avg blue_score Accuracy",blue_score_count/question_count)
    # print("Avg meteor_score Accuracy",meteor_score_count/question_count)
    # print("Avg rouge_score Accuracy",rouge_score_count/question_count)
    # print("Avg sentence_similarity_score Accuracy",sentence_similarity_score_count/question_count)
    # print("Avg Sim_hash_score_count Accuracy",Sim_hash_score_count/question_count)
    # print("Avg perplexity_score_count Accuracy",perplexity_score_count/question_count)
    # print("Avg bleurt_score1_count Accuracy",bleurt_score1_count/question_count)
    # print("Avg bert_score1_count Accuracy",bert_score1_count/question_count)


    result = {
        # "Avg_EM_Accuracy": count_value / question_count,
        # "Avg_F1_Accuracy": count_value_f1 / question_count,
        # "Avg_Blue_Score_Accuracy": blue_score_count / question_count,
        # "Avg_Meteor_Score_Accuracy": meteor_score_count / question_count,
        # "Avg_Rouge_Score_Accuracy": rouge_score_count / question_count,
        # "Avg_Sentence_Similarity_Score_Accuracy": sentence_similarity_score_count / question_count,
        # "Avg_Sim_Hash_Score_Accuracy": Sim_hash_score_count / question_count,
        # "Avg_Perplexity_Score_Accuracy": perplexity_score_count / question_count,
        # "Avg_Bleurt_Score1_Accuracy": bleurt_score1_count / question_count,
        # "Avg_Bert_Score1_Accuracy": bert_score1_count / question_count
        "Avg_EM_Accuracy": float(count_value / question_count),
        "Avg_F1_Accuracy": float(count_value_f1 / question_count),
        # "Avg_Blue_Score_Accuracy": float(blue_score_count / question_count),
        # "Avg_Meteor_Score_Accuracy": float(meteor_score_count / question_count),
        # "Avg_Rouge_Score_Accuracy": float(rouge_score_count / question_count),
        # "Avg_Sentence_Similarity_Score_Accuracy": float(sentence_similarity_score_count / question_count),
        "Avg_Sim_Hash_Score_Accuracy": float(Sim_hash_score_count / question_count),
        "Avg_Perplexity_Score_Accuracy": float(perplexity_score_count / question_count),
        # "Avg_Bleurt_Score1_Accuracy": float(bleurt_score1_count / question_count),
        # "Avg_Bert_Score1_Accuracy": float(bert_score1_count / question_count)
    }

    # 将字典保存为 JSON 格式的文件
    with open(save_path_and_name, "w") as json_file:
        json.dump(result, json_file, ensure_ascii=False, indent=4)

    # return [count_value, count_value_f1, count_value/question_count, count_value_f1/question_count, blue_score_count/question_count, meteor_score_count/question_count,
    #         rouge_score_count/question_count, sentence_similarity_score_count/question_count, Sim_hash_score_count/question_count, perplexity_score_count/question_count,
    #         bleurt_score1_count/question_count, bert_score1_count/question_count]