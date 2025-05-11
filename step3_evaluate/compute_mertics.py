import json
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import re
input_path = "//home/superbench/wujie/GRE_Rebuttal/GRE/RAG_Graph/Graph_Reasoning/Graph_construction/Raptor_Tree/Raptor/collapsed_raptor_MuSique_for_reasoning_L3_result.json"
data = []


f1_scores = []
from sklearn.metrics import precision_score, recall_score, f1_score
from collections import Counter

def calculate_f1(prediction, reference):
    # 将预测和参考句子都转换为小写，并按空格分词
    prediction_tokens = prediction.lower().split()
    reference_tokens = reference.lower().split()
    # if len(prediction_tokens) < 20:
    #     print(prediction_tokens)
    #     print(reference_tokens)
    # 统计预测和参考句子中各个词出现的频率
    prediction_counter = Counter(prediction_tokens)
    reference_counter = Counter(reference_tokens)

    # 计算预测和参考句子中共同词的数量
    common = prediction_counter & reference_counter
    num_same = sum(common.values())

    # 如果没有共同词，F1值为0
    if num_same == 0:
        return 0.0, 0.0, 0.0

    # 计算 Precision, Recall 和 F1 分数
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(reference_tokens)
    f1 = (2 * precision * recall) / (precision + recall)

    return precision, recall, f1


def extract_answer(response):
    # Try extracting **Answer**:
    match = re.search(r'\*\*Answer\*\*: ?(.+)', response)
    if match:
        answer = match.group(1).strip(' "')
    else:
        # If not found, try extracting Answer:
        match = re.search(r'Answer: ?(.+)', response)
        if match:
            answer = match.group(1).strip(' "')
        else:
            match = re.search(r'Answer": ?(.+)', response)
            if match:
                answer = match.group(1).strip(' "')
            else:
                match = re.search(r'Answer\*\*: ?(.+)', response)
                if match:
                    answer = match.group(1).strip(' "')
                else:
                    match = re.search(r'So the answer is?(.+)', response)
                    if match:
                        answer = match.group(1).strip(' "')
                    else:
                        answer = response
    cleaned_answer = answer.replace('\'', '').replace('.', '').replace('[', '').replace(']', '').replace('Answer', '').replace('*', '').replace(':', '').replace('Answer is:', '').strip()
    return cleaned_answer

with open(input_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 初始化 ROUGE 计算器
scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

# 初始化 BLEU 计算器
smoothing_function = SmoothingFunction().method4


def calculate_char_level_rouge(prediction, reference):
    prediction_chars = ' '.join(list(prediction.replace(' ', '')))
    reference_chars = ' '.join(list(reference.replace(' ', '')))
    char_scores = scorer.score(prediction_chars, reference_chars)
    return char_scores['rougeL'].fmeasure


# 计算每个样本的 ROUGE-L 分数、BLEU 分数和回答 Recall
results = []
em_scores = []  # 用于存储每个样本的EM得分
for item in data[:99]:
    response = item['response']
    label = item['label']
    try:
        # 计算单词级 ROUGE-L 分数
        word_rouge_scores = scorer.score(response, label)
        word_rouge_l_score = word_rouge_scores['rougeL'].fmeasure

        # 计算字符级 ROUGE-L 分数
        char_rouge_l_score = calculate_char_level_rouge(response, label)

        # 计算 BLEU 分数
        bleu_score = sentence_bleu([label], response, smoothing_function=smoothing_function)

        # 计算回答 Recall**Answer**:
        # response_answer = response.split("Answer:", 1)[-1].strip(' "')
        response_answer = extract_answer(response)
        label_answer = label

        precision, recall, f1 = calculate_f1(response_answer, label_answer)

        em_score = 1 if response_answer.strip().lower() == label_answer.strip().lower() else 0

        # 计算单词级回答 Recall
        word_answer_rl = scorer.score(response_answer, label_answer)
        word_answer_rl = word_answer_rl['rougeL'].fmeasure

        # 计算字符级回答 Recall
        char_answer_rl = calculate_char_level_rouge(response_answer, label_answer)

        # 计算回答 BLEU 分数
        answer_bleu_score = sentence_bleu([label_answer], response_answer, smoothing_function=smoothing_function)

        print()
        if len(response_answer) < 100:
            print('response:\n', response_answer)
            print('label:\n', label_answer)
            f1_scores.append(f1)
            em_scores.append(em_score)

            results.append({
                'response': response,
                'label': label,
                'word_rougeL': word_rouge_l_score,
                'char_rougeL': char_rouge_l_score,
                'bleu': bleu_score,
                'word_answer_rl': word_answer_rl,
                'char_answer_rl': char_answer_rl,
                'answer_bleu_score': answer_bleu_score
            })
    except Exception:
        continue

# 计算所有样本的平均单词级 ROUGE-L 分数、字符级 ROUGE-L 分数、BLEU 分数和 Recall
average_word_rouge_l = sum(result['word_rougeL'] for result in results) / len(results)
average_char_rouge_l = sum(result['char_rougeL'] for result in results) / len(results)
average_bleu = sum(result['bleu'] for result in results) / len(results)
average_word_answer_rl = sum(result['word_answer_rl'] for result in results) / len(results)
average_char_answer_rl = sum(result['char_answer_rl'] for result in results) / len(results)
average_answer_bleu = sum(result['answer_bleu_score'] for result in results) / len(results)
average_f1 = sum(f1_scores) / len(f1_scores)
average_em = sum(em_scores) / (len(em_scores)-1) # 计算平均EM得分
print(f"Average CoT Reasoning Word-level ROUGE-L: {average_word_rouge_l:.4f}")
print(f"Average CoT Reasoning Char-level ROUGE-L: {average_char_rouge_l:.4f}")
print(f"Average CoT Reasoning BLEU: {average_bleu:.4f}")
print(f"Average Answer Word-level ROUGE-L: {average_word_answer_rl:.4f}")
print(f"Average Answer Char-level ROUGE-L: {average_char_answer_rl:.4f}")
print(f"Average Answer BLEU: {average_answer_bleu:.4f}")
print(f"Average F1 Score: {average_f1:.4f}")
print(f"Average Exact Match (EM) Score: {average_em:.4f}")
print(len(results))
