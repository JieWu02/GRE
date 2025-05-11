import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import json
import tiktoken
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from FlagEmbedding import FlagModel
import concurrent.futures
import re
import torch


model = FlagModel("BAAI/bge-base-en-v1.5", use_fp16=True)
tokenizer = tiktoken.get_encoding('cl100k_base')

def fuzzy_match(query, texts):
    scores = [(i, fuzz.partial_ratio(query, text)) for i, text in enumerate(texts)]
    sorted_indices = [i for i, score in sorted(scores, key=lambda x: x[1], reverse=True)]
    return sorted_indices

def tfidf_match(question, texts):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([question] + texts)
    similarity_matrix = cosine_similarity(vectors[0], vectors[1:])[0]
    sorted_indices = np.argsort(similarity_matrix)[::-1]
    return sorted_indices

def clean_text_for_matching(text):
    # 清理文本以进行匹配
    text = re.sub(r'\s+', ' ', text)  # 将多个空格替换为一个空格
    text = text.strip()  # 去除首尾空格
    return text

def check_facts_in_chunks(supporting_facts, chunks):
    facts_in_chunks = {}

    cleaned_chunks = [chunk for chunk in chunks]

    for fact in supporting_facts:
        cleaned_fact = clean_text_for_matching(fact)
        facts_in_chunks[fact] = any(cleaned_fact in chunk for chunk in cleaned_chunks)

    return facts_in_chunks


def get_bge_embedding(text):
    text = text.replace("\n", " ").strip()
    embedding = model.encode([text])
    return embedding

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

file_path = 'chunked_datasets/Hotpot_test_chunked.json'

with open(file_path, 'r', encoding='utf-8') as file:
    json_data = json.load(file)

if __name__ == '__main__':
    sum_token = 0
    retrieved_results = []
    updated_json_data = []
    j = 0
    for item in json_data:
        data = item
        print(j)
        j=j+1
        chunks = []
        # passages = data['golden_context'] + data['noisy_context'] + data['distracting_context']
        passages = data['golden_context'] + data['noisy_context']
        # passages = data['golden_context']
        for passage in passages:
            for chunk in passage[1:]:
                chunks.append(chunk)

        question = data['question']
        answer = data['answer']
        supporting_facts = data['supporting_facts_context']

        # Encode the question and chunks
        question_embedding = get_bge_embedding(question)[0]
        chunk_embeddings = [get_bge_embedding(chunk)[0] for chunk in chunks]

        # Calculate cosine similarities
        similarities = cosine_similarity([question_embedding], chunk_embeddings)[0]

        # Get top-15 similar chunks
        top_15_indices = np.argsort(similarities)[-25:][::-1]
        top_15_chunks = [chunks[i] for i in top_15_indices]

        # 检查支持事实是否出现在top-15 chunks中
        # facts_check = check_facts_in_chunks(supporting_facts, top_15_chunks)

        # 按指定格式打印检查结果
        # for i, (fact, is_present) in enumerate(facts_check.items(), start=1):
        #     if is_present:
        #         print(f"Fact {i}: Found")
        #     else:
        #         print(f"Fact {i}: Not Found - {fact}")

        # Update the original data to only include top-15 chunks
        golden_contexts = []
        noisy_context = []
        distracting_contexts = []
        for passage in passages:
            title = passage[0]
            new_passage = []
            for ori_chunk in passage[1:]:
                if any(ori_chunk == top_chunk for top_chunk in top_15_chunks):
                    new_passage.append(ori_chunk)

            if len(new_passage) > 0:  # 如果new_passage中有超过一个元素，说明有chunk被保留
                is_golden = False
                is_noisy = False

                # 判断该passage属于golden_context
                for context in data['golden_context']:
                    if title == context[0]:
                        golden_contexts.append([title] + new_passage)
                        is_golden = True
                        break

                # 如果不是golden_context，判断是否属于noisy_context
                if not is_golden:
                    for context in data['noisy_context']:
                        if title == context[0]:
                            noisy_context.append([title] + new_passage)
                            is_noisy = True
                            break

                # 如果既不是golden_context也不是noisy_context，则属于distracting_context
                if not is_golden and not is_noisy:
                    distracting_contexts.append([title] + new_passage)

        # Add updated data to the new list
        data['golden_context'] = golden_contexts
        data['noisy_context'] = noisy_context
        data['distracting_context'] = distracting_contexts
        updated_json_data.append(data)

        # Write the updated data to a new JSON file
    with open('top25_chunks_datasets/Hotpot_top25_chunks.json', 'w', encoding='utf-8') as outfile:
        json.dump(updated_json_data, outfile, ensure_ascii=False, indent=4)
    print('done')