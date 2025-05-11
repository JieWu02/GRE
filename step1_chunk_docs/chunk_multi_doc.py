import json
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
import spacy
import nltk
from typing import List, Any
from text_spliter_util import SpacyTextSplitter
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
nltk.data.path.append("nltk/nltk_data")
nlp = spacy.load("en_core_web_sm")
# require nltk, spacy, langchain
# python -m spacy download en_core_web_sm

def clean_text(text):
    text = re.sub(r'[*—]', '', text)
    for ref in ["References", "External links", 'Alternate titles', 'See also']:
        text = re.sub(r'==\s*' + re.escape(ref) + r'\s*==', '', text, flags=re.IGNORECASE)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\s+', ' ', text)  
    text = text.strip()  
    return text

def process_context(context, item_index, context_index, chunk_size, separator, chunk_overlap):
    try:
        title, passage = context
        combined_text = f"{title}. {passage}"  # 将 title 和 passage 合并
        cleaned_text = passage

        text_splitter = SpacyTextSplitter(
            separator=' ',
            pipeline='en_core_web_sm',
            chunk_size=192,
            chunk_overlap=16,
            min_chunk_size=64
        )
        chunks = text_splitter.split_text(cleaned_text)

        print(f"Processing item {item_index}, context {context_index}: {title}")
        return [title] + chunks
    except Exception as e:
        print(f"Error processing item {item_index}, context {context_index}: {e}")
        return [f"Error: {e}"]

input_file_path = '../Raw_datasets/Hotpot_test.json'
output_file_path = 'chunked_datasets/Hotpot_test_chunked.json'

with open(input_file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

new_data = []

with ProcessPoolExecutor(max_workers=32) as executor:
    futures = []
    for item_index, item in enumerate(data):
        new_item = {
            'question': item['question'],
            'answer': item['answer'],
            'supporting_facts_context': item['supporting_facts_context'],
            'golden_context': [],
            'noisy_context': [],
            'distracting_context': []
        }
        for context_type in ['golden_context', 'distracting_context', 'noisy_context']:
            for context_index, context in enumerate(item[context_type]):
                future = executor.submit(process_context, context, item_index, context_index, 256, " ", 64)
                futures.append((future, context_type, new_item))
        new_data.append(new_item)

    for future in as_completed([f[0] for f in futures]):
        try:
            result = future.result()
            context_type = next(f[1] for f in futures if f[0] == future)
            new_item = next(f[2] for f in futures if f[0] == future)
            new_item[context_type].append(result)
        except Exception as e:
            print(f"Error processing future: {e}")

with open(output_file_path, 'w', encoding='utf-8') as file:
    json.dump(new_data, file, ensure_ascii=False, indent=4)
print('done')
