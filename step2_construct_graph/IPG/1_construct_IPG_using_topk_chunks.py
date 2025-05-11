import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import nltk
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity
from FlagEmbedding import FlagModel
from sentence_transformers import SentenceTransformer
import torch
import tiktoken
import pickle
import nltk
import re
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
def clean_text_for_matching(text):
    # 清理文本以进行匹配
    text = re.sub(r'\s+', ' ', text)  # 将多个空格替换为一个空格
    text = text.strip()  # 去除首尾空格
    return text

nltk.data.path.append("../../nltk/nltk_data")

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FlagModel("BAAI/bge-base-en-v1.5", use_fp16=True)
model_instruction = FlagModel('BAAI/bge-base-en-v1.5',
                          query_instruction_for_retrieval="Represent this sentence for searching relevant passages: ",
                          use_fp16=True)

def get_bge_embedding(text):
    text = text.replace("\n", " ").strip()
    embedding = model.encode([text])  # 获取单个文本的嵌入
    return embedding

def get_sentence_transformer_embedding(texts):
    embeddings = st_model.encode(texts, normalize_embeddings=True)
    return embeddings

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def fact_in_neighbors(fact, node_idx):
    # 检查节点自身内容
    if fact in node_contents[node_idx]:
        return node_labels[node_idx]

    # 检查入边邻居节点内容
    in_neighbors = [edge[0] for edge in G.in_edges(node_labels[node_idx])]
    for neighbor in in_neighbors:
        neighbor_idx = node_labels.index(neighbor)
        if fact in node_contents[neighbor_idx]:
            return neighbor

    # 检查出边邻居节点内容
    out_neighbors = [edge[1] for edge in G.out_edges(node_labels[node_idx])]
    for neighbor in out_neighbors:
        neighbor_idx = node_labels.index(neighbor)
        if fact in node_contents[neighbor_idx]:
            return neighbor

    return None

def merge_nodes(G, source_node, target_node, new_node_label):
    if nx.has_path(G, source_node, target_node):
        path = nx.shortest_path(G, source=source_node, target=target_node)
        print(f'{source_node} can reach {target_node}. Path: {path}', 'No changes made.')
        return False
    elif nx.has_path(G, target_node, source_node):
        path = nx.shortest_path(G, source=target_node, target=source_node)
        print(f'{target_node} can reach {source_node}. Path: {path}', 'No changes made.')
        return False
    source_content = G.nodes[source_node]['content']
    target_content = G.nodes[target_node]['content']

    new_node_content = source_content + target_content
    print(f'source_content: [{source_node}] ', source_content)
    print(f'target_content: [{target_node}] ', target_content)
    # getting response from llm
    # query = f"Following two paragraphs are related to the question **{question}**, \n" \
    #         f"Paragraph 1 is [{source_content}]\n" \
    #         f"Paragraph 2 is [{target_content}]\n" \
    #         f"Please summarise the relationship between the two paragraphs and argue whether they can help answer the question"
    # response_llm_zero_shot, history = inference(llm, template, query)
    # print('merging node with llm\n', query, '\n')
    # print('response by llm\n', response_llm_zero_shot,'\n')
    # new_node_content = new_node_content + response_llm_zero_shot

    G.add_node(new_node_label, content=new_node_content)
    print(f"node [{source_node}] and node [{target_node}] fuse into {new_node_label}")
    source_in_edges = list(G.in_edges(source_node))
    source_out_edges = list(G.out_edges(source_node))
    target_in_edges = list(G.in_edges(target_node))
    target_out_edges = list(G.out_edges(target_node))

    for edge in source_in_edges:
        G.add_edge(edge[0], new_node_label)
    for edge in target_in_edges:
        G.add_edge(edge[0], new_node_label)

    for edge in source_out_edges:
        G.add_edge(new_node_label, edge[1])
    for edge in target_out_edges:
        G.add_edge(new_node_label, edge[1])
    print('After fusion:', G.nodes[new_node_label]['content'])
    G.remove_node(source_node)
    G.remove_node(target_node)
    return True

def retrieve_nodes_with_edges(G, top_indices, node_contents, node_labels, question, answer, supporting_facts, hop2_indices):
    nodes_data = []
    for idx in top_indices:
        node_label = node_labels[idx]
        node_content = node_contents[idx]

        # Get in-edges
        in_edges = list(G.in_edges(node_label))
        in_edges_data = []
        for edge in in_edges:
            in_node_label = edge[0]
            in_node_idx = node_labels.index(in_node_label)
            in_edges_data.append({
                "node_label": in_node_label,
                "content": node_contents[in_node_idx]
            })
        out_edges = list(G.out_edges(node_label))
        out_edges_data = []
        # excluding original out edges
        for edge in out_edges:
            out_node_label = edge[1]
            out_node_idx = node_labels.index(out_node_label)
            if out_node_idx in hop2_indices:
                out_edges_data.append({
                    "node_label": out_node_label,
                    "content": node_contents[out_node_idx]
                })

        nodes_data.append({
            "node_label": node_label,
            "content": node_content,
            "out_edges": out_edges_data,
            "in_edges": in_edges_data
        })

    data = {
        "question": question,
        "answer": answer,
        "supporting_facts": supporting_facts,
        "nodes": nodes_data
    }
    return data

total_token = 0
total_doc_token = 0
sum_node = 0
all_saved_graphs = []
graph_dict = {}
input_path = r'../../step1_chunk_docs/top25_chunks_datasets/Hotpot_top25_chunks.json'
with open(input_path, 'r', encoding='utf-8') as file:
    json_data = json.load(file)[:100]
import time
start_time = time.time()
for item in json_data:
    data = item
    documents = []
    # without distracting data
    # data['noisy_context'].extend(data['distracting_context'])
    for passage in data['golden_context'] + data['noisy_context']:
        combined_text = ' '.join(passage)
        documents.append(combined_text)

    # Create a TF-IDF vectorizer and fit it to the documents
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)

    # Transform the question into the same TF-IDF space
    question_tfidf = vectorizer.transform([data['question']])

    # Compute the cosine similarity between the question and each passage
    cosine_similarities = np.dot(question_tfidf, tfidf_matrix.T).toarray()[0]

    # Rank the passages based on their similarity to the question
    ranked_indices = np.argsort(cosine_similarities)[::-1]

    """
    create subchain
    """
    # Create graphs
    G = nx.DiGraph()
    # Add nodes and edges for each passage with simplified labels in the order of relevance
    for i, idx in enumerate(ranked_indices):
        passage = data['golden_context'][idx] if idx < len(data['golden_context']) else data['noisy_context'][idx - len(data['golden_context'])]
        label_prefix = chr(65 + i)  # 'A' for first passage, 'B' for second, etc.
        for j in range(1, len(passage)):
            node_label = f'{label_prefix}{j}'
            G.add_node(node_label, content=passage[j])
            if j > 1:
                prev_node_label = f'{label_prefix}{j-1}'
                G.add_edge(prev_node_label, node_label)

    """
    computing topic similarity between chunks
    """
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    def preprocess_text(text):
        tokens = word_tokenize(text.lower())
        tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalpha() and word not in stop_words]
        return ' '.join(tokens)

    chunks = []
    chunk_labels = []

    for idx in ranked_indices:
        passage = data['golden_context'][idx] if idx < len(data['golden_context']) else data['noisy_context'][idx - len(data['golden_context'])]
        chunks.extend(passage[1:])
        chunk_labels.extend([passage[0]] * (len(passage) - 1))

    chunk_counter = 1
    chunk_node_mapping = []

    for i, idx in enumerate(ranked_indices):
        passage = data['golden_context'][idx] if idx < len(data['golden_context']) else data['noisy_context'][idx - len(data['golden_context'])]
        label_prefix = chr(65 + i)  # 'A' for first passage, 'B' for second, etc.
        for j in range(1, len(passage)):
            node_label = f'{label_prefix}{j}'
            chunk_node_mapping.append(node_label)
            # print(f"Chunk {chunk_counter}.{j} 对应节点编号: {node_label}")
        chunk_counter += 1


    preprocessed_chunks = [preprocess_text(chunk) for chunk in chunks]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(preprocessed_chunks)
    similarity_matrix = cosine_similarity(tfidf_matrix)

    for i, label_i in enumerate(chunk_labels):
        for j, label_j in enumerate(chunk_labels):
            if label_i == label_j:
                similarity_matrix[i, j] = float('inf')

    """
    Find the highest similarity chunk for each chunk in the most similar passage
    """
    def process_passage(passage_index, dead_nodes, only_one_pair=False):
        print(f'\nPassage {passage_index + 1}')
        # Identify the most similar passage to the question
        most_similar_passage_index = ranked_indices[passage_index]
        most_similar_passage = (data['golden_context'][most_similar_passage_index]
                                if most_similar_passage_index < len(data['golden_context'])
                                else data['noisy_context'][most_similar_passage_index - len(data['golden_context'])])
        # Get chunks from the most similar passage
        most_similar_chunks = most_similar_passage[1:]
        print("Most similar chunks to each chunk in the most similar passage:")

        max_similarity = -1
        best_chunk_pair = None

        for i, chunk in enumerate(most_similar_chunks):
            chunk_index = chunks.index(chunk)

            for j, other_chunk in enumerate(chunks):
                if (chunk_labels[chunk_index] != chunk_labels[j] and
                    chunk_node_mapping[j] not in dead_nodes and
                    chunk_node_mapping[chunk_index] not in dead_nodes):
                    if similarity_matrix[chunk_index, j] > max_similarity:
                        max_similarity = similarity_matrix[chunk_index, j]
                        best_chunk_pair = (chunk_index, j)

        if best_chunk_pair is not None and max_similarity > 0.0:
            chunk_index, most_similar_chunk_index = best_chunk_pair
            print(f"    {chunk_node_mapping[chunk_index]} Most similar to chunk ({chunk_node_mapping[most_similar_chunk_index]}) with similarity {max_similarity:.4f}")
            print(f"    源节点编号: {chunk_node_mapping[chunk_index]}; 融合点编号: {chunk_node_mapping[most_similar_chunk_index]}")
            target_node = chunk_node_mapping[chunk_index]
            source_node = chunk_node_mapping[most_similar_chunk_index]
            new_node_label = f'{source_node}_{target_node}'
            is_merged = merge_nodes(G, source_node, target_node, new_node_label)
            if is_merged:
                print(f"{source_node} and {target_node} has been merged.")
                dead_nodes.append(source_node)
                dead_nodes.append(target_node)
            else:
                print(f"{source_node} and {target_node} are connected and do not need to be aggregated.")
        else:
            print("    No similar chunk found.")

        return dead_nodes

    # def process_passage(passage_index, dead_nodes, only_one_pair=False):
    #     print(f'\nPassage {passage_index + 1}')
    #     # Identify the most similar passage to the question
    #     most_similar_passage_index = ranked_indices[passage_index]
    #     most_similar_passage = (data['golden_context'][most_similar_passage_index]
    #                             if most_similar_passage_index < len(data['golden_context'])
    #                             else data['noisy_context'][most_similar_passage_index - len(data['golden_context'])])
    #     # Get chunks from the most similar passage
    #     most_similar_chunks = most_similar_passage[1:]
    #     print("Most similar chunks to each chunk in the most similar passage:")
    #
    #     chunk_pairs = []
    #
    #     for i, chunk in enumerate(most_similar_chunks):
    #         chunk_index = chunks.index(chunk)
    #
    #         for j, other_chunk in enumerate(chunks):
    #             if (chunk_labels[chunk_index] != chunk_labels[j] and
    #                     chunk_node_mapping[j] not in dead_nodes and
    #                     chunk_node_mapping[chunk_index] not in dead_nodes):
    #                 if similarity_matrix[chunk_index, j] > 0.2:
    #                     chunk_pairs.append((chunk_index, j, similarity_matrix[chunk_index, j]))
    #
    #     if chunk_pairs:
    #         for chunk_index, most_similar_chunk_index, similarity in chunk_pairs:
    #             target_node = chunk_node_mapping[chunk_index]
    #             source_node = chunk_node_mapping[most_similar_chunk_index]
    #
    #             if target_node in G.nodes() and source_node in G.nodes():
    #                 print(
    #                     f"    {chunk_node_mapping[chunk_index]} Most similar to chunk ({chunk_node_mapping[most_similar_chunk_index]}) with similarity {similarity:.4f}")
    #                 print(
    #                     f"    源节点编号: {chunk_node_mapping[chunk_index]}; 融合点编号: {chunk_node_mapping[most_similar_chunk_index]}")
    #                 new_node_label = f'{source_node}_{target_node}'
    #                 is_merged = merge_nodes(G, source_node, target_node, new_node_label)
    #                 if is_merged:
    #                     print(f"{source_node} and {target_node} has been merged.")
    #                     dead_nodes.append(source_node)
    #                     dead_nodes.append(target_node)
    #                 else:
    #                     print(f"{source_node} and {target_node} are connected and do not need to be aggregated.")
    #             else:
    #                 print(f"    Either {source_node} or {target_node} is not in the graph.")
    #
    #     else:
    #         print("    No similar chunk found.")
    #
    #     return dead_nodes


    dead_nodes = []
    num_passages_to_process = len(ranked_indices)
    only_one_pair = True  # Set this to True or False based on your requirement

    for passage_index in range(num_passages_to_process):
        dead_nodes = process_passage(passage_index, dead_nodes, only_one_pair=only_one_pair)
    print(dead_nodes)

    graph_dict[item['question']] = G


    from collections import deque
    from rank_bm25 import BM25Okapi
    node_contents = []
    node_labels = []
    for node in G.nodes(data=True):
        node_label = node[0]
        node_content = node[1]['content']
        node_contents.append(node_content)
        node_labels.append(node_label)
    # 定义问题
    question = item['question']
    supporting_facts_context = item['supporting_facts_context']

    # 选择检索器
    retriever = 'bge_2_hop'

    if retriever == 'TF-IDF':
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(node_contents)
        question_tfidf = vectorizer.transform([question])
        cosine_similarities = cosine_similarity(question_tfidf, tfidf_matrix).flatten()
        top_indices = np.argsort(cosine_similarities)[-3:][::-1]

    elif retriever == 'BM25':
        tokenized_contents = [doc.split() for doc in node_contents]
        bm25 = BM25Okapi(tokenized_contents)
        tokenized_question = question.split()
        bm25_scores = bm25.get_scores(tokenized_question)
        top_indices = np.argsort(bm25_scores)[-3:][::-1]

    elif retriever == 'bge':
        question_embedding = get_bge_embedding(question).flatten()
        node_embeddings = np.array([get_bge_embedding(content).flatten() for content in node_contents])
        cosine_similarities = cosine_similarity([question_embedding], node_embeddings).flatten()
        top_indices = np.argsort(cosine_similarities)[-3:][::-1]

    elif retriever == 'bge_instruction':
        question_embedding = model_instruction.encode_queries([question]).flatten()
        node_embeddings = np.array([model_instruction.encode([content]).flatten() for content in node_contents])
        cosine_similarities = cosine_similarity([question_embedding], node_embeddings).flatten()
        top_indices = np.argsort(cosine_similarities)[-5:][::-1]
        hop2_indices = []

    elif retriever == 'bge_2_hop':
        question_embedding = model_instruction.encode_queries([question]).flatten()
        node_embeddings = np.array([model_instruction.encode([content]).flatten() for content in node_contents])
        cosine_similarities = cosine_similarity([question_embedding], node_embeddings).flatten()
        top_indices = np.argsort(cosine_similarities)[-9:][::-1]
        hop2_indices = []
        hop2_similarities = []
        for idx in top_indices:
            new_query = question + " " + node_contents[idx]
            new_query_embedding = model_instruction.encode_queries([new_query]).flatten()
            new_cosine_similarities = cosine_similarity([new_query_embedding], node_embeddings).flatten()
            new_top_indices = np.argsort(new_cosine_similarities)[::-1]

            neighbors = set()
            neighbors.update(set(G.successors(node_labels[idx])))
            neighbors.update(set(G.predecessors(node_labels[idx])))

            # filtered_indices = [new_idx for new_idx in new_top_indices if new_idx not in top_indices]
            # Ensure that filtered_indices not in top_indices and not in its neighbors
            filtered_indices = [new_idx for new_idx in new_top_indices if
                                new_idx not in top_indices and node_labels[new_idx] not in neighbors]

            filtered_indices = filtered_indices[:1]
            hop2_indices.extend(filtered_indices)
            hop2_similarities.extend(new_cosine_similarities[filtered_indices])

            node_label = node_labels[idx]
            if len(filtered_indices) > 0:
                G.add_edge(node_label, node_labels[filtered_indices[0]])
            if len(filtered_indices) > 1:
                G.add_edge(node_label, node_labels[filtered_indices[1]])
            if len(filtered_indices) > 2:
                G.add_edge(node_label, node_labels[filtered_indices[2]])

    encoding_name = 'cl100k_base'
    print('top_indices:\n',
          top_indices)
    print('hop2_indices:\n',
          hop2_indices)
    print("Most similar nodes:")
    for idx in top_indices:
        print(f"Node {node_labels[idx]}: {node_contents[idx]}")
        if retriever == 'TF-IDF':
            print(f"Similarity Score: {cosine_similarities[idx]:.4f}")
        elif retriever == 'BM25':
            print(f"BM25 Score: {bm25_scores[idx]:.4f}")
        elif retriever == 'bge' or retriever == 'bge_instruction':
            print(f"Cosine Similarity: {cosine_similarities[idx]:.4f}")
        print()

    # 统计token和node数量,不再统计out edges
    token_in_one_data = 0
    node_in_one_data = 0
    counted_nodes = set()
    for idx in top_indices:
        node_label = node_labels[idx]
        print(f"\nFind in edge and out edge for Top retrieved node {node_label}:")
        # Add current node to counted set and count tokens
        counted_nodes.add(node_label)
        token_in_one_data += num_tokens_from_string(node_contents[idx], encoding_name)
        # Process in-edges
        in_edges = list(G.in_edges(node_label))
        print("  In edges:")
        for edge in in_edges:
            in_node_label = edge[0]
            if in_node_label not in counted_nodes:
                in_node_content = node_contents[node_labels.index(in_node_label)]
                token_in_one_data += num_tokens_from_string(in_node_content, encoding_name)
                counted_nodes.add(in_node_label)
            print(f"    {in_node_label} -> {node_label}")
        # Process out-edges
        out_edges = list(G.out_edges(node_label))
        print("  Dont process Out edges Anymore")
        for edge in out_edges:
            out_node_label = edge[1]
            if out_node_label in [node_labels[i] for i in hop2_indices] and out_node_label not in counted_nodes:
                out_node_content = node_contents[node_labels.index(out_node_label)]
                token_in_one_data += num_tokens_from_string(out_node_content, encoding_name)
                counted_nodes.add(out_node_label)
            print(f"    {node_label} -> {out_node_label}")

    if retriever == 'bge_2_hop':
        for idx in hop2_indices:
            node_label = node_labels[idx]
            if node_label not in counted_nodes:
                token_in_one_data += num_tokens_from_string(node_contents[idx], encoding_name)
                counted_nodes.add(node_label)

    node_in_one_data = len(counted_nodes)
    print(f"Nodes in this data: {node_in_one_data}\n")
    print(f"Token in this data {token_in_one_data}\n")
    total_token += token_in_one_data
    sum_node += node_in_one_data
    print(f"Sum Retrieved Nodes num: {sum_node}\n")
    print(f"Sum Retrieved Toke num {total_token}\n")
    print()

    # fact_counter = 1
    # for fact in supporting_facts_context:
    #     fact = clean_text_for_matching(fact)
    #     found = False
    #     # Check in top_indices nodes
    #     for idx in top_indices:
    #         if fact in clean_text_for_matching(node_contents[idx]):
    #             print(f"Supporting fact {fact_counter} found in top_indices node {node_labels[idx]}: {fact}")
    #             found = True
    #             break
    #     # Check in in-edges neighbors
    #     if not found:
    #         for idx in top_indices:
    #             in_neighbors = [edge[0] for edge in G.in_edges(node_labels[idx])]
    #             for neighbor in in_neighbors:
    #                 neighbor_idx = node_labels.index(neighbor)
    #                 if fact in clean_text_for_matching(node_contents[neighbor_idx]):
    #                     print(
    #                         f"Supporting fact {fact_counter} found in in-edge neighbor {neighbor} of top_indices node {node_labels[idx]}: {fact}")
    #                     found = True
    #                     break
    #             if found:
    #                 break
    #     # Check in out-edges neighbors (excluding hop2_indices)
    #     if not found:
    #         for idx in top_indices:
    #             out_neighbors = [edge[1] for edge in G.out_edges(node_labels[idx])]
    #             for neighbor in out_neighbors:
    #                 neighbor_idx = node_labels.index(neighbor)
    #                 if fact in clean_text_for_matching(node_contents[neighbor_idx]) and neighbor not in [node_labels[i] for i in hop2_indices]:
    #                     print(
    #                         f"Supporting fact {fact_counter} found in out-edge neighbor {neighbor} of top_indices node {node_labels[idx]}: {fact}")
    #                     found = True
    #                     break
    #             if found:
    #                 break
    #     # Check in hop2_indices
    #     if not found and retriever == 'bge_2_hop':
    #         for idx in hop2_indices:
    #             if fact in clean_text_for_matching(node_contents[idx]):
    #                 print(f"Supporting fact {fact_counter} found in 2-hop node {node_labels[idx]}: {fact}")
    #                 found = True
    #                 break
    #     if not found:
    #         print(f"Supporting fact {fact_counter} NOT found: {fact}")
    #     # Increment the counter for the next supporting fact
    #     fact_counter += 1


    # 遍历图中所有节点，计算每个节点的 token 数量并累加
    for node in G.nodes(data=True):
        node_content = node[1]['content']
        total_doc_token += num_tokens_from_string(node_content, encoding_name)

    print(f"Total token count in the graph: {total_doc_token}")

    answer = data['answer']
    retrieved_data = retrieve_nodes_with_edges(G, top_indices, node_contents, node_labels, question, answer,
                                               supporting_facts_context, hop2_indices)
    all_saved_graphs.append(retrieved_data)
end_time = time.time()
print("Total time",end_time-start_time)

output_file = "IPG_results/IPG_Hotpot_9.json"
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(all_saved_graphs, f, ensure_ascii=False, indent=4)

print(f"Data saved to {output_file}")


# output_path = 'graph_formed_by_25_retrieved_5k.pkl'
# with open(output_path, 'wb') as f:
#     pickle.dump(graph_dict, f)
# print(f"All graphs have been saved to {output_path}")