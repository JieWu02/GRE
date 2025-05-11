import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import matplotlib.pyplot as plt
import tiktoken
from bs4 import BeautifulSoup as Soup
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import set_start_method
from openai import OpenAI
import umap.umap_ as umap
from scipy import spatial
import json
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from FlagEmbedding import FlagModel
from langchain_anthropic import ChatAnthropic
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from sklearn.mixture import GaussianMixture
from multiprocessing import Value, Manager
RANDOM_SEED = 224  # Fixed seed for reproducibility
client = OpenAI(
    api_key="sk-af4a75634c7c4f26bb5b610307e3820d",
    base_url="https://api.deepseek.com",
)
def chat(query: str) -> str:
    setting = [{"role": "user", "content": query}]
    while True:
        try:
            completion = client.chat.completions.create(
                model="deepseek-chat",
                messages=setting,
                temperature=0.0,
                max_tokens=200
            )
            result = completion.choices[0].message.content
            return result
        except Exception as e:
            error_message = str(e)
            if "400" in error_message and "high risk" in error_message:
                print(f"Bad request error: {error_message}. Skipping this query.")
                return None
            print(f"Request failed: {error_message}. Retrying in 10 seconds...")
            time.sleep(1)


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def global_cluster_embeddings(
        embeddings: np.ndarray,
        dim: int,
        n_neighbors: Optional[int] = None,
        metric: str = "cosine",
) -> np.ndarray:
    """
    Perform global dimensionality reduction on the embeddings using UMAP.

    Parameters:
    - embeddings: The input embeddings as a numpy array.
    - dim: The target dimensionality for the reduced space.
    - n_neighbors: Optional; the number of neighbors to consider for each point.
                   If not provided, it defaults to the square root of the number of embeddings.
    - metric: The distance metric to use for UMAP.

    Returns:
    - A numpy array of the embeddings reduced to the specified dimensionality.
    """
    if n_neighbors is None:
        n_neighbors = int((len(embeddings) - 1) ** 0.5)
    return umap.UMAP(
        n_neighbors=n_neighbors, n_components=dim, metric=metric
    ).fit_transform(embeddings)


def local_cluster_embeddings(
        embeddings: np.ndarray, dim: int, num_neighbors: int = 10, metric: str = "cosine"
) -> np.ndarray:
    """
    Perform local dimensionality reduction on the embeddings using UMAP, typically after global clustering.

    Parameters:
    - embeddings: The input embeddings as a numpy array.
    - dim: The target dimensionality for the reduced space.
    - num_neighbors: The number of neighbors to consider for each point.
    - metric: The distance metric to use for UMAP.

    Returns:
    - A numpy array of the embeddings reduced to the specified dimensionality.
    """
    return umap.UMAP(
        n_neighbors=num_neighbors, n_components=dim, metric=metric
    ).fit_transform(embeddings)


def get_optimal_clusters(
        embeddings: np.ndarray, max_clusters: int = 50, random_state: int = RANDOM_SEED
) -> int:
    """
    Determine the optimal number of clusters using the Bayesian Information Criterion (BIC) with a Gaussian Mixture Model.

    Parameters:
    - embeddings: The input embeddings as a numpy array.
    - max_clusters: The maximum number of clusters to consider.
    - random_state: Seed for reproducibility.

    Returns:
    - An integer representing the optimal number of clusters found.
    """
    max_clusters = min(max_clusters, len(embeddings))
    n_clusters = np.arange(1, max_clusters)
    bics = []
    for n in n_clusters:
        gm = GaussianMixture(n_components=n, random_state=random_state)
        gm.fit(embeddings)
        bics.append(gm.bic(embeddings))
    return n_clusters[np.argmin(bics)]


def GMM_cluster(embeddings: np.ndarray, threshold: float, random_state: int = 0):
    """
    Cluster embeddings using a Gaussian Mixture Model (GMM) based on a probability threshold.

    Parameters:
    - embeddings: The input embeddings as a numpy array.
    - threshold: The probability threshold for assigning an embedding to a cluster.
    - random_state: Seed for reproducibility.

    Returns:
    - A tuple containing the cluster labels and the number of clusters determined.
    """
    n_clusters = get_optimal_clusters(embeddings)
    gm = GaussianMixture(n_components=n_clusters, random_state=random_state)
    gm.fit(embeddings)
    probs = gm.predict_proba(embeddings)
    labels = [np.where(prob > threshold)[0] for prob in probs]
    return labels, n_clusters


def perform_clustering(
        embeddings: np.ndarray,
        dim: int,
        threshold: float,
) -> List[np.ndarray]:
    """
    Perform clustering on the embeddings by first reducing their dimensionality globally, then clustering
    using a Gaussian Mixture Model, and finally performing local clustering within each global cluster.

    Parameters:
    - embeddings: The input embeddings as a numpy array.
    - dim: The target dimensionality for UMAP reduction.
    - threshold: The probability threshold for assigning an embedding to a cluster in GMM.

    Returns:
    - A list of numpy arrays, where each array contains the cluster IDs for each embedding.
    """
    if len(embeddings) <= dim + 1:
        # Avoid clustering when there's insufficient data
        return [np.array([0]) for _ in range(len(embeddings))]

    # Global dimensionality reduction
    reduced_embeddings_global = global_cluster_embeddings(embeddings, dim)
    # Global clustering
    global_clusters, n_global_clusters = GMM_cluster(
        reduced_embeddings_global, threshold
    )

    all_local_clusters = [np.array([]) for _ in range(len(embeddings))]
    total_clusters = 0

    # Iterate through each global cluster to perform local clustering
    for i in range(n_global_clusters):
        # Extract embeddings belonging to the current global cluster
        global_cluster_embeddings_ = embeddings[
            np.array([i in gc for gc in global_clusters])
        ]

        if len(global_cluster_embeddings_) == 0:
            continue
        if len(global_cluster_embeddings_) <= dim + 1:
            # Handle small clusters with direct assignment
            local_clusters = [np.array([0]) for _ in global_cluster_embeddings_]
            n_local_clusters = 1
        else:
            # Local dimensionality reduction and clustering
            reduced_embeddings_local = local_cluster_embeddings(
                global_cluster_embeddings_, dim
            )
            local_clusters, n_local_clusters = GMM_cluster(
                reduced_embeddings_local, threshold
            )

        # Assign local cluster IDs, adjusting for total clusters already processed
        for j in range(n_local_clusters):
            local_cluster_embeddings_ = global_cluster_embeddings_[
                np.array([j in lc for lc in local_clusters])
            ]
            indices = np.where(
                (embeddings == local_cluster_embeddings_[:, None]).all(-1)
            )[1]
            for idx in indices:
                all_local_clusters[idx] = np.append(
                    all_local_clusters[idx], j + total_clusters
                )

        total_clusters += n_local_clusters

    return all_local_clusters


def embed(texts):
    """
    Generate embeddings for a list of text documents.

    This function assumes the existence of an embd object with a method embed_documents
    that takes a list of texts and returns their embeddings.

    Parameters:
    - texts: List[str], a list of text documents to be embedded.

    Returns:
    - numpy.ndarray: An array of embeddings for the given text documents.
    """
    # text_embeddings = embd.embed_documents(texts)
    cleaned_texts = [text.replace("\n", " ").strip() for text in texts]
    text_embeddings = embd.encode(cleaned_texts)
    # text_embeddings = embd.encode([texts.replace("\n", " ").strip()])[0]
    text_embeddings_np = np.array(text_embeddings)
    return text_embeddings_np


def embed_cluster_texts(texts):
    """
    Embeds a list of texts and clusters them, returning a DataFrame with texts, their embeddings, and cluster labels.

    This function combines embedding generation and clustering into a single step. It assumes the existence
    of a previously defined perform_clustering function that performs clustering on the embeddings.

    Parameters:
    - texts: List[str], a list of text documents to be processed.

    Returns:
    - pandas.DataFrame: A DataFrame containing the original texts, their embeddings, and the assigned cluster labels.
    """
    text_embeddings_np = embed(texts)  # Generate embeddings
    cluster_labels = perform_clustering(
        text_embeddings_np, 10, 0.5
    )  # Perform clustering on the embeddings
    df = pd.DataFrame()  # Initialize a DataFrame to store the results
    df["text"] = texts  # Store original texts
    df["embd"] = list(text_embeddings_np)  # Store embeddings as a list in the DataFrame
    df["cluster"] = cluster_labels  # Store cluster labels
    return df


def fmt_txt(df: pd.DataFrame) -> str:
    """
    Formats the text documents in a DataFrame into a single string.

    Parameters:
    - df: DataFrame containing the 'text' column with text documents to format.

    Returns:
    - A single string where all text documents are joined by a specific delimiter.
    """
    unique_txt = df["text"].tolist()
    return "--- --- \n --- --- ".join(unique_txt)


def embed_cluster_summarize_texts_by_deepseek(
        texts: List[str], level: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Embeds, clusters, and summarizes a list of texts. This function first generates embeddings for the texts,
    clusters them based on similarity, expands the cluster assignments for easier processing, and then summarizes
    the content within each cluster.

    Parameters:
    - texts: A list of text documents to be processed.
    - level: An integer parameter that could define the depth or detail of processing.

    Returns:
    - Tuple containing two DataFrames:
      1. The first DataFrame (df_clusters) includes the original texts, their embeddings, and cluster assignments.
      2. The second DataFrame (df_summary) contains summaries for each cluster, the specified level of detail,
         and the cluster identifiers.
    """

    # Embed and cluster the texts, resulting in a DataFrame with 'text', 'embd', and 'cluster' columns
    df_clusters = embed_cluster_texts(texts)

    # Prepare to expand the DataFrame for easier manipulation of clusters
    expanded_list = []

    # Expand DataFrame entries to document-cluster pairings for straightforward processing
    for index, row in df_clusters.iterrows():
        for cluster in row["cluster"]:
            expanded_list.append(
                {"text": row["text"], "embd": row["embd"], "cluster": cluster}
            )

    # Create a new DataFrame from the expanded list
    expanded_df = pd.DataFrame(expanded_list)

    # Retrieve unique cluster identifiers for processing
    all_clusters = expanded_df["cluster"].unique()

    # print(f"--Generated {len(all_clusters)} clusters--")

    # Summarization
    template = """
    You are a Summarizing Text Portal.
    Summarise the following in 3 concise sentences or less.    
    Contexts:
    {context}
    """
    # Format text within each cluster for summarization
    summaries = []
    sum_calls = 0
    sum_token = 0
    for i in all_clusters:
        df_cluster = expanded_df[expanded_df["cluster"] == i]
        formatted_txt = fmt_txt(df_cluster)
        context = template.format(context=formatted_txt)
        summary = chat(context)
        sum_calls += 1  # Increment the call count
        sum_token += num_tokens_from_string(summary, 'cl100k_base')
        summaries.append(summary)
        # print(f'\n Summary for {i} th cluster :\n', summary)
    print(sum_calls, sum_token)
    # Create a  DataFrame to store summaries with their corresponding cluster and level
    df_summary = pd.DataFrame(
        {
            "summaries": summaries,
            "level": [level] * len(summaries),
            "cluster": list(all_clusters),
        }
    )
    return df_clusters, df_summary


def recursive_embed_cluster_summarize(
        texts: List[str], level: int = 1, n_levels: int = 3
) -> Dict[int, Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Recursively embeds, clusters, and summarizes texts up to a specified level or until
    the number of unique clusters becomes 1, storing the results at each level.

    Parameters:
    - texts: List[str], texts to be processed.
    - level: int, current recursion level (starts at 1).
    - n_levels: int, maximum depth of recursion.

    Returns:
    - Dict[int, Tuple[pd.DataFrame, pd.DataFrame]], a dictionary where keys are the recursion
      levels and values are tuples containing the clusters DataFrame and summaries DataFrame at that level.
    """
    results = {}  # Dictionary to store results at each level

    # Perform embedding, clustering, and summarization for the current level
    df_clusters, df_summary= embed_cluster_summarize_texts_by_deepseek(texts, level)

    # Store the results of the current level
    results[level] = (df_clusters, df_summary)

    # Determine if further recursion is possible and meaningful
    unique_clusters = df_summary["cluster"].nunique()
    if level < n_levels and unique_clusters > 1:
        # Use summaries as the input texts for the next level of recursion
        new_texts = df_summary["summaries"].tolist()
        next_level_results = recursive_embed_cluster_summarize(
            new_texts, level + 1, n_levels
        )

        # Merge the results from the next level into the current results dictionary
        results.update(next_level_results)

    return results


def distances_from_embeddings(
    query_embedding: List[float],
    embeddings: List[List[float]],
    distance_metric: str = "cosine"
) -> List[float]:
    distance_metrics = {
        "cosine": spatial.distance.cosine,
        "L1": spatial.distance.cityblock,
        "L2": spatial.distance.euclidean,
        "Linf": spatial.distance.chebyshev,
    }

    if distance_metric not in distance_metrics:
        raise ValueError(
            f"Unsupported distance metric '{distance_metric}'. Supported metrics are: {list(distance_metrics.keys())}"
        )

    distances = [
        distance_metrics[distance_metric](query_embedding, embedding)
        for embedding in embeddings
    ]

    return distances

# Function to get indices of nearest neighbors
def indices_of_nearest_neighbors_from_distances(distances: List[float]) -> np.ndarray:
    return np.argsort(distances)

embd = FlagModel("BAAI/bge-base-en-v1.5", use_fp16=True)
file_path = '../../step1_chunk_docs/top25_chunks_datasets/Musique_top25_chunks.json'
with open(file_path, 'r', encoding='utf-8') as file:
    json_data = json.load(file)

# all_trees = []
# for data in tqdm(json_data, desc="Processing data"):
#     docs_texts = []
#     passages = data['golden_context'] + data['noisy_context'] + data['distracting_context']
#     for passage in passages:
#         for chunk in passage[1:]:
#             docs_texts.append(chunk)
#
#     texts_split = docs_texts
#     print(len(texts_split))
#
#     # Build tree
#     leaf_texts = docs_texts
#     results = recursive_embed_cluster_summarize(leaf_texts, level=1, n_levels=3)
#
#     all_texts = leaf_texts.copy()
#
#     for level in sorted(results.keys()):
#         # Extract summaries from the current level's DataFrame
#         summaries = results[level][1]["summaries"].tolist()
#         # Extend all_texts with the summaries from the current level
#         all_texts.extend(summaries)
#
#     tree_json = {
#         'question': data['question'],
#         'raptor_tree': all_texts,
#         'answer': data['answer'],
#         'supporting_facts_context': data['supporting_facts_context']
#     }
#     # tree_json_str = json.dumps(tree_json, indent=4)
#     all_trees.append(tree_json)
#
#     embeddings = embd.encode(all_texts)  # Generate embeddings for all texts
#     # Query and its embedding
#     query = data['question']
#     query_embedding = embd.encode([query])[0]
#     # Calculate distances
#     distances = distances_from_embeddings(query_embedding, embeddings)
#     # Get indices of nearest neighbors
#     indices = indices_of_nearest_neighbors_from_distances(distances)
#     # Retrieve top 10 texts
#     top_10_texts = [all_texts[i] for i in indices[:10]]
#     # print('\nTop 10 text', top_10_texts)
#
#     QA_template = """
#     Answer the question based on context.
#     Question is:
#     {question}
#     Context is:
#     {context}
#     Output format is:
#     Answer: [Answer]
#     """
#     query = QA_template.format(question=data['question'], context=top_10_texts)
#     prediction = chat(query)
#
#     print('Prediction', prediction)
#     print('label', data['answer'])
#
# # 将所有的树保存到一个文件中
# with open("rag_raptor_collapsed_trees_2WikiMQA_test.json", "w") as file:
#     json.dump(all_trees, file, indent=4)
#
# print('All trees written to file')


def process_data(data):
    docs_texts = []
    # passages = data['golden_context'] + data['noisy_context'] + data['distracting_context']
    passages = data['golden_context'] + data['noisy_context']
    for passage in passages:
        for chunk in passage[1:]:
            docs_texts.append(chunk)

    texts_split = docs_texts
    # print(len(texts_split))

    # Build tree
    leaf_texts = docs_texts
    results = recursive_embed_cluster_summarize(leaf_texts, level=1, n_levels=3)

    all_texts = leaf_texts.copy()

    for level in sorted(results.keys()):
        # Extract summaries from the current level's DataFrame
        summaries = results[level][1]["summaries"].tolist()
        # Extend all_texts with the summaries from the current level
        all_texts.extend(summaries)

    tree_json = {
        'question': data['question'],
        'raptor_tree': all_texts,
        'answer': data['answer'],
        'supporting_facts_context': data['supporting_facts_context']
    }

    embeddings = embd.encode(all_texts)  # Generate embeddings for all texts
    # Query and its embedding
    query = data['question']
    query_embedding = embd.encode([query])[0]
    # Calculate distances
    distances = distances_from_embeddings(query_embedding, embeddings)
    # Get indices of nearest neighbors
    indices = indices_of_nearest_neighbors_from_distances(distances)
    # Retrieve top 10 texts
    top_10_texts = [all_texts[i] for i in indices[:10]]

    QA_template = """
    Answer the question based on context.
    Question is:
    {question}
    Context is:
    {context}
    Output format is:
    Answer: [Answer]
    """
    query = QA_template.format(question=data['question'], context=top_10_texts)
    prediction = chat(query)

    print('Prediction', prediction)
    print('label', data['answer'])

    return tree_json

def main():
    import time
    all_trees = []
    start_time = time.time()  # 记录开始时间
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(process_data, data) for data in json_data]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing data"):
            all_trees.append(future.result())
    end_time = time.time()  # 记录结束时间
    elapsed_time = end_time - start_time  # 计算时间差
    print(f"程序运行时间: {elapsed_time:.2f} 秒")
    # 将所有的树保存到一个文件中
    with open("HSG_results/HSG_Musique.json", "w", encoding="utf-8") as file:
        json.dump(all_trees, file, ensure_ascii=False, indent=4)

    print('All trees written to file')

if __name__ == "__main__":
    set_start_method('spawn')
    main()