import os
import json
import time
import requests
from tqdm import tqdm
from openai import OpenAI
import concurrent.futures

import importlib.util

# spec = importlib.util.spec_from_file_location("call_gpt4", "/home/superbench/yaoxiang/code/DataSynthesis/call_api_local/call_api_2.py")
# test_api = importlib.util.module_from_spec(spec)
# spec.loader.exec_module(test_api)
# call_gpt = test_api.call_gpt4

def Deepseek_chat(prompt, api_key='sk-QBAKTRQN9Bb8d00543ddT3BLbkFJ88acBd84bBa24502B5bf', max_retries=3):
    # https://www.ohmygpt.com/ 中转站
    url = "https://cn2us02.opapi.win/v1/chat/completions"
    payload = json.dumps({
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    })
    headers = {
        'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
        'Content-Type': 'application/json',
        "Authorization": f'Bearer {api_key}',
    }

    retries = 0
    while retries < max_retries:
        try:
            response = requests.post(url, headers=headers, data=payload)
            response.raise_for_status()
            res = response.json()

            if 'choices' in res and len(res['choices']) > 0:
                res_content = res['choices'][0]['message']['content']
                return res_content
            else:
                print(f"Unexpected response structure: {res}")
                return None

        except requests.exceptions.RequestException as e:
            print(f"Request failed: {str(e)}. Retrying in 3 seconds... (Attempt {retries + 1}/{max_retries})")
            time.sleep(3)
            retries += 1
        except json.JSONDecodeError as e:
            print(f"Failed to decode JSON response: {str(e)}")
            return None

    print(f"Maximum retries reached. Failed to process prompt.")
    return None


def moonshotai_chat(query, api_key):
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.moonshot.cn/v1"
    )
    setting = [{
        "role": "user",
        "content": query
    }]
    try:
        completion = client.chat.completions.create(
            model="moonshot-v1-8k",
            messages=setting,
            temperature=0.0,
            max_tokens=20
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



# def GPT(query):
    # client = OpenAI(
    #     api_key=api_key,
    #     base_url="https://api.ohmygpt.com/v1",
    # )
    # setting = [{
    #     "role": "user",
    #     "content": query
    # }]
    # try:
    #     completion = client.chat.completions.create(
    #         model=model,
    #         messages=setting,
    #         temperature=0.,
    #         max_tokens=20
    #     )
    #     result = completion.choices[0].message.content
    #     return result
    # except Exception as e:
    #     error_message = str(e)
    #     print(f"Request failed: {error_message}. Retrying in 10 seconds...")
    #     time.sleep(1)
    #     return None
    # return call_gpt(query)[0]

def GPT(prompt, api_key='sk-QBAKTRQN9Bb8d00543ddT3BLbkFJ88acBd84bBa24502B5bf', max_retries=3):
    url = "https://cn2us02.opapi.win/v1/chat/completions"
    payload = json.dumps({
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    })
    headers = {
        'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
        'Content-Type': 'application/json',
        "Authorization": f'Bearer {api_key}',
    }

    retries = 0
    while retries < max_retries:
        try:
            response = requests.post(url, headers=headers, data=payload)
            response.raise_for_status()
            res = response.json()

            if 'choices' in res and len(res['choices']) > 0:
                res_content = res['choices'][0]['message']['content']
                return res_content
            else:
                print(f"Unexpected response structure: {res}")
                return None

        except requests.exceptions.RequestException as e:
            print(f"Request failed: {str(e)}. Retrying in 3 seconds... (Attempt {retries + 1}/{max_retries})")
            time.sleep(3)
            retries += 1
        except json.JSONDecodeError as e:
            print(f"Failed to decode JSON response: {str(e)}")
            return None

    print(f"Maximum retries reached. Failed to process prompt.")
    return None

# gpt4o_mini分析函数
def OpenAI_chat(prompt, api_key, max_retries=3):
    url = "https://cn2us02.opapi.win/v1/chat/completions"
    payload = json.dumps({
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    })
    headers = {
        'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
        'Content-Type': 'application/json',
        "Authorization": 'Bearer ' + api_key,
    }

    retries = 0
    while retries < max_retries:
        try:
            response = requests.post(url, headers=headers, data=payload)
            response.raise_for_status()  # Raise an HTTPError for bad responses
            res = response.json()

            # Ensure the response contains the 'choices' key
            if 'choices' in res and len(res['choices']) > 0:
                res_content = res['choices'][0]['message']['content']
                return res_content
            else:
                error_message = f"Unexpected response structure: {res}"
                print(error_message)
                return None

        except requests.exceptions.RequestException as e:
            error_message = f"Request failed: {str(e)}"
            print(f"{error_message}. Retrying in 10 seconds... (Attempt {retries + 1}/{max_retries})")
            time.sleep(3)
            retries += 1

        except json.JSONDecodeError as e:
            error_message = f"Failed to decode JSON response: {str(e)}"
            print(error_message)
            return None

    # If the maximum number of retries is reached without success
    print(f"Maximum retries reached. Failed to process prompt: {prompt}")
    return None

def process_query(item, api_type, api_key):
    query = item.get("query")
    if isinstance(query, list):
        query = ", ".join(query)
    label = item.get("answer")
    if query:
        if api_type == 'Deepseek':
            # api_key = 'sk-89a3b45138854e54b91a61e36e79dcf5'
            response = Deepseek_chat(query)
        elif api_type == 'MoonshotAI':
            api_key = 'sk-ZQ4N6Og1SBbPYbUfKqPADeK4JRAKa4ntJ0Ce7da9m9Y8yMpj'
            response = moonshotai_chat(query, api_key)
        elif api_type == 'GPT':
            api_key = 'sk-OrS5MQVIc7a90Ed1B21BT3BlBKFJ278dbdf5DD31405eA8C1'
            response = GPT(query)
        else:
            raise ValueError("Unsupported API type")

        result_item = {
            "query": query,
            "response": response,
            "label": label
        }
        return result_item
    else:
        print(f"No query found for item {i + 1}")
        return None


def main(api_type, dataset, structure, topk, api_key):
    os.environ['CUDA_VISIBLE_DEVICES'] = '6'

    print(f'API Type: {api_type}')

    # Construct paths
    # data_path = f'{structure}_eval_data/{dataset}/Top{topk}.json'
    # output_file_path = f'{structure}_eval_result/{dataset}/{api_type}/Top{topk}_result.json'
    data_path = '../step2_construct_graph/IPG/IPG_results/IPG_Hotpot_9_for_reasoning.json'
    output_file_path = '../step2_construct_graph/IPG/IPG_results/IPG_Hotpot_9_for_reasoning_.json'
    with open(data_path, 'r') as f:
        data = json.load(f)

    results = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=48) as executor:
        futures = [executor.submit(process_query, item, api_type, api_key) for item in data]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=f"Processing queries with {api_type}"):
            result = future.result()
            if result:
                results.append(result)

    if results:
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        with open(output_file_path, 'a') as outfile:
            json.dump(results, outfile, ensure_ascii=False, indent=4)

    print("All results saved to:", output_file_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process some parameters.")
    parser.add_argument('--api_type', type=str, required=True, help="API type (MoonshotAI, Deepseek, GPT)")
    parser.add_argument('--dataset', type=str, required=True, help="Datasets", default="Hotpot")
    parser.add_argument('--structure', type=str, required=True, help="Graph, Raptor, Sequential")
    parser.add_argument('--topk', type=str, required=False, help="Topk 10, 15, 20, 25")
    parser.add_argument('--api_key', type=str, required=False, help="API key for the selected API")

    args = parser.parse_args()

    main(args.api_type, args.dataset, args.structure, args.topk, args.api_key)
