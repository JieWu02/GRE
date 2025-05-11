import json


def build_node_as_query(data, filter=True):
    nodes = data.get('nodes', [])
    question = data.get('question', '')
    answer = data.get('answer', '')
    queries = [
        f"Answer the question based on the given content. \nQuestion is: **{question}**. "
        f"Following are documents content."]
    seen_content = set()

    content_counter = 1  # 初始化计数器

    for idx, node in enumerate(nodes):
        node_content = node.get('content', '')

        if not filter or node_content not in seen_content:
            queries.append(f"content [{content_counter}]: {node_content}")
            seen_content.add(node_content)
            content_counter += 1  # 计数器递增

    for node in nodes:
        for edge in node.get('in_edges', []):
            in_node_content = edge.get('content', '')

            if not filter or in_node_content not in seen_content:
                queries.append(f"content [{content_counter}]: {in_node_content}")
                seen_content.add(in_node_content)
                content_counter += 1  # 计数器递增

    # 添加所有 out_node_content
    # for node in nodes:
    #     for edge in node.get('out_edges', []):
    #         out_node_content = edge.get('content', '')
    #
    #         if not filter or out_node_content not in seen_content:
    #             queries.append(f"Content[{content_counter}]: {out_node_content}")
    #             seen_content.add(out_node_content)
    #             content_counter += 1  # 计数器递增

    queries.append("Answer the question based on the given contents.\n")
    queries.append(f"Question is: **{question}**\n")
    queries.append(f"Answer in less than 6 words. Your output format is **Answer**: Answer")
    return {'query': queries, 'answer': answer}



def build_node_prompt_as_query(data):
    nodes = data.get('nodes', [])
    question = data.get('question', '')
    answer = data.get('answer', '')
    queries = [
        f"Answer the question based on the given content. "
        f"Question is: {question}. "
        f"Following are documents content."
    ]
    seen_content = set()
    chains = []  # 用于存储链条

    for idx, node in enumerate(nodes):
        node_label = node.get('node_label', f"A{idx + 1}")
        node_content = node.get('content', '')

        if node_content not in seen_content:
            queries.append(f"Content[{node_label}]: {node_content}")
            seen_content.add(node_content)

        for edge in node.get('in_edges', []):
            in_node_label = edge.get('node_label', '')
            in_node_content = edge.get('content', '')

            if in_node_content not in seen_content:
                queries.append(f"Content[{in_node_label}]: {in_node_content}")
                seen_content.add(in_node_content)

        for edge in node.get('out_edges', []):
            out_node_label = edge.get('node_label', '')
            out_node_content = edge.get('content', '')

            if out_node_content not in seen_content:
                queries.append(f"Content[{out_node_label}]: {out_node_content}")
                seen_content.add(out_node_content)

    # 记录chains
    for idx, node in enumerate(nodes):
        node_label = node.get('node_label', f"A{idx + 1}")

        # 处理 in_edges
        has_in_edge = bool(node.get('in_edges', []))
        has_out_edge = bool(node.get('out_edges', []))

        for edge in node.get('in_edges', []):
            in_node_label = edge.get('node_label', '')

            # 添加链条
            if node.get('out_edges', []):
                for out_edge in node.get('out_edges', []):
                    out_node_label = out_edge.get('node_label', '')
                    chains.append(f"{in_node_label} -> {node_label} -> {out_node_label}")
            else:
                chains.append(f"{in_node_label} -> {node_label}")

        # 处理 out_edges（如果没有 in_edges，确保 out_edges 也被考虑）
        if not has_in_edge:
            for out_edge in node.get('out_edges', []):
                out_node_label = out_edge.get('node_label', '')
                chains.append(f"{node_label} -> {out_node_label}")

        # 如果节点既没有入边也没有出边，则将其单独作为一个链条
        if not has_in_edge and not has_out_edge:
            chains.append(f"{node_label}")

    queries.append("Here are the supporting chains of reasoning that may help you get to the answer.")
    for chain in chains:
        queries.append(chain)

    queries.append("Answer the question based on the given contents and supporting chains.")
    queries.append(f"Question is: {question}")
    queries.append(f"Answer in less than 6 words. Your output format is **Answer**:[Answer]")

    return {'query': queries, 'answer': answer}

def build_chains(data):
    nodes = data.get('nodes', [])
    question = data.get('question', '')
    answer = data.get('answer', '')
    queries = []
    seen_content = set()
    content_dict = {}  # 用于存储标签和内容的对应关系

    # Step 1: Collect content and build content_dict
    for idx, node in enumerate(nodes):
        node_label = node.get('node_label', f"A{idx + 1}")
        node_content = node.get('content', '')

        if node_content not in seen_content:
            seen_content.add(node_content)
            content_dict[node_label] = node_content
            print(f"Added node content: {node_label} -> {node_content}")

        for edge in node.get('in_edges', []):
            in_node_label = edge.get('node_label', '')
            in_node_content = edge.get('content', '')

            if in_node_content not in seen_content:
                seen_content.add(in_node_content)
                content_dict[in_node_label] = in_node_content
                print(f"Added in_edge content: {in_node_label} -> {in_node_content}")

        for edge in node.get('out_edges', []):
            out_node_label = edge.get('node_label', '')
            out_node_content = edge.get('content', '')

            if out_node_content not in seen_content:
                seen_content.add(out_node_content)
                content_dict[out_node_label] = out_node_content
                print(f"Added out_edge content: {out_node_label} -> {out_node_content}")

    # Step 2: Build chains
    chains = {}  # 用于存储链条，键是中间节点标签，值是链条列表
    for idx, node in enumerate(nodes):
        node_label = node.get('node_label', f"A{idx + 1}")

        has_in_edge = bool(node.get('in_edges', []))
        has_out_edge = bool(node.get('out_edges', []))

        for edge in node.get('in_edges', []):
            in_node_label = edge.get('node_label', '')

            if node.get('out_edges', []):
                for out_edge in node.get('out_edges', []):
                    out_node_label = out_edge.get('node_label', '')
                    try:
                        chain = f"{content_dict[in_node_label]} -> {content_dict[node_label]} -> {content_dict[out_node_label]}"
                        label_chain = f"{in_node_label} -> {node_label} -> {out_node_label}"
                        if node_label not in chains:
                            chains[node_label] = []
                        chains[node_label].append((label_chain, chain))
                    except KeyError as e:
                        print(f"KeyError: {e} not found in content_dict")
            else:
                try:
                    chain = f"{content_dict[in_node_label]} -> {content_dict[node_label]}"
                    label_chain = f"{in_node_label} -> {node_label}"
                    if node_label not in chains:
                        chains[node_label] = []
                    chains[node_label].append((label_chain, chain))
                except KeyError as e:
                    print(f"KeyError: {e} not found in content_dict")

        if not has_in_edge:
            for out_edge in node.get('out_edges', []):
                out_node_label = out_edge.get('node_label', '')
                try:
                    chain = f"{content_dict[node_label]} -> {content_dict[out_node_label]}"
                    label_chain = f"{node_label} -> {out_node_label}"
                    if node_label not in chains:
                        chains[node_label] = []
                    chains[node_label].append((label_chain, chain))
                except KeyError as e:
                    print(f"KeyError: {e} not found in content_dict")

        if not has_in_edge and not has_out_edge:
            chain = f"{content_dict[node_label]}"
            label_chain = f"{node_label}"
            if node_label not in chains:
                chains[node_label] = []
            chains[node_label].append((label_chain, chain))

    for node_label, node_chains in chains.items():
        queries.append(f"Chains for node {node_label}:")
        for label_chain, chain in node_chains:
            queries.append(f"{label_chain}: {chain}")

    return {'question': question, 'chains': chains, 'answer': answer}


def build_2_hop_query(data, filter=True):
    nodes = data.get('nodes', [])
    question = data.get('question', '')
    answer = data.get('answer', '')
    queries = [
        f"Answer the question based on the given content. \nQuestion is: **{question}**. "
        f"Following are the document contents."
    ]
    seen_content = set()
    content_counter = 1  # 计数器

    for node in nodes:
        node_content = node.get('content', '')

        # 第一层入节点的入节点内容
        for in_edge in node.get('in_edges', []):
            for in_in_edge in in_edge.get('in_edges', []):
                in_in_node_content = in_in_edge.get('content', '')
                if not filter or in_in_node_content not in seen_content:
                    queries.append(f"content [{content_counter}]: {in_in_node_content}")
                    seen_content.add(in_in_node_content)
                    content_counter += 1

        # 第一层入节点内容
        for in_edge in node.get('in_edges', []):
            in_node_content = in_edge.get('content', '')
            if not filter or in_node_content not in seen_content:
                queries.append(f"content [{content_counter}]: {in_node_content}")
                seen_content.add(in_node_content)
                content_counter += 1

        # 当前节点内容
        if not filter or node_content not in seen_content:
            queries.append(f"content [{content_counter}]: {node_content}")
            seen_content.add(node_content)
            content_counter += 1

        # 第一层出节点内容
        for out_edge in node.get('out_edges', []):
            out_node_content = out_edge.get('content', '')
            if not filter or out_node_content not in seen_content:
                queries.append(f"content [{content_counter}]: {out_node_content}")
                seen_content.add(out_node_content)
                content_counter += 1

            # 第一层出节点的出节点内容
            for out_out_edge in out_edge.get('out_edges', []):
                out_out_node_content = out_out_edge.get('content', '')
                if not filter or out_out_node_content not in seen_content:
                    queries.append(f"content [{content_counter}]: {out_out_node_content}")
                    seen_content.add(out_out_node_content)
                    content_counter += 1

    queries.append("Answer the question based on the given contents.\n")
    queries.append(f"Question is: **{question}**\n")
    queries.append(f"Answer in less than 6 words. Your output format is **Answer**: Answer")

    return {'query': queries, 'answer': answer}



# output_path = 'query_node_prompt_3.json'
# processed_data = []
# for data in data_list:
#     processed_data.append(build_node_prompt_as_query(data))
# with open(output_path, 'w', encoding='utf-8') as f:
#     json.dump(processed_data, f, ensure_ascii=False, indent=4)
with open("IPG_results/IPG_Hotpot_9.json", 'r', encoding='utf-8') as f:
    data_list = json.load(f)

output_path = 'IPG_results/IPG_Hotpot_9_for_reasoning.json'
processed_data = []
for data in data_list:
    processed_data.append(build_2_hop_query(data))
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(processed_data, f, ensure_ascii=False, indent=4)

print('Format done.')