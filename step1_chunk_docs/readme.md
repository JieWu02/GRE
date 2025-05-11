### text_spliter_util.py
文本分割工具类。它使用Spacy库将长文本按句子分割成更小的片段（chunks），使得每个片段大小适合模型处理，同时保持句子的完整性。

### chunk_multi_doc.py
这个脚本读取问答数据集（如HotpotQA），清理文本，然后使用上面的分割工具将长文档切分成小片段。它使用并行处理来加速，并将结果保存为新的JSON文件，保持原始数据的结构但文本被分成了更小的部分。
chunk 所有文档到 `chunked_datasets/Hotpot_test_chunked.json` 

### keep_top25_chunks.py
对齐查询（query），仅保留和查询语义相似度最高的25个chunk。
保留 `chunked_datasets/Hotpot_test_chunked.json`中的 top25 chunks 到 `step1_chunk_docs/top25_chunks_datasets` 目录下

## 使用方法

1. 首先执行 `chunk_multi_doc.py` 进行文档分块
2. 然后执行 `keep_top25_chunks.py` 筛选相关性最高的块


