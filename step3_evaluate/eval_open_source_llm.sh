python eval_open_source_llm.py --agent Llama3 \
                    --dataset none \
                    --structure none \
                    --topk none \
                    --type Finetuned \
                    --ckpt_dir /home/superbench/wujie/GRE/RAG_Graph/Graph_Reasoning/Model_and_Data/Hotpot/llama3_1-8b-instruct/v9-20240730-182046/checkpoint-680-merged
# merge lora:
# pip install ms-swift --upgrade (will coverage former environment)
# CUDA_VISIBLE_DEVICES=0 swift export --model /mnt/wujie/Code/Model_weights/Musique/Llama3_1_8B_instruct_Musique_3090_4/checkpoint-1420 --model_type llama --merge_lora true