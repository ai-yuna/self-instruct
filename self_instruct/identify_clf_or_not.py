import os
import json
import random
import tqdm
import re
import argparse
import pandas as pd
from collections import OrderedDict
from gpt3_api import make_requests as make_gpt3_requests
from templates.clf_task_template import template_1


random.seed(42)


templates = {
    "template_1": template_1
}

def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--template", type=str, default="template_1", help="Which template to use.")
    parser.add_argument(
        "--batch_dir",
        type=str,
        required=True,
        help="The directory where the batch is stored.",
    )
    parser.add_argument(
        "--num_instructions",
        type=int,
        help="if specified, only generate instance input for this many instructions",
    )
    parser.add_argument(
        "--template", 
        type=str, 
        default="template_1", 
        help="Which template to use. Currently only `template_1` is supported.",
    )
    parser.add_argument(
        "--engine",
        type=str,
        default="davinci",
        help="The engine to use."
    )
    parser.add_argument(
        "--request_batch_size",
        type=int,
        default=5,
        help="The number of requests to send in a batch."
    )
    parser.add_argument(
        "--api_key",
        type=str,
        help="The API key to use. If not specified, the key will be read from the environment variable `OPENAI_API_KEY`."
    )
    parser.add_argument(
        "--organization",
        type=str,
        help="The organization to use. If not specified, the default organization id will be used."
    )
    return parser.parse_args()


if __name__ == '__main__':
    """
    1. 读取需要判断分类的机器生成指令
    2. 初始化参数和变量
    3. 遍历指令通过gpt3生成是否分类的结果，并写入文件，对指令进行批处理，默认是5
        3.1 检查批次中的每个指令是否都已经存在于请求中，是的话：直接再次写入文件
        3.2 没有的话，就使用模板创建GPT-3的提示list，使用GPT-3 API请求结果，提取返回结果中的"is_classification"结果，写入文件 
    """
    args = parse_args()

    # 1. 读取需要判断分类的机器生成指令
    '''
    machine_generated_instructions.jsonl 存储的数据格式
    {
    "instruction": inst,
    "most_similar": most_similar_instructions,
    "avg_similarity_score": float(np.mean(rouge_scores)),
    "metadata": metadata,# gpt返回结果的元数据
    "request_idx": request_idx
    }
    '''
    with open(os.path.join(args.batch_dir, "machine_generated_instructions.jsonl")) as fin:
        lines = fin.readlines()
        if args.num_instructions is not None: #指定处理的指令数量
            lines = lines[:args.num_instructions]

    # 2. 初始化参数和变量
    output_path = os.path.join(args.batch_dir, f"is_clf_or_not_{args.engine}_{args.template}.jsonl")
    existing_requests = {}
    if os.path.exists(output_path):
        with open(output_path) as fin:
            for line in tqdm.tqdm(fin):
                try:
                    data = json.loads(line) # 本脚本的生成结果
                    existing_requests[data["instruction"]] = data
                except:
                    pass
        print(f"Loaded {len(existing_requests)} existing requests")

    progress_bar = tqdm.tqdm(total=len(lines))

    with open(output_path, "w") as fout:
        # 3. 遍历每个指令gpt3生成是否分类的结果，并写入文件
        for batch_idx in range(0, len(lines), args.request_batch_size):
            # 3.1 对指令进行批处理
            batch = [json.loads(line) for line in lines[batch_idx: batch_idx + args.request_batch_size]] #需要处理的指令
            # 3.2 检查批次中的每个指令是否都已经存在于请求中，是的话：直接再次写入文件
            if all(d["instruction"] in existing_requests for d in batch):#用于检查给定的可迭代对象中的所有元素是否都为 True。如果是，则返回 True，否则返回 False。
                # 如果存在，直接写入输出文件
                for d in batch:
                    data = existing_requests[d["instruction"]]
                    data = OrderedDict(
                        (k, data[k]) for k in \
                            ["instruction", "is_classification"]
                        )
                    '''
                    OrderedDict与常规的字典非常相似，但它保持了键的插入顺序
                    这段代码使用了列表推导来构建这个有序字典。
                    (k, data[k])：这是一个元组，其中 k 是键，data[k] 是对应的值。
                    for k in ["instruction", "is_classification"]：这表示迭代列表中的每个键，并从 data 中获取相应的值。
                    '''
                    fout.write(json.dumps(data, ensure_ascii=False) + "\n")
            else:
                # prefix = compose_prompt_prefix(human_written_tasks, batch[0]["instruction"], 8, 2)
                # 3.2 使用模板创建GPT-3的提示list，使用GPT-3 API请求结果，提取返回结果中的"is_classification"结果
                prefix = templates[args.template]
                prompts = [prefix + " " + d["instruction"].strip() + "\n" + "Is it classification?" for d in batch]
                # 使用GPT-3 API请求结果
                results = make_gpt3_requests(
                    engine=args.engine,
                    prompts=prompts,
                    max_tokens=3,
                    temperature=0,
                    top_p=0,
                    frequency_penalty=0,
                    presence_penalty=0,
                    stop_sequences=["\n", "Task"],
                    logprobs=1,
                    n=1,
                    best_of=1,
                    api_key=args.api_key,
                    organization=args.organization)

                # 处理API的返回结果
                for i in range(len(batch)):
                    data = batch[i]
                    if results[i]["response"] is not None:
                        data["is_classification"] = results[i]["response"]["choices"][0]["text"]
                    else:
                        data["is_classification"] = ""
                    data = {
                        "instruction": data["instruction"],
                        "is_classification": data["is_classification"]
                    }
                    data = OrderedDict(
                        (k, data[k]) for k in \
                            ["instruction", "is_classification"]
                        )
                    fout.write(json.dumps(data, ensure_ascii=False) + "\n")
            progress_bar.update(len(batch))
