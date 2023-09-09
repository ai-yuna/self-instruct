import os
import json
import random
import tqdm
import re
import argparse
import pandas as pd
from collections import OrderedDict
from gpt3_api import make_requests as make_gpt3_requests
from templates.instance_gen_template import output_first_template_for_clf, input_first_template_for_gen


random.seed(42)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_dir",
        type=str,
        required=True,
        help="The directory where the batch is stored.",
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default="machine_generated_instructions.jsonl"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="machine_generated_instances.jsonl",
    )
    parser.add_argument(
        "--num_instructions",
        type=int,
        help="if specified, only generate instance input for this many instructions",
    )
    parser.add_argument(
        "--max_instances_to_generate",
        type=int,
        default=5,
        help="The max number of instances to generate for each instruction.",
    )
    parser.add_argument(
        "--generation_tasks_only",
        action="store_true",
        help="If specified, only do for generation tasks.",
    )
    parser.add_argument(
        "--classification_tasks_only",
        action="store_true",
        help="If specified, only do for classification tasks.",
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
        help="The API key to use. If not specified, the key will be read from the environment variable OPENAI_API_KEY."
    )
    parser.add_argument(
        "--organization",
        type=str,
        help="The organization to use. If not specified, the default organization id will be used."
    )
    return parser.parse_args()


if __name__ == '__main__':
    """
    1. 获取生成instance的机器指令tasks list
    2. 获取每条指令对应的clf结果task_clf_types:{}
    3. 遍历处理所有的需要生成instance的指令任务，批处理默认5
        3.1 如果已经在输出文件中存在，则再次直接写入
        3.2 否则构造批prompts(遍历每个指令任务，判断是否是分类任务，分别构造输入prompt)，请求gpt3，返回结果
        3.3 处理批次返回结果，获取gpt3生成的raw_instances,构造输出数据并写入file
    """
    args = parse_args()

    with open(os.path.join(args.batch_dir, args.input_file)) as fin: #生成的机器指令
        lines = fin.readlines()
        if args.num_instructions is not None:
            lines = lines[:args.num_instructions]
        # 1. 需要生成instance的机器指令
        tasks = []
        '''
        data: {
        "instruction": inst,
        "most_similar": most_similar_instructions,
        "avg_similarity_score": float(np.mean(rouge_scores)),
        "metadata": metadata,# gpt返回结果的元数据单条
        "request_idx": request_idx
        }
        '''
        for line in lines:
            data = json.loads(line)
            if "metadata" in data:
                data["instruction_metadata"] = data["metadata"]
                del data["metadata"]
            tasks.append(data)

    task_clf_types = {} # 2. 每条指令对应的分类结果
    with open(os.path.join(args.batch_dir, "is_clf_or_not_davinci_template_1.jsonl")) as fin:
        for line in fin:
            '''
            data:{
                        "instruction": data["instruction"],
                        "is_classification": data["is_classification"]
                    }
            '''
            data = json.loads(line)
            task_clf_types[data["instruction"]] = data["is_classification"].strip() in ["Yes", "yes", "YES"]

    if args.classification_tasks_only:
        tasks = [task for task in tasks if task_clf_types[task["instruction"]]]
    
    if args.generation_tasks_only:
        tasks = [task for task in tasks if not task_clf_types[task["instruction"]]]

    # 定义输出路径
    output_path = os.path.join(args.batch_dir, args.output_file)
    existing_requests = {}
    if os.path.exists(output_path):
        with open(output_path) as fin:
            for line in tqdm.tqdm(fin):
                try:
                    data = json.loads(line)
                    existing_requests[data["instruction"]] = data
                except:
                    pass
        print(f"Loaded {len(existing_requests)} existing requests")

    progress_bar = tqdm.tqdm(total=len(tasks))
    # 3. 遍历处理所有的需要生成instance的指令任务，批处理默认5
    with open(output_path, "w") as fout:
        for batch_idx in range(0, len(tasks), args.request_batch_size):
            batch = tasks[batch_idx: batch_idx + args.request_batch_size]
            # 3.1 如果已经在输出文件中存在，则再次直接写入
            if all(d["instruction"] in existing_requests for d in batch):
                for d in batch:
                    data = existing_requests[d["instruction"]]
                    data = OrderedDict(
                        (k, data[k]) for k in \
                            ["instruction", "raw_instances", "instance_metadata", "instruction_metadata", 
                            "most_similar", "avg_similarity_score"]
                        )
                    fout.write(json.dumps(data, ensure_ascii=False) + "\n")
            else:
                # 3.2 否则构造批prompts(遍历每个指令任务，判断是否是分类任务，分别构造输入prompt)，请求gpt3，返回结果
                prompts = []
                # 遍历每个指令任务，判断是否是分类任务，分别构造输入prompt
                for task in batch:
                    if task_clf_types[task["instruction"]]: #分类任务
                        '''
                        output_first_template_for_clf:Given the classification task definition and the class labels, generate an input that corresponds to each of the class labels. If the task doesn't require input, just generate possible class labels.
                        '''
                        prompt = output_first_template_for_clf + " " + task["instruction"].strip() + "\n"
                        prompts.append(prompt)
                    else:# 生成任务
                        '''
                        input_first_template_for_gen:Come up with examples for the following tasks. Try to generate multiple examples when possible. If the task doesn't require additional input, you can generate the output directly
                        '''
                        prompt = input_first_template_for_gen + " " + task["instruction"].strip() + "\n"
                        prompts.append(prompt)
                # 请求gpt3
                results = make_gpt3_requests(
                    engine=args.engine,
                    prompts=prompts,
                    # because the clf template is longer, we need to decrease the max_tokens
                    max_tokens=300 if any(task_clf_types[task["instruction"]] for task in batch) else 350,
                    temperature=0,
                    top_p=0,
                    frequency_penalty=0,
                    presence_penalty=1.5,
                    stop_sequences=[f"Example {args.max_instances_to_generate + 1}", "Task:"],
                    logprobs=1,
                    n=1,
                    best_of=1,
                    api_key=args.api_key,
                    organization=args.organization)
                # 3.3 处理批返回结果，获取gpt3生成的raw_instances,构造输出数据并写入file
                for i in range(len(batch)):
                    '''
                    data: {
                        "instruction": inst,
                        "raw_instances":gpt返回instance结果的生成text部分,
                        "instance_metadata":gpt返回instance结果的元数据单条,
                        "instruction_metadata": metadata,# gpt返回instruction结果的元数据单条,
                        "most_similar": most_similar_instructions,
                        "avg_similarity_score": float(np.mean(rouge_scores))
                        }
                    '''
                    data = batch[i]
                    data["instance_metadata"] = results[i]
                    if results[i]["response"] is not None:
                        data["raw_instances"] = results[i]["response"]["choices"][0]["text"]
                    else:
                        data["raw_instances"] = ""
                    data = OrderedDict(
                        (k, data[k]) for k in \
                            ["instruction", "raw_instances", "instance_metadata", "instruction_metadata", 
                            "most_similar", "avg_similarity_score"]
                        )
                    fout.write(json.dumps(data, ensure_ascii=False) + "\n")
            progress_bar.update(len(batch))
