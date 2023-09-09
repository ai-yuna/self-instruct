import json
import tqdm
import os
import random
import openai
from datetime import datetime
import argparse
import time

# https://platform.openai.com/docs/api-reference/completions/create  api参数列表
# make_gpt3_requests(
#                 engine=args.engine,
#                 prompts=batch_inputs,
#                 max_tokens=1024, #the max_tokens parameter of GPT3.
#                 temperature=0.7,#gpt参数
#                 top_p=0.5,#gpt参数
#                 frequency_penalty=0,#gpt参数
#                 presence_penalty=2,#gpt参数
#                 stop_sequences=["\n\n", "\n16", "16.", "16 ."],#gpt参数
#                 logprobs=1,#gpt参数
#                 n=1,# "The `n` parameter of GPT3. The number of responses to generate."
#                 best_of=1, #"The `best_of` parameter of GPT3. The beam size on the GPT3 server."
#                 api_key=args.api_key,
#                 organization=args.organization,
#             )
"""
engine或者model：text-davinci-003  或者gpt-3.5-turbo
max_tokens: 模型生成的文本的最大tokens
temperature: 控制输出随机性的值。较高的值（如 0.8）会使输出更随机，而较低的值（如 0.2）会使输出更确定和一致。
top_p: 用于控制 "nucleus sampling"。这是一个0到1之间的值，表示从最可能的下一个令牌的概率分布中取一个 "核"，并仅从这个核中采样。例如，top_p=0.9 会从最可能的令牌中选择一个子集，直到这些令牌的累积概率超过0.9。
frequency_penalty: 范围从-2到2。这个值会影响模型是否偏向于频繁或不频繁地使用某些令牌。
presence_penalty: 范围从-2到2。这个值会影响模型是否偏向于包含或排除某些令牌。
stop: 一个令牌列表，指示模型在哪里停止生成。例如，["\n", "<end>"] 会告诉模型在遇到换行或 "<end>" 时停止生成。
logprobs: 一个整数，表示要返回多少个令牌的 log 概率。例如，如果设置为10，API将返回最可能的10个令牌的log概率。
n: 模型应该生成多少个独立的补全。
best_of: 表示应该生成多少个补全，并只返回一个最好的补全（基于模型的内部评分）
"""

def make_requests(
        engine, prompts, max_tokens, temperature, top_p, 
        frequency_penalty, presence_penalty, stop_sequences, logprobs, n, best_of, retries=3, api_key=None, organization=None
    ):
    """
    批prompts请求gpt3，获取并解析response，并把数据编码为results 结果字典列表
    :param engine:
    :param prompts:
    :param max_tokens:
    :param temperature:
    :param top_p:
    :param frequency_penalty:
    :param presence_penalty:
    :param stop_sequences:
    :param logprobs:
    :param n:
    :param best_of:
    :param retries:
    :param api_key:
    :param organization:
    :return: [data]
    data: {
                "prompt": prompt,
                "response": {"choices": response["choices"][j * n: (j + 1) * n]} if response else None,
                "created_at": str(datetime.now()),
            }
    """
    response = None
    target_length = max_tokens
    if api_key is not None:
        openai.api_key = api_key
    if organization is not None:
        openai.organization = organization
    retry_cnt = 0
    backoff_time = 30
    while retry_cnt <= retries: # 如果失败，重试3次
        try:
            response = openai.Completion.create(
                engine=engine,
                prompt=prompts,
                max_tokens=target_length,
                temperature=temperature,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                stop=stop_sequences,
                logprobs=logprobs,
                n=n,
                best_of=best_of,
            )
            break
        except openai.error.OpenAIError as e:
            print(f"OpenAIError: {e}.")
            if "Please reduce your prompt" in str(e):
                target_length = int(target_length * 0.8)
                print(f"Reducing target length to {target_length}, retrying...")
            else:
                print(f"Retrying in {backoff_time} seconds...")
                time.sleep(backoff_time)
                backoff_time *= 1.5
            retry_cnt += 1
    
    if isinstance(prompts, list):
        results = []
        # 遍历请求的prompts，解析response，并编码results 结果字典列表
        for j, prompt in enumerate(prompts):
            data = {
                "prompt": prompt,
                "response": {"choices": response["choices"][j * n: (j + 1) * n]} if response else None,
                "created_at": str(datetime.now()),
            }
            results.append(data)
        return results
    else:
        data = {
            "prompt": prompts,
            "response": response,
            "created_at": str(datetime.now()),
        }
        return [data]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file",
        type=str,
        help="The input file that contains the prompts to GPT3.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        help="The output file to save the responses from GPT3.",
    )
    parser.add_argument(
        "--engine",
        type=str,
        help="The openai GPT3 engine to use.",
    )
    parser.add_argument(
        "--max_tokens",
        default=500,
        type=int,
        help="The max_tokens parameter of GPT3.",
    )
    parser.add_argument(
        "--temperature",
        default=0.7,
        type=float,
        help="The temprature of GPT3.",
    )
    parser.add_argument(
        "--top_p",
        default=0.5,
        type=float,
        help="The `top_p` parameter of GPT3.",
    )
    parser.add_argument(
        "--frequency_penalty",
        default=0,
        type=float,
        help="The `frequency_penalty` parameter of GPT3.",
    )
    parser.add_argument(
        "--presence_penalty",
        default=0,
        type=float,
        help="The `presence_penalty` parameter of GPT3.",
    )
    parser.add_argument(
        "--stop_sequences",
        default=["\n\n"],
        nargs="+",
        help="The `stop_sequences` parameter of GPT3.",
    )
    parser.add_argument(
        "--logprobs",
        default=5,
        type=int,
        help="The `logprobs` parameter of GPT3"
    )
    parser.add_argument(
        "--n",
        type=int,
        help="The `n` parameter of GPT3. The number of responses to generate."
    )
    parser.add_argument(
        "--best_of",
        type=int,
        help="The `best_of` parameter of GPT3. The beam size on the GPT3 server."
    )
    parser.add_argument(
        "--use_existing_responses",
        action="store_true",
        help="Whether to use existing responses from the output file if it exists."
    )
    parser.add_argument(
        "--request_batch_size",
        default=20,
        type=int,
        help="The number of requests to send to GPT3 at a time."
    )
    return parser.parse_args()

    
if __name__ == "__main__":
    random.seed(123)
    args = parse_args()
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    # read existing file if it exists
    existing_responses = {}
    if os.path.exists(args.output_file) and args.use_existing_responses:
        with open(args.output_file, "r") as fin:
            for line in fin:
                data = json.loads(line)
                existing_responses[data["prompt"]] = data

    # do new prompts
    with open(args.input_file, "r") as fin:
        if args.input_file.endswith(".jsonl"):
            all_prompts = [json.loads(line)["prompt"] for line in fin]
        else:
            all_prompt = [line.strip().replace("\\n", "\n") for line in fin]

    with open(args.output_file, "w") as fout:
        for i in tqdm.tqdm(range(0, len(all_prompts), args.request_batch_size)):
            batch_prompts = all_prompts[i: i + args.request_batch_size]
            if all(p in existing_responses for p in batch_prompts):
                for p in batch_prompts:
                    fout.write(json.dumps(existing_responses[p]) + "\n")
            else:
                results = make_requests(
                    engine=args.engine,
                    prompts=batch_prompts,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    frequency_penalty=args.frequency_penalty,
                    presence_penalty=args.presence_penalty,
                    stop_sequences=args.stop_sequences,
                    logprobs=args.logprobs,
                    n=args.n,
                    best_of=args.best_of,
                )
                for data in results: #results:结果字典列表
                    fout.write(json.dumps(data) + "\n")