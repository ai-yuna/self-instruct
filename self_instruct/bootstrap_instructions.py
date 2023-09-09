import os
import json
import random
import re
import string
import tqdm
import argparse
import numpy as np
import pandas as pd
from multiprocessing import Pool
from functools import partial
from rouge_score import rouge_scorer
from gpt3_api import make_requests as make_gpt3_requests

random.seed(42)

def encode_prompt(prompt_instructions, classification=False):
    """
    把多条示例指令拼接为gpt输入prompt
    :param prompt_instructions:
    :param classification:
    :return:
    """
    if classification:
        prompt = "Come up with a series of classification tasks. Try to specify the possible output labels when possible.\n"
        # prompt = "Referring to a series of classification tasks, generate 8 more new tasks. Try to specify the possible output labels when possible.\n"
    else:
        prompt = "Come up with a series of tasks:\n" #源码
        # prompt = "Referring to these eight tasks, generate 8 more new tasks:\n"
    for idx, instruction in enumerate(prompt_instructions):
        instruction = re.sub(r"\s+", " ", instruction).strip().rstrip(":") # \s+ 匹配一个或多个空白字符（可以是空格、制表符、换行符等）替换为单个空格，并删除结尾的':'
        prompt += f"{idx+1}. {instruction}\n"
    prompt += f"{len(prompt_instructions) + 1}."
    return prompt


def sample_machine_instructions(machine_instructions, similarities, n):
    """
    从指令池中随机选择机器指令，默认n=2，还没有机器指令就是0
    :param machine_instructions:
    :param similarities:
    :param n:
    :return:
    """
    return random.sample(machine_instructions, min(n, len(machine_instructions)))


def find_word_in_string(w, s):
    """
    在指令中查找单词，给定单词都是不适合语言模型的指令关键词（图片、画图等）
    :param w:
    :param s:
    :return:
    """
    return re.compile(r'\b({0})\b'.format(w), flags=re.IGNORECASE).search(s)
    # \b 是一个词边界符号，确保我们匹配的是完整的单词，而不是部分单词。
    # {0} 是一个占位符，我们用 w 替换它。
    # flags=re.IGNORECASE 或者是re.I 确保匹配不区分大小写


def post_process_gpt3_response(response):
    """
    对 GPT-3 的响应（单条）进行后处理，提取指令，对指令进行清洗过滤，返回清洁的指令list
    过滤：
    1. 长度：3-150
    2. 不适合语言模型的关键词的指令
    3. 过滤掉以 "Write a program" 开头的指令
    4. 过滤掉以标点符号开头的指令
    5. 过滤掉以非英文字符开头的指令
    :param response:
    :return:
    """
    # 如果响应为空或由于达到最大长度而结束，则返回空列表
    if response is None or response["choices"][0]["finish_reason"] == "length":
        return []
    # 使用正则表达式分割模型输出，基于指令编号（如"1. ", "2. "等）进行分割
    raw_instructions = re.split(r"\n\d+\s?\. ", response["choices"][0]["text"])
    instructions = []
    for inst in raw_instructions:
        # 清理指令：删除多余的空格、修剪、首字母大写
        inst = re.sub(r"\s+", " ", inst).strip()
        inst = inst.strip().capitalize()
        if inst == "":
            continue

        # 过滤掉过短或过长的指令
        if len(inst.split()) <= 3 or len(inst.split()) > 150:
            continue
        # 过滤掉包含不适合语言模型的关键词的指令
        # 如果有任何元素为 True，则返回 True；否则，返回 False。
        if any(find_word_in_string(word, inst) for word in ["image", "images", "graph", "graphs", "picture", "pictures", "file", "files", "map", "maps", "draw", "plot", "go to"]):
            continue
        # 过滤掉以 "Write a program" 开头的指令，因为这导致了许多此类指令
        if inst.startswith("Write a program"):
            continue
        # 过滤掉以标点符号开头的指令
        if inst[0] in string.punctuation:
            continue
        # 过滤掉以非英文字符开头的指令
        if not inst[0].isascii():
            continue
        instructions.append(inst)
    return instructions


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_dir",
        type=str,
        required=True,
        default="data/gpt3_generations/",
        help="The directory where the batch is stored.",
    )
    parser.add_argument(
        "--seed_tasks_path",
        type=str,
        required=True,
        default="data/seed_tasks.jsonl",
        help="The path to the human written data.",
    )
    parser.add_argument(
        "--num_instructions_to_generate",
        type=int,
        default=100,
        help="th",
    )
    parser.add_argument(
        "--use_clf_seed_tasks_only",
        action="store_true",
        help="If specified, we will only use the classification seed tasks to prompt new instructions. This will lead to more classification instructions.",
    ) #对于 action="store_true", 它的意思是当命令行中包含了这个参数（在这里是 --use_clf_seed_tasks_only）时，将该参数的值设置为 True。如果命令行中没有包含这个参数，那么它的值默认为 False。
    parser.add_argument(
        "--engine",
        type=str,
        default="davinci",
        help="The engine to use."
    )
    parser.add_argument(
        "--num_prompt_instructions",
        type=int,
        default=8,
        help="The number of instructions to use in the prompt."
    )
    parser.add_argument(
        "--request_batch_size",
        type=int,
        default=5,
        help="The number of requests to send to GPT3 at a time."
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


if __name__ == "__main__":
    '''
    解析命令行参数
    加载种子指令 list
    创建或检查输出目录
    加载已经生成的机器指令list
    初始化 ROUGE 评分器rougeL
    进入主循环while，直到生成所需数量的指令：
        1. 为 GPT-3 准备批量输入：随机抽取种子指令和机器指令（6+2）拼接为prompt，默认生成5个输入prompt
        2. 输入batch_inputs，请求GPT-3返回对应的results
        3. gpt3生成结果后处理，提取所有的清洁机器指令list
        4. 遍历每个新生成的指令，评估新指令与现有指令的相似度，如果相似度低于阈值，则保存新指令到输出文件
    '''
    args = parse_args()
    ## s1:加载种子指令list
    seed_tasks = [json.loads(l) for l in open(args.seed_tasks_path, "r")] #种子任务
    if args.use_clf_seed_tasks_only: #只有分类任务
        seed_tasks = [t for t in seed_tasks if t["is_classification"]]
    seed_instructions = [t["instruction"] for t in seed_tasks] #种子指令
    print(f"Loaded {len(seed_instructions)} human-written seed instructions")
    
    os.makedirs(args.batch_dir, exist_ok=True)
    #os.makedirs: 这是一个函数，用于递归地创建目录。也就是说，如果你提供了一个路径，如 "a/b/c"，即使 "a" 和 "a/b" 都不存在，它也会为你创建整个路径
    #exist_ok=True: 这个参数的意思是，如果目标目录已经存在，函数不会引发错误
    request_idx = 0

    ## s2:加载已经生成的机器指令
    machine_instructions = []
    if os.path.exists(os.path.join(args.batch_dir, "machine_generated_instructions.jsonl")):
        with open(os.path.join(args.batch_dir, "machine_generated_instructions.jsonl"), "r") as fin:
            for line in fin:
                instruction_info = json.loads(line)
                machine_instructions.append(instruction_info["instruction"])
                request_idx = instruction_info["request_idx"] + 1
        print(f"Loaded {len(machine_instructions)} machine-generated instructions")

    # similarities = {}
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    '''
    创建了一个 RougeScorer 对象，它将用于计算 ROUGE-L 分数。如前所述，ROUGE-L 考虑了句子中的最长公共子序列
    ["rougeL"]: 这是一个列表，指定要计算的 ROUGE 变体。在这种情况下，我们只计算 ROUGE-L。ROUGE-L 考虑了句子中最长的公共子序列。
    use_stemmer=False: 这个参数指定是否在计算 ROUGE 分数时使用词干提取。词干提取是将单词转化为其基本形式或词根的过程。在这种情况下，我们选择不使用词干提取
    
    scores = scorer.score(reference_text, generated_text)
    这将返回一个包含 ROUGE-L 分数的字典，包括精确度、召回率和 F1 分数
    '''
    
    # now let's generate new instructions!
    progress_bar = tqdm.tqdm(total=args.num_instructions_to_generate)
    if machine_instructions:
        progress_bar.update(len(machine_instructions))


    ## s3:进入主循环，直到生成所需数量的指令：
    with open(os.path.join(args.batch_dir, "machine_generated_instructions.jsonl"), "a") as fout:
        while len(machine_instructions) < args.num_instructions_to_generate:
            ## s3.1：为 GPT-3 准备批量输入：随机抽取种子指令和机器指令（6+2），默认编码生成5个输入prompt
            batch_inputs = []
            for _ in range(args.request_batch_size):
                # sample machine instructions from the pool
                prompt_instructions = sample_machine_instructions(
                    machine_instructions, 
                    similarities=None,
                    n=2) # 随机抽取2个机器生成指令
                # sample human instructions from the pool
                prompt_instructions += random.sample(seed_instructions, args.num_prompt_instructions - len(prompt_instructions)) #一般随机抽取6个人工指令
                random.shuffle(prompt_instructions)
                prompt = encode_prompt(prompt_instructions, classification=args.use_clf_seed_tasks_only)
                batch_inputs.append(prompt)

            ## s3.2:输入batch_inputs，请求GPT-3返回对应的results
            results = make_gpt3_requests(
                engine=args.engine,
                prompts=batch_inputs,
                max_tokens=1024,
                temperature=0.7,
                top_p=0.5,
                frequency_penalty=0,
                presence_penalty=2,
                stop_sequences=["\n\n", "\n16", "16.", "16 ."],
                logprobs=1,
                n=1,
                best_of=1,
                api_key=args.api_key,
                organization=args.organization,
            )
            '''
            返回results是字典列表，其中每个元素的格式是：
            data = {
                "prompt": prompt,
                "response": {"choices": response["choices"][j * n: (j + 1) * n]} if response else None,
                "created_at": str(datetime.now()),
            }
            '''
            ## s3.3:gpt3生成结果后处理，提取清洁机器指令
            instructions = []
            all_metadata = []
            for result in results:
                '''
                post_process_gpt3_response:
                对 GPT-3 的响应（单条）进行后处理，提取指令，对指令进行清洗过滤，返回清洁的指令list
                过滤：
                1. 长度：3-150
                2. 不适合语言模型的关键词的指令
                3. 过滤掉以 "Write a program" 开头的指令
                4. 过滤掉以标点符号开头的指令
                5. 过滤掉以非英文字符开头的指令
                '''
                new_instructions = post_process_gpt3_response(result["response"])
                instructions += new_instructions
                # 对于每个新指令，将其关联的元数据添加到元数据列表中
                all_metadata += [result] * len(new_instructions)

            ## s3.4:遍历每个新生成的指令，评估新指令与现有all指令的相似度，如果相似度低于阈值，则保存新指令到输出文件
            for inst, metadata in zip(instructions, all_metadata):
                # 使用4个进程并行计算ROUGE分数
                with Pool(4) as p:
                    rouge_scores = p.map(partial(scorer.score, inst), seed_instructions + machine_instructions)
                rouge_scores = [score["rougeL"].fmeasure for score in rouge_scores]
                '''
                p 是一个 Pool 对象，它允许我们并行执行函数
                partial(scorer.score, inst) 是一个部分应用的函数。它将 scorer.score 函数的第一个参数固定为 inst，这样我们只需要提供第二个参数。
                seed_instructions + machine_instructions 是我们要计算 ROUGE 分数的参考文本列表。
                rouge_scores 是一个列表，其中每个元素都是与 inst（新生成的指令）的 ROUGE-L 分数的字典。
                '''
                # rouge_scores = [scorer.score(inst, e_inst)["rougeL"].fmeasure for e_inst in human_instructions + machine_instructions]
                if max(rouge_scores) > 0.7:#ROUGE 分数的值范围从 0 到 1，其中 0 表示完全不匹配，1 表示完全匹配。
                    continue
                all_instructions = seed_instructions + machine_instructions
                # 获取与新指令最相似的10个已有指令及其ROUGE分数
                most_similar_instructions = {
                        all_instructions[i] : rouge_scores[i] for i in np.argsort(rouge_scores)[-10:][::-1]
                    }
                '''
                这行代码的目的是从所有的 rouge_scores 中找出与 inst 最相似的前10个指令，然后创建一个字典，其中键是这10个指令，值是它们对应的 ROUGE 分数。
                以下是对此代码的详细解释：
                np.argsort(rouge_scores)：这个函数将返回一个整数列表，表示如果将 rouge_scores 进行升序排序，每个元素的索引位置。例如，对于列表 [0.2, 0.7, 0.1]，结果将是 [2, 0, 1]。
                [-10:]：这将选取上述列表的最后10个元素。这意味着我们正在获取10个最高的 ROUGE 分数的索引。
                [::-1]：这将反转列表，确保我们现在有的是10个最高 ROUGE 分数的索引，按降序排列。
                {all_instructions[i] : rouge_scores[i] for i in ...}：这是一个字典推导式，它为每个选定的索引 i 创建一个键值对。键是 all_instructions 中的指令，值是其对应的 ROUGE 分数。
                '''
                # 将新指令添加到机器生成的指令列表中
                machine_instructions.append(inst)
                fout.write(json.dumps({
                    "instruction": inst,
                    "most_similar": most_similar_instructions,
                    "avg_similarity_score": float(np.mean(rouge_scores)),
                    "metadata": metadata,# gpt返回结果的元数据
                    "request_idx": request_idx
                }) + "\n")
                progress_bar.update(1)
            request_idx += 1
