import os
import json
import argparse
import glob
import re
import random
import tqdm
import pandas as pd


random.seed(123)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--instance_files",
        nargs="+",
        default=["data/batch_221203/machine_generated_instances.jsonl"],
        type=str,
        help="The input files that contain the machine generated instances."
    )
    parser.add_argument(
        "--classification_type_files",
        nargs="+",
        default=["data/batch_221203/is_clf_or_not_davinci_template_1.jsonl"],
    )
    parser.add_argument(
        "--output_dir",
        default="data/gpt3_generations/batch_221203/finetuning/",
        type=str,
        help="The output dir to save the cleaned version of the generated instances, so that it can be used for GPT3 finetuning."
    )
    parser.add_argument(
        "--num_instructions",
        type=int,
        help="The number of instructions to load."
    )
    parser.add_argument(
        "--include_seed_tasks",
        action="store_true",
        help="Whether to include the seed human-written instances in the finetuning data."
    )
    parser.add_argument(
        "--seed_tasks_path",
        type=str,
        required=True,
        default="data/seed_tasks.jsonl",
        help="The path to the seed data.",
    )
    return parser.parse_args()


def encode_instance(instruction, input, output, random_template=True):
    """
    选择模版进行编码prompt，返回封装好的data字典
    """
    # 定义带有输入的编码模板列表
    encoding_templates_w_input = [
        ("{instruction}\nInput: {input}\nOutput:", " {output}<|endoftext|>"),
        ("{instruction}\n\nInput: {input}\n\nOutput:", " {output}<|endoftext|>"),
        ("Task: {instruction}\nInput: {input}\nOutput:", " {output}<|endoftext|>"),
        ("{instruction}\n\n{input}\n\nOutput:", " {output}<|endoftext|>"),
        ("{instruction}\n\n{input}\n\n", "{output}<|endoftext|>"),
        ("{instruction}\n{input}\n\n", "{output}<|endoftext|>"),
        ("Task: {instruction}\n\n{input}\n\n", "{output}<|endoftext|>"),
    ]
    # 定义不带输入的编码模板列表
    encoding_templates_wo_input = [
        ("{instruction} Output:", " {output}<|endoftext|>"),
        ("{instruction}\nOutput:", " {output}<|endoftext|>"),
        ("{instruction}\n\nOutput:", " {output}<|endoftext|>"),
        ("{instruction}\n", "{output}<|endoftext|>"),
        ("{instruction}\n\n", "{output}<|endoftext|>"),
        ("Task: {instruction}\n\n", "{output}<|endoftext|>"),
    ]
    # 根据是否随机选择模板来确定编码方式
    if random_template:
        # 如果有输入
        if input.strip() != "":
            prompt_template, completion_template = random.choice(encoding_templates_w_input) #choice 它从给定的非空序列（如列表或元组）中随机选择并返回一个元素
            prompt = prompt_template.format(instruction=instruction.strip(), input=input.strip())
            completion = completion_template.format(output=output.strip())
        else:
            prompt_template, completion_template = random.choice(encoding_templates_wo_input)
            prompt = prompt_template.format(instruction=instruction.strip())
            completion = completion_template.format(output=output.strip())
    # 如果不随机选择模板，则使用默认的编码方式
    else:
        prompt = instruction.strip() + "\n\n" + input.strip() + "\n\n"
        completion = output.strip() + "<|endoftext|>"

    data = {
        "prompt": prompt,
        "completion": completion,
        "instruction": instruction.strip(),
        "input": input.strip(),
        "output": output.strip(),
    }
    return data


def parse_input_output(response_text):
    """
    从raw_instances中解析输入输出 ，返回inst_input, inst_output
    :param response_text:
    :return:
    """
    # 如果响应文本中含有"Output"和数字
    if re.findall(r"Output\s*\d*\s*:", response_text):
        # 根据"Output"和数字分割响应文本，获取输入和输出
        inst_input = re.split(r"Output\s*\d*\s*:", response_text)[0].strip()
        inst_output = re.split(r"Output\s*\d*\s*:", response_text)[1].strip()
    # 否则，输入为空，输出为整个响应文本
    else:
        inst_input = ""
        inst_output = response_text.strip()
    # 如果输出中还含有"Input"和数字，我们只保留"Input"之前的部分作为输出
    if re.findall(r"Input\s*\d*\s*:", inst_output):
        inst_output = re.split(r"Input\s*\d*\s*:", inst_output)[0].strip()
    # 从字符串中移除"Input:"前缀
    inst_input = re.sub(r"^Input\s*\d*\s*:", "", inst_input).strip()
    return inst_input, inst_output


def filter_duplicate_instances(instances):
    '''
    过滤重复的实例
    :param instances:
    :return:
    '''
    # 如果实例具有相同的非空输入，但输出不同，我们将不使用这样的实例
    same_input_diff_output = False
    for i in range(1, len(instances)):
        # 遍历当前实例之前的所有实例
        for j in range(0, i):
            if instances[i][1] == "":
                continue
            # 检查当前实例和之前的实例是否具有相同的输入但不同的输出
            if instances[i][1] == instances[j][1] and instances[i][2] != instances[j][2]:
                same_input_diff_output = True
                break
    if same_input_diff_output:
        return []

    # remove duplicate instances
    instances = list(set(instances))
    return instances

def filter_invalid_instances(instances):
    '''
    过滤无效的实例
    instances：元组列表，(指令,输入,输出)
    :param instances:
    :return:
    '''
    filtered_instances = []
    # 遍历每个实例 (instruction.strip(), input_text.strip(), class_label.strip())
    for instance in instances:
        # 如果输入和输出相同，我们不使用这样的实例
        if instance[1] == instance[2]:
            continue
        # 如果输出为空，我们不使用这样的实例
        if instance[2] == "":
            continue
        # if input or output ends with a colon, these are usually imcomplete generation. We will not use such instances
        # 如果输入或输出以冒号结尾，这通常是不完整的生成。我们不使用这样的实例
        if instance[1].strip().endswith(":") or instance[2].strip().endswith(":"):
            continue
        filtered_instances.append(instance)
    return filtered_instances

def parse_instances_for_generation_task(raw_text, instruction, response_metadata):
    instances = []
    raw_text = raw_text.strip()
    # 如果原始文本中含有"Example"和数字，我们假设它是多个输入/输出对
    if re.findall("Example\s?\d*\.?", raw_text):
        #按照Example 1.等进行切分
        instance_texts = re.split(r"Example\s?\d*\.?", raw_text)
        instance_texts = [it.strip() for it in instance_texts if it.strip() != ""]
        for instance_text in instance_texts:
            # 解析返回每个输入/输出对
            inst_input, inst_output = parse_input_output(instance_text)
            # 将指令、输入和输出组成的元组添加到实例列表中
            instances.append((instruction.strip(), inst_input.strip(), inst_output.strip()))

    # 如果原始文本中只含有"Output"和数字，我们假设只有一个输入/输出对
    elif re.findall(r"Output\s*\d*\s*:", raw_text):
        # we assume only one input/output pair in this case
        inst_input, inst_output = parse_input_output(raw_text)
        instances.append((instruction.strip(), inst_input.strip(), inst_output.strip()))
    else:
        return []

    # if the generation stops because of length, we remove the last instance
    # 如果生成因长度而停止，我们移除最后一个实例
    if response_metadata["response"]["choices"][0]["finish_reason"] == "length":
        instances = instances[:-1]
    # 过滤无效的实例
    instances = filter_invalid_instances(instances)
    # 过滤重复的实例
    instances = filter_duplicate_instances(instances)
    return instances

def parse_instances_for_classification_task(raw_text, instruction, response_metadata):
    instances = []
    # 如果原始文本中不包含"Class label:"，则返回空列表
    if not "Class label:" in raw_text:
        return []
    # 根据"Class label:"分割原始文本，获取各个实例文本
    instance_texts = raw_text.split("Class label:")[1:]
    # 遍历每个实例文本
    for instance_text in instance_texts:
        instance_text = instance_text.strip()
        # 以换行符分割实例文本，获取类标签和输入文本
        fields = instance_text.split("\n", 1)
        # 如果切分出来的长度是2，第一个是类标签，其余部分是输入文本
        if len(fields) == 2:
            # the first field split by \n is the class label
            class_label = fields[0].strip()
            # the rest is the input
            input_text = fields[1].strip()
        # 如果只有一个字段，则其是类标签，输入文本为空
        elif len(fields) == 1:
            # the first field split by \n is the input
            class_label = fields[0].strip()
            input_text = ""
        else:
            # 抛出异常
            raise ValueError("Invalid instance text: {}".format(instance_text))

        # 将指令、输入文本和类标签组成的元组添加到实例列表中
        instances.append((instruction.strip(), input_text.strip(), class_label.strip()))

    # 如果生成因长度而停止，我们移除最后一个实例
    if response_metadata["response"]["choices"][0]["finish_reason"] == "length":
        instances = instances[:-1]
    # 过滤无效的实例
    instances = filter_invalid_instances(instances)
    # 过滤重复的实例
    instances = filter_duplicate_instances(instances)
    return instances


if __name__ == "__main__":
    """
    1. 加载生成的instance数据
    2. 加载每条指令对应的分类结果
    3. 遍历处理每个任务指令的raw_instances,解析并清洗后的3元组(指令,输入,输出)append进training_instances
    4. 将所有生成的实例三元组保存到文件中
    5. 如果设置了指令数量参数，则进行随机抽样
    6. 如果需要包含种子任务，则加载并添加到训练实例中
    7. 获取GPT-3的训练实例training_instances，遍历training_instances，随机选择模版编码为prompt，completion，并返回封装好的data字典
    8. 移除重复的实例，shuff，保存file
    """
    args = parse_args()

    training_instances = []
    
    generated_tasks = []
    # 1. 加载生成的instance数据
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
    for instance_file in args.instance_files:
        with open(instance_file) as fin:
            for line in fin:
                generated_tasks.append(json.loads(line))
    print(f"Loaded {len(generated_tasks)} raw generated tasks")

    # 2. 加载每条指令对应的分类结果
    task_clf_types = {}
    for file in args.classification_type_files:
        with open(file) as fin:
            for line in fin:
                data = json.loads(line)
                task_clf_types[data["instruction"]] = data["is_classification"].strip() in ["Yes", "yes", "YES"]

    # 3. 遍历处理每个任务指令的raw_instances,解析并清洗后的3元组(指令,输入,输出)append进training_instances
    for task in tqdm.tqdm(generated_tasks):
        # get instruction
        instruction = task["instruction"]
        task["is_classification"] = task_clf_types[instruction]

        # 根据是否分类分别解析实例
        if task["is_classification"]:
            # 分类任务instances解析：从raw_instances解析为3元组列表[(指令,输入,输出)]，然后过滤无效的实例，过滤重复实例，返回清洗后的实例元组列表
            task_instances = parse_instances_for_classification_task(task["raw_instances"], instruction, task["instance_metadata"])
        else:
            # 生成任务instances解析：从raw_instances解析为3元组列表[(指令,输入,输出)]，然后过滤无效的实例，过滤重复实例
            task_instances = parse_instances_for_generation_task(task["raw_instances"], instruction, task["instance_metadata"])

        # we only allow max 5 instances per task
        task_instances = random.sample(task_instances, min(len(task_instances), 5))
        
        if not task_instances:
            continue

        training_instances += task_instances

    # 创建输出目录（如果不存在）
    os.makedirs(args.output_dir, exist_ok=True)
    # 4. 将所有生成的实例三元组保存到文件中
    with open(os.path.join(args.output_dir, "all_generated_instances.jsonl"), "w") as fout:
        for instance in training_instances:
            fout.write(json.dumps({
                "instruction": instance[0],
                "input": instance[1],
                "output": instance[2],
            }) + "\n")
    print(f"Saved {len(training_instances)} instances")
    # 获取所有的指令
    unique_instructions = set([it[0] for it in training_instances])
    print(f"Unique instructions: {len(unique_instructions)}")
    # 获取分类任务指令列表
    clf_instructions = [instruction for instruction in unique_instructions if task_clf_types[instruction]]
    print(f"Classification instructions: {len(clf_instructions)}")
    # 获取生成任务指令列表
    non_clf_instructions = [instruction for instruction in unique_instructions if not task_clf_types[instruction]]
    print(f"Non-classification instructions: {len(non_clf_instructions)}")

    # 5. 如果设置了指令数量参数，则进行随机抽样
    if args.num_instructions is not None:
        print(f"Sampling {args.num_instructions} instructions")
        sampled_instructions = random.sample(unique_instructions, args.num_instructions)
        training_instances = [it for it in training_instances if it[0] in sampled_instructions]
        print(f"Only using {len(training_instances)} instances for these sampled instructions.")
        with open(os.path.join(args.output_dir, f"sampled_generated_instances_{args.num_instructions}.jsonl"), "w") as fout:
            for instance in training_instances:
                fout.write(json.dumps({
                    "instruction": instance[0],
                    "input": instance[1],
                    "output": instance[2],
                }) + "\n")

    # 6. 如果需要包含种子任务，则加载并添加到训练实例中
    if args.include_seed_tasks:
        seed_tasks = [json.loads(l) for l in open(args.seed_tasks_path, "r")]
        for task in seed_tasks:
            for instance in task["instances"]:
                training_instances.append((task["instruction"], instance["input"], instance["output"]))
        print(f"Included {len(seed_tasks)} seed tasks")

    # 7. 获取GPT-3的训练实例training_instances，遍历training_instances，随机选择模版编码为prompt，completion，并返回封装好的data字典
    gpt3_instances = []
    for instance in training_instances:
        # get input and do preprocessing
        inst_input = instance[1]
        # for some tasks, we check whether the input contains colon, and if so, we remove the part before the colon
        if random.random() < 0.5:
            colon_words = re.findall(r"(\w+):", inst_input)
            # if only one colon is found, we assume the instance only have one input and we remove the field name before the colon
            if len(set(colon_words)) == 1:
                inst_input = inst_input.split(":", 1)[1].strip()
            else:
                inst_input = inst_input.strip()
            # we also replace two consecutive new lines with one new line half of the time
            inst_input = inst_input.replace("\n\n", "\n")
        # 编码实例，返回data的字典列表
        gpt3_instances.append(encode_instance(instance[0], inst_input, instance[2]))

    # 8. 移除重复的实例，shuff，保存file
    filtered_instances = []
    prompt_completion_set = set()
    for instance in gpt3_instances:
        instance_pair = (instance["prompt"], instance["completion"])
        if instance_pair not in prompt_completion_set:
            prompt_completion_set.add((instance["prompt"], instance["completion"]))
            filtered_instances.append(instance)
    gpt3_instances = filtered_instances

    # shuffle
    random.shuffle(gpt3_instances)
    # 将GPT-3的训练数据保存到文件中
    with open(os.path.join(args.output_dir, f"gpt3_finetuning_data_{len(gpt3_instances)}.jsonl"), "w") as fout:
        for instance in gpt3_instances:
            fout.write(json.dumps({
                "prompt": instance["prompt"],
                "completion": instance["completion"],
            }) + "\n")