import os
import random
import json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from retrying import retry
import argparse
import traceback
import re
import requests

class GPT:
    def __init__(self, model_name, api_url, api_key):
        self.model_name = model_name
        self.api_url = api_url
        self.api_key = api_key
        print(f"Using model: {self.model_name}")

    def call(self, content, additional_args={}):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        payload = {
            "model": self.model_name,
            "messages": [{'role': 'user', 'content': content}],
            **additional_args,
        }
        response = requests.post(self.api_url, headers=headers, json=payload)
        response_data = response.json()

        if 'error' in response_data:
            raise ValueError(f"API Error: {response_data}")

        return response_data['choices'][0]['message']['content']

    @retry(wait_fixed=3000, stop_max_attempt_number=3)
    def retry_call(self, content, additional_args={"max_tokens": 8192}):
        return self.call(content, additional_args)

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to the input JSON data file.")
    parser.add_argument("--filter_data", action='store_true', help="Enable filtering of questions with LLMs.")
    parser.add_argument("--model_name", type=str, default="gpt-4", help="Name of the GPT model to use.")
    parser.add_argument("--api_key", type=str, required=True, help="OpenAI API key.")
    parser.add_argument("--api_url", type=str, default="https://api.openai.com/v1/chat/completions", help="OpenAI API URL.")
    parser.add_argument("--num_process", type=int, default=10, help="Number of parallel processes.")
    parser.add_argument("--limit_num", type=int, help="Limit the number of processed items.")
    parser.add_argument("--out_dir", type=str, help="Path to output.")
    return parser.parse_args()

def extract_bracket_content(text):
    # Extract content between the first '{' and the last '}'
    match = re.search(r'\{.*\}', text, re.DOTALL)
    return match.group(0) if match else None

def parse_gpt_response(response):
    try:       
        # 清理响应文本
        response = response.strip()
        if not response:
            raise ValueError("空响应")

        if not response.startswith('{'):
            response = extract_bracket_content(response)
        parsed_data = json.loads(response.replace('\n', ''))

        assert len(parsed_data) == 2, "Response JSON should contain exactly two keys."
        assert isinstance(parsed_data["Open-ended Verifiable Question"], str), "Open-ended Question must be a string."
        assert isinstance(parsed_data["Ground-True Answer"], str), "Ground-True Answer must be a string."

        return True, parsed_data
    except Exception as e:
        print(f"Error parsing GPT response: {e}")
        return False, None

def process_single_item(item, gpt_instance, save_directory, filter_prompt, reformat_prompt, filter_enabled):
    try:
        max_retries = 2
        save_path = os.path.join(save_directory, f"{item['process_id']}.json")

        # Generate options string for the question
        # item['options_str'] = '\n'.join([f"{key}. {value}" for key, value in item['options'].items()])
        question_text = f"{item['translated_question']}\n{item['translated_options']}"

        # Filter questions if enabled
        if filter_enabled:
            filter_query = filter_prompt.format(question_text, item['answer'])
            item['gpt_filter_query'] = filter_query
            response = gpt_instance.retry_call(filter_query)
            item['gpt_filter_response'] = response

            if 'pass' not in response.lower():
                with open(save_path, 'w', encoding='utf-8') as file:
                    json.dump(item, file, ensure_ascii=False, indent=2)
                return 1

        # Reformat questions into open-ended format
        reformat_query = reformat_prompt.format(question_text.replace("\n"," "), item['answer'])
        item['gpt_reformat_query'] = reformat_query

        for _ in range(max_retries):
            response = gpt_instance.retry_call(reformat_query)
            item['gpt_reformat_response'] = response
            valid, parsed_data = parse_gpt_response(response)
            if valid:
                item["Open-ended Verifiable Question"] = parsed_data["Open-ended Verifiable Question"]
                item["Ground-True Answer"] = parsed_data["Ground-True Answer"]
                break

        with open(save_path, 'w', encoding='utf-8') as file:
            json.dump(item, file, ensure_ascii=False, indent=2)

    except Exception as e:
        print(f"Error processing item {item['process_id']}: {e}")
        traceback.print_exc()
    return 1

def merge_saved_files(directory):
    _, _, filenames = next(os.walk(directory))
    json_files = [f for f in filenames if f.endswith('.json')]
    merged_data = []

    for file in json_files:
        try:
            with open(os.path.join(directory, file), 'r', encoding='utf-8') as f:
                data = json.load(f)
                assert 'Open-ended Verifiable Question' in data or 'gpt_filter_response' in data  or 'gpt4_response_filter' in data
                merged_data.append(data)
        except Exception as e:
            traceback.print_exc()
            print(f"Error merging file {file}: {e}")
    return merged_data

def deduplicate_data(data, processed_data):
    processed_ids = {item['process_id'] for item in processed_data}
    return [item for item in data if item['process_id'] not in processed_ids]

def main():
    args = parse_arguments()

    # Load input data
    with open(args.data_path, 'r') as file:
        input_data = json.load(file)

    # Assign unique process IDs to each item
    for idx, item in enumerate(input_data, start=1):
        item['process_id'] = idx

    if args.limit_num:
        input_data = input_data[:args.limit_num]

    print(f"Loaded {len(input_data)} items.")

    # Define task and save directory
    task_name = os.path.splitext(os.path.basename(args.data_path))[0]
    save_directory = os.path.join('output_data', task_name)
    os.makedirs(save_directory, exist_ok=True)

    # Check whether output dir exists
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
        print(f"目录 {args.out_dir} 已创建")
    else:
        print(f"目录 {args.out_dir} 已存在")

    gpt_instance = GPT(model_name=args.model_name, api_url=args.api_url, api_key=args.api_key)

    filter_prompt = """<Multiple-choice Question>
{}
Correct Answer: {}
</Multiple-choice Question>

You are an expert in filtering and evaluating multiple-choice questions for advanced reasoning tasks. Your job is to evaluate a given question and determine whether it meets the following criteria: 
1. **Depth of Reasoning:** The question should require deeper reasoning. If the question appears too simple, mark it as "Too Simple".
2. **Unambiguous Correct Answer:** The question must have a unique and unambiguous correct answer. If the question asks for "incorrect options" or allows for multiple correct answers, mark it as "Ambiguous Answer".
3. **Open-Ended Reformulation Feasibility:** The question should be suitable for reformatting into an open-ended format. If the question cannot be easily reformulated into an open-ended problem and a clear ground-truth answer, mark it as "Not Reformulatable".

For each question, provide one of the following evaluations:  
- "Pass" (The question meets all the criteria.)  
- "Too Simple"  
- "Ambiguous Answer"  
- "Not Reformulatable" """

    filter_prompt_Chinese = """<选择题>
{}
正确答案：{}
</选择题>

您是一位擅长筛选和评估选择题的专家，专门为高阶推理任务服务。您的任务是评估给定的选择题，并判断它是否符合以下标准：
1. **推理深度**：题目需要较深的推理能力。如果问题过于简单，请标记为"Too Simple"。
2. **答案明确性**：正确答案需要惟一且明确。如果题目要求选择"错误选项"或允许多个正确答案，请标记为"Ambiguous Answer"。
3. **可开放性改造**：题目应适合改写成开放式问题。如果无法改写成开放式问题并给出明确的标准答案，请标记为"Not Reformulatable""。

对每个问题，请给出以下评估结果之一：
- "Pass"（题目符合所有标准）
- "Too Simple"
- "Ambiguous Answer"
- "Not Reformulatable" """

    reformat_prompt = """I will provide you with a multiple-choice question, and your task is to rewrite it into an open-ended question, along with a Ground-True Answer. The requirements are:

1. The question must be specific, targeting the point being tested in the original multiple-choice question. Ensure it is open-ended, meaning no options are provided, but there must be a definitive Ground-True Answer.
2. The question must include all the key points from the original multiple-choice question. You must preserve these key elements exactly:
   - All numerical values (e.g., "4 cycles", "first cycle")
   - All medical conditions (e.g., "chemotherapy-induced thrombocytopenia")
   - All treatment contexts (e.g., "cancer chemotherapy")
3. Based on the correct answer from the original question, provide a concise Ground-True Answer. The answer should allow for precise matching to determine whether the model's response is correct.

Here is the multiple-choice question for you to rewrite:
<Multiple-choice Question>
{}
Correct Answer: {}
</Multiple-choice Question>

Please output the result in the following JSON format:
```json
{{
"Open-ended Verifiable Question": "...",
"Ground-True Answer": "..."
}}
```"""

    reformat_prompt_Chinese = """我将提供一个选择题，您的任务是将其改写成开放式问题，并提供一个标准答案。具体要求如下：

1. 问题必须具体明确，针对原选择题的考察要点。确保它是开放式的（不提供选项），但必须有一个确定的标准答案。
2. 问题必须包含原选择题中的所有关键点。您需要严格保留以下关键元素：
   - 所有数值（如"4个周期"、"第一个周期")
   - 所有医学状况（如"化疗引起的血小板减少症")
   - 所有治疗背景（如"癌症化疗")
3. 根据原问题的正确答案，提供一个简洁的标准答案。该答案应能精确匹配以判断模型的回答是否正确。

以下是需要改写的选择题：
<选择题>
{}
正确答案：{}
</选择题>

请按照以下JSON格式输出结果，且结果中不要有换行符:
```json
{{
    "Open-ended Verifiable Question": "改写后的开放式问题",
    "Ground-True Answer": "你生成的标准答案"
}}
```"""

    # Merge previously processed files
    processed_data = merge_saved_files(save_directory)
    print(f"Previously processed items: {len(processed_data)}")

    input_data = deduplicate_data(input_data, processed_data)
    print(f"Items remaining for processing: {len(input_data)}")

    # Process data using a thread pool
    with ThreadPoolExecutor(max_workers=args.num_process) as executor:
        list(tqdm(executor.map(lambda item: process_single_item(item, gpt_instance, save_directory, filter_prompt_Chinese, reformat_prompt_Chinese, args.filter_data), input_data), total=len(input_data), desc="Processing Items", unit="item"))

    # Merge and save final output
    final_data = merge_saved_files(save_directory)
    output_file = f"{task_name}_final_{len(final_data)}.json"
    output_path = os.path.join(args.out_dir, output_file)
    print(f"Processed {len(final_data)} items. Saving to {output_path}")

    with open(output_path, 'w', encoding='utf-8') as file:
        json.dump(final_data, file, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    main()
