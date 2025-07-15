# %%

import os
import random
import json
from tqdm import tqdm
import multiprocessing
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor
import random
import requests
from retrying import retry
import argparse
import re
import traceback
import copy

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
        response = requests.post(self.api_url, headers=headers, json=payload, timeout=500)
        response_data = response.json()

        if 'error' in response_data:
            raise ValueError(f"API Error: {response_data}")

        return response_data['choices'][0]['message']['content']

    @retry(wait_fixed=5000, stop_max_attempt_number=3)
    def retry_call(self, content, additional_args={"max_tokens": 8192}):
        return self.call(content, additional_args)


prompt_translate = """请将以下英文医学多选题及选项专业准确地翻译为中文，保持原意的同时符合中文医学表达习惯。

<question>
{}
</question>

<options>  
{}
</options>  

注意事项：
1. 医学术语需使用中国大陆通用译名
2. 复杂概念需保留英文原名括号标注
3. 题干中的"all of the above"等特殊表述需本地化处理

请返回严格合法的单行JSON，不要换行，字符串内部的换行符请转义为\\n。输出形式必须严格遵守下面的json格式，:
```json
{{
    "translated_question": "翻译后的中文问题题干", 
    "translated_options": "A.选项1翻译\\nB.选项2翻译\\nC.选项3翻译\\nD.选项4翻译"
}}
```"""

prompt_validate = """请作为医学翻译质检专家，严格评估以下中英对照多选题的翻译质量，按步骤处理：

<question>
{}
</question>

<options>  
{}
</options>  

<translated_question>
{}
</translated_question>

<translated_options>  
{}
</translated_options>  

==== 评估步骤 ====
1. 术语核查: 对比中英文医学术语的一致性（如MRI→磁共振成像）
2. 语义完整性：检查是否存在漏译/增译
3. 临床适配性：验证表述是否符合中文临床用语习惯
4. 逻辑一致性：确保选项与题干逻辑关系保持不变
5. 格式规范：核对选项编号顺序、标点符号等细节

==== 评估标准 ====
1. 对于每一项评估步骤的评价，优算20分，良算10分，差算0分
2. 将5项总分加起来得到综合评分，如果评分大于60，在translate_result中输入Pass，否则输出Reject

==== 输出格式 ====
1. 必须返回严格合法的单行 JSON，不能包含任何换行符或注释。
2. JSON 必须包含且仅包含以下字段：
```json
{{
    "translate_result": "Pass或Reject"
}}
```"""



def extract_bracket_content(text):
        # Extract content between the first '{' and the last '}'
        match = re.search(r'\{.*\}', text, re.DOTALL)
        return match.group(0) if match else None

def parse_translate_response(response):
    try:
        if '{' != response[0]:
            response = extract_bracket_content(response)
        da = json.loads(response)
        assert isinstance(da["translated_question"],str), "translated_question should be string"
        assert isinstance(da["translated_options"],str), "translated_options should be string"
        return True,da
    except Exception as e:
        print(e)
        traceback.print_exc()
        return False,None

def parse_gpt_varify(response):
    try:
        if '{' != response[0]:
            response = extract_bracket_content(response)
        da = json.loads(response)

        assert isinstance(da["translate_result"],str), "translate_result should be str"
        assert da["translate_result"] == "Pass" or da["translate_result"] == "Reject", "translate_result should Pass or Reject \\n"
        return True,da
    except Exception as e:
        print(e)
        traceback.print_exc()
        return False,None 
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to the input JSON data file.")
    parser.add_argument("--model_name", type=str, default="gpt-4", help="Name of the GPT model to use.")
    parser.add_argument("--api_key", type=str, required=True, help="OpenAI API key.")
    parser.add_argument("--api_url", type=str, default="https://api.openai.com/v1/chat/completions", help="OpenAI API URL.")
    # parser.add_argument("--max_search_attempts", type=int, default=1, help="Maximum number of search attempts.")
    # parser.add_argument("--max_search_depth", type=int, default=2, help="Maximum search depth.")
    # parser.add_argument("--efficient_search", type=bool, default=True, help="Enable efficient search strategy.")
    parser.add_argument("--num_process", type=int, default=32, help="Number of parallel processes.")
    parser.add_argument("--limit_num", type=int, help="Limit the number of processed items.")
    
    args = parser.parse_args()

    def filter_data(tmpdata):
        filtered_data = []
        for da in tmpdata:
            if 'question' not in da and 'options' not in da:
                continue
            filtered_data.append(da)

        print(f"Original data size: {len(tmpdata)}, Filtered data size: {len(filtered_data)}")
        return filtered_data

    with open(args.data_path) as f:
        tmpdata = json.load(f)

    tmp_id = 1
    for da in tmpdata:
        da['process_id'] = tmp_id
        tmp_id += 1
    data = filter_data(tmpdata)

    if args.limit_num:
        data = data[:args.limit_num]
        
    print(f"read data:{len(data)}")

    task_name = f'{os.path.split(args.data_path)[-1].replace(".json","")}_translate'
    save_dir = f'output_data/{task_name}'

    gpt_instance = GPT(model_name=args.model_name, api_url=args.api_url, api_key=args.api_key)


    def verify_gpt(translated_question,translated_options,d):
        query = prompt_validate.format(translated_question.replace('\n',''),translated_options.replace('\n',''),d["question"].replace('\n',''),d["options"].replace('\n',''))
        response = gpt_instance.retry_call(query)
        flag, response = parse_gpt_varify(response.replace("\n",""))

        if flag == False:
            raise ValueError("GPT 返回的验证响应解析失败")

        d['gpt4_verify_query'] = query
        d['gpt4_verify_response'] = response
        if 'pass' in response['translate_result'].lower():
            d['translate_result'] = "Pass"
            return True
        else:
            d['translate_result'] = "Reject"
            return False
        
    global wrongtime
    wrongtime = 0
    def write_piece_order_data(d):
        global wrongtime
        try:
            retry_time = 1
            d['translate_result'] = []
            d['translated_question'] = []
            d['translated_options'] = []
            d["gpt4_translate_response"] = []
            d['gpt4_verify_query'] = []
            d['gpt4_verify_response'] = []

            save_path = os.path.join(save_dir, str(d['process_id']) + ".json")

            # init reason
            query = prompt_translate.format(d['question'],d["options"])
            d['gpt4_translate_response'] = query

            response = gpt_instance.retry_call(query)
            flag, translate_response = parse_translate_response(response)
            # print("#############\n","flag:",flag,"\n",translate_response,"__________________\n")
            if flag:               
                d['translated_question'] = translate_response['translated_question']
                d['translated_options'] = translate_response['translated_options']
                verify_gpt(d['translated_question'],d['translated_options'],d)

            with open(save_path, mode="w", encoding="utf-8") as fw:
                json.dump(d, fw, ensure_ascii=False,indent=2)
                wrongtime = 0

        except Exception as e:
            traceback.print_exc()
            wrongtime += 1
            if wrongtime > 200:
                assert 1 == 0, 'wrong'
        return 1
            
    def deduplicate_data(data, processed_data):
        processed_ids = {item['process_id'] for item in processed_data}
        return [item for item in data if item['process_id'] not in processed_ids]


    def merge_saved_files(save_dir):
        _, _, filenames = [i for i in os.walk(save_dir)][0]
        json_files = [f for f in filenames if f.endswith('.json')]
        res = []
        for file_path in json_files:
            try:
                with open(os.path.join(save_dir, file_path), encoding="utf-8") as f:
                    da = json.loads(f.read())
                    assert 'translated_question' in da and 'translated_options' in da
                    # assert 'pass' in da['translate_result'].lower()
                    res.append(da)
            except Exception as e:
                continue
        return res
    
    os.makedirs(save_dir, exist_ok=True)

    # Merge previously processed files
    processed_data = merge_saved_files(save_dir)
    print(f"Previously processed items: {len(processed_data)}")

    input_data = deduplicate_data(data, processed_data)
    print(f"Items remaining for processing: {len(input_data)}")

    with ThreadPoolExecutor(max_workers=args.num_process) as executor:
        list(tqdm(executor.map(write_piece_order_data, input_data), total=len(input_data), desc="Processing samples", unit="sample"))

     # Merge and save final output
    final_data = merge_saved_files(save_dir)
    output_path = f"{task_name}_{len(final_data)}.json"
    print(f"Processed {len(final_data)} items. Saving to {output_path}")

    with open(output_path, 'w', encoding='utf-8') as file:
        json.dump(final_data, file, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()