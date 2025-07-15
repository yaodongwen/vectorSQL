import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

def clean_output_value(output_value):
    """清理output值，提取真正的答案字母"""
    if not output_value:
        return ""
    
    # 处理XML/HTML标签格式 <answer>A</answer>
    match = re.search(r"<answer>([A-Za-z])</answer>", str(output_value))
    if match:
        return match.group(1)
    
    # 处理其他可能格式
    output_value = str(output_value).strip()
    if output_value.startswith("<") and output_value.endswith(">"):
        return output_value[1:-1]  # 去掉尖括号
    
    return output_value  # 默认返回原值

def extract_answer(translated_options, output_key):
    """从translated_options中提取对应output_key的选项内容"""
    # 清理output_key
    output_key = clean_output_value(output_key)
    if not output_key:
        return None
    
    # 处理translated_options的不同格式
    if isinstance(translated_options, list):
        options = [str(opt).strip() for opt in translated_options]
    else:
        options = str(translated_options).split('\n')
    
    for option in options:
        option = str(option).strip()
        # 支持多种格式：A. xxx 或 A) xxx 或 A xxx
        if re.match(rf"^{re.escape(output_key)}[\.\)\s]+", option):
            return option[len(output_key):].lstrip(". )\t").strip()
    
    return None

def process_json_file(file_path):
    """处理单个JSON文件，添加answer字段"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        items = [data] if isinstance(data, dict) else data
        modified = False
        
        for item in items:
            if all(key in item for key in ['output', 'translated_options']) and 'answer' not in item:
                output_key = item['output']
                translated_options = item['translated_options']
                
                answer = extract_answer(translated_options, output_key)
                if answer:
                    item['answer'] = answer
                    modified = True
        
        if modified:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data if isinstance(data, dict) else items, f, ensure_ascii=False, indent=2)
            return True
        return False
    
    except Exception as e:
        print(f"\n处理文件 {file_path} 时出错: {str(e)}")
        return False

def batch_process_json_files(directory, num_process=4):
    """批量处理JSON文件（多线程带进度条）"""
    files_to_process = []
    
    print("扫描需要处理的文件...")
    for filename in tqdm(os.listdir(directory), desc="扫描文件"):
        if filename.endswith('.json'):
            file_path = os.path.join(directory, filename)
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                need_process = False
                if isinstance(data, dict):
                    if all(k in data for k in ['output', 'translated_options']) and 'answer' not in data:
                        need_process = True
                elif isinstance(data, list):
                    if any(all(k in item for k in ['output', 'translated_options']) and 'answer' not in item 
                          for item in data):
                        need_process = True
                
                if need_process:
                    files_to_process.append(file_path)
            
            except Exception as e:
                print(f"\n文件 {filename} 读取错误: {str(e)}")
                continue
    
    print(f"\n开始处理 {len(files_to_process)} 个文件...")
    with ThreadPoolExecutor(max_workers=num_process) as executor:
        results = list(tqdm(
            executor.map(process_json_file, files_to_process),
            total=len(files_to_process),
            desc="处理进度",
            unit="文件"
        ))
    
    print(f"\n处理完成: 共{len(files_to_process)}个文件，成功{sum(results)}个")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True, help="JSON文件目录路径")
    parser.add_argument("--num_process", type=int, default=4, help="线程数 (默认: 4)")
    args = parser.parse_args()
    
    batch_process_json_files(args.dir, args.num_process)