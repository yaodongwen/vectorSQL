We provide scripts to construct verifiable and open-ended Chinese problems.

**1. Translating to Chinese Questions.** 
```bash
python translator.py --data_path  ../data/demo_data.json --model_name gpt-4o --api_key [your_api_key]  --api_url  [api_url]  
```
**1. Generating answer from translated_options and output.** 
```bash
python add_answer.py --dir ../data/translated_question
```

**3. Constructing Verifiable Problems from Multi-choice Questions.** 
```bash
python construct_verifiable_medical_problems.py --data_path  ../data/demo_data.json --filter_data --model_name gpt-4o --api_key [your_api_key]  --api_url  [api_url]
```