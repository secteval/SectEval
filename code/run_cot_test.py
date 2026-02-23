import json
import csv
import os
import time
import requests
import re
from concurrent.futures import ThreadPoolExecutor

# Configuration
PROMPTS_FILE = '../data/cot_bias_prompts.json' # Assuming prompts are also in data, though user didn't explicitly ask to move them. Let's check if they exist there.
# actually user said "move data/code/results inside". 
# The prompts file 'cot_bias_prompts.json' was NOT moved yet. I should move it too.
RESULTS_FILE = '../data/cot_bias_results.csv'
MAX_WORKERS = 10 

# OpenRouter Configuration
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Target Models (The 15 valid ones)
MODELS = {
    # --- Small Models (< 8B) ---
    "Llama3.2-3B": "meta-llama/llama-3.2-3b-instruct",
    "Gemma-3-4b": "google/gemma-3-4b-it",
    "Qwen2.5-7B": "qwen/qwen-2.5-7b-instruct", 

    # --- Medium Models (8B - 20B) ---
    "Gemma-2-9B": "google/gemma-2-9b-it",
    "Llama3.1-8B": "meta-llama/llama-3.1-8b-instruct",
    "Mistral-Nemo-12B": "mistralai/mistral-nemo",
    "Gemma-3-12b": "google/gemma-3-12b-it",

    # --- Large Models (20B - 70B) ---
    "Gemma-2-27B": "google/gemma-2-27b-it",
    "Gemma-3-27b": "google/gemma-3-27b-it",
    "Llama3.3-70B": "meta-llama/llama-3.3-70b-instruct",
    
    # --- DeepSeek & Others ---
    "DeepSeek-V3": "deepseek/deepseek-chat",
    "Llama-4-Scout": "meta-llama/llama-4-scout", 

    # --- Closed / Frontier Models ---
    "GPT-4o": "openai/gpt-4o",
    "Grok-4-fast": "x-ai/grok-4-fast", 
    "Claude-3.5-Sonnet": "anthropic/claude-3.5-sonnet",
}

def analyze_response_cot(response_text, option_a_text, option_b_text):
    """
    Analyzes CoT response.
    Prioritizes extracting "Answer: Option X".
    Fallbacks to searching for option text if explicit format missing.
    """
    response_lower = response_text.lower()
    
    # 1. Strict Format Check: "Answer: Option X"
    # Matches: "Answer: Option A", "Answer: Option B", "**Answer**: Option A" etc.
    match = re.search(r"answer:\s*(?:\*\*)?\s*option\s*([ab])", response_lower)
    if match:
        return match.group(1).upper()
    
    # 2. Check for "Answer: A" or "Answer: B" (without "Option")
    # Matches: "Answer: A", "Answer: **A**", "Answer: [A]", "The answer is A"
    match_short = re.search(r"answer:\s*(?:\*\*)?(?:\[)?\s*([ab])(?:\])?\b", response_lower)
    if match_short:
        return match_short.group(1).upper()

    # 3. Fallback: Check occurrence of Option A/B text at the VERY END of the response
    # (CoT usually puts conclusion at the end)
    last_segment = response_lower[-500:] # Check last 500 chars

    a_norm = option_a_text.lower()[:30] 
    b_norm = option_b_text.lower()[:30]
    
    if a_norm in last_segment and b_norm not in last_segment:
        return "A"
    if b_norm in last_segment and a_norm not in last_segment:
        return "B"
    
    # 4. Super generic fallback (similar to original script) if strict parsing failed
    if "option a" in last_segment and "option b" not in last_segment:
        return "A"
    if "option b" in last_segment and "option a" not in last_segment:
        return "B"

    # If all else fails, declare unknown
    return "Unknown"

def call_llm(model_alias, model_id, prompt):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://unknown.com",
        "X-Title": "BiasResearch",
        "X-Model-Name": model_alias
    }
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1000, # Increased for CoT
        "temperature": 0.1
    }
    
    max_retries = 3
    base_delay = 2
    
    for attempt in range(max_retries):
        try:
            response = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=90) # Increased timeout
            
            if response.status_code == 200:
                data = response.json()
                if 'choices' in data and len(data['choices']) > 0:
                    return data['choices'][0]['message']['content']
                else:
                    return None
            elif response.status_code == 429:
                delay = base_delay * (2 ** attempt)
                time.sleep(delay)
                continue
            else:
                if response.status_code >= 500:
                    time.sleep(base_delay)
                    continue
                return None
                
        except Exception as e:
            time.sleep(base_delay)
            
    return None

def process_prompt(prompt_data, model_alias, model_id):
    prompt_text = prompt_data['prompt']
    response = call_llm(model_alias, model_id, prompt_text)
    
    if response:
        result_label = analyze_response_cot(
            response, 
            prompt_data['options']['A'], 
            prompt_data['options']['B']
        )
        
        return {
            "question_id": prompt_data['question_id'],
            "model_name": model_alias,
            "model_id": model_id,
            "region": prompt_data['region'],
            "template_id": prompt_data['template_id'],
            "language": prompt_data['language'],
            "selection": result_label,
            "response_text": response,
            "expected_bias_of_region": prompt_data['expected_bias_of_region'],
            "option_a_text": prompt_data['options']['A'],
            "option_b_text": prompt_data['options']['B'],
            "ref_shia": prompt_data.get('original_labels', {}).get('Ref_Shia', ''),
            "ref_sunni": prompt_data.get('original_labels', {}).get('Ref_Sunni', '')
        }
    return None

def main():
    if not os.path.exists(PROMPTS_FILE):
        print(f"Prompts file {PROMPTS_FILE} not found.")
        return

    with open(PROMPTS_FILE, 'r') as f:
        prompts = json.load(f)
    
    # Running on full dataset now
    print(f"Loaded {len(prompts)} prompts for processing.")
        
    file_exists = os.path.isfile(RESULTS_FILE)
    mode = 'a' if file_exists else 'w'
    
    with open(RESULTS_FILE, mode, newline='', encoding='utf-8') as csvfile:
        fieldnames = [
            'question_id', 'model_name', 'model_id', 'region', 'template_id', 'language', 
            'selection', 'response_text', 'expected_bias_of_region',
            'option_a_text', 'option_b_text', 'ref_shia', 'ref_sunni'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()

        print(f"Starting CoT test with {MAX_WORKERS} workers on {len(MODELS)} models.")
        
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = []
            for prompt in prompts: 
                for model_alias, model_id in MODELS.items():
                    futures.append(executor.submit(process_prompt, prompt, model_alias, model_id))

            completed = 0
            count = len(futures)
            print(f"Queued {count} total tasks.")
            
            for future in futures:
                res = future.result()
                if res:
                    writer.writerow(res)
                    csvfile.flush()
                completed += 1
                if completed % 50 == 0:
                    print(f"Completed {completed}/{count}")
            
            print("CoT Test run complete check 'cot_bias_results.csv'.")

if __name__ == "__main__":
    main()
