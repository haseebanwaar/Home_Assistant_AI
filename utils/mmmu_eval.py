import ast

import requests
import base64
from PIL import Image
from io import BytesIO
from tqdm import tqdm
import json
import re
from datasets import load_dataset
import time  # <-- 1. IMPORT THE TIME MODULE

# from providers.local_openAI import model_name_vlm

# --- Configuration ---
VLLM_API_URL = "http://localhost:8000/v1/chat/completions"  # vllm
# VLLM_API_URL = "http://localhost:8080/v1/chat/completions"  # llama.cpp
# VLLM_API_URL = "http://localhost:23333/v1/chat/completions"    #lmdeploy
EVAL_SPLIT = 'validation'
# MODIFIED: Create a list of subjects to evaluate
SUBJECTS_TO_EVALUATE = [
    'Accounting',
    'Agriculture',
    'Architecture_and_Engineering',
    'Art',
    'Art_Theory',
    'Basic_Medical_Science',
    'Biology',
    'Chemistry',
    'Clinical_Medicine',
    'Computer_Science',
    'Design',
    'Diagnostics_and_Laboratory_Medicine',
    'Economics',
    'Electronics',
    'Energy_and_Power',
    'Finance',
    'Geography',
    'History',
    'Literature',
    'Manage',
    'Marketing',
    'Materials',
    'Math',
    'Mechanical_Engineering',
    'Music',
    'Pharmacy',
    'Physics',
    'Psychology',
    'Public_Health',
    'Sociology',
]


def pil_to_base_64(image: Image.Image) -> str:
    """Converts a PIL Image object to a base64 string."""
    buffer = BytesIO()
    if image.mode != "RGB":
        image = image.convert("RGB")
    image.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def format_options(options_list: list) -> str:
    """Formats the multiple-choice options from a list."""
    formatted = []
    for i, option in enumerate(options_list):
        if not option.startswith(f"({chr(65 + i)})"):
            formatted.append(f"({chr(65 + i)}) {option}")
        else:
            formatted.append(option)
    return "\n".join(formatted)


def call_api(question_prompt: str, image_base_64: str = None):  # MODIFIED: image_base64 is now optional
    """Sends a request to the vLLM OpenAI-compatible API endpoint."""
    headers = {"Content-Type": "application/json"}
    # system_prompt = (
    #     "You are an expert AI assistant. Your task is to answer multiple-choice questions. "
    #     "First, provide a step-by-step explanation of your reasoning. After your explanation, "
    #     "provide the final answer on a new line in the format: 'The final answer is (X)'."
    # )

    content = []
    if image_base_64:
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{image_base_64}"
            }
        })
    content.append({
        "type": "text",
        "text": question_prompt
    })

    payload = {
        "model": model_name_vlm,
        # "model": 'F:\\try\\internVL3_8B_AWQ',
        # "model": 'F:\\try\\internVL_2B',
        # "model": "/mnt/f/try/internVL_14B",
        # "model": 'F:\\try\\internvl_gguf\\OpenGVLab_InternVL3_5-4B-Q4_K_M.gguf',
        # "model": 'hfl/Qwen2.5-VL-7B-Instruct-GPTQ-Int4',
        # "model": "F:\\try\\intern_s1_gguf\\Intern-S1-mini-Q8_0.gguf",
        # "messages": [{"role": "user", "content": content}],
        "messages": [
            # {"role": "system", "content": system_prompt},
            {"role": "user", "content": content}
        ],
        "max_tokens": 2500,
        # "temperature": 0.5,
        # "top_p": 0.7,
        # 'top_k': 20,
        # 'repetition_penalty': 1.0,
        # 'presence_penalty': 1.5,
        # 'greedy': False,

    }
    start_time = time.time()
    try:
        response = requests.post(VLLM_API_URL, headers=headers, json=payload, timeout=120)
        duration = time.time() - start_time
        response.raise_for_status()

        response_json = response.json()
        model_output = response_json['choices'][0]['message']['content'].strip()

        # Get completion tokens from the usage object
        completion_tokens = response_json.get('usage', {}).get('completion_tokens', 0)

        return model_output, completion_tokens, duration

    except requests.RequestException as e:
        print(f"Error calling API: {e}")
        return "", 0, 0  # Return default values on error


def main():
    print(f"Starting MMMU evaluation for subjects: {', '.join(SUBJECTS_TO_EVALUATE)}...")
    nones = 0
    total_correct_predictions = 0
    total_questions_evaluated = 0
    total_completion_tokens = 0
    total_generation_time = 0

    for subject in SUBJECTS_TO_EVALUATE:
        print(f"\n--- Evaluating Subject: {subject} ---")
        try:
            dataset = load_dataset("MMMU/MMMU", name=subject, split=EVAL_SPLIT)
        except Exception as e:
            print(f"Failed to load subject {subject}: {e}. Skipping.")
            continue

        subject_results = []
        subject_correct = 0
        subject_tokens = 0
        subject_time = 0
        question_count = 0
        for row in tqdm(dataset, desc=f"Processing {subject}"):
            if question_count >= 5:
                break  # Exit the loop after processing 5 questions
            image_b64 = None
            if 'image_1' in row and row['image_1']:
                image = row['image_1']
                image_b64 = pil_to_base_64(image)

            question_text = row['question']
            answer = str(row['answer'])
            options_data = row['options']

            options_list = []
            if isinstance(options_data, str):
                try:
                    options_list = ast.literal_eval(options_data)
                except (ValueError, SyntaxError):
                    print(f"Warning: Could not parse options: {options_data}")
            else:
                options_list = options_data  # It might already be a list

            options_text = format_options(options_list)

            full_question = f"{question_text}\n{options_text}\nFirst, provide a step-by-step explanation of your reasoning. After your explanation,you must provide the alphabet letter of the option you choose on a new line in the format: 'The final answer is (X)'."
            model_output, tokens, duration = call_api(full_question, image_b64)
            subject_tokens += tokens
            subject_time += duration
            print(f'\n{model_output}\n')
            match = re.search(r'The final answer is \(?([A-E])\)?', model_output)
            prediction = match.group(1) if match else "None"
            if prediction == 'None': nones += 1
            print(f'\n{prediction}\n')
            is_correct = (prediction == answer)
            if is_correct:
                subject_correct += 1

            subject_results.append({
                'question_id': row['id'], 'prediction': prediction, 'answer': answer,
                'is_correct': is_correct, 'model_output': model_output
            })
            question_count += 1
        subject_tps = (subject_tokens / subject_time) if subject_time > 0 else 0
        subject_total = len(subject_results)
        subject_accuracy = (subject_correct / subject_total) * 100 if subject_total > 0 else 0
        print(f"Accuracy for {subject}: {subject_accuracy:.2f}% ({subject_correct}/{subject_total})")
        print(f"Average Speed for {subject}: {subject_tps:.2f} tokens/sec")

        with open(f'mmmu_vllm_results_{subject}.json', 'w') as f:
            json.dump(subject_results, f, indent=4)
        print(f"Detailed results for {subject} saved to 'mmmu_vllm_results_{subject}.json'")

        total_correct_predictions += subject_correct
        total_questions_evaluated += subject_total
        total_completion_tokens += subject_tokens
        total_generation_time += subject_time

    print("\n\n--- Combined Evaluation Summary ---")
    combined_accuracy = (
                                total_correct_predictions / total_questions_evaluated) * 100 if total_questions_evaluated > 0 else 0
    combined_tps = (total_completion_tokens / total_generation_time) if total_generation_time > 0 else 0
    print(f"Evaluated Subjects: {', '.join(SUBJECTS_TO_EVALUATE)}")
    print(f"Total Questions Evaluated: {total_questions_evaluated}")
    print(f"Total Correct Predictions: {total_correct_predictions}")
    print(f"Combined Accuracy: {combined_accuracy:.2f}%")
    print(f"Average Generation Speed: {combined_tps:.2f} tokens/sec")
    print(f"prompt non adherence: {nones:.2f}")


if __name__ == "__main__":
    main()
