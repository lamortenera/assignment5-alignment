import argparse

import vllm_utils
import drgrpo_grader

import pathlib
import functools
import re
import json

parser = argparse.ArgumentParser(description="Evaluate a model on a set of queries.")
parser.add_argument("--prompt", type=str, required=True, help="The prompt to use for evaluation.")
parser.add_argument("--num_to_evaluate", type=int, default=10, help="The number of queries to evaluate.")


_MODEL_ID = "allenai/OLMo-2-0425-1B"
_DATA_PATH = (pathlib.Path(__file__).resolve().parent.parent) / "data"


_PROMPT_DIR = (pathlib.Path(__file__).resolve().parent) / "prompts"

def get_prompt(prompt_name):
    prompt_path = _PROMPT_DIR / f"{prompt_name}.prompt"
    with open(prompt_path, "r") as f:
        return f.read()


def get_sampling_params(prompt_name):
    #temperature 1.0, top-p 1.0, max generation length 512
    sampling_params = {
        "temperature": 1.0,
        "top_p": 1.0,
        "max_tokens": 512,
        "seed": 123,
        "n": 1
    }

    if prompt_name.startswith("r1_zero"):
        sampling_params['stop'] = ["</answer>"]
        sampling_params['include_stop_str_in_output'] = True
    
    return sampling_params

def grade_response(response, prompt_name, ground_truth):
    if prompt_name.startswith("r1_zero"):
        return drgrpo_grader.r1_zero_reward_fn(response, ground_truth)
    
    return drgrpo_grader.question_only_reward_fn(response, ground_truth)

def test_examples(num_to_evaluate):
    path = _DATA_PATH / "gsm8k" / "test.jsonl"
    with open(path, "r") as f:
        for i, line in enumerate(f):
            if i >= num_to_evaluate:
                break
            data = json.loads(line)
            question = data["question"]
            answer = data["answer"]
            numeric_answer = answer.split("####")[-1].strip()
            yield question, numeric_answer



if __name__ == "__main__":
    args = parser.parse_args()
    print(f"Evaluating model with prompt: {args.prompt} on {args.num_to_evaluate} queries.")
    vllm = vllm_utils.VLLMServer(model_id=_MODEL_ID, gpu=0)
    vllm.start()

    prompt_template = get_prompt(args.prompt)
    sampling_params = get_sampling_params(args.prompt)

    num_samples = 0
    total_reward = 0.0
    examples_by_grade = {}

    for question, numeric_answer in test_examples(args.num_to_evaluate):
        prompt = prompt_template.format(question=question)
        completion = vllm.generate_completions([prompt], sampling_params)
        response = completion[0].text

        grading = grade_response(response, args.prompt, numeric_answer)
        total_reward += grading["reward"]
        key = (grading["format_reward"], grading["answer_reward"])
        examples = examples_by_grade.setdefault(key, [])
        if len(examples) < 10:
            examples.append((question, numeric_answer, response))
        num_samples += 1

    print(f"Average reward: {total_reward / num_samples if num_samples > 0 else 0}")
    for (format_reward, answer_reward), examples in examples_by_grade.items():
        print(f"\nExamples with format reward {format_reward} and answer reward {answer_reward}:")
        for question, numeric_answer, response in examples:
            print(f"Question: {question}")
            print(f"Model Response: {response}")
            print(f"Expected Answer: {numeric_answer}\n")
            