import argparse

import vllm_utils
import drgrpo_grader

import pathlib
import functools
import re
import json
import torch
import time
from datetime import datetime
import gc
import pickle

import checkpoint
import assignment_lib

parser = argparse.ArgumentParser(description="Evaluate a model on a set of queries.")
parser.add_argument("--prompt", type=str, required=True, help="The prompt to use for evaluation.")
parser.add_argument("--num_eval_steps", type=int, default=10, help="The number of eval steps.")
parser.add_argument("--num_steps", type=int, default=100, help="The number of training steps.")
parser.add_argument("--batch_size", type=int, default=128, help="Number of examples in the train/eval batch")
parser.add_argument("--eval_every_n", type=int, default=10, help="The number of training steps.")
parser.add_argument("--checkpoint_every_n", type=int, default=10, help="After how many steps should we save a new checkpoint.")
parser.add_argument("--group_size", type=int, default=8, help="How many rollouts per prompt to make.")
parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate for AdamW optimizer")
parser.add_argument("--gradient_accumulation_steps", type=int, default=16, help="Number of microbatches per batch")
parser.add_argument("--max_grad_norm", type=float, default=1.0, help="The maximum gradient norm")
parser.add_argument("--debug_oom", type=bool, default=False, help="Whether to debug OOM")


_MODEL_ID = "allenai/OLMo-2-0425-1B"
_DATA_PATH = (pathlib.Path(__file__).resolve().parent.parent) / "data"
_TRAIN_PATH = _DATA_PATH / "gsm8k" / "train.jsonl"
_TEST_PATH = _DATA_PATH / "gsm8k" / "test.jsonl"

_PROMPT_DIR = (pathlib.Path(__file__).resolve().parent) / "prompts"

def get_prompt(prompt_name):
    prompt_path = _PROMPT_DIR / f"{prompt_name}.prompt"
    with open(prompt_path, "r") as f:
        return f.read()


def get_sampling_params(prompt_name, group_size):
    sampling_params = {
        "temperature": 1.0,
        "top_p": 1.0,
        "max_tokens": 512,
        "seed": 123,
        "n": group_size
    }

    if prompt_name.startswith("r1_zero"):
        sampling_params['stop'] = ["</answer>"]
        sampling_params['include_stop_str_in_output'] = True
    
    return sampling_params

def dict_to_string(stats):
    pieces = []
    for k, v in stats.items():
        if type(v) == float:
            pieces.append(f"{k}={v:.3f}")
        elif type(v) == dict:
            v_string = "{" + dict_to_string(v) + "}"
            pieces.append(f"{k}={v_string}")
        else:
            pieces.append(f"{k}={v}")
    return("\t".join(pieces))

def log_stats(stats):
    print(dict_to_string(stats))

def example_generator(path, infinite_loop=False):
    finished = False
    while not finished:
        with open(path, "r") as f:
            for line in f:
                yield json.loads(line)
        finished = not infinite_loop

def batch_generator(example_gen, batch_size):
    finished = False
    while not finished:
        questions = []
        ground_truths = []
        for _ in range(batch_size):
            try:
                data = next(example_gen)
            except StopIteration as e:
                yield questions, ground_truths
                finished = True
                break
            questions.append(data["question"])
            answer = data["answer"]
            clean_answer = answer.split("####")[-1].split()
            ground_truths.append(clean_answer)
        yield questions, ground_truths

def repeat_values(l, n):
    result = []
    for v in l:
        for _ in range(n):
            result.append(v)
    return result

def grade_response(response, prompt_name, ground_truth):
    if prompt_name.startswith("r1_zero"):
        return drgrpo_grader.r1_zero_reward_fn(response, ground_truth)
    
    return drgrpo_grader.question_only_reward_fn(response, ground_truth)

class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self  # This assigns the object to the variable in the 'as' clause

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time.perf_counter()
        self.elapsed = self.end - self.start

def run_eval(num_steps, test_path, batch_size, prompt_name, vllm):
    test_batch_gen = batch_generator(
        example_generator(test_path), batch_size)
    prompt_template = get_prompt(prompt_name)
    sampling_params = get_sampling_params(prompt_name, group_size=1)


    eval_metrics = {
        "num_examples": 0,
        "avg_reward": 0.0,
        "avg_format_reward": 0.0,
        "avg_answer_reward": 0.0,
    }

    for step, (questions, ground_truths) in enumerate(test_batch_gen):
        if step >= num_steps:
            break
        prompts = [prompt_template.format(question=q) for q in questions]
        completions = vllm.generate_completions(prompts, sampling_params)
        for c, gt in zip(completions, ground_truths):
            grading = grade_response(c.text, prompt_name, gt)
            eval_metrics["num_examples"] += 1
            eval_metrics["avg_reward"] += grading["reward"]
            eval_metrics["avg_format_reward"] += grading["format_reward"]
            eval_metrics["avg_answer_reward"] += grading["answer_reward"]
    
    num_examples = eval_metrics["num_examples"]
    for k, v in eval_metrics.items():
        if k.startswith("avg_"):
            eval_metrics[k] /= num_examples
    
    return eval_metrics

def flush_trainer_memory(model):
    # 1. Clear out internal python reference tracking cycles
    gc.collect()

    # 2. Tell PyTorch to explicitly drop remaining computational activation artifacts
    # (Note: If using DeepSpeed, the optimizer zeros gradients during the step,
    # but manually forcing it drops any missed tracking states)
    model.zero_grad(set_to_none=True)

    # 3. Force the underlying CUDA caching allocator to release empty memory back to the GPU OS
    torch.cuda.empty_cache()

    # 4. Optional: Reset peak tracking markers so you can monitor real transfer usage
    torch.cuda.reset_peak_memory_stats()


def estimate_static_memory(model):
    param_bytes = 0
    
    for p in model.parameters():
        # Size of elements * element count
        p_mem = p.element_size() * p.nelement()
        param_bytes += p_mem

    print(f"Calculated Parameters: {param_bytes / (1024**2):.2f} MB")

def print_allocated_memory(label):
    allocated_mb = torch.cuda.memory_allocated(device=0) / (1024 ** 2)
    print(f"Allocated memory (pytorch) {label}: {allocated_mb:.2f} MB")


def debug_oom(output_dir):
    assert torch.cuda.is_available()

    torch.cuda.memory._record_memory_history(
    enabled="all", 
    context="alloc", 
    stacks="python", 
    max_entries=100000)

    snapshot_path = str(output_dir / "memory_snapshot.pickle")
    
    def save_oom_snapshot(device, alloc, device_alloc, device_free):
        print("🚨 CUDA Out of Memory detected! Dumping snapshot...")
        print_allocated_memory("right before OOM")
        try:
            # Capture the memory state at the exact moment of OOM
            snapshot = torch.cuda.memory._snapshot()
            
            with open(snapshot_path, "wb") as f:
                pickle.dump(snapshot, f)
                
            print(f"✅ Snapshot saved successfully to {snapshot_path}!")
        except Exception as e:
            print(f"Failed to save snapshot: {e}")

    torch._C._cuda_attach_out_of_memory_observer(save_oom_snapshot)

if __name__ == "__main__":
    args = parser.parse_args()
    assert args.checkpoint_every_n % args.eval_every_n == 0, "You should save only evaluated checkpoints"
    assert args.batch_size % args.gradient_accumulation_steps == 0, "Num microbatches should divide batch size without remainder"
    microbatch_size = args.batch_size // args.gradient_accumulation_steps
    assert microbatch_size % args.group_size == 0, "The group size should divide the microbatch size without remainder"

    run_id = datetime.now().strftime("%Y%m%d%H%M%S")
    output_dir = _DATA_PATH / run_id
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    print(f"Run id: {run_id}, outputs will be written to: {output_dir}")
    
    if args.debug_oom:
        print("Adding debug for OOM")
        debug_oom(output_dir)

    promtp_batch_size = args.batch_size // args.group_size
    train_batch_gen = batch_generator(
        example_generator(_TRAIN_PATH, infinite_loop=True), 
        promtp_batch_size)
    

    # Spin up VLLM
    vllm = vllm_utils.VLLMServer(model_id=_MODEL_ID, gpu=1, gpu_memory_utilization=0.6)
    vllm.start()
    sampling_params = get_sampling_params(
        args.prompt, args.group_size)

    # Get model, tokenizer and optizer
    model, tokenizer = checkpoint.get_model_and_tokenizer(
        _MODEL_ID, device="cuda")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, 
        betas=(0.9, 0.95), weight_decay=0.0)
    
    estimate_static_memory(model)
    
    reward_fn = drgrpo_grader.question_only_reward_fn
    if args.prompt.startswith("r1_zero"):
        reward_fn = drgrpo_grader.r1_zero_reward_fn
    
    
    prompt_template = get_prompt(args.prompt)

    vllm.init_weight_sync(model.device)
                
    start_time = time.time()
    for iter in range(args.num_steps):
        print(f"Iter {iter}")
        with Timer() as step_timer:
            questions, ground_truths = next(train_batch_gen)
            prompts = [prompt_template.format(question=q) for q in questions]

            with Timer() as inference_timer:
                completions = vllm.generate_completions(
                    prompts, sampling_params)

            responses = [c.text for c in completions]
            print(f"Got {len(responses)} responses for {len(prompts)} prompts, group size is {args.group_size}")
            assert len(responses) == args.group_size * len(prompts)
            repeated_prompts = repeat_values(prompts, args.group_size)
            repeated_ground_truths = repeat_values(ground_truths, args.group_size)

            with Timer() as loss_timer:
                loss, loss_stats = assignment_lib.grpo_train_step(
                    model, tokenizer, optimizer, 
                    args.gradient_accumulation_steps,
                    args.max_grad_norm,
                    reward_fn, 
                    repeated_prompts, responses, repeated_ground_truths,
                    args.group_size)
            
            flush_trainer_memory(model)
            #print("Memory usage at the end of the batch: ")
            #print(torch.cuda.memory_summary(device=0, abbreviated=False))
            print_allocated_memory("end of batch")
            
            with Timer() as sync_timer:
                vllm.sync_policy_weights(model)


        stats = {
            "iter": iter,
            "elapsed": time.time() - start_time,
            "step_time": step_timer.elapsed,
            "inference_time": inference_timer.elapsed,
            "loss_time": loss_timer.elapsed,
            "sync_time": sync_timer.elapsed
        }
        stats |= loss_stats

        log_stats(stats)
        
        if (iter + 1) % args.eval_every_n == 0:
            print(f"Running eval for iter {iter}")
            with Timer() as eval_timer:
                eval_stats = run_eval(
                    args.num_eval_steps, _TEST_PATH, args.batch_size, args.prompt, vllm)

            eval_base_stats = {
                "iter": iter,
                "eval_time": eval_timer.elapsed
            }
            log_stats(eval_base_stats | eval_stats)

        if (iter + 1) % args.checkpoint_every_n == 0:
            print(f"Saving checkpoint for iter {iter}")
            checkpoint_dir = output_dir / f"checkpoint_{iter}"
            with Timer() as checkpoint_timer:
                model.save_pretrained(save_directory=checkpoint_dir)
                tokenizer.save_pretrained(save_directory=checkpoint_dir)
            log_stats({"iter": iter, "checkpoint_time": checkpoint_timer.elapsed})
        
