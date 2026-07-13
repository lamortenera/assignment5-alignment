import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase, PreTrainedModel
import torch.nn.functional as F
from typing import Any, Callable, Literal
import math

# def tokenize_prompt_and_output(
#     prompt_strs: list[str],
#     output_strs: list[str],
#     tokenizer: PreTrainedTokenizerBase,
#     ) -> dict[str, torch.Tensor]:
#     assert len(prompt_strs) == len(output_strs)
#     batch_size = len(prompt_strs)
#     tokenized_prompts = tokenizer(prompt_strs, add_special_tokens=False)
#     tokenized_outputs = tokenizer(output_strs, add_special_tokens=True)
#     all_input_ids = []
#     all_response_masks = []
#     for prompt_ids, output_ids in zip(tokenized_prompts['input_ids'], tokenized_outputs['input_ids']):
#         concat_ids = prompt_ids + output_ids
#         response_mask = [False]*len(prompt_ids) + [True]*len(output_ids)
#         all_input_ids.append(torch.tensor(concat_ids))
#         all_response_masks.append(torch.tensor(response_mask))
    
#     pad_token = tokenizer(tokenizer.pad_token)['input_ids']
#     assert len(pad_token) == 1
#     pad_token = pad_token[0]
#     all_input_ids = pad_sequence(all_input_ids, batch_first=True, padding_value=pad_token)
#     all_response_masks = pad_sequence(all_response_masks, batch_first=True, padding_value=False)

#     return {
#         "input_ids": all_input_ids[:, :-1],
#         "labels": all_input_ids[:, 1:],
#         "response_mask": all_response_masks[:, 1:]
#     }

def tokenize_prompt_and_output2(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
    ) -> dict[str, torch.Tensor]:
    assert len(prompt_strs) == len(output_strs)
    output = tokenizer(
        text=prompt_strs, text_pair=output_strs, padding="max_length",
        max_length=512, truncation=True,
        return_tensors='pt', return_token_type_ids=True)
    all_input_ids = output['input_ids']
    all_response_masks = output['token_type_ids']
    
    return {
        "input_ids": all_input_ids[:, :-1],
        "labels": all_input_ids[:, 1:],
        "response_mask": all_response_masks[:, 1:]
    }

def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor, # [Batch, SeqLen]
    return_token_entropy: bool = False,
    ) -> dict[str, torch.Tensor]:
    output = model(input_ids)
    logits = output.logits # [Batch, SeqLen, VocabSize]
    all_log_probs = F.log_softmax(logits, dim=-1)
    log_probs = torch.gather(all_log_probs, index=labels.unsqueeze(-1), dim=-1).squeeze(-1)
    result = {
        "log_probs": log_probs
    }
    if return_token_entropy:
        result["token_entropy"] = -torch.sum(torch.exp(all_log_probs) * all_log_probs, axis=-1)
    return result


def compute_rollout_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    ) -> tuple[torch.Tensor, dict[str, float]]:
    rewards = []
    stats = {}
    
    for response, ground_truth in zip(rollout_responses, repeated_ground_truths):
        grading = reward_fn(response, ground_truth)
        rewards.append(grading["reward"])
        stats["format_reward"] = stats.get("format_reward", 0) + grading["format_reward"]
        stats["answer_reward"] = stats.get("answer_reward", 0) + grading["answer_reward"]
    
    return torch.Tensor(rewards), stats    

def compute_group_normalized_rewards(
    raw_rewards: torch.Tensor,
    group_size: int,
    baseline: Literal["mean", "none"] = "mean",
    advantage_eps: float = 1e-6,
    advantage_normalizer: Literal["std", "none", "mean"] = "std",
    ):
    rewards = raw_rewards.view(-1, group_size)
    means = torch.mean(rewards, axis=-1)
    stds = torch.std(rewards, correction=1, axis=-1)
    bs = 0
    if baseline == "mean":
        bs = means
    
    f = 1
    if advantage_normalizer == "std":
        f = stds
    elif advantage_normalizer == "mean":
        f = means
    
    advantage = (rewards - bs)/(f + advantage_eps)
    return advantage.view(-1), {"mean": means, "std": stds}

def compute_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    response_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    return -raw_rewards_or_advantages[:, None] * policy_log_probs, {}

def aggregate_loss_across_microbatch(
    per_token_policy_gradient_loss: torch.Tensor,
    mask: torch.Tensor,
    loss_normalization: Literal["sequence", "constant"] = "sequence",
    normalization_constant: int | None = None,
    ) -> torch.Tensor:
    loss = torch.sum(per_token_policy_gradient_loss * mask, axis=-1)
    if loss_normalization == "sequence":
        loss /= torch.sum(mask, axis=1)
        return torch.mean(loss)
    else:
        return torch.sum(loss)/normalization_constant

def grpo_train_step(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    optimizer: torch.optim.Optimizer,
    gradient_accumulation_steps: int,
    max_grad_norm: float | None,
    reward_fn: Callable[[str, str], dict[str, float]],
    repeated_prompts: list[str],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    # Reward normalization
    baseline: Literal["mean", "none"] = "mean",
    advantage_eps: float = 1e-6,
    advantage_normalizer: Literal["std", "none", "mean"] = "std",
    # Importance reweighting and clipping
    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    # Loss normalization
    loss_normalization: Literal["sequence", "constant"] = "sequence",
    normalization_constant: int | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:

    def microbatch_loss(mb_repeated_prompts, mb_rollout_responses, mb_repeated_ground_truths):
        tokenization_output = tokenize_prompt_and_output2(mb_repeated_prompts, mb_rollout_responses, tokenizer)
        input_ids = tokenization_output["input_ids"].to(model.device)
        labels = tokenization_output["labels"].to(model.device)
        response_mask = tokenization_output["response_mask"].to(model.device)
        print(f"Input ids shape: {input_ids.shape}, labels shape: {labels.shape}")
        logprobs_output = get_response_log_probs(model, input_ids, labels)
        rewards_output = compute_rollout_rewards(reward_fn, mb_rollout_responses, mb_repeated_ground_truths)
        gn_rewards_output = compute_group_normalized_rewards(
            rewards_output[0].to(model.device), group_size, baseline, advantage_eps, advantage_normalizer)
        token_loss = compute_policy_gradient_loss(
            gn_rewards_output[0], logprobs_output["log_probs"], 
            importance_reweighting_method, old_log_probs, cliprange, response_mask)
        sequence_loss = aggregate_loss_across_microbatch(token_loss[0], 
            response_mask, loss_normalization, normalization_constant)
        
        stats = rewards_output[1]
        stats["reward"] = rewards_output[0].sum()

        return sequence_loss, stats
    
    assert len(repeated_prompts) == len(rollout_responses)
    assert len(repeated_prompts) == len(repeated_ground_truths)

    batch_size = len(repeated_prompts)
    microbatch_size = int(math.ceil(batch_size / gradient_accumulation_steps))
    print(f"Batch size: {batch_size}, grad accum steps: {gradient_accumulation_steps}, Microbatch size: {microbatch_size}")
    
    batch_loss = 0
    batch_stats = {}
    for i in range(gradient_accumulation_steps):
        mb_repeated_prompts = repeated_prompts[i*microbatch_size:(i+1)*microbatch_size]
        mb_rollout_responses = rollout_responses[i*microbatch_size:(i+1)*microbatch_size]
        mb_repeated_ground_truths = repeated_ground_truths[i*microbatch_size:(i+1)*microbatch_size]
        loss, stats = microbatch_loss(mb_repeated_prompts, mb_rollout_responses, mb_repeated_ground_truths)
        if loss_normalization == "sequence":
            loss *= len(mb_repeated_prompts)/batch_size
        loss.backward()
        batch_loss += loss.item()
        for k, v in stats.items():
            batch_stats[k] = batch_stats.get(k, 0) + v
        allocated_mb = torch.cuda.memory_allocated() / (1024 ** 2)
        print(f"Allocated memory (pytorch) at the end of the {i+1}-th microbatch: {allocated_mb:.2f} MB")
    
    total_norm = torch.nn.utils.get_total_norm(model.parameters())
    if max_grad_norm:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    
    optimizer.step()
    optimizer.zero_grad()

    allocated_mb = torch.cuda.memory_allocated() / (1024 ** 2)
    print(f"Allocated memory (pytorch) at the end of the end of the batch: {allocated_mb:.2f} MB")

    return batch_loss, {"grad_norm": total_norm, 
                        "avg_reward": batch_stats["reward"]/batch_size,
                        "avg_format_reward": batch_stats["format_reward"]/batch_size,
                        "avg_answer_reward": batch_stats["answer_reward"]/batch_size}
    
