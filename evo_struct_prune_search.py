import argparse
import random
import copy
import os
import math
from tqdm import trange
import time
from typing import List, Tuple, Sequence, Optional, Union

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    import wandb

    has_wandb = True
except ModuleNotFoundError:
    has_wandb = False

from src.data_utils import get_data
from src.common_utils import fix_seed
from src.metrics import compute_perplexity, compute_kl_div, compute_sparse_kl_div
from src.model_utils import layer_order_fn, group_layers

from struct_prune import load_sparse_config, load_compressed_finetuned_weights, load_compressed_weights
from utils import shrink


def load_layers(
    model: AutoModelForCausalLM,
    grouped_layer_names: Tuple[Sequence[str]],
    new_state: Tuple[Sequence[int]],
    sparse_weights_path: str,
):
    assert hasattr(model, "state")
    num_groups = len(grouped_layer_names)
    for i in range(num_groups):
        for layer_name, new_level, old_level in zip(grouped_layer_names[i], new_state[i], model.state[i]):
            if new_level != old_level:
                layer = model.get_submodule(layer_name)
                layer.weight.data = torch.load(
                    os.path.join(sparse_weights_path, layer_name, f"{new_level}.pth"), map_location=layer.weight.device
                ).to(layer.weight.dtype)
    # Update model state
    model.state = copy.deepcopy(new_state)


def compute_fitness(model, data, fitness_fn, 
                    target_logits: Optional[torch.Tensor] = None,
                    finetune_data_list: Optional[torch.Tensor] = None,
                    batch_size: int = 1,
                    finetune_lr: float = 1e-5,) -> float:
    if finetune_data_list is not None:
        # Fine-tune the model on the provided small subset
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=finetune_lr)
        num_samples = len(finetune_data_list)
        for i in trange(0, num_samples, batch_size, desc="Finetune the Candidate", leave=False):
            j = min(i + batch_size, num_samples)
            finetune_data = torch.cat(finetune_data_list[i:j])
            optimizer.zero_grad(set_to_none=True)
            finetune_data = finetune_data.to(model.device)
            outputs = model(finetune_data, labels=finetune_data)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            del outputs, loss, finetune_data
        optimizer = None
        del optimizer
    
    model.eval()
    
    if fitness_fn == "ppl":
        return compute_perplexity(model, data)
    elif fitness_fn == "kl":
        return compute_kl_div(model, data, target_logits)
    elif fitness_fn == "sparse_kl":
        return compute_sparse_kl_div(model, data, target_logits)


def selection(
    model,
    grouped_layer_names,
    sparse_weights_path: str,
    candidates,
    num_survive: int,
    calibration_data,
    num_tokens: int,
    fitness_fn: str = "ppl",
    target_logits: Optional[Union[List[torch.Tensor], Tuple[torch.Tensor]]] = None,
    finetune_data_list: Optional[torch.Tensor] = None 
):
    calibration_minibatch = []
    minibatch_ids = []
    target_logits_minibatch = []
    tokens_used = 0
    while tokens_used < num_tokens:  # generate minibatch with exactly num_tokens tokens
        minibatch_id = random.randint(0, len(calibration_data) - 1)
        if minibatch_id in minibatch_ids:  # avoid duplicates
            continue
        minibatch_ids.append(minibatch_id)
        if tokens_used + calibration_data[minibatch_id].shape[1] > num_tokens:
            calibration_minibatch.append(calibration_data[minibatch_id][:, : num_tokens - tokens_used])
            if fitness_fn == "kl":
                target_logits_minibatch.append(target_logits[minibatch_id][:, : num_tokens - tokens_used])
            elif fitness_fn == "sparse_kl":
                target_logits_minibatch.append(
                    (
                        target_logits[minibatch_id][0][:, : num_tokens - tokens_used],  # TopK indices
                        target_logits[minibatch_id][1][:, : num_tokens - tokens_used],  # TopK values
                    )
                )
            tokens_used = num_tokens
        else:
            calibration_minibatch.append(calibration_data[minibatch_id])
            if fitness_fn in ["kl", "sparse_kl"]:
                target_logits_minibatch.append(target_logits[minibatch_id])
            tokens_used += calibration_data[minibatch_id].shape[1]

    if len(target_logits_minibatch) == 0:
        target_logits_minibatch = None

    fitnesses = []
    fitnesses_before = []
    for candidate in candidates:
        model_copy = None
        model_copy = copy.deepcopy(model)
        # model_copy = model
        load_layers(model_copy, grouped_layer_names, candidate, sparse_weights_path)
        # shrink(model_copy, is_transformers=True)
        # if finetune_data_list is not None:
        #     fitness_before = compute_fitness(model_copy, calibration_minibatch, fitness_fn, target_logits_minibatch,
        #                             finetune_data_list=None)
        #     fitnesses_before.append(fitness_before)
        #     print("before fintune fitness is: ", fitness_before)
        fitness = compute_fitness(model_copy, calibration_minibatch, fitness_fn, target_logits_minibatch,
                                    finetune_data_list=finetune_data_list)
        if finetune_data_list is not None:
            print("after fintune fitness is: ", fitness)
        fitnesses.append(fitness)
        
        for param in model_copy.parameters():
            if param.grad is not None:
                param.grad = None  # Clear gradients
        del model_copy
        torch.cuda.empty_cache()

    # Keep only best
    best_ids = np.argsort(fitnesses)[:num_survive]
    best_fitness_before = None
    print("all finetune fitnesses are:", fitnesses)
    print("saved id is:", best_ids)

    if len(fitnesses_before) > 0:
        best_fitness_before = [fitnesses_before[i] for i in best_ids]
    return [candidates[i] for i in best_ids], [fitnesses[i] for i in best_ids], best_fitness_before


def parse_args():
    parser = argparse.ArgumentParser()
    # Model params
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="The name or path to the model being pruned",
    )
    parser.add_argument(
        "--sparse_finetuned_weight_path",
        type=str,
        help="The path to the sparse model after finetuning",
    )
    parser.add_argument(
        "--sparse_config_path",
        type=str,
        help="The  path to the sparse model config",
    )
    parser.add_argument(
        "--gradual_database",
        type=str,
        help="The  path to the sparse model config",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="The name or path to the tokenizer. By default use model tokenizer.",
    )
    # Data params
    parser.add_argument(
        "--calibration_data",
        type=str,
        required=True,
        help="The name or dataset or path used for calibration.",
    )
    parser.add_argument("--calibration_tokens", default=524288, type=int, help="Number of tokens for calibration.")
    parser.add_argument("--training_tokens", default=524288, type=int, help="Number of tokens for calibration.")
    parser.add_argument(
        "--calibration_sequence_length", default=None, type=int, help="Length of calibration sequences."
    )
    parser.add_argument(
        "--eval_datasets",
        nargs="+",
        type=str,
        default=["fineweb_edu", "wikitext2", "c4"],
        help="Datasets used for evaluation",
    )
    parser.add_argument("--eval_every", default=1, type=int, help="Eval every # generations.")
    parser.add_argument("--eval_tokens", default=524288, type=int, help="Number of tokens for evaluation.")
    parser.add_argument("--eval_sequence_length", default=None, type=int, help="Length of evaluation sequences.")
    parser.add_argument("--fitness_fn", choices=["ppl", "kl", "sparse_kl"], default="kl", help="Fitness function.")
    # Logging params
    parser.add_argument("--log_wandb", default=False, action="store_true", help="Whether to log to W&B")
    parser.add_argument(
        "--save_freq",
        type=int,
        default=100,
        help="steps to save",
    )
    # Evolutionary Search params
    parser.add_argument("--generations", type=int, required=True, help="Number of generations in evolutionary search")
    parser.add_argument("--offspring", type=int, required=True, help="Number of offspring generated in each generation")
    parser.add_argument(
        "--target_level",
        type=float,
        required=True,
        help="Base level for all layers. If no integer, initialize random with this average",
    )
    parser.add_argument(
        "--gradual_target_level",
        type=float,
        help="Base level for all layers. If no integer, initialize random with this average",
    )
    parser.add_argument("--sparse_weights_path", type=str, required=True, help="Path to quantized weights")
    parser.add_argument(
        "--survivors_per_selection",
        type=int,
        nargs="+",
        required=True,
        help="Number of survivors after each stage of selection",
    )
    parser.add_argument(
        "--tokens_per_selection",
        type=int,
        nargs="+",
        required=True,
        help="Number of calibration tokens at each stage of selection",
    )
    parser.add_argument(
        "--training_tokens_per_selection",
        type=int,
        nargs="+",
        required=True,
        help="Number of calibration tokens at each stage of selection",
    )
    parser.add_argument(
        "--initially_generated",
        type=int,
        help="Only for non-integer initial level: Number of search points generated in the beginning; fittest are selected for the initial population",
    )
    parser.add_argument(
        "--initial_tokens",
        type=int,
        help="Only for non-integer initial level: Number of calibration tokens used for the initial generation",
    )
    parser.add_argument(
        "--kl_topk",
        type=int,
        default=10,
        help="TopK logits in KL-divergence (for sparse_kl fitness function)",
    )
    # TODO infer automatically from configuration
    parser.add_argument(
        "--step_size",
        type=int,
        default=1,
        help="Step size between adjacent levels",
    )
    # Misc params
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "float32", "bfloat16"],
        help="dtype to load the model.",
    )
    parser.add_argument("--seed", default=0, type=int, help="Random seed.")
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default=None,
        choices=["eager", "sdpa", "flash_attention_2"],
        help="Attention implementation: eager, sdpa, or flash_attention_2",
    )
    parser.add_argument("--use_fast_tokenizer", action="store_true", help="Whether to use fast tokenizer.")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    # Sanity checks
    assert len(args.survivors_per_selection) == len(args.tokens_per_selection), "Must have same number of stages"
    assert args.survivors_per_selection[-1] == 1, "Last stage should have only one survivor"
    if int(args.target_level) != args.target_level:
        assert args.initially_generated is not None, "Need initially_generated for non-integer initial level"
        assert args.initial_tokens is not None, "Need initial_tokens for non-integer initial level"
    # Fix seed
    fix_seed(args.seed)
    # Init W&B logger
    if args.log_wandb:
        assert has_wandb, "`wandb` not installed, try pip install `wandb`"
        wandb.init(config=args)
    # init device
    device = f"cuda"
    if args.dtype != "auto":
        args.dtype = getattr(torch, args.dtype)
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        device_map="auto",
        low_cpu_mem_usage=True,
        torch_dtype=args.dtype,
        attn_implementation=args.attn_implementation,
    )
    model.config.use_cache = False  # do not use cache
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name or args.model_name_or_path, use_fast=args.use_fast_tokenizer
    )
    # Load calibration data
    args.calibration_sequence_length = args.calibration_sequence_length or min(
        model.config.max_position_embeddings, 8192
    )
    calibration_data = get_data(
        args.calibration_data, args.calibration_tokens, args.calibration_sequence_length, tokenizer, train=True
    )
    training_data = get_data(
        args.calibration_data,
        args.training_tokens, args.calibration_sequence_length, tokenizer, train=True, seed=123
    )
    # Load eval datasets
    args.eval_sequence_length = args.eval_sequence_length or min(model.config.max_position_embeddings, 8192)
    eval_datasets = []
    for eval_dataset_name in args.eval_datasets:
        eval_datasets.append(
            get_data(
                eval_dataset_name,
                args.eval_tokens,  # ignored for WikiText2 and C4
                args.eval_sequence_length,
                tokenizer,
                train=False,
            )
        )
    target_logits = []
    if args.fitness_fn == "kl":
        # Compute target logits (calibration)
        for i in trange(0, len(calibration_data), desc="Computing target logits (calib)", leave=False):
            with torch.no_grad():
                target_logits.append(model(calibration_data[i].to(device)).logits.cpu())

    elif args.fitness_fn == "sparse_kl":
        # Compute target logits (calibration)
        for i in trange(0, len(calibration_data), desc="Computing target logits (calib)", leave=False):
            with torch.no_grad():
                logits = model(calibration_data[i].to(device)).logits.cpu()
                topk_values, topk_indices = logits.topk(k=args.kl_topk, dim=-1)
                target_logits.append((topk_values, topk_indices))

    # Prepare layers and initial state
    layer_names = []
    for layer_name in os.listdir(args.sparse_weights_path):
        if os.path.isdir(os.path.join(args.sparse_weights_path, layer_name)):
            layer_names.append(layer_name)
    # Sort layers
    layer_names = sorted(layer_names, key=layer_order_fn)
    # Group layers
    grouped_layer_names = group_layers(model, layer_names, "name") # Only allow mutations between same type of layers
    print(grouped_layer_names)
    num_groups = len(grouped_layer_names)
    # Loaded state
    model.state = [[None] * len(names) for names in grouped_layer_names]

    target_bits = 0
    quantizable_weights = 0
    for group_id in range(len(grouped_layer_names)):
        for i, layer_name in enumerate(grouped_layer_names[group_id]):
            target_bits += int(model.get_submodule(layer_name).weight.numel() * args.target_level)
            quantizable_weights += model.get_submodule(layer_name).weight.numel()

    # Initialization
    parent = None
    ori_parent = None
    if args.gradual_target_level is not None:
        if args.sparse_config_path is not None:
            config = load_sparse_config(args.sparse_config_path)
            parent = [[], []]
            for key, val in config.items():
                if key in grouped_layer_names[0]:
                    parent[0].append(int(val))
                else:
                    parent[1].append(int(val))
            ori_parent = copy.deepcopy(parent)
            print("original parent:", ori_parent)
        resdual = (int(args.gradual_target_level) - int(args.target_level)) * len(grouped_layer_names[0])
        # random add the resdual value to new parent
        for i in range(resdual):
            while True:
                idx = random.randint(0, len(parent[0]) - 1)
                if parent[0][idx] + 1 > 9:
                    continue
                else:
                    parent[0][idx] += 1
                    break
            while True:
                idx = random.randint(0, len(parent[1]) - 1)
                if parent[1][idx] + 1 > 9:
                    continue
                else:
                    parent[1][idx] += 1
                    break
        print("New parent:", parent)
    else:
        parent = [[int(args.target_level) for _ in names] for names in grouped_layer_names]
    train_fitness = float("inf")
#     parent = [[5, 9, 9, 8, 6, 5, 6, 7, 4, 5, 7, 6, 5, 5, 5, 3, 5, 5, 7, 5, 4, 7, 6, 5, 7, 4, 3, 5, 4, 5, 7, 4, 4, 5, 3, 3, 3, 5, 3, 5, 3, 5, 4, 5, 4, 4, 1, 5],
# [3, 5, 6, 7, 5, 7, 7, 5, 7, 4, 5, 5, 8, 3, 9, 5, 5, 1, 5, 6, 4, 4, 5, 4, 7, 2, 6, 5, 6, 1, 4, 5, 3, 3, 5, 6, 6, 4, 6, 10, 3, 3, 5, 5, 3, 6, 5, 6]]

    if args.sparse_config_path is not None:
        model = load_compressed_finetuned_weights(model, args.sparse_weights_path, 
                                                args.sparse_finetuned_weight_path, args.sparse_config_path)
        args.sparse_weights_path = args.gradual_database
    if ori_parent is not None:
        model.state = copy.deepcopy(ori_parent)
    
    log_dict = {}
    for generation in range(args.generations):
        load_layers(model, grouped_layer_names, parent, args.sparse_weights_path)

        print(f"Generation {generation + 1}/{args.generations}")
        print(f"Current search point:")
        for group in parent:
            print(group)
        print(f"Train fitness: {train_fitness:.4e}")

        
        # Evaluate current search point
        if generation % args.eval_every == 0:
            for eval_dataset_name, eval_dataset in zip(args.eval_datasets, eval_datasets):
                ppl_eval = compute_perplexity(model, eval_dataset)
                print(f"{eval_dataset_name}: {ppl_eval:.2f}")
                log_dict[f"ppl_eval/{eval_dataset_name}"] = ppl_eval
            ppl_train = compute_perplexity(model, calibration_data)
            print(f"ppl_train: {ppl_train:.2f}")
            log_dict["ppl_train"] = ppl_train
        if args.log_wandb:
            wandb.log(log_dict)

        offspring_list = []

        while len(offspring_list) < args.offspring:
            offspring = copy.deepcopy(parent)
            # mutate offspring
            num_flips = min(random.randint(1, 3), random.randint(1, 3))  # bias towards lower values


            for _ in range(num_flips):
                # Select random group, proportional to the number of layers in a group
                group_id = random.choices(
                    range(len(grouped_layer_names)), weights=[len(g) for g in grouped_layer_names]
                )[0]
                group = grouped_layer_names[group_id]

                # Positions where compression can be decreased
                decr_ids = []
                for i, layer_name in enumerate(group):
                    level = offspring[group_id][i]
                    if os.path.exists(
                        os.path.join(args.sparse_weights_path, layer_name, f"{level - args.step_size}.pth")
                    ):
                        decr_ids.append(i)
                assert len(decr_ids) > 0, "There is no way to decrease compression level."
                decr_id = random.choice(decr_ids)
                # Positions where compression can be increased
                incr_ids = []
                for i, layer_name in enumerate(group):
                    level = offspring[group_id][i]
                    if os.path.exists(
                        os.path.join(args.sparse_weights_path, layer_name, f"{level + args.step_size}.pth")
                    ):
                        incr_ids.append(i)
                assert len(incr_ids) > 0, "There is no way to increase compression level."
                incr_id = random.choice(incr_ids)

                offspring[group_id][decr_id] -= args.step_size
                offspring[group_id][incr_id] += args.step_size

            if offspring in offspring_list or offspring in [parent]:  # Avoid duplicates
                continue
            offspring_list.append(offspring)
        training_data_ = None
        kl_full_finetune = []
        for offspring in offspring_list:
            model_copy = None
            model_copy = copy.deepcopy(model)
            load_layers(model_copy, grouped_layer_names, offspring, args.sparse_weights_path)
            fitness = compute_fitness(model_copy, calibration_data, args.fitness_fn, target_logits,
                                        finetune_data_list=training_data)
            # if finetune_data_list is not None:
            #     print("after fintune fitness is: ", fitness)
            kl_full_finetune.append(fitness)
        print(kl_full_finetune)


        for i, (num_survive, num_tokens, num_train_tokens) in enumerate(zip(args.survivors_per_selection, args.tokens_per_selection, args.training_tokens_per_selection)):
            # if num_survive == args.survivors_per_selection[-1]:
                # if parent not in offspring_list:  # Elitist EA
                #     offspring_list.append(parent)
            # if i == len(args.survivors_per_selection) - 1:
            #     training_data_ = training_data
            training_token_used = 0
            minibatch_ids = []
            training_minibatch = []
            while training_token_used < num_train_tokens:
                minibatch_id = random.randint(0, len(training_data) - 1)
                if minibatch_id in minibatch_ids:  # avoid duplicates
                    continue
                minibatch_ids.append(minibatch_id)
                training_minibatch.append(training_data[minibatch_id])
                training_token_used += training_data[minibatch_id].shape[1]
            # training_minibatch=None
            offspring_list, train_fitnesses, train_fitnesses_before_finetune = selection(
                model=model,
                grouped_layer_names=grouped_layer_names,
                sparse_weights_path=args.sparse_weights_path,
                candidates=offspring_list,
                num_survive=num_survive,
                calibration_data=calibration_data,
                num_tokens=num_tokens,
                fitness_fn=args.fitness_fn,
                target_logits=target_logits,
                finetune_data_list=training_minibatch,
            )
        # In the end we have lists with a single element (only 1 survivor in last selection step)
        train_fitness = train_fitnesses[0]
        parent = offspring_list[0]
        print(f"Train fitnesses: {train_fitness:.2e}")
        log_dict["train_fitness"] = train_fitness
        if train_fitnesses_before_finetune is not None:
            train_fitness_before_finetune = train_fitnesses_before_finetune[0]
            print(f"Train fitnesses before finetune: {train_fitness_before_finetune:.2e}")
            log_dict["train_fitness_before_finetune"] = train_fitness_before_finetune
        # Save configuration
        if (generation + 1) % args.save_freq == 0:
            model_name = args.model_name_or_path.split("/")[-1]
            configuration_name = f"evo-{args.fitness_fn}-configuration-{args.target_level}-{model_name}-{generation}step.txt"
            with open(os.path.join("./", configuration_name), "w") as f:
                for i in range(num_groups):
                    f.write(
                        "\n".join([f"{layer_name}: {level}" for layer_name, level in zip(grouped_layer_names[i], parent[i])])
                    )
                    if i != num_groups - 1:
                        f.write("\n")
    # Log final configuration
    print("Final configuration:")
    for group in parent:
        print(group)
    # Final evaluation
    for eval_dataset_name, eval_dataset in zip(args.eval_datasets, eval_datasets):
        ppl_eval = compute_perplexity(model, eval_dataset)
        print(f"{eval_dataset_name}: {ppl_eval:.2f}")
        log_dict[f"ppl_eval/{eval_dataset_name}"] = ppl_eval
    ppl_train = compute_perplexity(model, calibration_data)
    print(f"ppl_train: {ppl_train:.2f}")
    log_dict["ppl_train"] = ppl_train
    if args.log_wandb:
        wandb.log(log_dict)


if __name__ == "__main__":
    time_start = time.time()
    main()
    time_end = time.time()
    print('time cost: ', time_end - time_start,'s')
