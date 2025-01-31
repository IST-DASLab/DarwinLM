import os
import argparse
import time

import torch
import torch.nn.functional as F
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Union, Optional
import datetime

from src import dist_utils
from src.data_utils import get_data
from src.struct_pruner import ZipLMPruner
from src.struct_cap import SparseStructCAP
from src.common_utils import read_yaml_config
from utils import shrink, get_parameter_number

def load_compressed_weights(
    model: AutoModelForCausalLM,
    compressed_weights_path: Union[str, os.PathLike],
    compressed_config_path: Optional[str] = None,
    default_level: int = 0,
):
    # Load weights from configuration if provided
    if compressed_config_path:
        with open(os.path.join(compressed_config_path), "r") as f:
            for line in f:
                layer_name, level = line.split(":")
                layer = model.get_submodule(layer_name.strip(" "))
                orig_dtype = layer.weight.dtype
                layer.weight.data = torch.load(
                    os.path.join(compressed_weights_path, layer_name, f"{int(level)}.pth"),
                    map_location=layer.weight.device,
                ).to(orig_dtype)
    # Otherwise load uniform configuration
    else:
        for layer_name in sorted(os.listdir(compressed_weights_path)):
            if not os.path.isdir(os.path.join(compressed_weights_path, layer_name)):
                continue
            layer = model.get_submodule(layer_name.strip(" "))
            orig_dtype = layer.weight.dtype
            layer.weight.data = torch.load(
                os.path.join(compressed_weights_path, layer_name, f"{default_level}.pth"),
                map_location=layer.weight.device,
            ).to(orig_dtype)
    return model

def load_compressed_finetuned_weights(
    model: AutoModelForCausalLM,
    compressed_weights_path: Union[str, os.PathLike],
    finetuned_weight_path: Union[str, os.PathLike],
    compressed_config_path: Optional[str] = None,
    default_level: int = 0,
):
    model = load_compressed_weights(model, compressed_weights_path, compressed_config_path)
    shrink(model, is_transformers=True)
    print("Model shrinks, the remaining parameters are: ", get_parameter_number(model))
    state = torch.load(finetuned_weight_path, map_location="cpu")
    model.load_state_dict(state)
    print("Finetuned weights are loaded!")
    return model

def load_sparse_config(compressed_config_path):
    res = {}
    with open(os.path.join(compressed_config_path), "r") as f:
            for line in f:
                layer_name, level = line.split(":")
                res[layer_name] = level
    return res

def parse_args():
    parser = argparse.ArgumentParser(description="One-shot pruning with parallel OBC.")
    # Model params
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="The name or path to the model being pruned",
    )
    parser.add_argument(
        "--sparse_weight_path",
        type=str,
        help="The path to the sparse model database",
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
        "--tokenizer_name",
        type=str,
        default=None,
        help="The name or path to the tokenizer. By default use model tokenizer.",
    )
    
    parser.add_argument(
        "--prunable_modules",
        type=str,
        required=True,
        help="Regex for modules to prune",
    )
    parser.add_argument(
        "--pre_block_modules",
        nargs="+",
        type=str,
        required=True,
        help="Names of modules before transformer blocks",
    )
    parser.add_argument(
        "--block_modules",
        type=str,
        required=True,
        help="Name of transformer modules",
    )
    # Data params
    parser.add_argument(
        "--calibration_data",
        type=str,
        required=True,
        help="The name or dataset or path used for calibration.",
    )
    parser.add_argument("--calibration_tokens", default=int(2**23), type=int, help="Number of tokens for calibration.")
    parser.add_argument(
        "--calibration_sequence_length", default=None, type=int, help="Length of calibration sequences."
    )
    # Sparsification params
    parser.add_argument("--rel_damp", type=float, default=1e-2)
    parser.add_argument("--cols_removed_attn", type=int, nargs="+", default=None)
    parser.add_argument("--cols_removed_mlp", type=int, nargs="+", default=None)
    parser.add_argument("--mlp_prune_name", type=str, required=True)
    parser.add_argument("--attn_prune_name", type=str, required=True)
    # Save params
    parser.add_argument("--save_dir", type=str, required=True, help="where to save sparse model.")
    # Misc params
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "float32", "bfloat16"],
        help="dtype to load the model.",
    )
    parser.add_argument("--seed", default=0, type=int, help="random seed.")
    parser.add_argument(
        "--low_cpu_mem_usage", action="store_true", help="whether to load model with the use of `low_cpu_mem_usage`"
    )
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default=None,
        choices=["eager", "sdpa", "flash_attention_2"],
        help="Attention implementation: eager, sdpa, or flash_attention_2",
    )
    parser.add_argument("--cpu_offload_modules", action="store_true", help="whether to offload modules to CPU.")
    parser.add_argument("--cpu_offload_activations", action="store_true", help="whether to offload activations to CPU.")
    parser.add_argument("--verbose", action="store_true", help="whether to log progress.")
    
    # For StructCAP
    parser.add_argument("--use_cap", action="store_true", help="whether to use structCAP.")
    parser.add_argument("--block_size", default="1 d", type=str)
    parser.add_argument("--group_size_config_path", default=None, type=str)
    parser.add_argument("--rows_in_parallel", default=None, type=int)
    parser.add_argument("--grad_sparsity", default=0.0, type=float)
    parser.add_argument(
        "--find_block_size", default=False, action="store_true", help="Whether to search for block size."
    )
    parser.add_argument("--module_regex", type=str)


    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    # Distributed init
    if dist.is_available():
        dist.init_process_group(backend="nccl", init_method="env://", timeout=datetime.timedelta(hours=2.0))
    world_size = dist_utils.get_world_size()
    rank = dist_utils.get_rank()
    # init device
    device = f"cuda:{rank}"
    if args.dtype != "auto":
        args.dtype = getattr(torch, args.dtype)
    # Model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        torch_dtype=args.dtype,
        low_cpu_mem_usage=args.low_cpu_mem_usage,
        attn_implementation=args.attn_implementation,
    )
    config_dict = None
    if args.sparse_finetuned_weight_path is not None:
        config_dict = load_sparse_config(args.sparse_config_path)
        model = load_compressed_finetuned_weights(model, args.sparse_weight_path,
                                                  args.sparse_finetuned_weight_path,
                                                  args.sparse_config_path)

    if not args.cpu_offload_modules:
        model = model.to(device)
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name or args.model_name_or_path, use_fast=False)
    # Load calibration data
    args.calibration_sequence_length = args.calibration_sequence_length or min(
        model.config.max_position_embeddings, 8192
    )
    calibration_data = get_data(
        args.calibration_data, args.calibration_tokens, args.calibration_sequence_length, tokenizer, train=True
    )
    # take slice (if running on multiple workers)
    print("len of data is:", len(calibration_data))
    if dist_utils.is_dist_available_and_initialized():
        num_seq_per_rank = len(calibration_data) // world_size
        calibration_data = calibration_data[rank * num_seq_per_rank : (rank + 1) * num_seq_per_rank]
    if args.use_cap:
        calibration_data = [([], {"input_ids": input_ids}, input_ids) for input_ids in calibration_data]
    else:
        calibration_data = [([], {"input_ids": input_ids}) for input_ids in calibration_data]
    dist.barrier()
    # Pruner
    if not args.use_cap:
        pruner = ZipLMPruner(
            model,
            calibration_data,
            prunable_modules=args.prunable_modules,
            pre_block_modules=args.pre_block_modules,
            block_modules=args.block_modules,
            save_dir=args.save_dir,
            rel_damp=args.rel_damp,
            device=device,
            cpu_offload_modules=args.cpu_offload_modules,
            cpu_offload_activations=args.cpu_offload_activations,
            verbose=args.verbose,
            mlp_prune_name=args.mlp_prune_name,
            attn_prune_name=args.attn_prune_name,
            sparse_config=config_dict
        )
    else:
        # Define loss function
        def loss_fn(outputs, inputs):
            shift_logits = outputs.logits[:, :-1].contiguous()
            shift_labels = inputs[:, 1:]
            return F.cross_entropy(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))
        if args.group_size_config_path:
                group_size = read_yaml_config(args.group_size_config_path)["group_size"]
        # Calibration batch size is equal to grad_accum_steps

        pruner = SparseStructCAP(
                model,
                module_regex=args.module_regex,
                data_loader=calibration_data,
                loss_fn=loss_fn,
                device=device,
                damp=args.rel_damp,
                grad_sparsity=args.grad_sparsity,
                num_samples=len(calibration_data),
                block_size=512,
                group_size=group_size,
                find_block_size=args.find_block_size,
                mlp_prune_name=args.mlp_prune_name,
                attn_prune_name=args.attn_prune_name,
                sparse_config=config_dict,
                save_dir=args.save_dir
            )
    # TODO: add timing to compute cols_removed_attn and cols_removed_mlp automatically
    
    # Prepare save dir
    if dist_utils.is_main():
        os.makedirs(args.save_dir, exist_ok=True)
        torch.save(
            {"cols_removed_attn": args.cols_removed_attn, "cols_removed_mlp": args.cols_removed_mlp},
            os.path.join(args.save_dir, "metadata.pth"),
        )
    dist.barrier()
    t1 = time.perf_counter()

    
    pruner.struct_prune(args.cols_removed_attn, args.cols_removed_mlp)
    t2 = time.perf_counter()
    dist_utils.print_on_main(f"Pruning took {(t2 - t1)} s.")


if __name__ == "__main__":
    main()
