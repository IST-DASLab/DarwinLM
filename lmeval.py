# Copied from https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/__main__.py

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Union, Optional
import numpy as np
import torch
from transformers import AutoModelForCausalLM

from lm_eval import evaluator, utils
from lm_eval.api.registry import ALL_TASKS
from lm_eval.tasks import TaskManager
# from lm_eval.tasks import include_path, initialize_tasks
from lm_eval.utils import make_table

try:
    import wandb

    has_wandb = True
except ModuleNotFoundError:
    has_wandb = False

from utils import shrink, get_parameter_number, NoAttention, NoIntermediate


def _handle_non_serializable(o):
    if isinstance(o, np.int64) or isinstance(o, np.int32):
        return int(o)
    elif isinstance(o, set):
        return list(o)
    else:
        return str(o)


def parse_eval_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--model", "-m", default="hf", help="Name of model e.g. `hf`")
    parser.add_argument(
        "--tasks",
        "-t",
        default=None,
        metavar="task1,task2",
        help="To get full list of tasks, use the command lm-eval --tasks list",
    )
    parser.add_argument(
        "--model_args",
        "-a",
        default="",
        help="Comma separated string arguments for model, e.g. `pretrained=EleutherAI/pythia-160m,dtype=float32`",
    )
    parser.add_argument(
        "--num_fewshot",
        "-f",
        type=int,
        default=None,
        metavar="N",
        help="Number of examples in few-shot context",
    )
    parser.add_argument(
        "--batch_size",
        "-b",
        type=str,
        default=1,
        metavar="auto|auto:N|N",
        help="Acceptable values are 'auto', 'auto:N' or N, where N is an integer. Default 1.",
    )
    parser.add_argument(
        "--max_batch_size",
        type=int,
        default=None,
        metavar="N",
        help="Maximal batch size to try with --batch_size auto.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (e.g. cuda, cuda:0, cpu).",
    )
    parser.add_argument(
        "--output_path",
        "-o",
        default=None,
        type=str,
        metavar="DIR|DIR/file.json",
        help="The path to the output file where the result metrics will be saved. If the path is a directory and log_samples is true, the results will be saved in the directory. Else the parent directory will be used.",
    )
    parser.add_argument(
        "--limit",
        "-L",
        type=float,
        default=None,
        metavar="N|0<N<1",
        help="Limit the number of examples per task. " "If <1, limit is a percentage of the total number of examples.",
    )
    parser.add_argument(
        "--use_cache",
        "-c",
        type=str,
        default=None,
        metavar="DIR",
        help="A path to a sqlite db file for caching model responses. `None` if not caching.",
    )
    parser.add_argument("--decontamination_ngrams_path", default=None)  # TODO: not used
    parser.add_argument(
        "--check_integrity",
        action="store_true",
        help="Whether to run the relevant part of the test suite for the tasks.",
    )
    parser.add_argument(
        "--write_out",
        "-w",
        action="store_true",
        default=False,
        help="Prints the prompt for the first few documents.",
    )
    parser.add_argument(
        "--log_samples",
        "-s",
        action="store_true",
        default=False,
        help="If True, write out all model outputs and documents for per-sample measurement and post-hoc analysis. Use with --output_path.",
    )
    parser.add_argument(
        "--show_config",
        action="store_true",
        default=False,
        help="If True, shows the the full config of all tasks at the end of the evaluation.",
    )
    parser.add_argument(
        "--include_path",
        type=str,
        default=None,
        metavar="DIR",
        help="Additional path to include if there are external tasks to include.",
    )
    parser.add_argument(
        "--gen_kwargs",
        default=None,
        help=("String arguments for model generation on greedy_until tasks," " e.g. `temperature=0,top_k=0,top_p=0`."),
    )
    parser.add_argument(
        "--verbosity",
        "-v",
        type=str.upper,
        default="INFO",
        metavar="CRITICAL|ERROR|WARNING|INFO|DEBUG",
        help="Controls the reported logging error level. Set to DEBUG when testing + adding new task configurations for comprehensive log output.",
    )
    # Sparsification params
    parser.add_argument(
        "--sparse_weights_path",
        type=str,
        default=None,
        help="Path to sparse weights",
    )
    parser.add_argument(
        "--sparse_config_path",
        type=str,
        default=None,
        help="Path to sparsification config",
    )
    parser.add_argument(
        "--kv_ignore",
        action="store_true",
        help="Ignore the k,v matrices during pruning",
    )
    parser.add_argument(
        "--model_shrink",
        action="store_true",
        help="Shrink the model or not",
    )
    parser.add_argument(
        "--sparse_default_level",
        type=int,
        default=0,
        help="Default sparsity level",
    )
    # Logging params
    parser.add_argument("--log_wandb", default=False, action="store_true", help="Whether to log to W&B")

    return parser.parse_args()


# Compressed model loader
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
                if isinstance(model.get_submodule(".".join(layer_name.split('.')[:4]).strip(" ")),
                               NoAttention) or isinstance(model.get_submodule(".".join(layer_name.split('.')[:4]).strip(" ")),
                               NoIntermediate):
                               continue
                
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


def cli_evaluate(args: Union[argparse.Namespace, None] = None) -> None:
    if not args:
        # we allow for args to be passed externally, else we parse them ourselves
        args = parse_eval_args()

    # Backup original from_pretrained
    from_pretrained_orig = AutoModelForCausalLM.from_pretrained
    from_pretrained_overriden = from_pretrained_orig
    # Override from_pretrained
    if args.sparse_weights_path:
        sparse_weights_path = args.sparse_weights_path
        sparse_config_path = args.sparse_config_path
        kv_ignore = args.kv_ignore
        default_level = args.sparse_default_level
        model_shrink = args.model_shrink

        # Define new init
        def from_pretrained_overriden(*args, **kwargs):
            model = from_pretrained_orig(*args, **kwargs)
            
            model = load_compressed_weights(model, sparse_weights_path, sparse_config_path)
            if model_shrink:
                shrink(model, is_transformers=True, kv_ignore=kv_ignore)
                print("Model shrinks, the remaining parameters are: ", get_parameter_number(model))

            return model


    # Override init
    AutoModelForCausalLM.from_pretrained = staticmethod(from_pretrained_overriden)

    eval_logger = utils.eval_logger
    eval_logger.setLevel(getattr(logging, f"{args.verbosity}"))
    eval_logger.info(f"Verbosity set to {args.verbosity}")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # initialize_tasks(args.verbosity)

    if args.log_wandb:
        assert has_wandb, "`wandb` not installed, try pip install `wandb`"
        wandb.init(config=args)

    if args.limit:
        eval_logger.warning(
            " --limit SHOULD ONLY BE USED FOR TESTING." "REAL METRICS SHOULD NOT BE COMPUTED USING LIMIT."
        )
    # if args.include_path is not None:
    #     eval_logger.info(f"Including path: {args.include_path}")
    #     include_path(args.include_path)
    task_manager = TaskManager(args.verbosity, include_path=args.include_path)

    if args.tasks is None:
        task_names = ALL_TASKS
    elif args.tasks == "list":
        eval_logger.info("Available Tasks:\n - {}".format("\n - ".join(sorted(task_manager.all_tasks))))
        sys.exit()
    else:
        if os.path.isdir(args.tasks):
            import glob

            task_names = []
            yaml_path = os.path.join(args.tasks, "*.yaml")
            for yaml_file in glob.glob(yaml_path):
                config = utils.load_yaml_config(yaml_file)
                task_names.append(config)
        else:
            tasks_list = args.tasks.split(",")
            task_names = task_manager.match_tasks(tasks_list)
            for task in [task for task in tasks_list if task not in task_names]:
                if os.path.isfile(task):
                    config = utils.load_yaml_config(task)
                    task_names.append(config)
            task_missing = [
                task for task in tasks_list if task not in task_names and "*" not in task
            ]  # we don't want errors if a wildcard ("*") task name was used

            if task_missing:
                missing = ", ".join(task_missing)
                eval_logger.error(
                    f"Tasks were not found: {missing}\n"
                    f"{utils.SPACING}Try `lm-eval --tasks list` for list of available tasks",
                )
                raise ValueError(
                    f"Tasks not found: {missing}. Try `lm-eval --tasks list` for list of available tasks, or '--verbosity DEBUG' to troubleshoot task registration issues."
                )

    if args.output_path:
        path = Path(args.output_path)
        # check if file or 'dir/results.json' exists
        if path.is_file() or Path(args.output_path).joinpath("results.json").is_file():
            eval_logger.warning(f"File already exists at {path}. Results will be overwritten.")
            output_path_file = path.joinpath("results.json")
            assert not path.is_file(), "File already exists"
        # if path json then get parent dir
        elif path.suffix in (".json", ".jsonl"):
            output_path_file = path
            path.parent.mkdir(parents=True, exist_ok=True)
            path = path.parent
        else:
            path.mkdir(parents=True, exist_ok=True)
            output_path_file = path.joinpath("results.json")
    elif args.log_samples and not args.output_path:
        assert args.output_path, "Specify --output_path"

    eval_logger.info(f"Selected Tasks: {task_names}")

    results = evaluator.simple_evaluate(
        model=args.model,
        model_args=args.model_args,
        tasks=task_names,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        max_batch_size=args.max_batch_size,
        device=args.device,
        use_cache=args.use_cache,
        limit=args.limit,
        task_manager=task_manager,
        # decontamination_ngrams_path=args.decontamination_ngrams_path,
        check_integrity=args.check_integrity,
        write_out=args.write_out,
        log_samples=args.log_samples,
        gen_kwargs=args.gen_kwargs,
    )

    if results is not None:
        if args.log_samples:
            samples = results.pop("samples")
        dumped = json.dumps(results, indent=2, default=_handle_non_serializable, ensure_ascii=False)
        if args.show_config:
            print(dumped)

        batch_sizes = ",".join(map(str, results["config"]["batch_sizes"]))

        if args.output_path:
            output_path_file.open("w").write(dumped)

            if args.log_samples:
                for task_name, config in results["configs"].items():
                    output_name = "{}_{}".format(re.sub("/|=", "__", args.model_args), task_name)
                    filename = path.joinpath(f"{output_name}.jsonl")
                    samples_dumped = json.dumps(
                        samples[task_name],
                        indent=2,
                        default=_handle_non_serializable,
                        ensure_ascii=False,
                    )
                    filename.open("w").write(samples_dumped)

        print(
            f"{args.model} ({args.model_args}), gen_kwargs: ({args.gen_kwargs}), limit: {args.limit}, num_fewshot: {args.num_fewshot}, "
            f"batch_size: {args.batch_size}{f' ({batch_sizes})' if batch_sizes else ''}"
        )
        print(make_table(results))
        if "groups" in results:
            print(make_table(results, "groups"))

        if args.log_wandb:
            wandb.log(results)


cli_evaluate()
