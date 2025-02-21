# 🚀 DarwinLM: Evolutionary Structured Pruning for Language Models

🌟 [ArXiv Preprint](https://arxiv.org/pdf/2502.07780)

This repository contains the implementation of evolutionary structured pruning for language models, as introduced in our paper. The approach optimizes model efficiency while maintaining strong performance.

We provide six model variants on huggingface:
- Models after pruning and searching: [**2.7B-Pruned Model**](https://huggingface.co/Shengkun/DarwinLM-2.7B-Pruned), [**4.6B-Pruned Model**](https://huggingface.co/Shengkun/DarwinLM-4.6B-Pruned), [**8.4B-Pruned Model**](https://huggingface.co/Shengkun/DarwinLM-8.4B-Pruned)
- Models after post-training: [**2.7B Model**](https://huggingface.co/Shengkun/DarwinLM-2.7B), [**4.6B Model**](https://huggingface.co/Shengkun/DarwinLM-4.6B), [**8.4B Model**](https://huggingface.co/Shengkun/DarwinLM-8.4B)



---


## ⚙️ Installation

To set up the environment, ensure you have the necessary dependencies installed:

```bash
conda env create -f environment.yml
conda activate darwinlm
```

## ✂️ Database Preparation

Before running the searching steps, you need to generate a structured database by running:

```bash
# For llama-2-7B
bash scripts/ziplm_llama2-7B.sh

# For llama-3.1-8B
bash scripts/ziplm_llama3.1-8B.sh

# For Qwen2.5-14B-Ins
bash scripts/ziplm_qwen2.5-14B-instruct.sh
```

> **Note:** Currently, you need to manually specify the number of columns removed for each compression step.

## 🚀 Evolutionary Search

To perform structured pruning, run the following ```scripts/struct_prune_search.sh```:

```bash
# The example is for Llama-2-7B, you can set other models and set the path of generated database to COMPR_PATH 
bash scripts/struct_prune_search.sh
```

This will conduct a structured pruning search based on the defined configurations. After the searching, a ```.txt``` file will be generated. You can stitch the model with the database and ```.txt``` file.

## 🛠 Post-Training

After pruning, you can further fine-tune the model with [Fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) dataset using the [llm-foundry](https://github.com/mosaicml/llm-foundry) repository. You can check the parameter settings in our paper for replication.


<!-- ## 🏗 Model Variants

We provide six model variants on huggingface:

- Models after pruning and searching: [**2.7B-Pruned Model**](https://huggingface.co/Shengkun/DarwinLM-2.7B-Pruned), [**4.6B-Pruned Model**](https://huggingface.co/Shengkun/DarwinLM-4.6B-Pruned), [**8.4B-Pruned Model**](https://huggingface.co/Shengkun/DarwinLM-8.4B-Pruned)
- Models after post-training: [**2.7B Model**](https://huggingface.co/Shengkun/DarwinLM-2.7B), [**4.6B Model**](https://huggingface.co/Shengkun/DarwinLM-4.6B), [**8.4B Model**](https://huggingface.co/Shengkun/DarwinLM-8.4B) -->



## 📊 Evaluation
First of all, you should install the [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) from their guidelines.


- Option 1: After that, if you want to replicate the results in our paper, we provide the weights above, which are fully supported by `transformers` packages directly. You can simply run the script
```bash
# Simply modify MODEL_ID to different models
bash scripts/run_lmeval_hf.sh
```

- Option 2: If you want to evaluate your searched structure, the evolutionary structured pruning search (`evo_struct_prune_search.py`) produces a configuration file that you need to pass as `sparse_config_path`. Also, pass the database path to `database_path` and run:

```bash
bash scripts/run_lmeval_config.sh
```
> **Note:** Currently, `transformers` packages do not support heads pruning for Llama and Qwen models. Therefore, if you install the package from the official repo, you should set `model_shrink=false`, which will keep the pruned weights as 0 but not be actually removed. If you want the actual speed, you can install the `transformers` packages from [my implementation](https://github.com/Tangshengku/transformers/tree/llama3.1) from source.


## 📖 Citation

If you find this work useful, please cite our paper:

```bibtex
@article{tang2025darwinlm,
  title={DarwinLM: Evolutionary Structured Pruning of Large Language Models},
  author={Tang, Shengkun and Sieberling, Oliver and Kurtic, Eldar and Shen, Zhiqiang and Alistarh, Dan},
  journal={arXiv preprint arXiv:2502.07780},
  year={2025}
}
```

---

For any issues or questions, please open an issue or contact us directly. 🚀

