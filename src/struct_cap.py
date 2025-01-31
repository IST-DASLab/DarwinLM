import re
import os
import math
import warnings
from tqdm import tqdm
from typing import Callable, Iterable, Optional, Tuple, Union, List, Dict, Any

import torch
import torch.nn as nn
import torch.distributed as dist
from torch import Tensor
from torch.nn.modules.conv import _ConvNd

from src import dist_utils, model_utils, linalg_utils
from src.common_utils import to


torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


class StructCAPLayerWrapper:
    def __init__(
        self,
        layer: nn.Module,
        num_samples: int,  # has to be known in advance
        group_size: int = 1,
        damp: float = 1e-6,
        block_size: Union[int, str] = "1 d",
        verbose: bool = False,
        F_inv_rank: int = 0,
        eps: float = 1e-9,
        find_block_size: bool = False
    ):
        self._validate_layer(layer)
        self.layer = layer
        self.W = self.layer.weight
        self.d_row, self.d_col = model_utils.get_number_of_rows_and_cols(layer)
        self.block_size = model_utils.parse_block_size(block_size, self.d_col)
        block_size_is_valid = (self.block_size % self.d_col == 0) or (self.d_col % self.block_size == 0)
        if find_block_size and not block_size_is_valid:
            # Find largest block size divisible by d_col
            # TODO there should exist more elegant algorithm (probably via DP)
            block_size = 1
            for i in range(1, self.block_size):
                if self.d_col % i == 0:
                    block_size = i
            self.block_size = block_size
        else:    
            assert block_size_is_valid
        self.group_size = group_size
        # OBS hyperparameters
        self.num_samples = num_samples
        self.damp = damp
        # backup layer properties
        self.W_device = self.W.device
        self.W_dtype = self.W.dtype
        self.W_shape = self.W.shape
        # init hessian
        self.num_blocks = self.W.numel() // self.block_size
        self.F_inv = None
        # misc args
        self.verbose = verbose
        self.F_inv_rank = F_inv_rank
        self.eps = eps

    @staticmethod
    def _validate_layer(layer):
        assert isinstance(layer, (nn.Linear, _ConvNd)), "OBC supports only linear and convolutional layers."

    # preparatory methods
    @torch.no_grad()
    def update(self, grad: Tensor) -> None:
        """
        Update the estimate of Fisher inverse given a grad.

        Args:
            grad: batch of layer inputs
        """
        rank = dist_utils.get_rank()
        world_size = dist_utils.get_world_size()

        # init hessian
        if dist_utils.is_main() and self.F_inv is None:
            self.F_inv = (
                (1.0 / self.damp * torch.eye(n=self.block_size, device=f"cuda:{rank}"))
                .unsqueeze(0)
                .repeat(self.num_blocks, 1, 1)
            )
        if dist_utils.is_dist_available_and_initialized():
            dist.barrier()

        # send grads
        if dist_utils.is_main():
            grads = [torch.empty_like(self.W).view(-1) for _ in range(world_size)]
        else:
            grads = None

        if dist_utils.is_dist_available_and_initialized():
            dist.gather(grad, grads, dst=0)

        if dist_utils.is_main():
            for g in grads:
                # reshape grad and cast to float32
                g = g.view(self.num_blocks, self.block_size).to(torch.float32)
                # batched Fs_inv x g: (n_B, B, B) x (n_B, B) -> (n_B, B)
                F_inv_g = torch.bmm(self.F_inv, g.unsqueeze(-1)).squeeze(-1)
                # scalar denominator for each block (n_B)
                denominator = (self.num_samples + (g * F_inv_g).sum(dim=-1)).sqrt().unsqueeze(1)
                F_inv_g.div_(denominator)
                # update inv_blocks with new outer product: (n_B, B) x (n_B, B) -> (n_B, B, B)
                self.F_inv.baddbmm_(F_inv_g.unsqueeze(2), F_inv_g.unsqueeze(1), alpha=-1)

    def reset(self) -> None:
        self.W = self.layer.weight
        self.F_inv = None
        torch.cuda.empty_cache()

    @torch.no_grad()
    def pruning_pre_step(self) -> None:
        """
        Preparatory step with hessian regularization and weight reshaping.
        """
        # 1) Hessian preparation
        # init Hessian on other workers
        rank = dist_utils.get_rank()
        if not dist_utils.is_main():
            self.F_inv = torch.empty(self.num_blocks, self.block_size, self.block_size, device=f"cuda:{rank}")
        # synchronize F_inv across devices
        if dist_utils.is_dist_available_and_initialized():
            dist.broadcast(self.F_inv, src=0)
        # 2) Weight preparation
        # copy weight, flatten and convert to float
        self.W = self.W.clone().float().view(-1, self.block_size)
        # flag pre step as completed
        self.pre_step_completed = True

    def step_single(self, sparsity: float) -> Tensor:
        # 1) define constants and chunk
        block_size, d_row, d_col, device, dtype = (self.block_size, self.d_row, self.d_col, self.W_device, self.W_dtype)
        blocks_per_dim = d_col // block_size

        if dist_utils.get_rank() == self.F_inv_rank:
            w, F_inv, mask = self.W, self.F_inv, torch.ones(d_col, device=device, dtype=torch.bool)

            cols_to_prune = round(sparsity * d_col)
            for _ in range(cols_to_prune):
                # 1) compute scores
                F_inv_d = F_inv.diagonal(dim1=-2, dim2=-1)
                scores = (w.pow(2) / F_inv_d).view(-1, d_col).sum(dim=0)
                scores[~mask] = torch.inf
                # 2) mask selection
                p_id = scores.argmin()
                p1, p2 = p_id // block_size, p_id % block_size
                mask[p_id] = False
                # 3) weight update
                w_p = w[p1::blocks_per_dim]  # weight part that will be updated
                F_inv_p = F_inv[p1::blocks_per_dim]  # hessian part that will be updated
                F_inv_pr = F_inv_p[:, p2]
                F_inv_pd = F_inv_p[:, p2, p2]
                w_p.add_(F_inv_pr * (w_p[:, p2] / F_inv_pd).unsqueeze(1), alpha=-1)
                w_p[:, p2] = 0
                # 4) hessian update
                dF = F_inv_pr / F_inv_pd.sqrt().unsqueeze(-1)
                F_inv_p.baddbmm_(dF.unsqueeze(-1), dF.unsqueeze(-2), alpha=-1)
                F_inv_p[:, :, p2] = 0.0
                F_inv_p[:, p2, :] = 0.0
                F_inv_p[:, p2, p2] = 1.0

            assert not torch.isnan(w).any().item(), "NaN encountered!"
            W_sparse = w.to(device=device, dtype=dtype)
        else:
            W_sparse = torch.empty_like(self.W, device=device, dtype=dtype)

        if dist_utils.is_dist_available_and_initialized():
            dist.barrier()
            dist.broadcast(W_sparse, src=self.F_inv_rank)

        return W_sparse.view(d_row, d_col).to(dtype)

    def step_grouped(self, groups_to_prune) -> Tensor:
        num_levels = len(groups_to_prune)
        # 1) define constants and chunk
        block_size, d_row, d_col, gs,  device, dtype = (
            self.block_size,
            self.d_row,
            self.d_col,
            self.group_size,
            self.W_device,
            self.W_dtype,
        )

        blocks_per_dim = d_col // block_size
        ng = d_col // gs  # number of groups per input dimension
        gpb = block_size // gs  # number of groups per block
        add_zeros = False
        sparse_weights = []

        if dist_utils.get_rank() == self.F_inv_rank:
            # groups_to_prune = round(sparsity * ng)
            if groups_to_prune[0] == 0:
                sparse_weights.append(self.layer.weight.data.clone().to(device=device, dtype=dtype))
                groups_to_prune = groups_to_prune[1:]
                print('0 error 0.0')
        
            if groups_to_prune[-1] == d_col // self.group_size:
                add_zeros = True
                groups_to_prune = groups_to_prune[:-1]

            w, F_inv, mask = self.W, self.F_inv, torch.ones(ng, device=device, dtype=torch.bool)
            for dropped in range(ng + 1):
                # 1) compure scores
                F_inv_db = (
                    F_inv.view(-1, gpb, gs, gpb, gs).diagonal(dim1=1, dim2=3).movedim(-1, 1)
                )  # shape (nb, gpb, gs, gs)
                inv_F_inv_db = linalg_utils.inv_sym(F_inv_db)  # shape (nb, gpb, gs, gs)
                w_g = w.view(-1, gpb, gs, 1)  # shape (nb, gpb, gs, 1)
                inv_F_inv_db_w = inv_F_inv_db @ w_g  # shape (nb, gpb, gs, 1)
                scores = (w_g * inv_F_inv_db_w).view(d_row, -1, gs).sum(dim=(0, 2))  # shape (ng,)
                scores[~mask] = torch.inf
                # 2) mask selection
                p_id = scores.argmin(dim=0).item()
                p1, p2 = p_id // gpb, p_id % gpb
                p2_ids = gs * p2 + torch.arange(gs)
                mask[p_id] = False
                # 3) weight update
                p_slc = slice(p1, None, blocks_per_dim)
                w_p = w[p_slc]  # weight part that will be updated
                F_inv_p = F_inv[p_slc]  # hessian part that will be updated
                inv_F_inv_pdb = inv_F_inv_db[p_slc, p2]  # shape (d_o, gs, gs)
                dw = torch.bmm(inv_F_inv_db_w[p_slc, p2 : p2 + 1, :, 0], F_inv_p[:, p2_ids]).squeeze(1)
                w_p.add_(dw, alpha=-1)
                w_p[:, p2_ids] = 0
                # 4) hessian update
                F_inv_p.add_(F_inv_p[:, :, p2_ids] @ inv_F_inv_pdb @ F_inv_p[:, p2_ids, :], alpha=-1)
                # isolate pruned columns
                F_inv_p[:, p2_ids, :] = 0
                F_inv_p[:, :, p2_ids] = 0
                F_inv_p[:, p2_ids, p2_ids] = 1
                while dropped + 1 == groups_to_prune[0]:
                    sparse_weights.append(w.clone().reshape(self.layer.weight.shape).to(device=device, dtype=dtype))
                    print('%4d error' % groups_to_prune[0])
                    groups_to_prune.pop(0)
                    if not len(groups_to_prune):
                        break
                if not len(groups_to_prune):
                    break
            assert not torch.isnan(w).any().item(), "NaN encountered!"
            # W_sparse = w.to(device=device, dtype=dtype)
            if add_zeros:
                sparse_weights.append(torch.zeros_like(self.layer.weight, device=device, dtype=dtype))
                print('removed all')
        else:
            sparse_weights = [torch.empty_like(self.layer.weight, device=device, dtype=dtype) for _ in range(num_levels)]

        if dist_utils.is_dist_available_and_initialized():
            dist.barrier()
            for i, _ in enumerate(range(num_levels)):
                dist.broadcast(sparse_weights[i], src=self.F_inv_rank)

        return sparse_weights

    def step(self, group_num_prune) -> Tensor:
            return self.step_grouped(group_num_prune)

    def prune(self, group_num_prune) -> Tensor:
        self.pruning_pre_step()
        W_sparse = self.step(group_num_prune)
        return W_sparse
    

class SparseStructCAP:

    def __init__(
        self, 
        model: nn.Module, 
        module_regex: str, 
        data_loader: Iterable, 
        loss_fn: Callable,
        damp: float,
        grad_sparsity: float,
        num_samples: int, # Number of batches
        mlp_prune_name: str,
        attn_prune_name: str,
        save_dir: Union[str, os.PathLike],
        block_size: Union[Union[int, str], List[Tuple[str, Union[int, str]]]] = "1 d",
        group_size: Union[int, List[Tuple[str, int]]] = 1,
        find_block_size: bool = False,
        device: torch.device = None,
        grad_accum_steps: int = 1,
        sparse_config=None,
    ):
        self.model = model
        self.module_regex = module_regex
        self.data_loader = data_loader
        self.loss_fn = loss_fn
        self.damp = damp
        self.block_size = block_size
        self.group_size = group_size
        self.grad_sparsity = grad_sparsity
        self.num_samples = num_samples
        self.samples_per_proc = num_samples // dist_utils.get_world_size()
        self.find_block_size = find_block_size
        self.device = device or next(self.model.parameters()).device  
        self.grad_accum_steps = grad_accum_steps 
        self.mlp_prune_name = mlp_prune_name
        self.attn_prune_name = attn_prune_name
        self.sparse_config = sparse_config
        self.save_dir = save_dir
        # Gradient buffers
        if self.grad_sparsity > 0:
            self.iG = None # Top-K indices
            self.vG = None # Top-K values
        else:
            self.G = None # gradient buffer

    @property
    def prunable_modules(self) -> List[nn.Module]:
        prunable_modules = []
        for module_name, module in self.model.named_modules():
            if re.search(self.module_regex, module_name):
                prunable_modules.append(module)
        return prunable_modules

    @property
    def prunable_module_names(self) -> List[str]:
        prunable_modules = []
        for module_name, _ in self.model.named_modules():
            if re.search(self.module_regex, module_name):
                prunable_modules.append(module_name)
        return prunable_modules

    @property
    def prunable_weights(self) -> List[torch.Tensor]:
        return [m.weight for m in self.prunable_modules]

    def _init_buffers(self):
        if self.grad_sparsity > 0:
            self.iG = []
            self.vG = []
            for p in self.prunable_weights:
                nnz = math.ceil((1 - self.grad_sparsity) * p.numel())
                self.iG.append(torch.zeros(self.samples_per_proc, nnz, dtype=torch.int32, device='cpu', pin_memory=True))
                self.vG.append(torch.zeros(self.samples_per_proc, nnz, dtype=p.dtype, device='cpu', pin_memory=True))
        else:
            self.G = []
            for p in self.prunable_weights:
                self.G.append(torch.zeros(self.samples_per_proc, p.numel(), dtype=p.dtype, device='cpu', pin_memory=True))

    @torch.no_grad()
    def _update_gradient_buffers(self, batch_id: int):
        for p_id, p in enumerate(self.prunable_weights):
            g = p.grad.view(-1)
            if self.grad_sparsity == 0:
                self.G[p_id][batch_id, :] = g
                p.grad = None
            else:
                k = math.ceil((1 - self.grad_sparsity) * p.numel())
                _, indices = torch.topk(input=g.abs(), k=k, sorted=False)
                # Save the sparse gradient to the CPU buffers
                self.iG[p_id][batch_id, :] = indices.cpu()
                self.vG[p_id][batch_id, :] = g[indices].cpu()
                # Zero grad for pruned ids
                g[indices] = 0
    
    def _prepare_gradients(self):
        device = self.device or next(self.model.parameters()).device
        microbatches_per_proc = self.grad_accum_steps * self.samples_per_proc
        for microbatch_id, (inp_args, inp_kwargs, targets) in tqdm(
            enumerate(self.data_loader), 
            desc=f'Computing gradients',
            total=microbatches_per_proc,
            disable=not dist_utils.is_main()
        ):
            if microbatch_id == microbatches_per_proc:
                # Manifestly stop if sufficient number of gradients are collected
                break
            outputs = self.model(*to(inp_args, device=device), **to(inp_kwargs, device=device))
            loss = self.loss_fn(outputs, targets.to(device))
            loss.backward()
            if (microbatch_id + 1) % self.grad_accum_steps == 0:
                # Update 
                batch_id = microbatch_id // self.grad_accum_steps
                self._update_gradient_buffers(batch_id)

    @torch.no_grad()
    def _prepare_block_slice(self, W: torch.Tensor, F_inv: torch.Tensor, b1: int, b2: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int, torch.Tensor]:
        nb = b2 - b1
        # get slice of rows
        w = W[b1:b2].clone()
        mask = w.ne(0)
        # get minimal number of zeros in a slice
        min_zeros = (~mask).sum(dim=1).min().item()
        # get zero row and col ids
        row_ids, col_ids = torch.nonzero(~mask).T
        # create N copies (d_row, d_col) -> (nb, d_col, d_col)
        F_inv = F_inv[b1:b2]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # mask rows with zeroed weights
            F_inv[row_ids, col_ids, :] = 0
            F_inv[row_ids, :, col_ids] = 0
            F_inv[row_ids, col_ids, col_ids] = 1
        return w, mask, F_inv, min_zeros, nb, torch.arange(nb)
        
    def struct_prune(self, rows_removed_attention: List[int], rows_removed_mlp: List[int]):
        self._init_buffers()
        self._prepare_gradients()
        for layer_id, _ in tqdm(
            enumerate(self.prunable_modules), 
            desc="Pruning layer", 
            total=len(self.prunable_modules),
            disable=not dist_utils.is_main()
        ):
            self._prune_layer(layer_id, rows_removed_attention, rows_removed_mlp)

    def _get_pruner_kwargs_for_layer(self, layer_name: str) -> Dict[str, Any]:
        pruner_kwargs_for_layer = {}
        # set specific block size for a given group of layers
        if isinstance(self.block_size, list):
            for group_regex, block_size in self.block_size:
                if re.search(group_regex, layer_name):
                    pruner_kwargs_for_layer["block_size"] = block_size
                    break
        else:
            pruner_kwargs_for_layer["block_size"] = self.block_size
        # set specific group size (number of weights pruned at once) for a given group of layers
        if isinstance(self.group_size, list):
            for group_regex, group_size in self.group_size:
                if re.search(group_regex, layer_name):
                    pruner_kwargs_for_layer["group_size"] = group_size
                    break
        else:
            pruner_kwargs_for_layer["group_size"] = self.group_size
        return pruner_kwargs_for_layer

    @torch.no_grad()
    def _prune_layer(self, layer_id, rows_removed_attention: List[int], rows_removed_mlp: List[int]):
        layer = self.prunable_modules[layer_id]
        layer_name = self.prunable_module_names[layer_id]
        if self.grad_sparsity > 0:
            iG = self.iG[layer_id]
            vG = self.vG[layer_id]
        else:
            G = self.G[layer_id]
        # Init handle
        layer_wrapper = StructCAPLayerWrapper(
            layer, 
            num_samples=self.num_samples, 
            damp=self.damp,
            find_block_size=self.find_block_size,
            **self._get_pruner_kwargs_for_layer(layer_name)
        )
        layer_wrapper.is_attn = self.attn_prune_name in layer_name
        # Estimate Fisher inverse
        g = torch.zeros_like(layer.weight)
        for i in range(self.samples_per_proc):
            g = g.view(-1)
            if self.grad_sparsity > 0:
                iG_i, vG_i = iG[i].to(g.device).long(), vG[i].to(g.device) 
                g[iG_i] = vG_i
            else:
                g = G[i].to(g.device)
            layer_wrapper.update(g)
            g.zero_()
        # Prune layer
        if self.attn_prune_name in layer_name:
            groups_to_prune = [ele//layer_wrapper.group_size for ele in rows_removed_attention]
        else:
            groups_to_prune = rows_removed_mlp
        
        sparse_weights = layer_wrapper.prune(groups_to_prune)

        if dist_utils.is_main():
            for level, sparse_weight in enumerate(sparse_weights):
                # For gradual pruning, the first one is the original pruned one
                if self.sparse_config is not None:
                    level += int(self.sparse_config[layer_name])
                
                os.makedirs(os.path.join(self.save_dir, layer_name), exist_ok=True)
                # Map tensor to CPU before saving
                
                torch.save(sparse_weight.cpu(), os.path.join(self.save_dir, layer_name, f"{level}.pth"))
        layer_wrapper.reset()
        del layer_wrapper
        del sparse_weights
        torch.cuda.empty_cache()
        if dist_utils.is_dist_available_and_initialized():
            dist.barrier()