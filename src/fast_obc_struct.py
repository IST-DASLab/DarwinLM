from typing import List

import torch
import torch.nn as nn
import torch.distributed as dist
from torch import Tensor
from torch.nn.modules.conv import _ConvNd

from src import dist_utils
from src import model_utils
from src import linalg_utils

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


class FastOBCStruct:

    def __init__(self, layer: nn.Module, rel_damp: float = 1e-2, verbose: bool = False):
        self._validate_layer(layer)
        self.layer = layer
        self.W = self.layer.weight
        self.d_row, self.d_col = model_utils.get_number_of_rows_and_cols(layer)
        # FastOBC hyperparameters
        self.rel_damp = rel_damp
        # backup layer properties
        self.W_device = self.W.device
        self.W_dtype = self.W.dtype
        self.W_shape = self.W.shape
        # init hessian
        self.H = None
        self.num_samples = 0
        # misc args
        self.verbose = verbose

    @staticmethod
    def _validate_layer(layer):
        assert isinstance(layer, (nn.Linear, _ConvNd)), "FastOBC supports only linear and convolutional layers."

    # preparatory methods
    @torch.no_grad()
    def update(self, input: Tensor) -> None:
        """
        Update the estimate of Hessian matrix from a batch of data.

        Args:
            input: batch of layer inputs
        """
        # get batch size
        batch_size = input.shape[0]
        # init hessian
        if self.H is None:
            self.H = torch.zeros((self.d_col, self.d_col), device=input.device, dtype=torch.float32)
        # input reshaping
        if isinstance(self.layer, nn.Linear):
            input = input.reshape(-1, input.shape[-1])
        else:
            unfold = nn.Unfold(
                self.layer.kernel_size,
                dilation=self.layer.dilation,
                padding=self.layer.padding,
                stride=self.layer.stride,
            )
            # output size (batch_size, channels * \prod kernel_size, num_patches)
            input = unfold(input)
            input = input.transpose(1, 2).flatten(0, 1)
        # cast input to float32 before addition
        input = input.float()
        # hessian update
        beta = self.num_samples / (self.num_samples + batch_size)
        alpha = 2.0 / (self.num_samples + batch_size)
        self.H.addmm_(input.T, input, beta=beta, alpha=alpha)
        # update number of collected samples
        self.num_samples += batch_size

    def reset(self) -> None:
        self.W = self.layer.weight
        self.H = None
        self.num_samples = 0
        torch.cuda.empty_cache()

    @torch.no_grad()
    def pruning_pre_step(self) -> None:
        """
        Preparatory step with hessian regularization and weight reshaping.
        """
        # 1) Hessian preparation
        assert self.H is not None, "One has to process at least one sample of calibration data to run pruning"
        # synchronize Hessians
        if dist_utils.is_dist_available_and_initialized():
            dist.all_reduce(self.H, op=dist.ReduceOp.AVG)
        # get ids of pruned channels
        pruned_ids = torch.diag(self.H) == 0
        self.H[pruned_ids, pruned_ids] = 1
        # Hessian regularization
        damp = self.rel_damp * torch.diag(self.H).mean()
        self.H[range(self.d_col), range(self.d_col)] += damp
        # 2) Weight preparation
        # copy weight, flatten and convert to float
        self.W = self.W.clone().float()
        if isinstance(self.layer, _ConvNd):
            self.W = self.W.flatten(1, -1)
        self.W[:, pruned_ids] = 0
        # flag pre step as completed
        self.pre_step_completed = True

    # mostly copy pasted from ZipLM prune_struct method
    # I assumethis can be significantly optimized by not iterating over every single column to remove (do in blocks)
    def step(self, rows_removed_attention: List[int], rows_removed_mlp: List[int], headsize: int) -> List[Tensor]:
        num_levels = len(rows_removed_attention)
        d_col, device, dtype = self.d_col, self.W_device, self.W_dtype
        if dist_utils.is_main():
            torch.cuda.empty_cache()
            # prepare empty list for sparse weights
            sparse_weights = []
            # prepare weight and Cholesky of H^{-1}
            W, Hinv = self._prepare()

            if self.is_attn:
                pruned = [ele//headsize for ele in rows_removed_attention]
                size = headsize
            else:
                pruned = rows_removed_mlp
                size = 1

            count = d_col // size
            Losses = torch.zeros(count + 1, device=device, dtype=torch.float)
            mask = torch.zeros(count, device=device).bool()
            rangecount = torch.arange(count, device=device)
            rangecolumns = torch.arange(d_col, device=device)

            assert len(pruned) == len(set(pruned)), "Duplicate pruning steps!"
            pruned = sorted(pruned)
        
            add_zeros = False
            if pruned[0] == 0:
                sparse_weights.append(self.layer.weight.data.clone().to(device=device, dtype=dtype))
                pruned = pruned[1:]
                print('0 error 0.0')
            if pruned[-1] == d_col // size:
                add_zeros = True
                pruned = pruned[:-1]
            if size == 1:
                for dropped in range(count + 1):
                    diag = torch.diagonal(Hinv)
                    scores = torch.sum(W ** 2, 0) / diag
                    scores[mask] = float('inf')
                    j = torch.argmin(scores)
                    Losses[dropped] = scores[j]
                    row = Hinv[j, :]
                    d = diag[j]
                    W -= ((W[:, j] / d).unsqueeze(1)).matmul(row.unsqueeze(0))
                    mask[j] = True
                    W[:, mask] = 0
                    while dropped + 1 == pruned[0]:
                        sparse_weights.append(W.clone().reshape(self.layer.weight.shape).to(device=device, dtype=dtype))
                        print('%4d error' % pruned[0], torch.sum(Losses).item() / 2)
                        pruned.pop(0)
                        if not len(pruned):
                            break
                    if not len(pruned):
                        break
                    row /= torch.sqrt(d)
                    Hinv -= row.unsqueeze(1).matmul(row.unsqueeze(0))
            else:
                mask1 = torch.zeros(d_col, device=device).bool()
                for dropped in range(count + 1):
                    blocks = Hinv.reshape(count, size, count, size)
                    blocks = blocks[rangecount, :, rangecount, :]
                    try:
                        invblocks = torch.cholesky_inverse(torch.linalg.cholesky(blocks))
                    except:
                        invblocks = torch.linalg.pinv(blocks, hermitian=True)
                    W1 = W.reshape((self.d_row, count, size)).transpose(0, 1)
                    lambd = torch.bmm(W1, invblocks)
                    scores = torch.sum(lambd * W1, (1, 2))
                    scores[mask] = float('inf')
                    j = torch.argmin(scores)
                    Losses[dropped] = scores[j]
                    rows = Hinv[(size * j):(size * (j + 1)), :]
                    d = invblocks[j]
                    W -= lambd[j].matmul(rows)
                    mask[j] = True
                    mask1[(size * j):(size * (j + 1))] = True
                    W[:, mask1] = 0
                    while dropped + 1 == pruned[0]:
                        sparse_weights.append(W.clone().reshape(self.layer.weight.shape).to(device=device, dtype=dtype))
                        print('%4d error' % pruned[0], torch.sum(Losses).item() / 2)
                        pruned.pop(0)
                        if not len(pruned):
                            break
                    if not len(pruned):
                        break
                    Hinv -= rows.t().matmul(d.matmul(rows))
                    Hinv[rangecolumns[mask1], rangecolumns[mask1]] = 1
            if add_zeros:
                sparse_weights.append(torch.zeros_like(self.W, device=device, dtype=dtype))
                print('removed all')
        else:
            sparse_weights = [torch.empty_like(self.W, device=device, dtype=dtype) for _ in range(num_levels)]


        if dist_utils.is_dist_available_and_initialized():
            dist.barrier()
            for i, _ in enumerate(range(num_levels)):
                dist.broadcast(sparse_weights[i], src=0)
 
        return sparse_weights

    def prune_struct(self, rows_removed_attention: List[int], rows_removed_mlp: List[int], headsize: int) -> List[Tensor]:
        self.pruning_pre_step()
        sparse_weights = self.step(rows_removed_attention, rows_removed_mlp, headsize)
        return sparse_weights

    @torch.no_grad()
    def _prepare(self):
        w = self.W
        # get columns with all zeros
        zero_cols = torch.nonzero(w.eq(0).all(dim=0))
        H = self.H
        # mask rows with zero input channels
        H[zero_cols, :] = 0
        H[:, zero_cols] = 0
        H[zero_cols, zero_cols] = 1
        # invert
        H = linalg_utils.inv_sym(H)
        ###H_inv_cho = torch.linalg.cholesky(H, upper=True)
        return w, H #H_inv_cho
