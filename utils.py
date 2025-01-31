import torch
from torch.nn import Module
from typing import Optional, Tuple
from transformers.modeling_utils import prune_linear_layer


class NoAttention(Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ):
        return (0, None, None)


class NoIntermediate(Module):
    def forward(self, hidden_states):
        return hidden_states


class NoOutput(Module):
    def forward(self, hidden_states):
        return 0


def shrink(model, update_mask=False, is_transformers=False, kv_ignore=False):
    if is_transformers:
        layers = model.model.layers
    else:
        layers = model.model.model.layers
    for layer in layers:
        if not isinstance(layer.self_attn, NoAttention):
            weight = layer.self_attn.o_proj.weight
            if torch.all(weight == 0):
                layer.self_attn = NoAttention()
            else:
                # mask = torch.all(weight == 0, 0)
                # if torch.any(mask):
                #     idx = torch.nonzero(~mask).flatten()

                mask = torch.all(
                    weight.t().reshape((-1, weight.shape[0] * layer.self_attn.head_dim)) == 0, 1
                )
                # if print_:
                    
                #     mask_ = mask
                    # layer.self_attn.o_proj.weight.data = layer.self_attn.o_proj.weight * mask_.unsqueeze(0)
                    # layer.self_attn.q_proj.weight.data = layer.self_attn.q_proj.weight * mask
                    # layer.self_attn.v_proj.weight.data = layer.self_attn.v_proj.weight * mask
                    # layer.self_attn.k_proj.weight.data = layer.self_attn.k_proj.weight * mask
                if update_mask:
                    mask_ = (~mask.unsqueeze(1)) * torch.ones_like(weight.t(), device=mask.device).reshape(
                            (-1, weight.shape[0] * layer.attn.head_size)) 
                    mask_ = mask_.reshape(-1, weight.shape[0])

                    layer.self_attn.o_proj = layer.self_attn.o_proj * mask_.t()
                    layer.self_attn.q_proj = layer.self_attn.q_proj * mask
                    layer.self_attn.v_proj = layer.self_attn.v_proj * mask
                    layer.self_attn.k_proj = layer.self_attn.k_proj * mask
                    layer.self_attn.o_proj.register_buffer("mask", mask_.t(), persistent=False)
                    layer.self_attn.in_proj_linear_q.register_buffer("mask", mask_, persistent=False)
                    layer.self_attn.in_proj_linear_k.register_buffer("mask", mask_, persistent=False)
                    layer.self_attn.in_proj_linear_v.register_buffer("mask", mask_, persistent=False)
                # else:
                #     idx = torch.nonzero(~mask).flatten()
                #     layer.self_attn.q_proj = prune_linear_layer(layer.self_attn.q_proj, idx)
                #     layer.self_attn.k_proj = prune_linear_layer(layer.self_attn.k_proj, idx)
                #     layer.self_attn.v_proj = prune_linear_layer(layer.self_attn.v_proj, idx)
                #     layer.self_attn.o_proj = prune_linear_layer(layer.self_attn.o_proj, idx, dim=1)
                else:
                    idx = []
                    count = 0
                    for i in range(mask.numel()):
                        # pruned_heads = layer.self_attn.pruned_heads
                        # while count in pruned_heads:
                        #     count += 1
                        if mask[i]:
                            idx.append(count)
                        count += 1
                    if torch.any(mask):
                        # idx = torch.nonzero(~mask).squeeze(-1).tolist()
                        layer.self_attn.prune_heads(idx, kv_ignore=kv_ignore)
                    
        if not isinstance(layer.mlp.down_proj, NoOutput):
            weight = layer.mlp.down_proj.weight
            if torch.all(weight == 0):
                layer.mlp.up_proj = NoIntermediate()
                layer.mlp.gate_proj = NoIntermediate()
                layer.mlp.down_proj = NoOutput()
            else:
                mask = torch.all(weight == 0, 0)
                if update_mask:
                    mask_ = (~mask.unsqueeze(0)) * torch.ones_like(weight, device=mask.device)
                    # layer.mlp.down_proj.mask = mask_
                    # layer.mlp.up_proj = mask_.t()
                    # layer.mlp.gate_proj = mask_.t()
                    
                    layer.mlp.down_proj.register_buffer("mask", mask_, persistent=False)
                    layer.mlp.up_proj.register_buffer("mask", mask_.t(), persistent=False)
                    layer.mlp.gate_proj.register_buffer("mask", mask_.t(), persistent=False)
                elif torch.any(mask):
                    idx = torch.nonzero(~mask).flatten()
                    layer.mlp.gate_proj = prune_linear_layer(layer.mlp.gate_proj, idx)
                    layer.mlp.up_proj = prune_linear_layer(layer.mlp.up_proj, idx)
                    layer.mlp.down_proj = prune_linear_layer(layer.mlp.down_proj, idx, dim=1)

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}