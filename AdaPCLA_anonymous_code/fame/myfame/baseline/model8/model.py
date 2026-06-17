'''
HALO generator + prior correction (logit adjustment).

Based on fame/myfame/baseline/HALO/model.py, with the following changes:
- Uses BCEWithLogitsLoss (via binary_cross_entropy_with_logits) for stability
- Supports per-code logit adjustment vector (prior correction) during training/eval and sampling
'''

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class Conv1D(nn.Module):
    def __init__(self, nf, nx):
        super(Conv1D, self).__init__()
        self.nf = nf
        w = torch.empty(nx, nf)
        nn.init.normal_(w, std=0.02)
        self.weight = nn.Parameter(w)
        self.bias = nn.Parameter(torch.zeros(nf))

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(*size_out)
        return x


class Attention(nn.Module):
    def __init__(self, nx, n_ctx, config, scale=False):
        super(Attention, self).__init__()
        n_state = nx
        assert n_state % config.n_head == 0
        self.register_buffer("bias", torch.tril(torch.ones(n_ctx, n_ctx)).view(1, 1, n_ctx, n_ctx))
        self.n_head = config.n_head
        self.split_size = n_state
        self.scale = scale
        self.c_attn = Conv1D(n_state * 3, nx)
        self.c_proj = Conv1D(n_state, nx)

    def _attn(self, q, k, v):
        w = torch.matmul(q, k)
        if self.scale:
            w = w / math.sqrt(v.size(-1))
        nd, ns = w.size(-2), w.size(-1)
        b = self.bias[:, :, ns - nd : ns, :ns]
        w = w * b - 1e10 * (1 - b)
        w = nn.Softmax(dim=-1)(w)
        return torch.matmul(w, v)

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)

    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)
        if k:
            return x.permute(0, 2, 3, 1)
        return x.permute(0, 2, 1, 3)

    def forward(self, x, layer_past=None):
        x = self.c_attn(x)
        query, key, value = x.split(self.split_size, dim=2)
        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)
        if layer_past is not None:
            past_key, past_value = layer_past[0].transpose(-2, -1), layer_past[1]
            key = torch.cat((past_key, key), dim=-1)
            value = torch.cat((past_value, value), dim=-2)
        present = torch.stack((key.transpose(-2, -1), value))
        a = self._attn(query, key, value)
        a = self.merge_heads(a)
        a = self.c_proj(a)
        return a, present


class MLP(nn.Module):
    def __init__(self, n_state, config):
        super(MLP, self).__init__()
        nx = config.n_embd
        self.c_fc = Conv1D(n_state, nx)
        self.c_proj = Conv1D(nx, n_state)
        self.act = gelu

    def forward(self, x):
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        return h2


class Block(nn.Module):
    def __init__(self, n_ctx, config, scale=False):
        super(Block, self).__init__()
        nx = config.n_embd
        self.ln_1 = LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.attn = Attention(nx, n_ctx, config, scale)
        self.ln_2 = LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.mlp = MLP(4 * nx, config)

    def forward(self, x, layer_past=None):
        a, present = self.attn(self.ln_1(x), layer_past=layer_past)
        x = x + a
        m = self.mlp(self.ln_2(x))
        x = x + m
        return x, present


class CoarseTransformerModel(nn.Module):
    def __init__(self, config):
        super(CoarseTransformerModel, self).__init__()
        self.n_layer = config.n_layer
        self.n_embd = config.n_embd
        self.n_vocab = config.total_vocab_size

        self.vis_embed_mat = nn.Linear(config.total_vocab_size, config.n_embd, bias=False)
        self.pos_embed_mat = nn.Embedding(config.n_positions, config.n_embd)
        block = Block(config.n_ctx, config, scale=True)
        self.h = nn.ModuleList([copy.deepcopy(block) for _ in range(config.n_layer)])
        self.ln_f = LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

    def forward(self, input_visits, position_ids=None, past=None):
        if past is None:
            past_length = 0
            past = [None] * len(self.h)
        else:
            past_length = past[0][0].size(-2)
        if position_ids is None:
            position_ids = torch.arange(
                past_length, input_visits.size(1) + past_length, dtype=torch.long, device=input_visits.device
            )
            position_ids = position_ids.unsqueeze(0).expand(input_visits.size(0), input_visits.size(1))

        inputs_embeds = self.vis_embed_mat(input_visits)
        position_embeds = self.pos_embed_mat(position_ids)
        hidden_states = inputs_embeds + position_embeds
        for block, layer_past in zip(self.h, past):
            hidden_states, _ = block(hidden_states, layer_past)
        hidden_states = self.ln_f(hidden_states)
        return hidden_states


class AutoregressiveLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)
        self.register_buffer("mask", torch.tril(torch.ones(in_features, out_features)).int())

    def forward(self, input):
        return F.linear(input, self.mask * self.weight, self.bias)


class FineAutoregressiveHead(nn.Module):
    def __init__(self, config):
        super(FineAutoregressiveHead, self).__init__()
        self.auto1 = AutoregressiveLinear(config.n_embd + config.total_vocab_size, config.n_embd + config.total_vocab_size)
        self.auto2 = AutoregressiveLinear(config.n_embd + config.total_vocab_size, config.n_embd + config.total_vocab_size)
        self.n_embd = config.n_embd
        self.tot_vocab = config.total_vocab_size

    def forward(self, history, input_visits):
        history = history[:, :-1, :]
        input_visits = input_visits[:, 1:, :]
        code_logits = self.auto2(torch.relu(self.auto1(torch.cat((history, input_visits), dim=2))))[:, :, self.n_embd - 1 : -1]
        return code_logits

    def sample(self, history, input_visits):
        # Match the original behavior but avoid concatenating the full sequence when
        # only the last step is used (HALO2-style sampling optimization).
        if history.size(1) >= 2 and input_visits.size(1) >= 2:
            history_vec = history[:, -2, :]
            token_vec = input_visits[:, -1, :]
            curr_visit = torch.cat((history_vec, token_vec), dim=1).unsqueeze(1)
        else:
            history = history[:, :-1, :]
            input_visits = input_visits[:, 1:, :]
            curr_visit = torch.cat((history, input_visits), dim=2)[:, -1, :].unsqueeze(1)
        code_logits = self.auto2(torch.relu(self.auto1(curr_visit)))[:, :, self.n_embd - 1 : -1]
        return code_logits


def _validate_logit_adjust(logit_adjust, *, total_vocab_size: int, device, dtype):
    if logit_adjust is None:
        return None
    if not isinstance(logit_adjust, torch.Tensor):
        raise ValueError("logit_adjust must be a 1D torch.Tensor or None")
    if logit_adjust.dim() != 1 or int(logit_adjust.numel()) != int(total_vocab_size):
        raise ValueError(f"logit_adjust shape mismatch: got {tuple(logit_adjust.shape)} expected ({int(total_vocab_size)},)")
    return logit_adjust.to(device=device, dtype=dtype)


class HALOModel(nn.Module):
    def __init__(self, config):
        super(HALOModel, self).__init__()
        self.transformer = CoarseTransformerModel(config)
        self.ehr_head = FineAutoregressiveHead(config)
        self.config = config

    def forward(
        self,
        input_visits,
        position_ids=None,
        ehr_labels=None,
        ehr_masks=None,
        past=None,
        pos_loss_weight=None,
        *,
        logit_adjust=None,
        apply_logit_adjust_to_outputs: bool = False,
    ):
        hidden_states = self.transformer(input_visits, position_ids, past)
        logits = self.ehr_head(hidden_states, input_visits)

        adj = _validate_logit_adjust(
            logit_adjust, total_vocab_size=int(logits.size(-1)), device=logits.device, dtype=logits.dtype
        )

        out_logits = logits
        if apply_logit_adjust_to_outputs and adj is not None:
            out_logits = out_logits + adj.view(1, 1, -1)
        probs = torch.sigmoid(out_logits)

        if ehr_labels is None:
            return probs

        shift_labels = ehr_labels[..., 1:, :].contiguous()

        # Optional positive reweighting (kept compatible with original baseline).
        pos_weight = None
        if pos_loss_weight is not None:
            pos_weight = torch.full((logits.size(-1),), float(pos_loss_weight), device=logits.device, dtype=logits.dtype)

        loss_logits = logits
        if adj is not None:
            loss_logits = loss_logits + adj.view(1, 1, -1)

        loss_elem = F.binary_cross_entropy_with_logits(
            loss_logits,
            shift_labels.to(dtype=loss_logits.dtype),
            pos_weight=pos_weight,
            reduction="none",
        )
        if ehr_masks is not None:
            mask = ehr_masks.to(dtype=loss_elem.dtype, device=loss_elem.device)
            loss_elem = loss_elem * mask
            denom = mask.sum().clamp(min=1.0) * float(loss_elem.size(-1))
            loss = loss_elem.sum() / denom
            return loss, probs * mask, shift_labels * mask

        loss = loss_elem.mean()
        return loss, probs, shift_labels

    def sample(self, input_visits, random=True, *, logit_adjust=None):
        sig = nn.Sigmoid()
        hidden_states = self.transformer(input_visits)
        adj = None
        if logit_adjust is not None:
            adj = _validate_logit_adjust(
                logit_adjust, total_vocab_size=int(self.ehr_head.tot_vocab), device=input_visits.device, dtype=input_visits.dtype
            )
        i = 0
        while i < self.ehr_head.tot_vocab:
            next_logits = self.ehr_head.sample(hidden_states, input_visits)
            if adj is not None:
                next_logits = next_logits + adj.view(1, 1, -1)
            next_probs = sig(next_logits)
            if random:
                visit = torch.bernoulli(next_probs)
            else:
                visit = torch.round(next_probs)

            remaining_visit = visit[:, 0, i:]
            nonzero = torch.nonzero(remaining_visit, as_tuple=True)[1]
            if nonzero.numel() == 0:
                break

            first_nonzero = nonzero.min()
            input_visits[:, -1, i + first_nonzero] = visit[:, 0, i + first_nonzero]
            i = i + first_nonzero + 1

        return input_visits
