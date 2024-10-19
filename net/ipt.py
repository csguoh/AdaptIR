# 2021.05.07-Changed for IPT
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
import os

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from net import common
import math
import torch
import torch.nn.functional as F
from torch import nn, Tensor
import copy
from torchvision import ops
from matplotlib import pyplot as plt
import numpy as np
from functools import partial, reduce
from operator import mul


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


def tensor_to_image(tensor):
    if tensor.max() > 1:
        tensor = tensor / 255.
    # Ensure the input tensor is on the CPU and in the range [0, 1]
    tensor = tensor.cpu()
    # tensor = tensor.clamp(0, 1)

    # Convert the tensor to a NumPy array
    image = tensor.squeeze(0).permute(1, 2, 0).numpy()
    plt.imshow(image)
    plt.savefig('./test.jpg')


class IPT(nn.Module):
    def __init__(self, args):
        super(IPT, self).__init__()
        self.TASK_MAP = {'lr4_noise30': 2, 'lr4_jpeg30': 2, 'sr_2': 0, 'sr_3': 1, 'sr_4': 2,
                         'derainH': 3,'derainL': 3, 'denoise_30': 4, 'denoise_50': 5, 'low_light': 5, }
        if isinstance(args.de_type,list):
            self.task_idx = None
        else:
            self.task_idx = self.TASK_MAP[args.de_type] if type(args.de_type) is not list else 5
        conv = common.default_conv
        self.scales = [2, 3, 4, 1, 1, 1]
        n_feats = 64
        self.patch_size = 48
        n_colors = 3
        kernel_size = 3
        rgb_range = 255
        act = nn.ReLU(True)

        self.sub_mean = common.MeanShift(rgb_range)
        self.add_mean = common.MeanShift(rgb_range, sign=1)

        self.head = nn.ModuleList([
            nn.Sequential(
                conv(n_colors, n_feats, kernel_size),
                common.ResBlock(conv, n_feats, 5, act=act),
                common.ResBlock(conv, n_feats, 5, act=act)
            ) for _ in self.scales
        ])

        self.body = VisionTransformer(img_dim=48, patch_dim=3,
                                      num_channels=n_feats, embedding_dim=n_feats * 3 * 3,
                                      num_heads=12, num_layers=12,
                                      hidden_dim=n_feats * 3 * 3 * 4, num_queries=len(self.scales),
                                      dropout_rate=0, mlp=False, pos_every=False,
                                      no_pos=False, no_norm=False)

        self.tail = nn.ModuleList([
            nn.Sequential(
                common.Upsampler(conv, s, n_feats, act=False),
                conv(n_feats, 3, kernel_size)
            ) for s in self.scales
        ])

    def forward(self, x, de_id=None):
        x = x * 255.
        if not self.training:
            return self.forward_chop(x) / 255.
        else:
            return self.forward_train(x, de_id) / 255.

    def forward_train(self, x,de_id=None):
        if de_id is not None:
            self.task_idx = self.TASK_MAP[de_id[0]] if type(de_id[0]) is not list else 5
        x = self.sub_mean(x)
        x = self.head[self.task_idx](x)

        res = self.body(x, self.task_idx)
        res += x

        x = self.tail[self.task_idx](res)
        x = self.add_mean(x)

        return x

    def set_scale(self, task_idx):
        self.task_idx = task_idx

    def forward_chop(self, x):
        x.cpu()
        batchsize = 64
        h, w = x.size()[-2:]
        padsize = int(self.patch_size)
        shave = int(self.patch_size / 2)

        scale = self.scales[self.task_idx]

        h_cut = (h - padsize) % (int(shave / 2))
        w_cut = (w - padsize) % (int(shave / 2))

        x_unfold = torch.nn.functional.unfold(x, padsize, stride=int(shave / 2)).transpose(0,
                                                                                           2).contiguous()  # [num_patch, 48*48*3, N]

        x_hw_cut = x[..., (h - padsize):, (w - padsize):]
        y_hw_cut = self.forward_train(x_hw_cut.cuda()).cpu()

        x_h_cut = x[..., (h - padsize):, :]
        x_w_cut = x[..., :, (w - padsize):]
        y_h_cut = self.cut_h(x_h_cut, h, w, h_cut, w_cut, padsize, shave, scale, batchsize)
        y_w_cut = self.cut_w(x_w_cut, h, w, h_cut, w_cut, padsize, shave, scale, batchsize)

        x_h_top = x[..., :padsize, :]
        x_w_top = x[..., :, :padsize]
        y_h_top = self.cut_h(x_h_top, h, w, h_cut, w_cut, padsize, shave, scale, batchsize)
        y_w_top = self.cut_w(x_w_top, h, w, h_cut, w_cut, padsize, shave, scale, batchsize)

        x_unfold = x_unfold.view(x_unfold.size(0), -1, padsize, padsize)
        y_unfold = []

        x_range = x_unfold.size(0) // batchsize + (x_unfold.size(0) % batchsize != 0)
        x_unfold.cuda()
        for i in range(x_range):
            y_unfold.append(self.forward_train(x_unfold[i * batchsize:(i + 1) * batchsize, ...]).cpu())
        y_unfold = torch.cat(y_unfold, dim=0)

        y = torch.nn.functional.fold(y_unfold.view(y_unfold.size(0), -1, 1).transpose(0, 2).contiguous(),
                                     ((h - h_cut) * scale, (w - w_cut) * scale), padsize * scale,
                                     stride=int(shave / 2 * scale))

        y[..., :padsize * scale, :] = y_h_top
        y[..., :, :padsize * scale] = y_w_top

        y_unfold = y_unfold[..., int(shave / 2 * scale):padsize * scale - int(shave / 2 * scale),
                   int(shave / 2 * scale):padsize * scale - int(shave / 2 * scale)].contiguous()
        y_inter = torch.nn.functional.fold(y_unfold.view(y_unfold.size(0), -1, 1).transpose(0, 2).contiguous(),
                                           ((h - h_cut - shave) * scale, (w - w_cut - shave) * scale),
                                           padsize * scale - shave * scale, stride=int(shave / 2 * scale))

        y_ones = torch.ones(y_inter.shape, dtype=y_inter.dtype)
        divisor = torch.nn.functional.fold(
            torch.nn.functional.unfold(y_ones, padsize * scale - shave * scale, stride=int(shave / 2 * scale)),
            ((h - h_cut - shave) * scale, (w - w_cut - shave) * scale), padsize * scale - shave * scale,
            stride=int(shave / 2 * scale))

        y_inter = y_inter / divisor

        y[..., int(shave / 2 * scale):(h - h_cut) * scale - int(shave / 2 * scale),
        int(shave / 2 * scale):(w - w_cut) * scale - int(shave / 2 * scale)] = y_inter

        y = torch.cat([y[..., :y.size(2) - int((padsize - h_cut) / 2 * scale), :],
                       y_h_cut[..., int((padsize - h_cut) / 2 * scale + 0.5):, :]], dim=2)
        y_w_cat = torch.cat([y_w_cut[..., :y_w_cut.size(2) - int((padsize - h_cut) / 2 * scale), :],
                             y_hw_cut[..., int((padsize - h_cut) / 2 * scale + 0.5):, :]], dim=2)
        y = torch.cat([y[..., :, :y.size(3) - int((padsize - w_cut) / 2 * scale)],
                       y_w_cat[..., :, int((padsize - w_cut) / 2 * scale + 0.5):]], dim=3)
        return y.cuda()

    def cut_h(self, x_h_cut, h, w, h_cut, w_cut, padsize, shave, scale, batchsize):

        x_h_cut_unfold = torch.nn.functional.unfold(x_h_cut, padsize, stride=int(shave / 2)).transpose(0,
                                                                                                       2).contiguous()

        x_h_cut_unfold = x_h_cut_unfold.view(x_h_cut_unfold.size(0), -1, padsize, padsize)
        x_range = x_h_cut_unfold.size(0) // batchsize + (x_h_cut_unfold.size(0) % batchsize != 0)
        y_h_cut_unfold = []
        x_h_cut_unfold.cuda()
        for i in range(x_range):
            y_h_cut_unfold.append(self.forward_train(x_h_cut_unfold[i * batchsize:(i + 1) * batchsize, ...]).cpu())
        y_h_cut_unfold = torch.cat(y_h_cut_unfold, dim=0)

        y_h_cut = torch.nn.functional.fold(
            y_h_cut_unfold.view(y_h_cut_unfold.size(0), -1, 1).transpose(0, 2).contiguous(),
            (padsize * scale, (w - w_cut) * scale), padsize * scale, stride=int(shave / 2 * scale))
        y_h_cut_unfold = y_h_cut_unfold[..., :,
                         int(shave / 2 * scale):padsize * scale - int(shave / 2 * scale)].contiguous()
        y_h_cut_inter = torch.nn.functional.fold(
            y_h_cut_unfold.view(y_h_cut_unfold.size(0), -1, 1).transpose(0, 2).contiguous(),
            (padsize * scale, (w - w_cut - shave) * scale), (padsize * scale, padsize * scale - shave * scale),
            stride=int(shave / 2 * scale))

        y_ones = torch.ones(y_h_cut_inter.shape, dtype=y_h_cut_inter.dtype)
        divisor = torch.nn.functional.fold(
            torch.nn.functional.unfold(y_ones, (padsize * scale, padsize * scale - shave * scale),
                                       stride=int(shave / 2 * scale)), (padsize * scale, (w - w_cut - shave) * scale),
            (padsize * scale, padsize * scale - shave * scale), stride=int(shave / 2 * scale))
        y_h_cut_inter = y_h_cut_inter / divisor

        y_h_cut[..., :, int(shave / 2 * scale):(w - w_cut) * scale - int(shave / 2 * scale)] = y_h_cut_inter
        return y_h_cut

    def cut_w(self, x_w_cut, h, w, h_cut, w_cut, padsize, shave, scale, batchsize):
        x_w_cut_unfold = torch.nn.functional.unfold(x_w_cut, padsize, stride=int(shave / 2)).transpose(0,
                                                                                                       2).contiguous()

        x_w_cut_unfold = x_w_cut_unfold.view(x_w_cut_unfold.size(0), -1, padsize, padsize)
        x_range = x_w_cut_unfold.size(0) // batchsize + (x_w_cut_unfold.size(0) % batchsize != 0)
        y_w_cut_unfold = []
        x_w_cut_unfold.cuda()
        for i in range(x_range):
            y_w_cut_unfold.append(self.forward_train(x_w_cut_unfold[i * batchsize:(i + 1) * batchsize, ...]).cpu())
        y_w_cut_unfold = torch.cat(y_w_cut_unfold, dim=0)

        y_w_cut = torch.nn.functional.fold(
            y_w_cut_unfold.view(y_w_cut_unfold.size(0), -1, 1).transpose(0, 2).contiguous(),
            ((h - h_cut) * scale, padsize * scale), padsize * scale, stride=int(shave / 2 * scale))
        y_w_cut_unfold = y_w_cut_unfold[..., int(shave / 2 * scale):padsize * scale - int(shave / 2 * scale),
                         :].contiguous()
        y_w_cut_inter = torch.nn.functional.fold(
            y_w_cut_unfold.view(y_w_cut_unfold.size(0), -1, 1).transpose(0, 2).contiguous(),
            ((h - h_cut - shave) * scale, padsize * scale), (padsize * scale - shave * scale, padsize * scale),
            stride=int(shave / 2 * scale))

        y_ones = torch.ones(y_w_cut_inter.shape, dtype=y_w_cut_inter.dtype)
        divisor = torch.nn.functional.fold(
            torch.nn.functional.unfold(y_ones, (padsize * scale - shave * scale, padsize * scale),
                                       stride=int(shave / 2 * scale)), ((h - h_cut - shave) * scale, padsize * scale),
            (padsize * scale - shave * scale, padsize * scale), stride=int(shave / 2 * scale))
        y_w_cut_inter = y_w_cut_inter / divisor

        y_w_cut[..., int(shave / 2 * scale):(h - h_cut) * scale - int(shave / 2 * scale), :] = y_w_cut_inter
        return y_w_cut


class VisionTransformer(nn.Module):
    def __init__(
            self,
            img_dim,
            patch_dim,
            num_channels,
            embedding_dim,
            num_heads,
            num_layers,
            hidden_dim,
            num_queries,
            positional_encoding_type="learned",
            dropout_rate=0,
            no_norm=False,
            mlp=False,
            pos_every=False,
            no_pos=False
    ):
        super(VisionTransformer, self).__init__()

        assert embedding_dim % num_heads == 0
        assert img_dim % patch_dim == 0
        self.no_norm = no_norm
        self.mlp = mlp
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.patch_dim = patch_dim
        self.num_channels = num_channels

        self.img_dim = img_dim
        self.pos_every = pos_every
        self.num_patches = int((img_dim // patch_dim) ** 2)
        self.seq_length = self.num_patches
        self.flatten_dim = patch_dim * patch_dim * num_channels

        self.out_dim = patch_dim * patch_dim * num_channels

        self.no_pos = no_pos

        if self.mlp == False:
            self.linear_encoding = nn.Linear(self.flatten_dim, embedding_dim)
            self.mlp_head = nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim),
                nn.Dropout(dropout_rate),
                nn.ReLU(),
                nn.Linear(hidden_dim, self.out_dim),
                nn.Dropout(dropout_rate)
            )

            self.query_embed = nn.Embedding(num_queries, embedding_dim * self.seq_length)

        encoder_layer = TransformerEncoderLayer(embedding_dim, num_heads, hidden_dim, dropout_rate, self.no_norm)
        self.encoder = TransformerEncoder(encoder_layer, num_layers)

        decoder_layer = TransformerDecoderLayer(embedding_dim, num_heads, hidden_dim, dropout_rate, self.no_norm)
        self.decoder = TransformerDecoder(decoder_layer, num_layers)

        if not self.no_pos:
            self.position_encoding = LearnedPositionalEncoding(
                self.seq_length, self.embedding_dim, self.seq_length
            )

        self.dropout_layer1 = nn.Dropout(dropout_rate)

        if no_norm:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=1 / m.weight.size(1))

    def forward(self, x, query_idx, con=False):

        x = torch.nn.functional.unfold(x, self.patch_dim, stride=self.patch_dim).transpose(1, 2).transpose(0,
                                                                                                           1).contiguous()

        if self.mlp == False:
            x = self.dropout_layer1(self.linear_encoding(x)) + x

            query_embed = self.query_embed.weight[query_idx].view(-1, 1, self.embedding_dim).repeat(1, x.size(1), 1)
        else:
            query_embed = None

        if not self.no_pos:
            pos = self.position_encoding(x).transpose(0, 1)

        if self.pos_every:
            x = self.encoder(x, pos=pos)
            x = self.decoder(x, x, pos=pos, query_pos=query_embed)
        elif self.no_pos:
            x = self.encoder(x)
            x = self.decoder(x, x, query_pos=query_embed)
        else:  # here
            x = self.encoder(x + pos)
            x = self.decoder(x, x, query_pos=query_embed)

        if self.mlp == False:
            x = self.mlp_head(x) + x

        x = x.transpose(0, 1).contiguous().view(x.size(1), -1, self.flatten_dim)

        if con:
            con_x = x
            x = torch.nn.functional.fold(x.transpose(1, 2).contiguous(), int(self.img_dim), self.patch_dim,
                                         stride=self.patch_dim)
            return x, con_x

        x = torch.nn.functional.fold(x.transpose(1, 2).contiguous(), int(self.img_dim), self.patch_dim,
                                     stride=self.patch_dim)

        return x


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, max_position_embeddings, embedding_dim, seq_length):
        super(LearnedPositionalEncoding, self).__init__()
        self.pe = nn.Embedding(max_position_embeddings, embedding_dim)
        self.seq_length = seq_length

        self.register_buffer(
            "position_ids", torch.arange(self.seq_length).expand((1, -1))
        )

    def forward(self, x, position_ids=None):
        if position_ids is None:
            position_ids = self.position_ids[:, : self.seq_length]

        position_embeddings = self.pe(position_ids)
        return position_embeddings


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        d_model = 576

    def forward(self, src, pos=None):
        output = src
        for idx, layer in enumerate(self.layers):
            output = layer(output, pos=pos)
        return output


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, no_norm=False,
                 activation="relu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, bias=False)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model) if not no_norm else nn.Identity()
        self.norm2 = nn.LayerNorm(d_model) if not no_norm else nn.Identity()
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)

        self.adaptir = AdaptIR(d_model)

        nn.init.kaiming_uniform_(self.self_attn.in_proj_weight, a=math.sqrt(5))

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, src, pos=None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, src2)

        src = src + self.dropout1(src2[0])
        src2 = self.norm2(src)
        adapt = self.adaptir(src2)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2+adapt)
        return src


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, tgt, memory, pos=None, query_pos=None):
        output = tgt
        for idx, layer in enumerate(self.layers):
            output = layer(output, memory, pos=pos, query_pos=query_pos)
        return output


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, no_norm=False,
                 activation="relu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, bias=False)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, bias=False)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model) if not no_norm else nn.Identity()
        self.norm2 = nn.LayerNorm(d_model) if not no_norm else nn.Identity()
        self.norm3 = nn.LayerNorm(d_model) if not no_norm else nn.Identity()
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

        self.adaptir = AdaptIR(d_model)

    def with_pos_embed(self, tensor, pos):
        if pos is not None and pos.shape[0] < tensor.shape[0]:
            pos = torch.cat(
                [torch.zeros(tensor.shape[0] - pos.shape[0], tensor.shape[1], tensor.shape[2]).to(pos.device), pos],
                dim=0)
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory, pos=None, query_pos=None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2)[0]
        tgt = tgt + self.dropout1(tgt2)

        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory)[0]
        tgt = tgt + self.dropout2(tgt2)

        tgt2 = self.norm3(tgt)
        adapt = self.adaptir(tgt2)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))

        tgt = tgt + self.dropout3(tgt2+adapt)
        return tgt


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class AdaptIR(nn.Module):
    def __init__(self, d_model):
        super(AdaptIR, self).__init__()
        self.hidden = d_model // 24
        self.rank = self.hidden // 2
        self.kernel_size = 3
        self.group = self.hidden
        self.head = nn.Conv2d(d_model, self.hidden, 1, 1)

        self.BN = nn.BatchNorm2d(self.hidden)

        self.conv_weight_A = nn.Parameter(torch.randn(self.hidden, self.rank))
        self.conv_weight_B = nn.Parameter(
            torch.randn(self.rank, self.hidden // self.group * self.kernel_size * self.kernel_size))
        self.conv_bias = nn.Parameter(torch.zeros(self.hidden))
        nn.init.kaiming_uniform_(self.conv_weight_A, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.conv_weight_B, a=math.sqrt(5))

        self.amp_fuse = nn.Conv2d(self.hidden, self.hidden, 1, 1, groups=self.hidden)
        self.pha_fuse = nn.Conv2d(self.hidden, self.hidden, 1, 1, groups=self.hidden)
        nn.init.ones_(self.pha_fuse.weight)
        nn.init.ones_(self.amp_fuse.weight)
        nn.init.zeros_(self.amp_fuse.bias)
        nn.init.zeros_(self.pha_fuse.bias)

        self.compress = nn.Conv2d(self.hidden, 1, 1, 1)
        self.proj = nn.Sequential(
            nn.Linear(self.hidden, self.hidden // 2),
            nn.GELU(),
            nn.Linear(self.hidden // 2, self.hidden),
        )
        self.tail = nn.Conv2d(self.hidden, d_model, 1, 1, bias=False)
        nn.init.zeros_(self.tail.weight)


        self.channel_interaction = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.hidden, self.hidden // 8, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(self.hidden // 8,self.hidden, kernel_size=1)
        )
        nn.init.zeros_(self.channel_interaction[3].weight)
        nn.init.zeros_(self.channel_interaction[3].bias)


        self.spatial_interaction = nn.Conv2d(self.hidden, 1, kernel_size=1)
        nn.init.zeros_(self.spatial_interaction.weight)
        nn.init.zeros_(self.spatial_interaction.bias)


    def forward(self, x):
        L, N, C = x.shape
        H = W = int(math.sqrt(L))
        x = x.view(H, W, N, C).permute(2, 3, 0, 1).contiguous()  # N,C,H,W
        x = self.BN(self.head(x))

        global_x = torch.fft.rfft2(x, dim=(2, 3), norm='ortho')
        mag_x = torch.abs(global_x)
        pha_x = torch.angle(global_x)
        Mag = self.amp_fuse(mag_x)
        Pha = self.pha_fuse(pha_x)
        real = Mag * torch.cos(Pha)
        imag = Mag * torch.sin(Pha)
        global_x = torch.complex(real, imag)
        global_x = torch.fft.irfft2(global_x, s=(H, W), dim=(2, 3), norm='ortho')  # N,C,H,W
        global_x = torch.abs(global_x)

        conv_weight = (self.conv_weight_A @ self.conv_weight_B) \
            .view(self.hidden, self.hidden // self.group, self.kernel_size, self.kernel_size).contiguous()
        local_x = F.conv2d(x, weight=conv_weight, bias=self.conv_bias, stride=1, padding=1, groups=self.group)

        score = self.compress(x).view(N, 1, H * W).permute(0, 2, 1).contiguous()  # N,HW,1
        score = F.softmax(score, dim=1)
        out = x.view(N, self.hidden, H * W)  # N,C,HW
        out = out @ score  # N,C,1
        out = out.permute(2, 0, 1)  # 1,N,C
        out = self.proj(out)
        channel_score = out.permute(1, 2, 0).unsqueeze(-1).contiguous()  # N,C,1,1

        channel_gate = self.channel_interaction(global_x).sigmoid()
        spatial_gate = self.spatial_interaction(local_x).sigmoid()
        spatial_x = channel_gate*local_x+spatial_gate*global_x

        x = self.tail(channel_score*spatial_x)
        x = x.view(N, C, H * W).permute(2, 0, 1).contiguous()
        return x


