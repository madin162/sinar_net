# ------------------------------------------------------------------------
# Reference:
# https://github.com/facebookresearch/detr/blob/main/models/position_encoding.py
# https://github.com/tatp22/multidim-positional-encoding
# ------------------------------------------------------------------------

import math
import torch
from torch import nn
import numpy as np
def get_emb(sin_inp):
    """
    Gets a base embedding for one dimension with sin and cos intertwined
    """
    emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
    return torch.flatten(emb, -2, -1)

class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x):
        if type(x).__name__ == 'NestedTensor':
            tensor_list = x
            x = tensor_list.tensors
            mask = tensor_list.mask
            assert mask is not None
            not_mask = ~mask
            y_embed = not_mask.cumsum(1, dtype=torch.float32)
            x_embed = not_mask.cumsum(2, dtype=torch.float32)
        else:
            if len(x.shape)==4:
                bs, c, h, w = x.shape
                y_embed = torch.arange(1, h + 1, device=x.device).unsqueeze(0).unsqueeze(2)
                y_embed = y_embed.repeat(bs, 1, w)
                x_embed = torch.arange(1, w + 1, device=x.device).unsqueeze(0).unsqueeze(1)
                x_embed = x_embed.repeat(bs, h, 1)

                if self.normalize:
                    eps = 1e-6
                    y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
                    x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

                dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
                dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

                pos_x = x_embed[:, :, :, None] / dim_t
                pos_y = y_embed[:, :, :, None] / dim_t
                pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
                pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
                pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)

            elif len(x.shape)==3:
                bs, c, w = x.shape
                x_embed = torch.arange(1, w + 1, device=x.device).unsqueeze(0)
                x_embed = x_embed.repeat(bs, 1)

                if self.normalize:
                    eps = 1e-6
                    x_embed = x_embed / (x_embed[:, -1:] + eps) * self.scale

                dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
                dim_t = self.temperature ** (2 * (dim_t) / self.num_pos_feats)

                pos_x = x_embed[:, :, None] / dim_t
                pos_x =pos_x.sin()
                pos = pos_x.permute(0, 1, 2)
                
                #pos = torch.cat((pos_x), dim=3).permute(0, 2, 1)

        return pos


class PositionalEncoding3D(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding3D, self).__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 6) * 2)
        if channels % 2:
            channels += 1
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)
        self.register_buffer("cached_penc", None, persistent=False)

    def forward(self, tensor):
        """
        :param tensor: A 5d tensor of size (batch_size, ch, x, y, z)
        :return: Positional Encoding Matrix of size (batch_size, ch, x, y, z)
        """
        tensor = tensor.permute(0,2,3,4,1)
        if len(tensor.shape) != 5:
            raise RuntimeError("The input tensor has to be 5d!")

        if self.cached_penc is not None and self.cached_penc.shape == tensor.shape:
            return self.cached_penc

        self.cached_penc = None
        batch_size, x, y, z, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device, dtype=self.inv_freq.dtype)
        pos_y = torch.arange(y, device=tensor.device, dtype=self.inv_freq.dtype)
        pos_z = torch.arange(z, device=tensor.device, dtype=self.inv_freq.dtype)
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        sin_inp_z = torch.einsum("i,j->ij", pos_z, self.inv_freq)
        emb_x = get_emb(sin_inp_x).unsqueeze(1).unsqueeze(1)
        emb_y = get_emb(sin_inp_y).unsqueeze(1)
        emb_z = get_emb(sin_inp_z)
        emb = torch.zeros(
            (x, y, z, self.channels * 3),
            device=tensor.device,
            dtype=tensor.dtype,
        )
        emb[:, :, :, : self.channels] = emb_x
        emb[:, :, :, self.channels : 2 * self.channels] = emb_y
        emb[:, :, :, 2 * self.channels :] = emb_z

        self.cached_penc = emb[None, :, :, :, :orig_ch].repeat(batch_size, 1, 1, 1, 1)
        self.cached_penc = self.cached_penc.permute(0,4,1,2,3)
        return self.cached_penc


class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """
    def __init__(self, num_pos_feats=256):
        super().__init__()
        self.row_embed = nn.Embedding(50, num_pos_feats)
        self.col_embed = nn.Embedding(50, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, x):
        h, w = x.shape[-2:]
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        return pos


def build_position_encoding(args, N_steps = None):
    if N_steps is None:
        N_steps = args.hidden_dim // 2
    if args.position_embedding in ('v2', 'sine'):
        # TODO find a better way of exposing other arguments
        position_embedding = PositionEmbeddingSine(N_steps, normalize=True)
    elif args.position_embedding in ('v3', 'learned'):
        position_embedding = PositionEmbeddingLearned(N_steps)
    else:
        raise ValueError(f"not supported {args.position_embedding}")

    return position_embedding


def build_index_encoding(args):
    N_steps = args.hidden_dim // 2
    if args.index_embedding in ('v2', 'sine'):
        # TODO find a better way of exposing other arguments
        position_embedding = PositionEmbeddingSine(N_steps, normalize=True)
    elif args.index_embedding in ('v3', 'learned'):
        position_embedding = PositionEmbeddingLearned(N_steps)
    else:
        raise ValueError(f"not supported {args.position_embedding}")

    return position_embedding
