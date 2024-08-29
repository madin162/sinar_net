import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms

import time
import numpy as np

from models.detr_transformer import build_transformer
from models.position_encoding import PositionEmbeddingLearned, build_position_encoding

from .backbone import build_backbone
from .token_encoder import Transformer, build_token_encoder, build_transformer_encoder, build_transformer_decoder
from .transformer_block import MultiScaleEncoder, TransformerBlock
from .detr_transformer import MLP
import collections
from itertools import repeat
import math
import random

class SInAR_Net(nn.Module):
    def __init__(self, args):
        super(SInAR_Net, self).__init__()

        self.dataset = args.dataset
        self.num_class = args.num_activities

        # model parameters
        self.num_frame = args.num_frame
        self.hidden_dim = args.hidden_dim
        self.num_tokens = args.num_tokens

        # feature extraction
        self.backbone = build_backbone(args)
        #self.token_encoder = build_token_encoder(args)
        self.token_encoder = TransformerBlock(
        t_encoder_embed_dim=args.hidden_dim,
        t_encoder_num_heads=args.nheads,
        t_encoder_depth=args.enc_layers,
        mlp_ratio = args.ffn_dim//args.hidden_dim,
        drop=args.drop_rate,
        attn_type='cross_att'
        )

        self.query_embed = nn.Embedding(self.num_tokens, self.hidden_dim)
        self.input_proj = nn.Conv2d(self.backbone.num_channels, args.hidden_dim, kernel_size=1)
        
        self.num_mem_token = 4 
        if self.dataset == 'Volleyball':
            self.num_caption_token = 3
        elif self.dataset == 'nba':
            self.num_caption_token = 12
        
        self.mask_token = nn.Parameter(torch.randn(args.hidden_dim))
        self.query_a2s = nn.Embedding(self.num_caption_token, self.hidden_dim)
        self.actor2semantic_decoder =  build_transformer_decoder(
            d_model=args.hidden_dim,
            nheads=args.nheads,
            nlayers=args.enc_layers//2,
            dim_feedforward=args.ffn_dim,
            dropout=args.drop_rate,
            normalize_before=args.pre_norm
        )
        self.position_embedding_a2s = build_position_encoding(args, N_steps=args.hidden_dim)
        self.query_s2s = nn.Embedding(self.num_caption_token, self.hidden_dim)
        self.scene2semantic_decoder =  build_transformer_decoder(
            d_model=args.hidden_dim,
            nheads=args.nheads,
            nlayers=args.enc_layers//2,
            dim_feedforward=args.ffn_dim,
            dropout=args.drop_rate,
            normalize_before=args.pre_norm
        )
        self.position_embedding_s2s = build_position_encoding(args, N_steps=args.hidden_dim)

        if self.dataset == 'Volleyball':
            self.actsem_pos = nn.Parameter(torch.randn(1+self.num_tokens+(self.num_frame)+self.num_caption_token,args.hidden_dim))
        elif self.dataset == 'nba':
            self.actsem_pos = nn.Parameter(torch.randn(1+self.num_tokens+(self.num_frame-8)+self.num_caption_token,args.hidden_dim))
        self.scesem_pos = nn.Parameter(torch.randn(self.num_frame+self.num_caption_token,args.hidden_dim))
        self.norm_actsem = nn.LayerNorm(self.hidden_dim)
        
        self.group_encoder =  build_transformer_encoder(
            d_model = args.hidden_dim,
            nheads = args.nheads,
            nlayers = args.enc_layers//2,
            dim_feedforward = args.ffn_dim,
            dropout = args.drop_rate
        )
        
        if self.dataset == 'Volleyball':
            self.conv1 = nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=3, stride=1, padding=1)
            self.conv2 = nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=3, stride=1, padding=1)
        elif self.dataset == 'nba':
            self.conv1 = nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=5, stride=1)
            self.conv2 = nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=5, stride=1)
            self.conv3 = nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=5, stride=1)
        else:
            assert False

        self.self_attn = nn.MultiheadAttention(args.hidden_dim, args.nheads_agg, dropout=args.drop_rate)
        self.dropout1 = nn.Dropout(args.drop_rate)
        self.norm1 = nn.LayerNorm(self.hidden_dim)
        self.norm2 = nn.LayerNorm(self.hidden_dim)
        self.classifier = nn.Linear(self.hidden_dim, self.num_class) if self.num_class > 0 else nn.Identity()
        self.classifier_scene = nn.Linear(self.hidden_dim, self.num_class) if self.num_class > 0 else nn.Identity()

        self.relu = F.relu
        self.gelu = F.gelu

        for name, m in self.named_modules():
            if 'backbone' not in name and 'token_encoder' not in name:
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)


        self.mlp_embed_caption = MLP(1024, args.hidden_dim, args.hidden_dim, 3)
        self.caption_cls = nn.Linear(self.hidden_dim, self.num_class)
        self.norm_caption = nn.LayerNorm(self.hidden_dim)
        
        self.len_capt = self.num_caption_token
        if self.dataset == 'Volleyball':
            self.conv1_scene = nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=[3,3], stride=[1,1], padding=[1,0])
            self.conv2_scene = nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=[3,3], stride=[1,1], padding=[1,0])
        
        elif self.dataset == 'nba':
            self.conv1_scene = nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=[5,3], stride=[1,3])
            self.conv2_scene = nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=[5,3], stride=[1,3])

        self.pool_frame = nn.AdaptiveAvgPool2d((1, 1))
        self.norm2_scene = nn.LayerNorm(self.hidden_dim)
        self.self_attn_scene = nn.MultiheadAttention(args.hidden_dim, args.nheads_agg, dropout=args.drop_rate)
        self.dropout1_scene = nn.Dropout(args.drop_rate)
        
        self.dropout2 = nn.Dropout(args.drop_rate)
        self.classifier_frame = nn.Linear(self.hidden_dim, self.num_class)
        self.norm1_frame = nn.LayerNorm(self.hidden_dim)
        self.cls_token = nn.Parameter(torch.randn(1,1,args.hidden_dim))
    
    def forward(self, x, caption_feat = None, label_embed = None):
        """
        :param 
        x: [B, T, 3, H, W]
        caption_feat: [B, S, Dim]
        :return:
        """
        out = {}
        b, t, _, h, w = x.shape
        x = x.reshape(b * t, 3, h, w)
        
        src, pos = self.backbone(x)      # [B x T, C, H', W']
        _, c, oh, ow = src.shape
        
        src = self.input_proj(src) #Project C to F dimension
        obj_queries  = self.query_embed.weight.unsqueeze(0).repeat(b*t,1,1)
        src = src.flatten(2).permute(0, 2, 1)
        pos = pos.flatten(2).permute(0, 2, 1)
        representations, att_stacks = self.token_encoder(obj_queries, k = src, pos=pos, get_att = True)
        att_map = att_stacks[-1]
        
        representations = representations.reshape(b, t, self.num_tokens, -1)      # [B, T, K, D]
        if self.dataset == 'Volleyball':
            rep_base = representations.clone()
            # Aggregation along T dimension (Temporal conv), then K dimension
            representations = representations.permute(0, 2, 3, 1).contiguous()                  # [B, K, D, T]
            representations = representations.reshape(b * self.num_tokens, -1, t)               # [B x K, D, T]
            representations = self.conv1(representations)
            representations = self.relu(representations)
            representations = self.conv2(representations)
            representations = self.relu(representations)
            representations = torch.mean(representations, dim=2)
            representations = self.norm1(representations)
            # transformer encoding
            representations = representations.reshape(b, self.num_tokens, -1)                   # [B, K, D]

            rep_base = rep_base.reshape(b*t, self.num_tokens, -1)    # [B, K, D]
            rep_base = rep_base.permute(1, 0, 2).contiguous()  # [K, B*T, D]
            q = k = v = rep_base
            rep_base2, _ = self.self_attn(q, k, v)
            rep_base = rep_base + self.dropout2(rep_base2)
            rep_base = self.norm1_frame(rep_base)
            rep_base = rep_base.permute(1, 0, 2).contiguous()  # [B*T, K, D]
            scene_rep = rep_base.reshape(b,t, self.num_tokens, -1).clone().detach() 
            scene_rep = scene_rep.permute(0,3,1,2)     # [B, D, T, K]
            rep_base = torch.mean(rep_base, dim=1)  # [B*T, D]
            rep_base = rep_base.reshape(b*t, -1)
            activities_scores_frame = self.classifier_frame(rep_base)
            out['score_frame'] = activities_scores_frame

            pos_a2s = self.position_embedding_a2s(representations.permute(0,2,1))
            rec_token_a2s,_ = self.actor2semantic_decoder(self.query_a2s.weight, representations.permute(1,0,2).detach(), pos=pos_a2s.permute(1,0,2))

            pos_s2s = self.position_embedding_s2s(rep_base.reshape(b,t,-1).permute(0,2,1))
            rec_token_s2s,_ = self.scene2semantic_decoder(self.query_s2s.weight, rep_base.reshape(b,t,-1).permute(1,0,2).detach(), pos=pos_s2s.permute(1,0,2))

            rec_token = rec_token_a2s.permute(1,0,2) + rec_token_s2s.permute(1,0,2)

            if self.training:
                caption_feat = self.mlp_embed_caption(caption_feat).type(representations.dtype)
                caption_feat_emb = caption_feat.clone()
                _, len_capt, _ = caption_feat.shape
                mask_ratio = 0.7
                len_mask = int(mask_ratio * len_capt)
                if len_mask>0:
                    idx_mask = torch.randperm(len_capt)[:len_mask]
                    caption_feat[:,idx_mask] =self.mask_token
            
            else:
                caption_feat = rec_token
            
            representations = representations.permute(1, 0, 2).contiguous()                     # [K, B, D]
            q = k = v = representations
            representations2, _ = self.self_attn(q, k, v)
            representations = representations + self.dropout1(representations2)
            representations = self.norm2(representations)
            representations = representations.permute(1, 0, 2).contiguous()                     # [B, K, D]
            representations_cls = torch.mean(representations, dim=1)

            scene_rep = self.conv1_scene(scene_rep)
            scene_rep = self.relu(scene_rep)
            scene_rep = self.conv2_scene(scene_rep)
            scene_rep = self.relu(scene_rep)
            scene_rep = torch.mean(scene_rep, dim=-1)    # [B, D, T2]
            scene_rep_cls = torch.mean(scene_rep, dim=-1)    # [B, D]
            scene_rep = scene_rep.permute(0,2,1)

            cls_token = self.cls_token.repeat(b,1,1)
            rep_final = torch.cat((cls_token,representations.detach(),scene_rep.detach(),caption_feat), dim=1)
            actsem_pos = self.actsem_pos.unsqueeze(0).repeat(b,1,1)
            rep_final = self.group_encoder(src = rep_final,pos=actsem_pos)
            score_caption = self.caption_cls(rep_final[0])
            rep_cls_enc = torch.mean(rep_final[1:self.num_tokens],dim=0)
            scene_cls_enc = torch.mean(rep_final[1+self.num_tokens:1+self.num_tokens+(self.num_frame)],dim=0)

        elif self.dataset == 'nba':
            rep_base = representations.clone()
            representations = representations.permute(0, 2, 3, 1).contiguous()                  # [B, K, D, T]
            
            # Partial Context Embedding
            representations = representations.reshape(b * self.num_tokens, -1, t)               # [B x K, D, T]
            representations = self.conv1(representations)
            representations = self.relu(representations)
            representations = self.conv2(representations)
            representations = self.relu(representations)
            representations = self.conv3(representations)
            representations = self.relu(representations)

            # Average Pool to combine the temporal dimensions
            representations = torch.mean(representations, dim=2)  # [B x K, D]
            representations = self.norm1(representations)
            
            # transformer encoding
            representations = representations.reshape(b, self.num_tokens, -1)  # [B, K, D]

            rep_base = rep_base.reshape(b*t, self.num_tokens, -1)    # [B, K, D]
            rep_base = rep_base.permute(1, 0, 2).contiguous()  # [K, B*T, D]
            q = k = v = rep_base
            rep_base2, _ = self.self_attn(q, k, v)
            rep_base = rep_base + self.dropout2(rep_base2)
            rep_base = self.norm1_frame(rep_base)
            rep_base = rep_base.permute(1, 0, 2).contiguous()  # [B*T, K, D]
            scene_rep = rep_base.reshape(b,t, self.num_tokens, -1).clone().detach() 
            scene_rep = scene_rep.permute(0,3,1,2)     # [B, D, T, K]
            rep_base = torch.mean(rep_base, dim=1)  # [B*T, D]
            rep_base = rep_base.reshape(b*t, -1)
            activities_scores_frame = self.classifier_frame(rep_base)
            out['score_frame'] = activities_scores_frame

            pos_a2s = self.position_embedding_a2s(representations.permute(0,2,1))
            rec_token_a2s,_ = self.actor2semantic_decoder(self.query_a2s.weight, representations.permute(1,0,2).detach(), pos=pos_a2s.permute(1,0,2))
            
            pos_s2s = self.position_embedding_s2s(rep_base.reshape(b,t,-1).permute(0,2,1))
            rec_token_s2s,_ = self.scene2semantic_decoder(self.query_s2s.weight, rep_base.reshape(b,t,-1).permute(1,0,2).detach(), pos=pos_s2s.permute(1,0,2))
            rec_token = rec_token_a2s.permute(1,0,2) + rec_token_s2s.permute(1,0,2)

            if self.training:
                caption_feat = self.mlp_embed_caption(caption_feat).type(representations.dtype)
                caption_feat_emb = caption_feat.clone()
                _, len_capt, _ = caption_feat.shape
                mask_ratio = 0.7
                len_mask = int(mask_ratio * len_capt)
                if len_mask>0:
                    idx_mask = torch.randperm(len_capt)[:len_mask]
                    caption_feat[:,idx_mask] =self.mask_token
                
            else:
                
                caption_feat = rec_token

            representations = representations.permute(1, 0, 2).contiguous()  # [K, B, D]
            q = k = v = representations
            representations2, _ = self.self_attn(q, k, v)
            representations = representations + self.dropout1(representations2)
            representations = self.norm2(representations)
            representations = representations.permute(1, 0, 2).contiguous()   # [B, K, D]
            representations_cls = torch.mean(representations, dim=1)     # [B, D]

            scene_rep = self.conv1_scene(scene_rep)
            scene_rep = self.relu(scene_rep)
            scene_rep = self.conv2_scene(scene_rep)
            scene_rep = self.relu(scene_rep)
            scene_rep = torch.mean(scene_rep, dim=-1)    # [B, D, T2]
            scene_rep_cls = torch.mean(scene_rep, dim=-1)    # [B, D]
            scene_rep = scene_rep.permute(0,2,1)
            
            cls_token = self.cls_token.repeat(b,1,1)
            rep_final = torch.cat((cls_token,representations.detach(),scene_rep.detach(),caption_feat), dim=1)
            actsem_pos = self.actsem_pos.unsqueeze(0).repeat(b,1,1)
            rep_final = self.group_encoder(src = rep_final,pos=actsem_pos)
            score_caption = self.caption_cls(rep_final[0])
            rep_cls_enc = torch.mean(rep_final[1:self.num_tokens],dim=0)
            scene_cls_enc = torch.mean(rep_final[1+self.num_tokens:1+self.num_tokens+(self.num_frame-8)],dim=0)

        activities_scores = self.classifier(representations_cls + rep_cls_enc) + self.classifier_scene(scene_rep_cls + scene_cls_enc)    # [B, C]
        
        activities_scores = activities_scores + score_caption
        out['score'] = activities_scores
        out['att_map']=att_map

        # add rec_token decoder here
        if self.training:
            out['rec_token'] = rec_token
            out['caption_feat'] = caption_feat_emb
            out['video_feature'] = representations

        return out

class BaseModel(nn.Module):
    def __init__(self, args):
        super(BaseModel, self).__init__()

        self.num_class = args.num_activities

        # model parameters
        self.num_frame = args.num_frame
        self.hidden_dim = args.hidden_dim

        # feature extraction
        self.backbone = build_backbone(args)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(self.backbone.num_channels, self.num_class)

        for name, m in self.named_modules():
            if 'backbone' not in name and 'token_encoder' not in name:
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        :param x: [B, T, 3, H, W]
        :return:
        """
        b, t, _, h, w = x.shape
        x = x.reshape(b * t, 3, h, w)

        src, pos = self.backbone(x)                                                             # [B x T, C, H', W']
        _, c, oh, ow = src.shape

        representations = self.avg_pool(src)
        representations = representations.reshape(b, t, c)

        representations = representations.reshape(b * t, self.backbone.num_channels)        # [B, T, F]
        activities_scores = self.classifier(representations)
        activities_scores = activities_scores.reshape(b, t, -1).mean(dim=1)

        return activities_scores

class SpatioTemporalResBlock(nn.Module):
    r"""Single block for the ResNet network. Uses SpatioTemporalConv in 
        the standard ResNet block layout (conv->batchnorm->ReLU->conv->batchnorm->sum->ReLU)
        
        Args:
            in_channels (int): Number of channels in the input tensor.
            out_channels (int): Number of channels in the output produced by the block.
            kernel_size (int or tuple): Size of the convolving kernels.
            downsample (bool, optional): If ``True``, the output size is to be smaller than the input. Default: ``False``
        """
    def __init__(self, in_channels, out_channels, kernel_size, downsample=False):
        super(SpatioTemporalResBlock, self).__init__()
        
        # If downsample == True, the first conv of the layer has stride = 2 
        # to halve the residual output size, and the input x is passed 
        # through a seperate 1x1x1 conv with stride = 2 to also halve it.

        # no pooling layers are used inside ResNet
        self.downsample = downsample
        
        # to allow for SAME padding
        # hack to allow padding with 0 for temporal
        if(type(kernel_size).__name__=='int'):
            padding = kernel_size//2
        else:
            #print(type(kernel_size))
            if((kernel_size[0]==1)):
                padding=[0,kernel_size[1]//2,kernel_size[2]//2]
            else:
                padding=[kernel_size[0]//2,0,0]
     
        if self.downsample:
            # downsample with stride =2 the input x
            if(type(kernel_size).__name__!='int'):
                if(kernel_size[0]==1):
                    self.downsampleconv = SpatioTemporalConv(in_channels, out_channels, 1, stride=[1,2,2])
                    self.downsamplebn = nn.BatchNorm3d(out_channels)
                    # downsample with stride = 2when producing the residual
                    self.conv1 = SpatioTemporalConv(in_channels, out_channels, kernel_size, padding=padding, stride=[1,2,2])
                
            else:
                self.downsampleconv = SpatioTemporalConv(in_channels, out_channels, 1, stride=2)
                self.downsamplebn = nn.BatchNorm3d(out_channels)
                # downsample with stride = 2when producing the residual
                self.conv1 = SpatioTemporalConv(in_channels, out_channels, kernel_size, padding=padding, stride=2)

        else:
            self.conv1 = SpatioTemporalConv(in_channels, out_channels, kernel_size, padding=padding)

        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu1 = nn.ReLU()

        # standard conv->batchnorm->ReLU
        #self.conv2 = SpatioTemporalConv(out_channels, out_channels, kernel_size, padding=padding)
        #self.bn2 = nn.BatchNorm3d(out_channels)
        #self.outrelu = nn.ReLU()

    def forward(self, x):
        res = self.relu1(self.bn1(self.conv1(x)))    
        res = self.bn2(self.conv2(res))
        if self.downsample:
            x = self.downsamplebn(self.downsampleconv(x))

        return self.outrelu(x + res)

class SpatioTemporalResLayer(nn.Module):
    r"""Forms a single layer of the ResNet network, with a number of repeating 
    blocks of same output size stacked on top of each other
        
        Args:
            in_channels (int): Number of channels in the input tensor.
            out_channels (int): Number of channels in the output produced by the layer.
            kernel_size (int or tuple): Size of the convolving kernels.
            layer_size (int): Number of blocks to be stacked to form the layer
            block_type (Module, optional): Type of block that is to be used to form the layer. Default: SpatioTemporalResBlock. 
            downsample (bool, optional): If ``True``, the first block in layer will implement downsampling. Default: ``False``
        """

    def __init__(self, in_channels, out_channels, kernel_size, layer_size, block_type=SpatioTemporalResBlock, downsample=False):
        
        super(SpatioTemporalResLayer, self).__init__()

        # implement the first block
        self.block1 = block_type(in_channels, out_channels, kernel_size, downsample)

        # prepare module list to hold all (layer_size - 1) blocks
        self.blocks = nn.ModuleList([])
        for i in range(layer_size - 1):
            # all these blocks are identical, and have downsample = False by default
            self.blocks += [block_type(out_channels, out_channels, kernel_size)]
        
        self.block_type = block_type

    def forward(self, x, x_sp=None):
        if(isinstance(x, tuple)):
            x=x[0]
        x = self.block1(x)
        for block in self.blocks:
            x = block(x)

        return x    


class SpatioTemporalConv(nn.Module):
    r"""Applies a factored 3D convolution over an input signal composed of several input 
    planes with distinct spatial and time axes, by performing a 2D convolution over the 
    spatial axes to an intermediate subspace, followed by a 1D convolution over the time 
    axis to produce the final output.
    Args:
        in_channels (int): Number of channels in the input tensor
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to the sides of the input during their respective convolutions. Default: 0
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(SpatioTemporalConv, self).__init__()

        # if ints are entered, convert them to iterables, 1 -> [1, 1, 1]
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)

        # decomposing the parameters into spatial and temporal components by
        # masking out the values with the defaults on the axis that
        # won't be convolved over. This is necessary to avoid unintentional
        # behavior such as padding being added twice
        spatial_kernel_size =  [1, kernel_size[1], kernel_size[2]]
        spatial_stride =  [1, stride[1], stride[2]]
        spatial_padding =  [0, padding[1], padding[2]]

        temporal_kernel_size = [kernel_size[0], 1, 1]
        temporal_stride =  [stride[0], 1, 1]
        temporal_padding =  [padding[0], 0, 0]

        # compute the number of intermediary channels (M) using formula 
        # from the paper section 3.5
        #intermed_channels = int(math.floor((kernel_size[0] * kernel_size[1] * kernel_size[2] * in_channels * out_channels)/ \
        #                    (kernel_size[1]* kernel_size[2] * in_channels + kernel_size[0] * out_channels)))
        intermed_channels = out_channels

        # the spatial conv is effectively a 2D conv due to the 
        # spatial_kernel_size, followed by batch_norm and ReLU
        self.spatial_conv = nn.Conv3d(in_channels, intermed_channels, spatial_kernel_size,
                                    stride=spatial_stride, padding=spatial_padding, bias=bias)
        self.bn = nn.BatchNorm3d(intermed_channels)
        self.relu = nn.ReLU()

        # the temporal conv is effectively a 1D conv, but has batch norm 
        # and ReLU added inside the model constructor, not here. This is an 
        # intentional design choice, to allow this module to externally act 
        # identical to a standard Conv3D, so it can be reused easily in any 
        # other codebase
        self.temporal_conv = nn.Conv3d(intermed_channels, out_channels, temporal_kernel_size, 
                                    stride=temporal_stride, padding=temporal_padding, bias=bias)

    def forward(self, x):
        x = self.relu(self.bn(self.spatial_conv(x)))
        x = self.temporal_conv(x)
        return x

def _ntuple(n, name="parse"):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return tuple(x)
        return tuple(repeat(x, n))

    parse.__name__ = name
    return parse

_triple = _ntuple(3, "_triple")

def batched_index_select(input, dim, index):
    # input: B x * x ... x *
    # dim: 0 < scalar
    # index: B x M
    for ii in range(1, len(input.shape)):
        if ii != dim:
            index = index.unsqueeze(ii)
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.expand(expanse)
    return torch.gather(input, dim, index)