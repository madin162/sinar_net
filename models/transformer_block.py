from itertools import repeat
import collections.abc

from typing import Optional
from torch import nn, Tensor


class MultiScaleEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        self.num_scale_modules = args.num_scale_modules
        self.tf_blocks = nn.ModuleList([
        TransformerBlock(
            t_encoder_embed_dim=args.hidden_dim,
            t_encoder_num_heads=args.nheads,
            t_encoder_depth=args.enc_layers//self.num_scale_modules,
            mlp_ratio = args.ffn_dim//args.hidden_dim,
            drop=args.drop_rate,
            attn_type='cross_att',attn_add_mode=True) 
            for i in range(self.num_scale_modules)
        ])

        self.z_tf_blocks = nn.ModuleList([
        TransformerBlock(
            t_encoder_embed_dim=args.hidden_dim,
            t_encoder_num_heads=args.nheads,
            t_encoder_depth=args.enc_layers//self.num_scale_modules,
            mlp_ratio = 4,
            drop=args.drop_rate) 
            for i in range(self.num_scale_modules-1)
        ])
        
        self.src_pools = nn.ModuleList(
        [nn.Identity()] +    
        [nn.AvgPool2d(args.scale_pool_size,args.scale_pool_size)
            for i in range(self.num_scale_modules-1)
        ])
        self.num_tokens = args.num_tokens

    def forward(self, obj_queries, src, get_att=False):
        src_list = []
        shape_list = []
        src_now = src
        for i in range(self.num_scale_modules):
            src_now = self.src_pools[i](src_now)
            src_list.append(src_now)
            shape_list.append([src_now.shape[0], self.num_tokens] + list(src_now.shape[2:]))
            
        att_map = 0
        representations = obj_queries
        att_stacks = []
        for i,blk in enumerate(self.tf_blocks):
            if i>0:
                representations = self.z_tf_blocks[i-1](representations)
            
            id_src = self.num_scale_modules-1-i
            src = src_list[id_src]
            src = src.flatten(2).permute(0, 2, 1) #sequence-first
            representations, att_map = self.tf_blocks[i](representations, k = src, get_att = True, attn_add=att_map)
            att_stacks.append(att_map)
            att_map = att_map.reshape(shape_list[id_src])
            att_map = nn.functional.interpolate(att_map,size=shape_list[max(0,id_src-1)][-2:],mode='nearest')
            #print('here' + str(att_map.shape))
        
        #src_1 = self.src_pool1(src)
        #src_2 = self.src_pool2(src_1)
        
        #src_3 = src_2.flatten(2).permute(0, 2, 1) #sequence-first
        #src_2 = src_1.flatten(2).permute(0, 2, 1) #sequence-first
        #src_1 = src.flatten(2).permute(0, 2, 1) #sequence-first
        
        #representations, att_map = self.tf_lvl_2(obj_queries, k = src_2, get_att = True)
        #representations, att_map = self.tf_lvl_1(obj_queries, k = src, get_att = True)
        if get_att:
            return representations, att_stacks
        else:
            return representations
        
class TransformerBlock(nn.Module):
    def __init__(self, t_encoder_embed_dim, t_encoder_num_heads, t_encoder_depth, mlp_ratio, attn_add_mode = False,
                 norm_layer=nn.LayerNorm, drop=0., attn_drop=0., drop_path=0., qkv_bias=True, attn_type = 'self_att'):
        super().__init__()
        self.attn_type = attn_type
        self.blocks = nn.ModuleList([
            Block(t_encoder_embed_dim, t_encoder_num_heads, mlp_ratio, qkv_bias=qkv_bias,
                  drop=drop, attn_drop=attn_drop, drop_path=drop_path, norm_layer=norm_layer, attn_type=attn_type)
            for i in range(t_encoder_depth)
        ])
        self.attn_add_mode = attn_add_mode

    def forward(self, x, k=None, src_key_padding_mask=None, pos=None, get_att=None, attn_add=0):
        att_stacks = []
        for blk in self.blocks:
            if self.attn_type=='cross_att':
                x = [x, k]
            
            if not self.attn_add_mode:
                attn_add = 0

            blk_out = blk(x, src_key_padding_mask, pos, get_att, attn_add=attn_add)
            if get_att:
                x, att = blk_out
                att_stacks.append(att)
            else:
                x = blk_out
            
        if get_att:
            return x, att_stacks
        else:
            return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, attn_type = 'self_att'):
        super().__init__()
        self.norm1 = norm_layer(dim)
        if attn_type == 'self_att':
            self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        elif attn_type == 'cross_att':
            self.attn = CrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
            self.z_attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
            self.norm_z = norm_layer(dim)
        
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.attn_type = attn_type

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, x, src_key_padding_mask=None, pos=None, get_att=None, attn_add=0):
        if self.attn_type == 'cross_att':
            [x, k] = x
        
        #if pos is None:
        if self.attn_type == 'cross_att':
            k = self.with_pos_embed(k, pos)
        else:
            x = self.with_pos_embed(x, pos)
        
        # add Z encoder
        if self.attn_type == 'cross_att':
            x2 = self.norm_z(x)
            x2 = self.z_attn(x2)
            x = x + self.drop_path(x2)
        # end of Z encoder

        x2 = self.norm1(x)
        if self.attn_type == 'cross_att':
            x2 = [x2,k]
        
        if get_att:
            x2, att = self.attn(x2, attn_add=attn_add, get_att = get_att)
            x = x + self.drop_path(x2)
        else:
            x = x + self.drop_path(self.attn(x2))
        
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        if get_att:
            return x, att
        else:
            return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q_em = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x_k, attn_add=0, get_att = None):
        x,k = x_k
        B, N_x, C = x.shape
        B, N_k, C = k.shape
        
        q = self.q_em(x).reshape(B, N_x, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(k).reshape(B, N_k, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        if type(attn_add) is not int:
            #print(attn.shape)
            attn_add = attn_add.flatten(2).unsqueeze(1)
            attn_add = attn_add.repeat(1,self.num_heads,1,1)
            #print(attn.shape)
            attn = attn*attn_add
            
            #attn = attn+attn_add.detach()
            attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N_x, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        if get_att:
            attn = attn.mean(dim=1) #average across heads
            return x, attn
        else:
            return x


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None, scale_by_keep=True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)


# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple