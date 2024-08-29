# ------------------------------------------------------------------------
# Reference:
# https://github.com/facebookresearch/detr/blob/main/models/backbone.py
# ------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from typing import Optional
from spatial_correlation_sampler import SpatialCorrelationSampler

from .position_encoding import build_position_encoding

class Backbone(nn.Module):
    def __init__(self, args):
        super(Backbone, self).__init__()
        
        if args.backbone in ('resnet18', 'resnet34'):
            backbone = getattr(torchvision.models, args.backbone)(
                replace_stride_with_dilation=[False, False, args.dilation], pretrained=True)
        else:
            backbone = getattr(torchvision.models, args.backbone)(pretrained=True)

        self.num_frames = args.num_frame
        self.num_channels = 512 if args.backbone in ('resnet18', 'resnet34') else 2048

        self.motion = args.motion
        self.motion_layer = args.motion_layer
        self.corr_dim = args.corr_dim

        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        if self.motion:
            self.layer_channel = [64, 128, 256, 512]

            self.channel_dim = self.layer_channel[self.motion_layer - 1]

            self.corr_input_proj = nn.Sequential(
                nn.Conv2d(self.channel_dim, self.corr_dim, kernel_size=1, bias=False),
                nn.ReLU()
            )

            self.neighbor_size = args.neighbor_size
            self.ps = 2 * args.neighbor_size + 1
            # P = 11 = 2 * [5] + 1. Thus, args.neighbor_size = 5 = l (maximum displacement)

            self.correlation_sampler = SpatialCorrelationSampler(kernel_size=1, patch_size=self.ps, stride=1, padding=0, dilation_patch=1)

            self.corr_output_proj = nn.Sequential(
                nn.Conv2d(self.ps * self.ps, self.channel_dim, kernel_size=1, bias=False),
                nn.ReLU()
            )

    def get_local_corr(self, x):
        x = self.corr_input_proj(x)
        x = F.normalize(x, dim=1)
        x = x.reshape((-1, self.num_frames) + x.size()[1:])
        b, t, c, h, w = x.shape

        x = x.permute(0, 2, 1, 3, 4).contiguous()                                       # [B, C, T, H, W]

        # new implementation
        x_pre = x[:, :, :].permute(0, 2, 1, 3, 4).contiguous().view(-1, c, h, w)
        x_post = torch.cat([x[:, :, 1:], x[:, :, -1:]], dim=2).permute(0, 2, 1, 3, 4).contiguous().view(-1, c, h, w)
        corr = self.correlation_sampler(x_pre, x_post)                                  # [B x T, P, P, H, W]
        corr = corr.view(-1, self.ps * self.ps, h, w)                                   # [B x T, P * P, H, W]
        corr = F.relu(corr)

        corr = self.corr_output_proj(corr)

        return corr

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)

        # motion-augmented feature extraction
        # computes local correlation between two adjacent intermediate feature maps, F(t) and F(t+1)
        #By restricting the maximum displacement to l, the correlation scores of spatial position x 
        #are computed only in its local neighborhood of size P = 2l + 1
        if self.motion:
            if self.motion_layer == 1:
                corr = self.get_local_corr(x)
                x = x + corr
                x = self.layer2(x)
                x = self.layer3(x)
                x = self.layer4(x)
            elif self.motion_layer == 2:
                x = self.layer2(x)
                corr = self.get_local_corr(x)
                x = x + corr
                x = self.layer3(x)
                x = self.layer4(x)
            elif self.motion_layer == 3:
                x = self.layer2(x)
                x = self.layer3(x)
                corr = self.get_local_corr(x)
                x = x + corr
                x = self.layer4(x)
            elif self.motion_layer == 4:
                x = self.layer2(x)
                x = self.layer3(x)
                x = self.layer4(x)
                corr = self.get_local_corr(x)
                x = x + corr
            else:
                assert False
        else:
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

        return x


class MultiCorrBackbone(nn.Module):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, args):
        super(MultiCorrBackbone, self).__init__()
        
        if args.backbone in ('resnet18', 'resnet34'):
            backbone = getattr(torchvision.models, args.backbone)(
            replace_stride_with_dilation=[False, False, args.dilation],
            weights = 'IMAGENET1K_V1')
        else:
            backbone = getattr(torchvision.models, args.backbone)(pretrained=True)

        self.num_frames = args.num_frame
        self.num_channels = 512 if args.backbone in ('resnet18', 'resnet34') else 2048

        self.motion = args.motion
        self.motion_layer = args.motion_layer
        self.corr_dim = args.corr_dim

        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.layer_channel = [64, 128, 256, 512]

        self.channel_dim = self.layer_channel[self.motion_layer - 1]

        self.corr_input_proj1 = nn.Sequential(
            nn.Conv2d(self.layer_channel[2], self.corr_dim, kernel_size=1, bias=False),
            nn.ReLU()
        )
        self.corr_input_proj2 = nn.Sequential(
            nn.Conv2d(self.layer_channel[3], self.corr_dim, kernel_size=1, bias=False),
            nn.ReLU()
        )

        self.neighbor_size = args.neighbor_size
        self.ps = 2 * args.neighbor_size + 1

        # kernel_size = 1 for only correlation to neighboring frame
        self.correlation_sampler = SpatialCorrelationSampler(kernel_size=1, patch_size=self.ps,
                                                             stride=1, padding=0, dilation_patch=1)

        self.corr_output_proj1 = nn.Sequential(
            nn.Conv2d(self.ps * self.ps, self.layer_channel[2], kernel_size=1, bias=False),
            nn.ReLU()
        )
        self.corr_output_proj2 = nn.Sequential(
            nn.Conv2d(self.ps * self.ps, self.layer_channel[3], kernel_size=1, bias=False),
            nn.ReLU()
        )

    def get_local_corr(self, x, idx):
        if idx == 0:
            x = self.corr_input_proj1(x)
        else:
            x = self.corr_input_proj2(x)
        x = F.normalize(x, dim=1)
        x = x.reshape((-1, self.num_frames) + x.size()[1:]) # reshape to separate B and T dimension
        b, t, c, h, w = x.shape

        x = x.permute(0, 2, 1, 3, 4).contiguous()                                       # [B, C, T, H, W]

        # new implementation
        # frame at t
        x_pre = x[:, :, :].permute(0, 2, 1, 3, 4).contiguous().view(-1, c, h, w)
        # frame at t+1
        x_post = torch.cat([x[:, :, 1:], x[:, :, -1:]], dim=2).permute(0, 2, 1, 3, 4).contiguous().view(-1, c, h, w)
        # correlation between frames
        corr = self.correlation_sampler(x_pre, x_post)                                  # [B x T, P, P, H, W]
        corr = corr.view(-1, self.ps * self.ps, h, w)                                   # [B x T, P * P, H, W]
        corr = F.relu(corr)
        
        # Project back to ResNet conv channel
        if idx == 0:
            corr = self.corr_output_proj1(corr)
        else:
            corr = self.corr_output_proj2(corr)

        return corr

    def forward(self, x, local_corr=True):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        if local_corr is True: 
            corr = self.get_local_corr(x, 0)
            x = x + corr

        x = self.layer4(x)

        if local_corr is True: 
            corr = self.get_local_corr(x, 1)
            x = x + corr

        return x


class IncMultiCorrBackbone(nn.Module):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, args):
        super(IncMultiCorrBackbone, self).__init__()
        
        backbone = getattr(torchvision.models, args.backbone)(pretrained=True)

        self.num_frames = args.num_frame
        self.num_channels = 512 if args.backbone in ('resnet18', 'resnet34') else 2048

        self.motion = args.motion
        self.motion_layer = args.motion_layer
        self.corr_dim = args.corr_dim

        self.aux_logits = backbone.aux_logits
        self.transform_input = backbone.transform_input
        self.Conv2d_1a_3x3 = backbone.Conv2d_1a_3x3
        self.Conv2d_2a_3x3 = backbone.Conv2d_2a_3x3
        self.Conv2d_2b_3x3 = backbone.Conv2d_2b_3x3
        self.maxpool1 = backbone.maxpool1
        self.Conv2d_3b_1x1 = backbone.Conv2d_3b_1x1
        self.Conv2d_4a_3x3 = backbone.Conv2d_4a_3x3
        self.maxpool2 = backbone.maxpool2
        self.Mixed_5b = backbone.Mixed_5b
        self.Mixed_5c = backbone.Mixed_5c
        self.Mixed_5d = backbone.Mixed_5d 
        self.Mixed_6a = backbone.Mixed_6a
        self.Mixed_6b = backbone.Mixed_6b
        self.Mixed_6c = backbone.Mixed_6c 
        self.Mixed_6d = backbone.Mixed_6d
        self.Mixed_6e = backbone.Mixed_6e 
        self.AuxLogits: Optional[nn.Module] = None
        #if aux_logits:
        #    self.AuxLogits = inception_aux(768, num_classes)
        self.Mixed_7a = backbone.Mixed_7a
        self.Mixed_7b = backbone.Mixed_7b
        self.Mixed_7c = backbone.Mixed_7c
        self.avgpool = backbone.avgpool
        #self.dropout = nn.Dropout(p=dropout)
        #self.fc = nn.Linear(2048, num_classes)

        self.layer_channel = [32, 64, 80, 192, 288, 768, 1280, 2048]

        self.channel_dim = self.layer_channel[self.motion_layer - 1]

        self.corr_input_proj1 = nn.Sequential(
            nn.Conv2d(self.layer_channel[5], self.corr_dim, kernel_size=1, bias=False),
            nn.ReLU()
        )
        self.corr_input_proj2 = nn.Sequential(
            nn.Conv2d(self.layer_channel[7], self.corr_dim, kernel_size=1, bias=False),
            nn.ReLU()
        )

        self.neighbor_size = args.neighbor_size
        self.ps = 2 * args.neighbor_size + 1

        # kernel_size = 1 for only correlation to neighboring frame
        self.correlation_sampler = SpatialCorrelationSampler(kernel_size=1, patch_size=self.ps,
                                                             stride=1, padding=0, dilation_patch=1)

        self.corr_output_proj1 = nn.Sequential(
            nn.Conv2d(self.ps * self.ps, self.layer_channel[5], kernel_size=1, bias=False),
            nn.ReLU()
        )
        self.corr_output_proj2 = nn.Sequential(
            nn.Conv2d(self.ps * self.ps, self.layer_channel[7], kernel_size=1, bias=False),
            nn.ReLU()
        )

    def get_local_corr(self, x, idx):
        if idx == 0:
            x = self.corr_input_proj1(x)
        else:
            x = self.corr_input_proj2(x)
        x = F.normalize(x, dim=1)
        x = x.reshape((-1, self.num_frames) + x.size()[1:]) # reshape to separate B and T dimension
        b, t, c, h, w = x.shape

        x = x.permute(0, 2, 1, 3, 4).contiguous()                                       # [B, C, T, H, W]

        # new implementation
        # frame at t
        x_pre = x[:, :, :].permute(0, 2, 1, 3, 4).contiguous().view(-1, c, h, w)
        # frame at t+1
        x_post = torch.cat([x[:, :, 1:], x[:, :, -1:]], dim=2).permute(0, 2, 1, 3, 4).contiguous().view(-1, c, h, w)
        # correlation between frames
        corr = self.correlation_sampler(x_pre, x_post)                                  # [B x T, P, P, H, W]
        corr = corr.view(-1, self.ps * self.ps, h, w)                                   # [B x T, P * P, H, W]
        corr = F.relu(corr)
        
        # Project back to ResNet conv channel
        if idx == 0:
            corr = self.corr_output_proj1(corr)
        else:
            corr = self.corr_output_proj2(corr)

        return corr

    def forward(self, x, local_corr=True):
        # N x 3 x 299 x 299
        x = self.Conv2d_1a_3x3(x)
        # N x 32 x 149 x 149
        x = self.Conv2d_2a_3x3(x)
        # N x 32 x 147 x 147
        x = self.Conv2d_2b_3x3(x)
        # N x 64 x 147 x 147
        x = self.maxpool1(x)
        # N x 64 x 73 x 73
        x = self.Conv2d_3b_1x1(x)
        # N x 80 x 73 x 73
        x = self.Conv2d_4a_3x3(x)
        # N x 192 x 71 x 71
        x = self.maxpool2(x)
        # N x 192 x 35 x 35
        x = self.Mixed_5b(x)
        # N x 256 x 35 x 35
        x = self.Mixed_5c(x)
        # N x 288 x 35 x 35
        x = self.Mixed_5d(x)
        # N x 288 x 35 x 35
        
        x = self.Mixed_6a(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6b(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6c(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6d(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6e(x)
        # N x 768 x 17 x 17
        corr = self.get_local_corr(x, 0)
        x = x + corr

        # N x 768 x 17 x 17
        x = self.Mixed_7a(x)
        # N x 1280 x 8 x 8
        x = self.Mixed_7b(x)
        # N x 2048 x 8 x 8
        x = self.Mixed_7c(x)
        # N x 2048 x 8 x 8
        
        corr = self.get_local_corr(x, 1)
        x = x + corr
        
        
        
        
        # Adaptive average pooling
        #x = self.avgpool(x)
        # N x 2048 x 1 x 1
        #x = self.dropout(x)
        # N x 2048 x 1 x 1
        #x = torch.flatten(x, 1)
        # N x 2048
        #x = self.fc(x)
        # N x 1000 (num_classes)
        return x

class MyRes18(nn.Module):
    def __init__(self, pretrained = False):
        super(MyRes18, self).__init__()
        res18 = torchvision.models.resnet18(pretrained = pretrained)
        self.features = nn.Sequential(
            res18.conv1,
            res18.bn1,
            res18.relu,
            res18.maxpool,
            res18.layer1,
            res18.layer2,
            res18.layer3,
            res18.layer4
        )

    def forward(self, x):
        x = self.features(x)
        return [x]
    

class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, x, local_corr=True):
        #features = self[0](x, local_corr=local_corr)
        features = self[0](x)
        if type(features) is list:
            pos = []
            pos.append(self[1](features[0]).to(x.dtype))
            pos.append(self[1](features[1]).to(x.dtype))
        else:
            pos = self[1](features).to(x.dtype)
        
        return features, pos


def build_backbone(args,vlm_prop=None):
    position_embedding = build_position_encoding(args)
    #multi_corr is on for NBA
    # For NBA dataset, we insert two motion feature modules
    # after 4th and 5th residual block. 
    # For Volleyball dataset, we insert one motion feature 
    # module after the last residual block.
    if args.backbone == 'inception_v3':
        backbone = IncMultiCorrBackbone(args)
        num_channel = backbone.num_channels
    
    else:
        if args.multi_corr:
            backbone = MultiCorrBackbone(args)
        else:
            backbone = Backbone(args)
        num_channel = backbone.num_channels
    
    # this part make the output also include positional embedding
    model = Joiner(backbone, position_embedding)
    model.num_channels = num_channel
    
    return model
