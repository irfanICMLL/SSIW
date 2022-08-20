import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

#from mmseg.models.builder import BACKBONES
from mmseg.utils import get_root_logger
from mmcv.runner import load_checkpoint
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule

import math
#from .decode_heads.decode_head import BaseDecodeHead
from utils.decode_heads.aspp_head import ASPPHead, ASPPModule
from mmcv.cnn.bricks import build_norm_layer
from mmseg.ops import resize

# pip install timm==0.3.2
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import logging
from utils.decode_heads.segformer_head import SegFormerHead
logger = logging.getLogger(__name__)

class FCNHead(nn.Module):
    """Fully Convolution Networks for Semantic Segmentation.
    This head is implemented of `FCNNet <https://arxiv.org/abs/1411.4038>`_.
    Args:
        num_convs (int): Number of convs in the head. Default: 2.
        kernel_size (int): The kernel size for convs in the head. Default: 3.
        concat_input (bool): Whether concat the input and output of convs
            before classification layer.
    """

    def __init__(self,
                 num_convs=1,
                 kernel_size=3,
                 in_channels=320,
                 num_classes=150,
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU'),
                 **kwargs):
        assert num_convs >= 0
        self.kernel_size = kernel_size
        super(FCNHead, self).__init__()
        self.in_channels = in_channels
        self.channels = 256
        self.num_classes = num_classes
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        convs = []
        convs.append(
            ConvModule(
                self.in_channels,
                self.channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,  #
                conv_cfg=None,
                norm_cfg=self.norm_cfg, # sync bn
                act_cfg=self.act_cfg)) # relu

        self.convs = nn.Sequential(*convs)
        self.cls_seg = nn.Conv2d(in_channels=self.channels, out_channels=self.num_classes,
                                 kernel_size=1)

    def forward(self, inputs):
        """Forward function."""
        x = inputs[-2]
        output = self.convs(x)
        output = self.cls_seg(output)
        return output

class DepthwiseSeparableASPPModule(ASPPModule):
    """Atrous Spatial Pyramid Pooling (ASPP) Module with depthwise separable
    conv."""

    def __init__(self, **kwargs):
        super(DepthwiseSeparableASPPModule, self).__init__(**kwargs)
        for i, dilation in enumerate(self.dilations):
            if dilation > 1:
                self[i] = DepthwiseSeparableConvModule(
                    self.in_channels,
                    self.channels,
                    3,
                    dilation=dilation,
                    padding=dilation,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg)


class DynHead(nn.Module):
    def __init__(self,
                 in_channels,
                 num_classes,
                 norm_cfg,
                 act_cfg,
                 upsample_f,
                 dyn_ch,
                 mask_ch,
                 use_low_level_info=False,
                 channel_reduce_factor=2,
                 zero_init=False,
                 supress_std=True):
        super(DynHead, self).__init__()

        channels = dyn_ch
        num_bases = 0
        if use_low_level_info:
            num_bases = mask_ch
        num_out_channel = (2 + num_bases) * channels + \
                          channels + \
                          channels * channels + \
                          channels + \
                          channels * num_classes + \
                          num_classes

        self.classifier = nn.Sequential(
            ConvModule(
                in_channels,
                in_channels // channel_reduce_factor,
                3,
                padding=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg, ),
            nn.Conv2d(in_channels // channel_reduce_factor, num_out_channel, 1)
        )


        if zero_init:
            nn.init.constant_(self.classifier[-1].weight, 0)
        else:
            nn.init.xavier_normal_(self.classifier[-1].weight)
            if supress_std:
                param = self.classifier[-1].weight / num_out_channel
                self.classifier[-1].weight = nn.Parameter(param)
        nn.init.constant_(self.classifier[-1].bias, 0)

    def forward(self, feature):
        return self.classifier(feature)


#@HEADS.register_module()
class BilinearPADHead_fast_xavier_init(ASPPHead):
    """Encoder-Decoder with Atrous Separable Convolution for Semantic Image
    Segmentation.
    This head is the implementation of `DeepLabV3+
    <https://arxiv.org/abs/1802.02611>`_.
    Args:
        c1_in_channels (int): The input channels of c1 decoder. If is 0,
            the no decoder will be used.
        c1_channels (int): The intermediate channels of c1 decoder.
    """

    def __init__(self, c1_in_channels, c1_channels,
                 upsample_factor,
                 dyn_branch_ch,
                 mask_head_ch,
                 pad_out_channel_factor=4,
                 channel_reduce_factor=2,
                 zero_init=False,
                 supress_std=True,
                 feature_strides=None,
                 **kwargs):
        super(BilinearPADHead_fast_xavier_init, self).__init__(**kwargs)
        assert c1_in_channels >= 0
        self.pad_out_channel = self.num_classes
        self.upsample_f = upsample_factor
        self.dyn_ch = dyn_branch_ch
        self.mask_ch = mask_head_ch
        self.use_low_level_info = True
        self.channel_reduce_factor = channel_reduce_factor

        self.aspp_modules = DepthwiseSeparableASPPModule(
            dilations=self.dilations,
            in_channels=self.in_channels,
            channels=self.channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        last_stage_ch = self.channels
        self.classifier = DynHead(last_stage_ch,
                                  self.pad_out_channel,
                                  self.norm_cfg,
                                  self.act_cfg,
                                  self.upsample_f,
                                  self.dyn_ch,
                                  self.mask_ch,
                                  self.use_low_level_info,
                                  self.channel_reduce_factor,
                                  zero_init,
                                  supress_std)

        if c1_in_channels > 0:
            self.c1_bottleneck = nn.Sequential(
                ConvModule(
                    c1_in_channels,
                    c1_channels,
                    3,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg),
                ConvModule(
                    c1_channels,
                    self.mask_ch,
                    1,
                    conv_cfg=self.conv_cfg,
                    act_cfg=None,
                ),
            )
        else:
            self.c1_bottleneck = None

        _, norm = build_norm_layer(self.norm_cfg, 2 + self.mask_ch)
        self.add_module("cat_norm", norm)
        nn.init.constant_(self.cat_norm.weight, 1)
        nn.init.constant_(self.cat_norm.bias, 0)

        coord_tmp = self.computer_locations_per_level(640, 640)
        self.register_buffer("coord", coord_tmp.float(), persistent=False)

    def computer_locations_per_level(self, height, width, h=8, w=8):
        shifts_x = torch.arange(0, 1, step=1/w, dtype=torch.float32)
        shifts_y = torch.arange(0, 1, step=1/h, dtype=torch.float32)
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        locations = torch.stack((shift_x, shift_y), dim=0)
        stride_h = height // 32
        stride_w = width // 32
        coord = locations.repeat(stride_h*stride_w, 1, 1, 1)
        return coord


    def forward(self, inputs):
        """Forward function."""
        # inputs: [1/32 stage, 1/4 stage]
        x = inputs[0] # 1/32 stage

        aspp_outs = [
            resize(
                self.image_pool(x),
                size=x.size()[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        ]
        aspp_outs.extend(self.aspp_modules(x))
        aspp_outs = torch.cat(aspp_outs, dim=1)
        output = self.bottleneck(aspp_outs)

        plot = False

        if self.c1_bottleneck is not None:
            c1_output = self.c1_bottleneck(inputs[1])
            if plot:
                output2 = output
                output3 = c1_output
        if self.upsample_f != 8:
            c1_output = resize(
                c1_output,
                scale_factor=self.upsample_f // 8,
                mode='bilinear',
                align_corners=self.align_corners)
        output = self.classifier(output)
        output = self.interpolate_fast(output, c1_output, self.cat_norm)
        if plot:
            outputs = []
            outputs.append(output)
            outputs.append(output2)
            outputs.append(output3)
            return outputs

        return output

    def interpolate(self, x, x_cat=None, norm=None):
        dy_ch = self.dyn_ch
        B, conv_ch, H, W = x.size()
        x = x.view(B, conv_ch, H * W).permute(0, 2, 1)
        x = x.reshape(B * H * W, conv_ch)
        weights, biases = self.get_subnetworks_params(x, channels=dy_ch)
        f = self.upsample_f
        self.coord_generator(H, W)
        coord = self.coord.reshape(1, H, W, 2, f, f).permute(0, 3, 1, 4, 2, 5).reshape(1, 2, H * f, W * f)
        coord = coord.repeat(B, 1, 1, 1)
        if x_cat is not None:
            coord = torch.cat((coord, x_cat), 1)
            coord = norm(coord)

        B_coord, ch_coord, H_coord, W_coord = coord.size()
        coord = coord.reshape(B_coord, ch_coord, H, f, W, f).permute(0, 2, 4, 1, 3, 5).reshape(1,
                                                                                               B_coord * H * W * ch_coord,
                                                                                               f, f)
        output = self.subnetworks_forward(coord, weights, biases, B * H * W)
        output = output.reshape(B, H, W, self.pad_out_channel, f, f).permute(0, 3, 1, 4, 2, 5)
        output = output.reshape(B, self.pad_out_channel, H * f, W * f)
        return output

    def interpolate_fast(self, x, x_cat=None, norm=None):
        dy_ch = self.dyn_ch
        B, conv_ch, H, W = x.size()
        weights, biases = self.get_subnetworks_params_fast(x, channels=dy_ch)
        f = self.upsample_f
        #self.coord_generator(H, W)
        coord = self.coord.reshape(1, H, W, 2, f, f).permute(0, 3, 1, 4, 2, 5).reshape(1, 2, H * f, W * f)
        coord = coord.repeat(B, 1, 1, 1)
        if x_cat is not None:
            coord = torch.cat((coord, x_cat), 1)
            coord = norm(coord)

        output = self.subnetworks_forward_fast(coord, weights, biases, B * H * W)
        return output

    def get_subnetworks_params(self, attns, num_bases=0, channels=16):
        assert attns.dim() == 2
        n_inst = attns.size(0)
        if self.use_low_level_info:
            num_bases = self.mask_ch
        else:
            num_bases = 0

        w0, b0, w1, b1, w2, b2 = torch.split_with_sizes(attns, [
            (2 + num_bases) * channels, channels,
            channels * channels, channels,
            channels * self.pad_out_channel, self.pad_out_channel
        ], dim=1)

        w0 = w0.reshape(n_inst * channels, 2 + num_bases, 1, 1)
        b0 = b0.reshape(n_inst * channels)
        w1 = w1.reshape(n_inst * channels, channels, 1, 1)
        b1 = b1.reshape(n_inst * channels)
        w2 = w2.reshape(n_inst * self.pad_out_channel, channels, 1, 1)
        b2 = b2.reshape(n_inst * self.pad_out_channel)

        return [w0, w1, w2], [b0, b1, b2]

    def get_subnetworks_params_fast(self, attns, num_bases=0, channels=16):
        assert attns.dim() == 4
        B, conv_ch, H, W = attns.size()
        if self.use_low_level_info:
            num_bases = self.mask_ch
        else:
            num_bases = 0

        w0, b0, w1, b1, w2, b2 = torch.split_with_sizes(attns, [
            (2 + num_bases) * channels, channels,
            channels * channels, channels,
            channels * self.pad_out_channel, self.pad_out_channel
        ], dim=1)

        w0 = resize(w0, scale_factor=self.upsample_f, mode='nearest')
        b0 = resize(b0, scale_factor=self.upsample_f, mode='nearest')
        w1 = resize(w1, scale_factor=self.upsample_f, mode='nearest')
        b1 = resize(b1, scale_factor=self.upsample_f, mode='nearest')
        w2 = resize(w2, scale_factor=self.upsample_f, mode='nearest')
        b2 = resize(b2, scale_factor=self.upsample_f, mode='nearest')

        return [w0, w1, w2], [b0, b1, b2]

    def subnetworks_forward(self, inputs, weights, biases, n_subnets):
        assert inputs.dim() == 4
        n_layer = len(weights)
        x = inputs
        # NOTE: x has to be treated as min_batch size 1
        for i, (w, b) in enumerate(zip(weights, biases)):
            x = F.conv2d(
                x, w, bias=b,
                stride=1, padding=0,
                groups=n_subnets
            )
            if i < n_layer - 1:
                x = F.relu(x)
        return x

    def subnetworks_forward_fast(self, inputs, weights, biases, n_subnets):
        assert inputs.dim() == 4
        n_layer = len(weights)
        x = inputs
        if self.use_low_level_info:
            num_bases = self.mask_ch
        else:
            num_bases = 0
        for i, (w, b) in enumerate(zip(weights, biases)):
            if i == 0:
                x = self.padconv(x, w, b, cin=2 + num_bases, cout=self.dyn_ch, relu=True)
            if i == 1:
                x = self.padconv(x, w, b, cin=self.dyn_ch, cout=self.dyn_ch, relu=True)
            if i == 2:
                x = self.padconv(x, w, b, cin=self.dyn_ch, cout=self.pad_out_channel, relu=False)
        return x

    def padconv(self, input, w, b, cin, cout, relu):
        input = input.repeat(1, cout, 1, 1)
        x = input * w
        conv_w = torch.ones((cout, cin, 1, 1), device=input.device)
        x = F.conv2d(
            x, conv_w, stride=1, padding=0,
            groups=cout
        )
        x = x + b
        if relu:
            x = F.relu(x)
        return x

    def coord_generator(self, height, width):
        f = self.upsample_f
        coord = compute_locations_per_level(f, f)
        H = height
        W = width
        coord = coord.repeat(H * W, 1, 1, 1)
        self.coord = coord.to(device='cuda')


def compute_locations_per_level(h, w):
    shifts_x = torch.arange(
        0, 1, step=1 / w,
        dtype=torch.float32, device='cuda'
    )
    shifts_y = torch.arange(
        0, 1, step=1 / h,
        dtype=torch.float32, device='cuda'
    )
    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
    locations = torch.stack((shift_x, shift_y), dim=0)
    return locations



class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        return x


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W


class MixVisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1]):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths

        # patch_embed
        self.patch_embed1 = OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=4, in_chans=in_chans,
                                              embed_dim=embed_dims[0]) # 1/4
        self.patch_embed2 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1]) # 1/8
        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2]) # auxilary output
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 16, patch_size=3, stride=2, in_chans=embed_dims[2],
                                              embed_dim=embed_dims[3]) # 1/32

        # transformer encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        self.block1 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])
            for i in range(depths[0])])
        self.norm1 = norm_layer(embed_dims[0])

        cur += depths[0]
        self.block2 = nn.ModuleList([Block(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for i in range(depths[1])])
        self.norm2 = norm_layer(embed_dims[1])

        cur += depths[1]
        self.block3 = nn.ModuleList([Block(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2])
            for i in range(depths[2])])
        self.norm3 = norm_layer(embed_dims[2])

        cur += depths[2]
        self.block4 = nn.ModuleList([Block(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[3])
            for i in range(depths[3])])
        self.norm4 = norm_layer(embed_dims[3])

        # classification head
        # self.head = nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, map_location='cpu', strict=False, logger=logger)

    def reset_drop_path(self, drop_path_rate):
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        cur = 0
        for i in range(self.depths[0]):
            self.block1[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[0]
        for i in range(self.depths[1]):
            self.block2[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[1]
        for i in range(self.depths[2]):
            self.block3[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[2]
        for i in range(self.depths[3]):
            self.block4[i].drop_path.drop_prob = dpr[cur + i]

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}  # has pos_embed may be better

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        outs = []

        # stage 1
        x, H, W = self.patch_embed1(x)
        for i, blk in enumerate(self.block1):
            x = blk(x, H, W)
        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 2
        x, H, W = self.patch_embed2(x)
        for i, blk in enumerate(self.block2):
            x = blk(x, H, W)
        x = self.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 3
        x, H, W = self.patch_embed3(x)
        for i, blk in enumerate(self.block3):
            x = blk(x, H, W)
        x = self.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 4
        x, H, W = self.patch_embed4(x)
        for i, blk in enumerate(self.block4):
            x = blk(x, H, W)
        x = self.norm4(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        return outs

    def forward(self, x):
        x = self.forward_features(x)
        # x = self.head(x)

        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W).contiguous()
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x



#@BACKBONES.register_module()
class mit_b0(MixVisionTransformer):
    def __init__(self, **kwargs):
        super(mit_b0, self).__init__(
            patch_size=4, embed_dims=[32, 64, 160, 256], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


#@BACKBONES.register_module()
class mit_b1(MixVisionTransformer):
    def __init__(self, **kwargs):
        super(mit_b1, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


#@BACKBONES.register_module()
class mit_b2(MixVisionTransformer):
    def __init__(self, **kwargs):
        super(mit_b2, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


#@BACKBONES.register_module()
class mit_b3(MixVisionTransformer):
    def __init__(self, **kwargs):
        super(mit_b3, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 18, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


#@BACKBONES.register_module()
class mit_b4(MixVisionTransformer):
    def __init__(self, **kwargs):
        super(mit_b4, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 8, 27, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


#@BACKBONES.register_module()
class mit_b5(MixVisionTransformer):
    def __init__(self, **kwargs):
        super(mit_b5, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 6, 40, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)

class SegFormer(nn.Module):
    def __init__(self, num_classes, load_imagenet_model, imagenet_ckpt_fpath, **kwargs):
        super(SegFormer, self).__init__(**kwargs)

        self.encoder = mit_b5()
        # self.head = BilinearPADHead_fast_xavier_init(num_classes=num_classes,
        #                                                 c1_in_channels=64,
        #                                                 c1_channels=48,
        #                                                 upsample_factor=8,
        #                                                 dyn_branch_ch=16,
        #                                                 mask_head_ch=16,
        #                                                 pad_out_channel_factor=4,
        #                                                 channel_reduce_factor=2,
        #                                                 zero_init=False,
        #                                                 supress_std=True,
        #                                                 feature_strides=None,
        #                                                 in_channels=512,
        #                                                 channels=512,
        #                                                 in_index=3,
        #                                                 dilations=(1, 3, 6, 9),
        #                                                 dropout_ratio=0.1,
        #                                                 norm_cfg=dict(type='SyncBN', requires_grad=True),
        #                                                 align_corners=False,)

        self.head = SegFormerHead(num_classes=num_classes,
                                  in_channels=[64, 128, 320, 512],
                                  channels=128,
                                  in_index=[0,1,2,3],
                                  feature_strides=[4, 8, 16, 32],
                                  #decoder_params=dict(embed_dim=768),
                                  dropout_ratio=0.1,
                                  norm_cfg=dict(type='SyncBN', requires_grad=True),
                                  align_corners=False)
        self.auxi_net = FCNHead(num_convs=1,
                                kernel_size=3,
                                concat_input=True,
                                in_channels=320,
                                num_classes=num_classes,
                                norm_cfg=dict(type='SyncBN', requires_grad=True))
        self.init_weights(load_imagenet_model, imagenet_ckpt_fpath)

    def init_weights(self, load_imagenet_model: bool=False, imagenet_ckpt_fpath: str='') -> None:
        """ For training, we use a models pretrained on ImageNet. Irrelevant at inference.
            Args:
            -   pretrained_fpath: str representing path to pretrained models
            Returns:
            -   None
        """
        logger.info('=> init weights from normal distribution')
        if not load_imagenet_model:
            return
        if os.path.isfile(imagenet_ckpt_fpath):
            print('===========> loading pretrained models {}'.format(imagenet_ckpt_fpath))
            self.encoder.init_weights(pretrained=imagenet_ckpt_fpath)
        else:
            # logger.info(pretrained)
            print('cannot find ImageNet models path, use random initialization')
            raise RuntimeError('no pretrained models found at {}'.format(imagenet_ckpt_fpath))

    def forward(self, inputs):
        h = inputs.size()[2]
        w = inputs.size()[3]
        x = self.encoder(inputs)
        #out = self.head([x[3], x[0]])
        out = self.head(x)
        auxi_out = self.auxi_net(x)
        high_out = F.interpolate(out, size=(h,w), mode='bilinear', align_corners=True)
        return high_out, out, auxi_out


class SegModel(nn.Module):
    def __init__(self, criterions, num_classes, load_imagenet_model, imagenet_ckpt_fpath, **kwargs):
        super(SegModel, self).__init__(**kwargs)
        self.segmodel = SegFormer(num_classes=num_classes,
                                  load_imagenet_model=load_imagenet_model,
                                  imagenet_ckpt_fpath=imagenet_ckpt_fpath)
        self.criterion = None
    def forward(self, inputs, gt=None, label_space=None, others=None):
        high_reso, low_reso, auxi_out = self.segmodel(inputs)
        return high_reso, None, None

def get_seg_model(
    criterion: list,
    n_classes: int,
    load_imagenet_model: bool = False,
    imagenet_ckpt_fpath: str = '',
    **kwargs
    ) -> nn.Module:
    model = SegModel(criterions=criterion,
                     num_classes=n_classes,
                     load_imagenet_model=load_imagenet_model,
                     imagenet_ckpt_fpath=imagenet_ckpt_fpath)
    assert isinstance(model, nn.Module)
    return model

def get_configured_segformer(
    n_classes: int,
    criterion: list,
    load_imagenet_model: bool = False,
    imagenet_ckpt_fpath: str = '',
    ) -> nn.Module:
    """
        Args:
        -   n_classes: integer representing number of output classes
        -   load_imagenet_model: whether to initialize from ImageNet-pretrained models
        -   imagenet_ckpt_fpath: string representing path to file with weights to
                initialize models with
        Returns:
        -   models: HRNet models w/ architecture configured according to models yaml,
                and with specified number of classes and weights initialized
                (at training, init using imagenet-pretrained models)
    """

    model = get_seg_model(criterion, n_classes, load_imagenet_model, imagenet_ckpt_fpath)
    return model


if __name__=='__main__':
    imagenet_ckpt_fpath = ''
    load_imagenet_model = False
    criterions=[]
    from mseg_semantic.model.criterion import Cross_sim_loss

    loss_method = Cross_sim_loss(data_index=['universal'],
                                 data_root='./data',
                                 ignore_label=255,
                                 emd_method='wiki_embeddings')
    criterions.append(loss_method)
    model = get_configured_segformer(180, criterions, load_imagenet_model, imagenet_ckpt_fpath)
    num_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(num_p)
