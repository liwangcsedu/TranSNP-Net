import torch
import torch.nn as nn
from timm.models.layers import DropPath
import numpy as np
import torch.nn.functional as F
from models.SwinTransformers import SwinTransformer

def ConvSNP(in_planes, out_planes, k=3, s=1, p=1, b=False):
    return nn.Sequential(
            nn.GELU(),
            nn.Conv2d(in_planes, out_planes, kernel_size=k, stride=s, padding=p, bias=b),
            nn.BatchNorm2d(out_planes)
            )

class TranSNP_Net(nn.Module):
    def __init__(self):
        super(TranSNP_Net, self).__init__()

        self.rgb_swin = SwinTransformer(embed_dim=128, depths=[2,2,18,2], num_heads=[4,8,16,32])
        self.depth_swin = SwinTransformer(embed_dim=128, depths=[2,2,18,2], num_heads=[4,8,16,32])
        self.up2 = nn.UpsamplingBilinear2d(scale_factor = 2)
        self.up4 = nn.UpsamplingBilinear2d(scale_factor = 4)

        self.SNP_Fusion_1 = SNP_Fusion(2048, 2048)
        self.SNP_Fusion_2 = SNP_Fusion(1024, 1024)
        self.SNP_Fusion_3 = SNP_Fusion(512, 512)
        self.SNP_Fusion_4 = SNP_Fusion(256, 256)

        self.SNP_D_1 = SNP_D(dim=256)
        self.SNP_D_2 = SNP_D(dim=128)
        self.SNP_D_3 = SNP_D(dim=64)

        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.ConvSNP_D_1 =  nn.Sequential(       # ConvSNP
            nn.GELU(),#
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            self.upsample2
        )
        self.ConvSNP_D_2 = nn.Sequential(       # ConvSNP
            nn.GELU(),#
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            self.upsample2
        )
        self.ConvSNP_D_3 = nn.Sequential(       # ConvSNP
            nn.GELU(),#
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            self.upsample2
        )
        self.ConvSNP_D_4 = nn.Sequential(       # ConvSNP
            nn.GELU(),#
            nn.Conv2d(in_channels=256, out_channels=64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            self.upsample2
        )
        self.ConvSNP_Predice = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            self.upsample2,
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, padding=1, bias=True),
            )
        self.predtrans2 = nn.Conv2d(128, 1, kernel_size=3, padding=1)
        self.predtrans3 = nn.Conv2d(256, 1, kernel_size=3, padding=1)
        self.predtrans4 = nn.Conv2d(512, 1, kernel_size=3, padding=1)
        self.dwc3 = ConvSNP(256, 128)
        self.dwc2 = ConvSNP(512, 256)
        self.dwc1 = ConvSNP(1024, 512)
        self.dwcon_1 = ConvSNP(2048, 1024)
        self.dwcon_2 = ConvSNP(1024, 512)
        self.dwcon_3 = ConvSNP(512, 256)
        self.dwcon_4 = ConvSNP(256, 128)



    def forward(self,x ,d):
        rgb_list = self.rgb_swin(x)
        depth_list = self.depth_swin(d)

        r4 = rgb_list[0]
        r3 = rgb_list[1]
        r2 = rgb_list[2]
        r1 = rgb_list[3]
        d4 = depth_list[0]
        d3 = depth_list[1]
        d2 = depth_list[2]
        d1 = depth_list[3]

        r3_up = F.interpolate(self.dwc3(r3), size=96, mode='bilinear')
        r2_up = F.interpolate(self.dwc2(r2), size=48, mode='bilinear')
        r1_up = F.interpolate(self.dwc1(r1), size=24, mode='bilinear')
        d3_up = F.interpolate(self.dwc3(d3), size=96, mode='bilinear')
        d2_up = F.interpolate(self.dwc2(d2), size=48, mode='bilinear')
        d1_up = F.interpolate(self.dwc1(d1), size=24, mode='bilinear')

        r1_con = torch.cat((r1, r1), 1)
        r1_con = self.dwcon_1(r1_con)
        d1_con = torch.cat((d1, d1), 1)
        d1_con = self.dwcon_1(d1_con)
        r2_con = torch.cat((r2, r1_up), 1)
        r2_con = self.dwcon_2(r2_con)
        d2_con = torch.cat((d2, d1_up), 1)
        d2_con = self.dwcon_2(d2_con)
        r3_con = torch.cat((r3, r2_up), 1)
        r3_con = self.dwcon_3(r3_con)
        d3_con = torch.cat((d3, d2_up), 1)
        d3_con = self.dwcon_3(d3_con)
        r4_con = torch.cat((r4, r3_up), 1)
        r4_con = self.dwcon_4(r4_con)
        d4_con = torch.cat((d4, d3_up), 1)
        d4_con = self.dwcon_4(d4_con)


        xf_1 = self.SNP_Fusion_1(r1_con, d1_con)  # 1024,12,12
        xf_2 = self.SNP_Fusion_2(r2_con, d2_con)  # 512,24,24
        xf_3 = self.SNP_Fusion_3(r3_con, d3_con)  # 256,48,48
        xf_4 = self.SNP_Fusion_4(r4_con, d4_con)  # 128,96,96


        df_f_1 = self.ConvSNP_D_1(xf_1)

        xc_1_2 = torch.cat((df_f_1, xf_2), 1)
        df_f_2 = self.ConvSNP_D_2(xc_1_2)
        df_f_2 = self.DNP_D_1(df_f_2)

        xc_1_3 = torch.cat((df_f_2, xf_3), 1)
        df_f_3 = self.ConvSNP_D_3(xc_1_3)
        df_f_3 = self.SNP_D_2(df_f_3)

        xc_1_4 = torch.cat((df_f_3, xf_4), 1)
        df_f_4 = self.ConvSNP_D_4(xc_1_4)
        df_f_4 = self.SNP_D_3(df_f_4)
        y1 = self.ConvSNP_Predice(df_f_4)
        y2 = F.interpolate(self.predtrans2(df_f_3), size=384, mode='bilinear')
        y3 = F.interpolate(self.predtrans3(df_f_2), size=384, mode='bilinear')
        y4 = F.interpolate(self.predtrans4(df_f_1), size=384, mode='bilinear')
        return y1,y2,y3,y4

    def load_pre(self, pre_model):
        self.rgb_swin.load_state_dict(torch.load(pre_model)['model'],strict=False)
        print(f"RGB SwinTransformer loading pre_model ${pre_model}")
        self.depth_swin.load_state_dict(torch.load(pre_model)['model'], strict=False)
        print(f"Depth SwinTransformer loading pre_model ${pre_model}")


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class SA_Enhance(nn.Module):    # Special Attention Enhance
    def __init__(self, kernel_size=7):
        super(SA_Enhance, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = max_out
        x = self.conv1(x)
        return self.sigmoid(x)


class SNP_Fusion(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(SNP_Fusion, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_end = nn.Conv2d(oup, oup // 2, kernel_size=1, stride=1, padding=0)
        self.self_SA_Enhance = SA_Enhance()

    def forward(self, rgb, depth):
        x = torch.cat((rgb, depth), dim=1)

        n, c, h, w = x.size()
        x_h = self.pool_h(x).permute(0, 1, 3, 2)
        x_w = self.pool_w(x)                            # .permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=3)                # dim=2
        y = self.act(y)                                 # SNP------>sigmode
        y = self.conv1(y)
        y = self.bn1(y)

        x_h, x_w = torch.split(y, [h, w], dim=3)        # dim=2
        #x_w = x_w.permute(0, 1, 3, 2)
        x_h = x_h.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()                # CA
        a_w = self.conv_w(x_w).sigmoid()                # CA

        out_ca = x * a_w * a_h
        out_sa = self.self_SA_Enhance(out_ca)           # SA
        out = x.mul(out_sa)
        out = self.conv_end(out)

        return out


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(normalized_shape), requires_grad=True)
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise ValueError(f"not support data format '{self.data_format}'")
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            # [batch_size, channels, height, width]
            mean = x.mean(1, keepdim=True)
            var = (x - mean).pow(2).mean(1, keepdim=True)
            x = (x - mean) / torch.sqrt(var + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class SNP_D(nn.Module):
    def __init__(self, dim, drop_rate=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = nn.GroupNorm(num_groups=1, num_channels=dim)  # use GroupNorm
        self.pwconv1 = nn.Conv2d(dim, 4 * dim, kernel_size=1)  # pointwise/1x1 conv
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(4 * dim, dim, kernel_size=1)  # pointwise/1x1 conv
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim,)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_rate) if drop_rate > 0. else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.dwconv(x)
        x = self.norm(x)  # apply GroupNorm directly on [N, C, H, W] format
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma.unsqueeze(0).unsqueeze(2).unsqueeze(3) * x  # ensure gamma shape matches x
        x = shortcut + self.drop_path(x)
        return x
