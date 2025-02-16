from functools import partial
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

from models.ffc import *
from models import pvt_v2
from timm.models.vision_transformer import _cfg
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.in_layers = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        )

        self.out_layers = nn.Sequential(
            nn.GroupNorm(32, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )

        if out_channels == in_channels:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        h = self.in_layers(x)
        h = self.out_layers(h)
        return h + self.skip(x)

class TB_Encoder(nn.Module):
    def __init__(self):

        super().__init__()

        backbone = pvt_v2.PyramidVisionTransformerV2(
            patch_size=4,
            embed_dims=[64, 128, 320, 512],
            num_heads=[1, 2, 5, 8],
            mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            depths=[3, 4, 18, 3],
            sr_ratios=[8, 4, 2, 1],
        )

        checkpoint = torch.load("/home/moon/child_proj/abdomen_system/segmentation/weights/pvt_v2_b3.pth")
        backbone.default_cfg = _cfg()
        backbone.load_state_dict(checkpoint)
        self.backbone = torch.nn.Sequential(*list(backbone.children()))[:-1]
       
        for i in [1, 4, 7, 10]:
            self.backbone[i] = torch.nn.Sequential(*list(self.backbone[i].children()))


    def get_pyramid_1_3_stage(self, x):
        pyramid = []
        B = x.shape[0]
        for i, module in enumerate(self.backbone[:int(0.75 * len(self.backbone))]):
            if i in [0, 3, 6]:
                x, H, W = module(x)
            elif i in [1, 4, 7]:
                for sub_module in module:
                    x = sub_module(x, H, W)
            else:
                x = module(x)
                x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
                pyramid.append(x)

        return pyramid

    def forward(self, x):
        encoding_pyramid_feature = self.get_pyramid_1_3_stage(x)
    

        return encoding_pyramid_feature

class TB_middle_Encoder(TB_Encoder):
    def __init__(self):
        super().__init__()
        
    def get_shared_pyramid(self, pyramid):
        x = pyramid[-1]
        B = x.shape[0]
        middle_feature = []
        for i, module in enumerate(self.backbone[int(0.75 * len(self.backbone)):]):
            if i == 0:
                x, H, W = module(x)
            elif i == 1:
                for sub_module in module:
                    x = sub_module(x, H, W)
            else:
                x = module(x)
                x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
                middle_feature.append(x)

        return middle_feature

    def forward(self, x):
        middle_encoder_feature = self.get_shared_pyramid(x)
        
        return middle_encoder_feature

class FeatureMixingBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.start_layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(32, out_channels),
            nn.SiLU()
        )    
        self.FFC_seq = nn.Sequential(FFC_BN_ACT(out_channels, out_channels, kernel_size=3, ratio_gin=0.5, ratio_gout=0.5, padding=1), 
                                     FFC_BN_ACT(out_channels, out_channels, kernel_size=3, ratio_gin=0.5, ratio_gout=0.5, padding=1),
                                    )
        #self.conv_1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.out_channel = out_channels
    def forward(self, x):
        x1, x2 = x if type(x) is tuple else (x, 0)
        x_merge = torch.cat((x1, x2), 1)
        
        x_step1 = self.start_layer(x_merge)
        x_step1_split = torch.split(x_step1, split_size_or_sections=self.out_channel // 2, dim=1)
        x_step1_l, x_step1_g = self.FFC_seq(x_step1_split)
        x_step2_l, x_step2_g = self.FFC_seq((x_step1_l, x_step1_g))
        x_step3_l, x_step3_g = self.FFC_seq((x_step2_l, x_step2_g))
        x_step3_all = torch.cat((x_step3_l, x_step3_g), 1)
        out_x = x_step3_all + x1
        out_x = out_x + x2
        out_x_l, out_x_g = torch.split(out_x, split_size_or_sections=self.out_channel // 2, dim=1)
        #print(out_x_l.size())
        #print(out_x_g.size())
        return (out_x_l, out_x_g)

class FPN_Decoder(nn.Module):

    def __init__(self, num_classes=1):
        super(FPN_Decoder, self).__init__()
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        # Top layer
        self.toplayer = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)  # Reduce channels

        # Smooth layers
        self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        # Lateral layers
        self.latlayer1 = nn.Conv2d(320, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(64, 256, kernel_size=1, stride=1, padding=0)

		# Semantic branch
        #self.semantic_branch = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.semantic_branch = nn.Sequential(
            RB(256, 192), RB(192, 128)
        )
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, self.num_classes, kernel_size=1, stride=1, padding=0)
        
        self.skip_conv4 = nn.Conv2d(640, 128, kernel_size=3, padding=1)
        self.skip_conv3 = nn.Conv2d(448, 128, kernel_size=3, padding=1)
        self.skip_conv2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.skip_conv1 = nn.Conv2d(192, 128, kernel_size=3, padding=1)
        # num_groups, num_channels
        self.gn1 = nn.GroupNorm(32, 128) 
        self.gn2 = nn.GroupNorm(32, 256)
        self.conv1x1 = nn.Conv2d(128, 1, kernel_size=1, stride=1, padding=0)

    def _upsample(self, x, h, w):
        return F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)
    
    def _upsample_add(self, x, y):
        _,_,H,W = y.size()
        return F.interpolate(x, size=(H,W), mode='bilinear', align_corners=True) + y


    def forward(self, x):
        low_level_features = x
        c1 = low_level_features[0]
        c2 = low_level_features[1]
        c3 = low_level_features[2]
        c4 = low_level_features[3]
    
        p4 = self.toplayer(c4)
        #print(p4.size())
        p3 = self._upsample_add(p4, self.latlayer1(c3))
        #print(p3.size())
        p2 = self._upsample_add(p3, self.latlayer2(c2))
        #print(p2.size())
        p1 = self._upsample_add(p2, self.latlayer3(c1))

        # Smooth
        p3 = self.smooth1(p3)
        p2 = self.smooth2(p2)
        p1 = self.smooth2(p1)
    
        # Semantic
        _, _, h, w = c1.size()
        
        # 256->256
        s4 = F.relu(self.gn2(self.conv2(p4)))
        s4 = F.relu(self.gn1(self.semantic_branch(s4)))
        s4_up = self._upsample(s4, h, w)
        #print('s4: ', s4.size())
        # 256->128
        s3 = F.relu(self.gn1(self.semantic_branch(p3)))
        s3 = torch.cat((s3, c3), 1)
        s3 = self.skip_conv3(s3)
        #print('s3: ', s3.size())
        s3_up = self._upsample(s3, h, w)
        
        s2 = F.relu(self.gn1(self.semantic_branch(p2)))
        s2 = torch.cat((s2, c2), 1)
        s2 = self.skip_conv2(s2)
        #print('s2: ', s2.size())
        s2_up = self._upsample(s2, h, w)
        
        s1 = F.relu(self.gn1(self.semantic_branch(p1)))
        s1 = torch.cat((s1, c1), 1)
        s1 = self.skip_conv1(s1)
        #print('s1: ', s1.size())
        s1_up = self._upsample(s1, h, w)
        
        return [self._upsample((s1_up + s2_up + s3_up + s4_up), 4 * h, 4 * w), s1, s2, s3, s4]

class HGA(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(HGA, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x1, x2):
        batch_size, channels, height, width = x1.size()

        # reshape tensor 1 to (batch_size, channels, -1) for attention calculation
        proj_query = self.query_conv(x1).view(batch_size, -1, height * width).permute(0, 2, 1) # (batch_size, h*w, channels)
        proj_key = self.key_conv(x2).view(batch_size, -1, height * width) # (batch_size, channels, h*w)
        energy = torch.bmm(proj_query, proj_key) # (batch_size, h*w, h*w)

        # apply softmax to get attention weights
        attention = F.softmax(energy, dim=-1)

        # apply attention to value
        proj_value = self.value_conv(x2).view(batch_size, -1, height * width) # (batch_size, channels, h*w)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1)) # (batch_size, channels, h*w)

        # reshape back to original shape
        out = out.view(batch_size, channels, height, width)

        return out

class FPN_Decoder_all(nn.Module):

    def __init__(self, num_classes=1):
        super(FPN_Decoder_all, self).__init__()
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        # Top layer
        self.toplayer = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)  # Reduce channels

        # Smooth layers
        self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        # Lateral layers
        self.latlayer1 = nn.Conv2d(320, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(64, 256, kernel_size=1, stride=1, padding=0)

		# Semantic branch
        #self.semantic_branch = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.semantic_branch = nn.Sequential(
            RB(256, 192), RB(192, 128)
        )
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, self.num_classes, kernel_size=1, stride=1, padding=0)
        
        self.skip_conv4 = nn.Conv2d(640, 128, kernel_size=3, padding=1)
        self.skip_conv3 = nn.Conv2d(448, 128, kernel_size=3, padding=1)
        self.skip_conv2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.skip_conv1 = nn.Conv2d(192, 128, kernel_size=3, padding=1)

        # num_groups, num_channels
        self.gn1 = nn.GroupNorm(32, 128) 
        self.gn2 = nn.GroupNorm(32, 256)
        self.conv1x1 = nn.Conv2d(128, 1, kernel_size=1)
        
        self.HGA = HGA(128, 128)
        
    def _upsample(self, x, h, w):
        return F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)
    
    def _upsample_add(self, x, y):
        _,_,H,W = y.size()
        return F.interpolate(x, size=(H,W), mode='bilinear', align_corners=True) + y


    def forward(self, x, x_head):
        low_level_features = x
        c1 = low_level_features[0]
        #print(c1.size())
        c2 = low_level_features[1]
        #print(c2.size())
        c3 = low_level_features[2]
        #print(c3.size())
        c4 = low_level_features[3]
        #print(c4.size())
        
        head_features = []
        for guidance in x_head:
            #head_features.append(self.conv1x1(guidance))
            #print(self.conv1x1(guidance).size())
            head_features.append(guidance)
        
        #head_1_pred = F.interpolate(head_features[-1], scale_factor=32, mode='bilinear')
        #head_2_pred = F.interpolate(head_features[-2], scale_factor=16, mode='bilinear')
        #head_3_pred = F.interpolate(head_features[-3], scale_factor=8, mode='bilinear')
        #head_4_pred = F.interpolate(head_features[-4], scale_factor=4, mode='bilinear')
        
        p4 = self.toplayer(c4)
        #print(p4.size())
        p3 = self._upsample_add(p4, self.latlayer1(c3))
        #print(p3.size())
        p2 = self._upsample_add(p3, self.latlayer2(c2))
        #print(p2.size())
        p1 = self._upsample_add(p2, self.latlayer3(c1))

        # Smooth
        p3 = self.smooth1(p3)
        p2 = self.smooth2(p2)
        p1 = self.smooth2(p1)
    
        # Semantic
        _, _, h, w = c1.size()
        
        # 256->256
        s4 = F.relu(self.gn2(self.conv2(p4)))
        s4 = F.relu(self.gn1(self.semantic_branch(s4)))
        s4_up = self._upsample(s4, h, w)
        #print('s4: ', s4.size())
        s4_hga = self.HGA(s4, head_features[-1])
        s4 = s4 + s4_hga
        # 256->128
        s3 = F.relu(self.gn1(self.semantic_branch(p3)))
        s3 = torch.cat((s3, c3), 1)
        s3 = self.skip_conv3(s3)
        s3_hga = self.HGA(s3, head_features[-2])
        #print('s3: ', s3.size())
        s3 = s3 + s3_hga 
        s3_up = self._upsample(s3, h, w)
        
        s2 = F.relu(self.gn1(self.semantic_branch(p2)))
        s2 = torch.cat((s2, c2), 1)
        s2 = self.skip_conv2(s2)
        s2_hga = self.HGA(s2, head_features[-3])
        s2 = s2 + s2_hga
        #print('s2: ', s2.size())
        s2_up = self._upsample(s2, h, w)
        
        s1 = F.relu(self.gn1(self.semantic_branch(p1)))
        s1 = torch.cat((s1, c1), 1)
        s1 = self.skip_conv1(s1)
        s1_hga = self.HGA(s1, head_features[-4])
        s1 = s1 + s1_hga
        #print('s1: ', s1.size())
        s1_up = self._upsample(s1, h, w)
        
        return [self._upsample((s1_up + s2_up + s3_up + s4_up), 4 * h, 4 * w), s1, s2, s3, s4]

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class ExtractBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.in_layers = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1),
        )
        self.out_layers = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
        )
        
        self.conv_1x1 = nn.Conv2d(in_channels // 2, in_channels // 2, kernel_size=1)
        self.conv_1x1_re = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)
        self.conv_3x3 = nn.Conv2d(in_channels // 2, in_channels // 2, kernel_size=3, padding=1)
        self.conv_5x5 = nn.Conv2d(in_channels // 2, in_channels // 2, kernel_size=5, padding=2)
        self.sa = SpatialAttention()
        self.ca = ChannelAttention(in_channels // 2, in_channels // 2)
        self.silu = nn.SiLU()

    def forward(self, x):
        h = self.in_layers(x)
        h_1x1 = self.conv_1x1(h)
        #print(h_1x1.size())
        h_3x3 = self.conv_3x3(h)
        #print(h_3x3.size())
        h_5x5 = self.conv_5x5(h)
        #print(h_5x5.size())
        h_ca = self.ca(h) * h
        #print(h_ca.size())
        h_sa = self.sa(h_ca) * h
        #print(h_sa.size())
        all_feature = torch.cat((h_1x1, h_3x3, h_5x5, h_sa), 1)
        all_feature_1x1 = self.conv_1x1_re(all_feature)
        #print(all_feature_1x1.size())
        all_feature_1x1 += x
        all_feature_final = self.silu(all_feature_1x1)
        return all_feature_final


class XNet(nn.Module):
    def __init__(self, size=448):

        super().__init__()

        self.TB_Encoder1 = TB_Encoder()
        self.TB_Encoder2 = TB_Encoder()
        self.TB_middle_Encoder = TB_middle_Encoder()
        self.decoder = FPN_Decoder()
        self.decoder_all = FPN_Decoder_all()
        self.feature_mixing_block = nn.Sequential(FeatureMixingBlock(1024, 512),
                                                  FeatureMixingBlock(512, 256),
                                                  FeatureMixingBlock(256, 128),)
        self.PH_all = nn.Sequential(
            ExtractBlock(128), ExtractBlock(128), nn.Conv2d(128, 1, kernel_size=1)
        )
        self.PH_head = nn.Sequential(
            ExtractBlock(128), ExtractBlock(128), nn.Conv2d(128, 1, kernel_size=1)
        )
        self.conv1x1 = nn.Conv2d(640, 512, kernel_size=1)
        self.conv1x1_final = nn.Conv2d(128, 1, kernel_size=1)

    def forward(self, x1, x2):
        encoding_features_1 = self.TB_Encoder1(x1)
        encoding_features_2 = self.TB_Encoder2(x2)
        
        encoding_middle_feature_1 = self.TB_middle_Encoder(encoding_features_1)
        encoding_middle_feature_2 = self.TB_middle_Encoder(encoding_features_2)
        mixed_feature = self.feature_mixing_block((encoding_middle_feature_1[-1], encoding_middle_feature_2[-1]))
        mixed_feature = torch.concat((mixed_feature[0], mixed_feature[1]), 1)
        
        all_encoding_features_1 = [features for features in encoding_features_1]
        encoding_middle_mixed_1 = torch.cat((mixed_feature, encoding_middle_feature_1[-1]), 1)
        encoding_middle_mixed_1 = self.conv1x1(encoding_middle_mixed_1)
        all_encoding_features_1.append(encoding_middle_mixed_1)
        
        all_encoding_features_2 = [features for features in encoding_features_2]
        encoding_middle_mixed_2 = torch.cat((mixed_feature, encoding_middle_feature_2[-1]), 1)
        encoding_middle_mixed_2 = self.conv1x1(encoding_middle_mixed_2)
        all_encoding_features_2.append(encoding_middle_mixed_2)
        
        decoder_outputs_head = self.decoder(all_encoding_features_2)
        decoder_outputs_all = self.decoder_all(all_encoding_features_1, decoder_outputs_head)
        
        decoder_output_1 = decoder_outputs_all[0]
        decoder_output_2 = decoder_outputs_head[0]
        
        all_result_1 = self.PH_all(decoder_output_1)
        all_result_2 = self.PH_all(decoder_output_2)
        
        head_result_1 = self.PH_head(decoder_output_1)
        head_result_2 = self.PH_head(decoder_output_2)
        
        #return decoder_output_1, decoder_output_2
        return all_result_1, all_result_2, head_result_1, head_result_2
        #return x1, x2

