import torch
import torch.nn as nn
import math
from collections import OrderedDict
from .CTrans import ChannelTransformer

def get_blocks_to_be_concat(model, x):
    shapes = set()
    blocks = OrderedDict()
    hooks = []
    count = 0

    def register_hook(module):

        def hook(module, input, output):
            try:
                nonlocal count
                if module.name == f'blocks_{count}_output_batch_norm':
                    count += 1
                    shape = output.size()[-2:]
                    if shape not in shapes:
                        shapes.add(shape)
                        blocks[module.name] = output
                elif module.name == 'head_swish':
                    blocks.popitem()
                    blocks[module.name] = output

            except AttributeError:
                pass

        if (
                not isinstance(module, nn.Sequential)
                and not isinstance(module, nn.ModuleList)
                and not (module == model)
        ):
            hooks.append(module.register_forward_hook(hook))
    model.apply(register_hook)
    model(x)
    for h in hooks:
        h.remove()

    return blocks

def get_activation(activation_type):
    activation_type = activation_type.lower()
    if hasattr(nn, activation_type):
        return getattr(nn, activation_type)()
    else:
        return nn.ReLU()

def _make_nConv(in_channels, out_channels, nb_Conv, activation='ReLU'):
    layers = []
    layers.append(ConvBatchNorm(in_channels, out_channels, activation))

    for _ in range(nb_Conv - 1):
        layers.append(ConvBatchNorm(out_channels, out_channels, activation))
    return nn.Sequential(*layers)

class ConvBatchNorm(nn.Module):

    def __init__(self, in_channels, out_channels, activation='ReLU'):
        super(ConvBatchNorm, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = get_activation(activation)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        return self.activation(out)


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3, stride = 1, expansion_factor = 6, nb_Conv=2, activation='ReLU'):
        super(DownBlock, self).__init__()
        self.stride = stride
        self.expansion_factor = expansion_factor
        self.maxpool = nn.MaxPool2d(2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)
        mid_channels = (in_channels * expansion_factor)

        self.bottleneck = nn.Sequential(
            Conv1x1BNAct(in_channels, mid_channels),
            ConvBNAct(mid_channels, mid_channels, kernel_size, stride, groups=mid_channels),
            SEBlock(mid_channels),
            Conv1x1BN(mid_channels, out_channels)
        )

        if self.stride == 1:
            self.shortcut = Conv1x1BN(in_channels, out_channels)
    def forward(self, x):
        out = self.bottleneck(x)
        out = (out + self.shortcut(x)) if self.stride == 1 else out
        out = self.maxpool(x)
        return self.nConvs(out)
    
class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super(UpBlock, self).__init__()

        # self.up = nn.Upsample(scale_factor=2)
        self.up = nn.ConvTranspose2d(in_channels//2,in_channels//2,(2,2),2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x, skip_x):
        out = self.up(x)
        x = torch.cat([out, skip_x], dim=1)  # dim 1 is the channel dimension
        return self.nConvs(x)

class CCA(nn.Module):
    def __init__(self, F_g, F_x, kernel_size = 7):
        super().__init__()
        self.mlp_x = nn.Sequential(
            Flatten(),
            nn.Linear(F_x, F_x))
        self.mlp_g = nn.Sequential(
            Flatten(),
            nn.Linear(F_g, F_x))
        self.relu = nn.ReLU(inplace=True)
        self.sa = SpatialAttention(kernel_size=kernel_size)

    def forward(self, g, x):
        avg_pool_x = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
        channel_att_x = self.mlp_x(avg_pool_x)
        avg_pool_g = F.avg_pool2d( g, (g.size(2), g.size(3)), stride=(g.size(2), g.size(3)))
        channel_att_g = self.mlp_g(avg_pool_g)
        channel_att_sum = (channel_att_x + channel_att_g)/2.0
        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        x_after_channel = x * scale

        out = x_after_channel * self.sa(x_after_channel)
        result = self.relu(out)
        
        return result

class UpBlock_attention(nn.Module):
    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2)
        self.coatt = CCA(F_g=in_channels//2, F_x=in_channels//2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x, skip_x):
        up = self.up(x)
        skip_x_att = self.coatt(g=up, x=skip_x)
        x = torch.cat([skip_x_att, up], dim=1)  # dim 1 is the channel dimension
        return self.nConvs(x)
    
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)  # 7,3     3,1
        # self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return torch.sigmoid(x)
    
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class Unet(nn.Module):
    def __init__(self, n_channels=3, n_classes=2,subtype='efficientnet_b0',
                 img_size=224, vis=False):
        super().__init__()
        self.depth_div = 8
        self.stage1 = ConvBNAct(3, self._calculate_width(64), kernel_size=3, stride=2)
        self.stage2 = self.make_layer(self._calculate_width(64), self._calculate_width(128), kernel_size=3, stride=2,
                                      block=self._calculate_depth(1))
        self.stage3 = self.make_layer(self._calculate_width(128), self._calculate_width(256), kernel_size=3, stride=2,
                                      block=self._calculate_depth(2))
        self.stage4 = self.make_layer(self._calculate_width(256), self._calculate_width(512), kernel_size=5, stride=2,
                                      block=self._calculate_depth(2))
        self.stage5 = self.make_layer(self._calculate_width(512), self._calculate_width(512), kernel_size=3, stride=2,
                                      block=self._calculate_depth(3))
        self.vis = vis
        self.n_channels = n_channels
        self.n_classes = n_classes
        in_channels = 64
        self.inc = ConvBatchNorm(n_channels, in_channels)
        self.down1 = DownBlock(in_channels, in_channels * 2, nb_Conv=2)
        self.down2 = DownBlock(in_channels * 2, in_channels * 4, nb_Conv=2)
        self.down3 = DownBlock(in_channels * 4, in_channels * 8, nb_Conv=2)
        self.down4 = DownBlock(in_channels * 8, in_channels * 8, nb_Conv=2)
        self.mtc = ChannelTransformer(vis = vis, img_size = img_size,
                                      channel_num=[in_channels, in_channels * 2, in_channels * 4, in_channels * 8],
                                      patchSize=[16, 8, 4, 2])
        self.up4 = UpBlock_attention(in_channels * 16, in_channels * 4, nb_Conv=2)
        self.up3 = UpBlock_attention(in_channels * 8, in_channels * 2, nb_Conv=2)
        self.up2 = UpBlock_attention(in_channels * 4, in_channels, nb_Conv=2)
        self.up1 = UpBlock_attention(in_channels * 2, in_channels, nb_Conv=2)
        self.outc = nn.Conv2d(in_channels, n_classes, kernel_size=(1, 1), stride=(1, 1))

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
            elif isinstance(m, nn.Linear):
                init_range = 1.0 / math.sqrt(m.weight.shape[1])
                nn.init.uniform_(m.weight, -init_range, init_range)

    def _calculate_width(self, x):
        x *= self.width_coeff
        new_x = max(self.depth_div, int(x + self.depth_div / 2) // self.depth_div * self.depth_div)
        if new_x < 0.9 * x:
            new_x += self.depth_div
        return int(new_x)

    def _calculate_depth(self, x):
        return int(math.ceil(x * self.depth_coeff))

    def make_layer(self, in_places, places, kernel_size, stride, block):
        layers = []
        layers.append(MBConvBlock(in_places, places, kernel_size, stride))
        for i in range(1, block):
            layers.append(MBConvBlock(places, places, kernel_size))
        
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.float()
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x1, x2, x3, x4, att_weights = self.mtc(x1, x2, x3, x4)
        x = self.up4(x5, x4)
        x = self.up3(x, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)
        if self.n_classes == 1:
            logits = torch.sigmoid(self.outc(x))
        else:
            logits = self.outc(x)
        if self.vis:
            return logits, att_weights
        else:
            return logits

    
    def freeze_backbone(self):
        if self.backbone == "b5":
            for param in self.encoder.parameters():
                param.requires_grad = False
        elif self.backbone == "resnet50":
            for param in self.resnet.parameters():
                param.requires_grad = False

    def unfreeze_backbone(self):
        if self.backbone == "b5":
            for param in self.encoder.parameters():
                param.requires_grad = True
        elif self.backbone == "resnet50":
            for param in self.resnet.parameters():
                param.requires_grad = True

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


def ConvBNAct(in_channels,out_channels,kernel_size=3, stride=1,groups=1):
    return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=groups),
            nn.BatchNorm2d(out_channels),
            Swish()
        )


def Conv1x1BNAct(in_channels,out_channels):
    return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels),
            Swish()
        )

def Conv1x1BN(in_channels,out_channels):
    return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels)
        )

def Conv1(in_planes, places, stride=2):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_planes,out_channels=places,kernel_size=7,stride=stride,padding=3, bias=False),
        nn.BatchNorm2d(places),
        Swish(),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)


class SEBlock(nn.Module):
    def __init__(self, channels, ratio=16):
        super().__init__()
        mid_channels = channels // ratio
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, mid_channels, kernel_size=1, stride=1, padding=0, bias=True),
            Swish(),
            nn.Conv2d(mid_channels, channels, kernel_size=1, stride=1, padding=0, bias=True),
        )

    def forward(self, x):
        return x * torch.sigmoid(self.se(x))


class MBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, expansion_factor=6, activation = "ReLU"):
        super(MBConvBlock, self).__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.stride = stride
        self.expansion_factor = expansion_factor
        mid_channels = (in_channels * expansion_factor)

        self.bottleneck = nn.Sequential(
            Conv1x1BNAct(in_channels, mid_channels),
            ConvBNAct(mid_channels, mid_channels, kernel_size, stride, groups=mid_channels),
            SEBlock(mid_channels),
            Conv1x1BN(mid_channels, out_channels)
        )

        if self.stride == 1:
            self.shortcut = Conv1x1BN(in_channels, out_channels)

    def forward(self, x):
        out = self.maxpool(x)
        out = self.bottleneck(x)
        out = (out + self.shortcut(x)) if self.stride==1 else out
        
        return out
