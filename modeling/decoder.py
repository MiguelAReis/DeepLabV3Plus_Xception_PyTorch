import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d

import brevitas.nn as qnn
from brevitas.quant import Int8Bias as BiasQuant


weight_bit_width=4
activ_bit_width=6

class Decoder(nn.Module):
    def __init__(self, num_classes, backbone, BatchNorm):
        super(Decoder, self).__init__()


        if backbone == 'resnet' or backbone == 'drn':
            low_level_inplanes = 256
        elif backbone == 'xception':
            low_level_inplanes = 128
        elif backbone == 'mobilenet':
            low_level_inplanes = 24
        elif backbone == 'xceptionQuant':
            low_level_inplanes = 128
        else:
            raise NotImplementedError

        self.conv1 = qnn.QuantConv2d(low_level_inplanes, 48, 1, bias=False,weight_bit_width=weight_bit_width, bias_quant=BiasQuant, return_quant_tensor=True)
        self.bn1 = BatchNorm(48)
        self.relu = qnn.QuantReLU(bit_width=activ_bit_width, return_quant_tensor=True)
        self.last_conv = nn.Sequential(qnn.QuantConv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False,weight_bit_width=weight_bit_width, bias_quant=BiasQuant, return_quant_tensor=True),
                                       BatchNorm(256),
                                       qnn.QuantReLU(bit_width=activ_bit_width, return_quant_tensor=True),
                                       nn.Dropout(0.5),
                                       qnn.QuantConv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False,weight_bit_width=weight_bit_width, bias_quant=BiasQuant, return_quant_tensor=True),
                                       BatchNorm(256),
                                       qnn.QuantReLU(bit_width=activ_bit_width, return_quant_tensor=True),
                                       nn.Dropout(0.1),
                                       qnn.QuantConv2d(256, num_classes, kernel_size=1, stride=1,weight_bit_width=weight_bit_width, bias_quant=BiasQuant, return_quant_tensor=True))
        self._init_weight()


    def forward(self, x, low_level_feat):
        low_level_feat = self.conv1(low_level_feat)
        low_level_feat = self.bn1(low_level_feat)
        low_level_feat = self.relu(low_level_feat)

        x = F.interpolate(x, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x, low_level_feat), dim=1)
        x = self.last_conv(x)

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

def build_decoder(num_classes, backbone, BatchNorm):
    return Decoder(num_classes, backbone, BatchNorm)