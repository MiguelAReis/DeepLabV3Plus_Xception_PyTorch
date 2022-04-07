#This file was generated with brevitasConverter.py
weightBitWidth=8
activationBitWidth=8

import brevitas.nn as qnn
from brevitas.quant import Int8Bias as BiasQuant

#Engine declaration
from brevitas.inject import ExtendedInjector
from brevitas.quant.solver import WeightQuantSolver, ActQuantSolver
from brevitas.core.bit_width import BitWidthImplType
from brevitas.core.quant import QuantType
from brevitas.core.restrict_val import RestrictValueType, FloatToIntImplType
from brevitas.core.scaling import ScalingImplType
from brevitas.core.zero_point import ZeroZeroPoint
from brevitas.inject.enum import ScalingImplType, StatsOp, RestrictValueType
from dependencies import value

class CustomQuant(ExtendedInjector):
    bit_width_impl_type = BitWidthImplType.CONST
    restrict_scaling_type = RestrictValueType.POWER_OF_TWO
    zero_point_impl = ZeroZeroPoint
    float_to_int_impl_type = FloatToIntImplType.ROUND
    scaling_impl_type = ScalingImplType.STATS
    scaling_stats_op = StatsOp.MAX
    scaling_per_output_channel = False
    bit_width = None
    narrow_range = True
    signed = True
    
    @value
    def quant_type():
        global weightBitWidth
        if weightBitWidth == 1:
            return QuantType.BINARY
        #elif  weightBitWidth ==2:
        #    return QuantType.TERNARY
        else:
            return QuantType.INT

class CustomWeightQuant(CustomQuant,WeightQuantSolver):
    scaling_const = 1.0        

class CustomActQuant(CustomQuant, ActQuantSolver):
    min_val = 0
    max_val = 10

#Global Variables

#End of Engine declaration

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo


class DeepLabQuant(nn.Module):
    def __init__(self, backbone='resnet', output_stride=16, num_classes=21, freeze_bn=False):
        super(DeepLabQuant, self).__init__()

        self.imageQuant = qnn.QuantIdentity(bit_width=8, act_quant=CustomActQuant, return_quant_tensor=True)


        self.backbone = AlignedXception(output_stride)
        self.aspp = build_aspp(backbone, output_stride)
        self.decoder = build_decoder(num_classes, backbone)

        self.freeze_bn = freeze_bn

    def setBitWidths(weight,activation):
        global weightBitWidth
        global activationBitWidth
        weightBitWidth=weight
        activationBitWidth=activation

    def forward(self, input):
        x= self.imageQuant(input)
        x, low_level_feat = self.backbone(x)

        x = self.aspp(x)
        x = self.decoder(x, low_level_feat)
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)

        return x

    

#BACKBONE XCEPTION

def fixed_padding(inputs, kernel_size, dilation):
    kernel_size_effective = kernel_size + (kernel_size - 1) * (dilation - 1)
    pad_total = kernel_size_effective - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    padded_inputs = F.pad(inputs, (pad_beg, pad_end, pad_beg, pad_end))
    return padded_inputs


class SeparableConv2d(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()



        self.conv1 = qnn.QuantConv2d(inplanes, inplanes, kernel_size, stride, 0, dilation, groups=inplanes, bias=bias, weight_bit_width=weightBitWidth, bias_quant=BiasQuant, weight_quant=CustomWeightQuant, return_quant_tensor=False)
        self.bn = nn.BatchNorm2d(inplanes)
        self.pointwise = qnn.QuantConv2d(inplanes, planes, 1, 1, 0, 1, 1, bias=bias, weight_bit_width=weightBitWidth, bias_quant=BiasQuant, weight_quant=CustomWeightQuant, return_quant_tensor=True)

    def forward(self, x):
        x = fixed_padding(x, self.conv1.kernel_size[0], dilation=self.conv1.dilation[0])

        x = self.conv1(x)
        x = self.bn(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):
    def __init__(self, inplanes, planes, reps, stride=1, dilation=1, start_with_relu=True, grow_first=True, is_last=False):
        super(Block, self).__init__()



        if planes != inplanes or stride != 1:
            self.skip = qnn.QuantConv2d(inplanes, planes, 1, stride=stride, bias=False, weight_bit_width=weightBitWidth, bias_quant=BiasQuant, weight_quant=CustomWeightQuant, return_quant_tensor=False)
            self.skipbn = nn.BatchNorm2d(planes)
        else:
            self.skip = None

        self.relu = qnn.QuantReLU(inplace=True, bit_width=activationBitWidth, return_quant_tensor=True, act_quant=CustomActQuant) if weightBitWidth not in (1,2) else qnn.QuantIdentity(bit_width=activationBitWidth, act_quant=CustomActQuant, return_quant_tensor=True)
        rep = []

        filters = inplanes
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(inplanes, planes, 3, 1, dilation))
            rep.append(nn.BatchNorm2d(planes))
            filters = planes

        for i in range(reps - 1):
            rep.append(self.relu)
            rep.append(SeparableConv2d(filters, filters, 3, 1, dilation))
            rep.append(nn.BatchNorm2d(filters))

        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(inplanes, planes, 3, 1, dilation))
            rep.append(nn.BatchNorm2d(planes))

        if stride != 1:
            rep.append(self.relu)
            rep.append(SeparableConv2d(planes, planes, 3, 2))
            rep.append(nn.BatchNorm2d(planes))

        if stride == 1 and is_last:
            rep.append(self.relu)
            rep.append(SeparableConv2d(planes, planes, 3, 1))
            rep.append(nn.BatchNorm2d(planes))

        if not start_with_relu:
            rep = rep[1:]

        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x = x + skip

        return x


class AlignedXception(nn.Module):
    """
    Modified Alighed Xception
    """
    def __init__(self, output_stride,pretrained=True):
        super(AlignedXception, self).__init__()


        if output_stride == 16:
            entry_block3_stride = 2
            middle_block_dilation = 1
            exit_block_dilations = (1, 2)
        elif output_stride == 8:
            entry_block3_stride = 1
            middle_block_dilation = 2
            exit_block_dilations = (2, 4)
        else:
            raise NotImplementedError


        # Entry flow
        self.conv1 = qnn.QuantConv2d(3, 32, 3, stride=2, padding=1, bias=False, weight_bit_width=weightBitWidth, bias_quant=BiasQuant, weight_quant=CustomWeightQuant, return_quant_tensor=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = qnn.QuantReLU(inplace=True, bit_width=activationBitWidth, return_quant_tensor=True, act_quant=CustomActQuant) if weightBitWidth not in (1,2) else qnn.QuantIdentity(bit_width=activationBitWidth, act_quant=CustomActQuant, return_quant_tensor=True)
        self.relu2 = qnn.QuantReLU(inplace=True, bit_width=activationBitWidth, return_quant_tensor=True, act_quant=CustomActQuant) if weightBitWidth not in (1,2) else qnn.QuantIdentity(bit_width=activationBitWidth, act_quant=CustomActQuant, return_quant_tensor=True)
        self.relu3 = qnn.QuantReLU(inplace=True, bit_width=activationBitWidth, return_quant_tensor=True, act_quant=CustomActQuant) if weightBitWidth not in (1,2) else qnn.QuantIdentity(bit_width=activationBitWidth, act_quant=CustomActQuant, return_quant_tensor=True)
        self.relu4 = qnn.QuantReLU(inplace=True, bit_width=activationBitWidth, return_quant_tensor=True, act_quant=CustomActQuant) if weightBitWidth not in (1,2) else qnn.QuantIdentity(bit_width=activationBitWidth, act_quant=CustomActQuant, return_quant_tensor=True)
        self.relu5 = qnn.QuantReLU(inplace=True, bit_width=activationBitWidth, return_quant_tensor=True, act_quant=CustomActQuant) if weightBitWidth not in (1,2) else qnn.QuantIdentity(bit_width=activationBitWidth, act_quant=CustomActQuant, return_quant_tensor=True)
        self.relu6 = qnn.QuantReLU(inplace=True, bit_width=activationBitWidth, return_quant_tensor=True, act_quant=CustomActQuant) if weightBitWidth not in (1,2) else qnn.QuantIdentity(bit_width=activationBitWidth, act_quant=CustomActQuant, return_quant_tensor=True)
        self.relu7 = qnn.QuantReLU(inplace=True, bit_width=activationBitWidth, return_quant_tensor=True, act_quant=CustomActQuant) if weightBitWidth not in (1,2) else qnn.QuantIdentity(bit_width=activationBitWidth, act_quant=CustomActQuant, return_quant_tensor=True)

        self.conv2 = qnn.QuantConv2d(32, 64, 3, stride=1, padding=1, bias=False, weight_bit_width=weightBitWidth, bias_quant=BiasQuant, weight_quant=CustomWeightQuant, return_quant_tensor=False)
        self.bn2 = nn.BatchNorm2d(64)

        self.block1 = Block(64, 128, reps=2, stride=2,  start_with_relu=False)
        self.block2 = Block(128, 256, reps=2, stride=2, start_with_relu=False,grow_first=True)
        self.block3 = Block(256, 728, reps=2, stride=entry_block3_stride,start_with_relu=True, grow_first=True, is_last=True)

        # Middle flow
        self.block4  = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation, start_with_relu=True, grow_first=True)
        self.block5  = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation, start_with_relu=True, grow_first=True)
        self.block6  = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation, start_with_relu=True, grow_first=True)
        self.block7  = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation, start_with_relu=True, grow_first=True)
        self.block8  = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation, start_with_relu=True, grow_first=True)
        self.block9  = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation, start_with_relu=True, grow_first=True)
        self.block10 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation, start_with_relu=True, grow_first=True)
        self.block11 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation, start_with_relu=True, grow_first=True)
        self.block12 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation, start_with_relu=True, grow_first=True)
        self.block13 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation, start_with_relu=True, grow_first=True)
        self.block14 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation, start_with_relu=True, grow_first=True)
        self.block15 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation, start_with_relu=True, grow_first=True)
        self.block16 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation, start_with_relu=True, grow_first=True)
        self.block17 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation, start_with_relu=True, grow_first=True)
        self.block18 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation, start_with_relu=True, grow_first=True)
        self.block19 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation, start_with_relu=True, grow_first=True)

        # Exit flow
        self.block20 = Block(728, 1024, reps=2, stride=1, dilation=exit_block_dilations[0], start_with_relu=True, grow_first=False, is_last=True)

        self.conv3 = SeparableConv2d(1024, 1536, 3, stride=1, dilation=exit_block_dilations[1])
        self.bn3 = nn.BatchNorm2d(1536)

        self.conv4 = SeparableConv2d(1536, 1536, 3, stride=1, dilation=exit_block_dilations[1])
        self.bn4 = nn.BatchNorm2d(1536)

        self.conv5 = SeparableConv2d(1536, 2048, 3, stride=1, dilation=exit_block_dilations[1])
        self.bn5 = nn.BatchNorm2d(2048)


    def forward(self, x):
        # Entry flow

        x = self.conv1(x)

        x = self.bn1(x)

        x = self.relu1(x)
 

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.block1(x)
        # add relu here
        x = self.relu3(x)
        low_level_feat = x
        x = self.block2(x)
        x = self.block3(x)

        # Middle flow
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)
        x = self.block13(x)
        x = self.block14(x)
        x = self.block15(x)
        x = self.block16(x)
        x = self.block17(x)
        x = self.block18(x)
        x = self.block19(x)

        # Exit flow
        x = self.block20(x)
        x = self.relu4(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu5(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu6(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu7(x)

        return x, low_level_feat



### ASPP

class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation):
        super(_ASPPModule, self).__init__()


        self.atrous_conv = qnn.QuantConv2d(inplanes, planes, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation, bias=False, weight_bit_width=weightBitWidth, bias_quant=BiasQuant, weight_quant=CustomWeightQuant, return_quant_tensor=False)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = qnn.QuantReLU(bit_width=activationBitWidth, return_quant_tensor=True, act_quant=CustomActQuant) if weightBitWidth not in (1,2) else qnn.QuantIdentity(bit_width=activationBitWidth, act_quant=CustomActQuant, return_quant_tensor=True)


    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)



class ASPP(nn.Module):
    def __init__(self, backbone, output_stride):
        super(ASPP, self).__init__()


        if backbone == 'drn':
            inplanes = 512
        elif backbone == 'mobilenet':
            inplanes = 320
        else:
            inplanes = 2048
        if output_stride == 16:
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = _ASPPModule(inplanes, 256, 1, padding=0, dilation=dilations[0])
        self.aspp2 = _ASPPModule(inplanes, 256, 3, padding=dilations[1], dilation=dilations[1])
        self.aspp3 = _ASPPModule(inplanes, 256, 3, padding=dilations[2], dilation=dilations[2])
        self.aspp4 = _ASPPModule(inplanes, 256, 3, padding=dilations[3], dilation=dilations[3])
        
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            qnn.QuantConv2d(inplanes, 256, 1, stride=1, bias=False, weight_bit_width=weightBitWidth, bias_quant=BiasQuant, weight_quant=CustomWeightQuant, return_quant_tensor=False),
            nn.BatchNorm2d(256),
            qnn.QuantReLU(bit_width=activationBitWidth, return_quant_tensor=True, act_quant=CustomActQuant) if weightBitWidth not in (1,2) else qnn.QuantIdentity(bit_width=activationBitWidth, act_quant=CustomActQuant, return_quant_tensor=True)
            )

        
        self.conv1 = qnn.QuantConv2d(1280, 256, 1, bias=False, weight_bit_width=weightBitWidth, bias_quant=BiasQuant, weight_quant=CustomWeightQuant, return_quant_tensor=False)
        self.bn1 = nn.BatchNorm2d(256)
        self.relu = qnn.QuantReLU(bit_width=activationBitWidth, return_quant_tensor=True, act_quant=CustomActQuant) if weightBitWidth not in (1,2) else qnn.QuantIdentity(bit_width=activationBitWidth, act_quant=CustomActQuant, return_quant_tensor=True)
        self.dropout = qnn.QuantDropout(0.5, return_quant_tensor=True)

        self.quant_inp = qnn.QuantIdentity(bit_width=activationBitWidth,act_quant=CustomActQuant, return_quant_tensor=True)
        #self.concat =qnn.QuantCat(return_quant_tensor=True)


    def forward(self, x):
        x1 = self.aspp1(x)
        x1 = self.quant_inp(x1)
        x2 = self.aspp2(x)
        x2 = self.quant_inp(x2)
        x3 = self.aspp3(x)
        x3 = self.quant_inp(x3)
        x4 = self.aspp4(x)
        x4 = self.quant_inp(x4)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)

        x5 = self.quant_inp(x5)
        

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return self.dropout(x)


def build_aspp(backbone, output_stride):
    return ASPP(backbone, output_stride)

###DECODER

class Decoder(nn.Module):
    def __init__(self, num_classes, backbone):
        super(Decoder, self).__init__()

        low_level_inplanes = 128


        self.conv1 = qnn.QuantConv2d(low_level_inplanes, 48, 1, bias=False, weight_bit_width=weightBitWidth, bias_quant=BiasQuant, weight_quant=CustomWeightQuant, return_quant_tensor=True)
        self.bn1 = nn.BatchNorm2d(48)
        self.relu = qnn.QuantReLU(bit_width=activationBitWidth, return_quant_tensor=True, act_quant=CustomActQuant) if weightBitWidth not in (1,2) else qnn.QuantIdentity(bit_width=activationBitWidth, act_quant=CustomActQuant, return_quant_tensor=True)
        
        self.last_conv = nn.Sequential(
            qnn.QuantConv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False, weight_bit_width=weightBitWidth, bias_quant=BiasQuant, weight_quant=CustomWeightQuant, return_quant_tensor=True),
            nn.BatchNorm2d(256),
            qnn.QuantReLU(bit_width=activationBitWidth, return_quant_tensor=True, act_quant=CustomActQuant) if weightBitWidth not in (1,2) else qnn.QuantIdentity(bit_width=activationBitWidth, act_quant=CustomActQuant, return_quant_tensor=True),
            qnn.QuantDropout(0.5, return_quant_tensor=True),
            qnn.QuantConv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False, weight_bit_width=weightBitWidth, bias_quant=BiasQuant, weight_quant=CustomWeightQuant, return_quant_tensor=True),
            nn.BatchNorm2d(256),
            qnn.QuantReLU(bit_width=activationBitWidth, return_quant_tensor=True, act_quant=CustomActQuant) if weightBitWidth not in (1,2) else qnn.QuantIdentity(bit_width=activationBitWidth, act_quant=CustomActQuant, return_quant_tensor=True),
            qnn.QuantDropout(0.1, return_quant_tensor=True),
            qnn.QuantConv2d(256, num_classes, kernel_size=1, stride=1, weight_bit_width=weightBitWidth, bias_quant=BiasQuant, weight_quant=CustomWeightQuant, return_quant_tensor=False)
            )
        self.quant_inp = qnn.QuantIdentity(bit_width=activationBitWidth,act_quant=CustomActQuant, return_quant_tensor=True)
        #self.concat =qnn.QuantCat(return_quant_tensor=True)


    def forward(self, x, low_level_feat):
        low_level_feat = self.conv1(low_level_feat)
        low_level_feat = self.bn1(low_level_feat)
        low_level_feat = self.relu(low_level_feat)
        low_level_feat = self.quant_inp(low_level_feat)

        x = F.interpolate(x, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)
        x = self.quant_inp(x)
        x = torch.cat((x, low_level_feat), dim=1)
        x = self.last_conv(x)
        

        return x


def build_decoder(num_classes, backbone):
    return Decoder(num_classes, backbone)

if __name__ == "__main__":
    model = DeepLabQuant(backbone='mobilenet', output_stride=16)
    model.eval()
    input = torch.rand(1, 3, 513, 513)
    output = model(input)
    print(output.size())


