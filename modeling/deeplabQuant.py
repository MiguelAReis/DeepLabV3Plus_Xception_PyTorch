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
	narrow_range = False
	signed = True
	
	@value
	def quant_type():
		global weightBitWidth
		if weightBitWidth == 1:
			return QuantType.BINARY
		#elif  weightBitWidth ==2:
		#	return QuantType.TERNARY
		else:
			return QuantType.INT

class CustomWeightQuant(CustomQuant,WeightQuantSolver):
	scaling_const = 1.0		

class CustomActQuant(CustomQuant, ActQuantSolver):
    signed=False
    float_to_int_impl_type = FloatToIntImplType.FLOOR

class CustomSignedActQuant(CustomQuant, ActQuantSolver):
    signed=True
    float_to_int_impl_type = FloatToIntImplType.FLOOR

#Global Variables

#End of Engine declaration

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

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

		self.depthwise = qnn.QuantConv2d(inplanes, inplanes, kernel_size, stride, 0, dilation, groups=inplanes, bias=bias, weight_bit_width=weightBitWidth, bias_quant=BiasQuant, weight_quant=CustomWeightQuant, return_quant_tensor=True)
		self.bn = nn.BatchNorm2d(inplanes)
		self.identity = qnn.QuantIdentity( bit_width=activationBitWidth, return_quant_tensor=True, act_quant=CustomSignedActQuant)
		self.pointwise = qnn.QuantConv2d(inplanes, planes, 1, 1, 0, 1, 1, bias=bias, weight_bit_width=weightBitWidth, bias_quant=BiasQuant, weight_quant=CustomWeightQuant, return_quant_tensor=True)

	def forward(self, x):
		x = fixed_padding(x, self.depthwise.kernel_size[0], dilation=self.depthwise.dilation[0])
		x = self.depthwise(x)
		x = self.bn(x)
		x = self.identity(x)
		x = self.pointwise(x)
		return x


class Block(nn.Module):
	def __init__(self, inplanes, planes, reps, stride=1, dilation=1, start_with_relu=True, grow_first=True, is_last=False):
		super(Block, self).__init__()

		#self.relu = qnn.QuantReLU(inplace=True, bit_width=activationBitWidth, return_quant_tensor=True, act_quant=CustomActQuant) if weightBitWidth not in (1,2) else qnn.QuantIdentity(bit_width=activationBitWidth, act_quant=CustomActQuant, return_quant_tensor=True)
		rep = []

		filters = inplanes
		if grow_first:
			rep.append(qnn.QuantReLU(inplace=True, bit_width=activationBitWidth, return_quant_tensor=True, act_quant=CustomActQuant) if weightBitWidth not in (1,2) else qnn.QuantIdentity(bit_width=activationBitWidth, act_quant=CustomActQuant, return_quant_tensor=True)
					   )
			rep.append(SeparableConv2d(inplanes, planes, 3, 1, dilation))
			rep.append(nn.BatchNorm2d(planes))
			filters = planes

		for i in range(reps - 1):
			rep.append(qnn.QuantReLU(inplace=True, bit_width=activationBitWidth, return_quant_tensor=True, act_quant=CustomActQuant) if weightBitWidth not in (1,2) else qnn.QuantIdentity(bit_width=activationBitWidth, act_quant=CustomActQuant, return_quant_tensor=True)
					   )
			rep.append(SeparableConv2d(filters, filters, 3, 1, dilation))
			rep.append(nn.BatchNorm2d(filters))

		if not grow_first:
			rep.append(qnn.QuantReLU(inplace=True, bit_width=activationBitWidth, return_quant_tensor=True, act_quant=CustomActQuant) if weightBitWidth not in (1,2) else qnn.QuantIdentity(bit_width=activationBitWidth, act_quant=CustomActQuant, return_quant_tensor=True)
					   )
			rep.append(SeparableConv2d(inplanes, planes, 3, 1, dilation))
			rep.append(nn.BatchNorm2d(planes))

		if stride != 1:
			rep.append(qnn.QuantReLU(inplace=True, bit_width=activationBitWidth, return_quant_tensor=True, act_quant=CustomActQuant) if weightBitWidth not in (1,2) else qnn.QuantIdentity(bit_width=activationBitWidth, act_quant=CustomActQuant, return_quant_tensor=True)
					   )
			rep.append(SeparableConv2d(planes, planes, 3, 2))
			rep.append(nn.BatchNorm2d(planes))

		if stride == 1 and is_last:
			rep.append(qnn.QuantReLU(inplace=True, bit_width=activationBitWidth, return_quant_tensor=True, act_quant=CustomActQuant) if weightBitWidth not in (1,2) else qnn.QuantIdentity(bit_width=activationBitWidth, act_quant=CustomActQuant, return_quant_tensor=True)
					   )
			rep.append(SeparableConv2d(planes, planes, 3, 1))
			rep.append(nn.BatchNorm2d(planes))

		if not start_with_relu:
			rep = rep[1:]

		self.rep = nn.Sequential(*rep)


	   # if planes != inplanes or stride != 1:
	   #	 self.skip = qnn.QuantConv2d(inplanes, planes, 1, stride=stride, bias=False, weight_bit_width=weightBitWidth, bias_quant=BiasQuant, weight_quant=CustomWeightQuant, return_quant_tensor=True)
	   #	 self.skipbn = nn.BatchNorm2d(planes)
	   #	 self.skiprelu = self.finalrelu
	   # else:
	   #	 self.skip = None

	def forward(self, inp):
		x = self.rep(inp)


		#if self.skip is not None:
		#	skip = self.skip(inp)
		#	skip = self.skipbn(skip)
		#	skip = self.skiprelu(skip)

		return x

class _ASPPModule(nn.Module):
	def __init__(self, inplanes, planes, kernel_size, padding, dilation):
		super(_ASPPModule, self).__init__()
		self.atrous_conv = qnn.QuantConv2d(inplanes, planes, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation, bias=False, weight_bit_width=weightBitWidth, bias_quant=BiasQuant, weight_quant=CustomWeightQuant, return_quant_tensor=True)
		self.bn = nn.BatchNorm2d(planes)



	def forward(self, x):
		x = self.atrous_conv(x)
		x = self.bn(x)

		return x


class DeepLabQuant(nn.Module):
	def __init__(self, backbone='resnet', output_stride=16, num_classes=21, freeze_bn=False):
		super(DeepLabQuant, self).__init__()
		global weightBitWidth
		global activationBitWidth

		self.imageQuant = qnn.QuantIdentity(bit_width=activationBitWidth, act_quant=CustomActQuant, return_quant_tensor=True)

		#BACKBONE

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
		self.conv1 = qnn.QuantConv2d(3, 32, 3, stride=2, padding=1, bias=False, weight_bit_width=weightBitWidth, bias_quant=BiasQuant, weight_quant=CustomWeightQuant, return_quant_tensor=True)
		self.bn1 = nn.BatchNorm2d(32)
		self.relu1 = qnn.QuantReLU(inplace=True, bit_width=activationBitWidth, return_quant_tensor=True, act_quant=CustomActQuant) if weightBitWidth not in (1,2) else qnn.QuantIdentity(bit_width=activationBitWidth, act_quant=CustomActQuant, return_quant_tensor=True)
		

		self.conv2 = qnn.QuantConv2d(32, 64, 3, stride=1, padding=1, bias=False, weight_bit_width=weightBitWidth, bias_quant=BiasQuant, weight_quant=CustomWeightQuant, return_quant_tensor=True)
		self.bn2 = nn.BatchNorm2d(64)
		self.relu2 = qnn.QuantReLU(inplace=True, bit_width=activationBitWidth, return_quant_tensor=True, act_quant=CustomActQuant) if weightBitWidth not in (1,2) else qnn.QuantIdentity(bit_width=activationBitWidth, act_quant=CustomActQuant, return_quant_tensor=True)


		self.block1 = Block(64, 128, reps=2, stride=2,  start_with_relu=False) #skip
		self.block1skip = qnn.QuantConv2d(64, 128, 1, stride=2, bias=False, weight_bit_width=weightBitWidth, bias_quant=BiasQuant, weight_quant=CustomWeightQuant, return_quant_tensor=True)
		self.block1skipbn = nn.BatchNorm2d(128)
		self.block1relu = qnn.QuantReLU(inplace=True, bit_width=activationBitWidth, return_quant_tensor=True, act_quant=CustomActQuant) if weightBitWidth not in (1,2) else qnn.QuantIdentity(bit_width=activationBitWidth, act_quant=CustomActQuant, return_quant_tensor=True)
		self.block1identity = qnn.QuantIdentity( bit_width=activationBitWidth, return_quant_tensor=True, act_quant=CustomActQuant)

		self.block2 = Block(128, 256, reps=2, stride=2, start_with_relu=False,grow_first=True) #skip
		self.block2skip = qnn.QuantConv2d(128, 256, 1, stride=2, bias=False, weight_bit_width=weightBitWidth, bias_quant=BiasQuant, weight_quant=CustomWeightQuant, return_quant_tensor=True)
		self.block2skipbn = nn.BatchNorm2d(256)
		self.block2relu = qnn.QuantReLU(inplace=True, bit_width=activationBitWidth, return_quant_tensor=True, act_quant=CustomActQuant) if weightBitWidth not in (1,2) else qnn.QuantIdentity(bit_width=activationBitWidth, act_quant=CustomActQuant, return_quant_tensor=True)
		self.block2identity = qnn.QuantIdentity( bit_width=activationBitWidth, return_quant_tensor=True, act_quant=CustomActQuant)

		self.block3 = Block(256, 728, reps=2, stride=entry_block3_stride,start_with_relu=False, grow_first=True, is_last=True) #skip
		self.block3skip = qnn.QuantConv2d(256, 728, 1, stride=entry_block3_stride, bias=False, weight_bit_width=weightBitWidth, bias_quant=BiasQuant, weight_quant=CustomWeightQuant, return_quant_tensor=True)
		self.block3skipbn = nn.BatchNorm2d(728)
		self.block3relu1 = qnn.QuantReLU(inplace=True, bit_width=activationBitWidth, return_quant_tensor=True, act_quant=CustomActQuant) if weightBitWidth not in (1,2) else qnn.QuantIdentity(bit_width=activationBitWidth, act_quant=CustomActQuant, return_quant_tensor=True)
		self.block3relu2 = qnn.QuantReLU(inplace=True, bit_width=activationBitWidth, return_quant_tensor=True, act_quant=CustomActQuant) if weightBitWidth not in (1,2) else qnn.QuantIdentity(bit_width=activationBitWidth, act_quant=CustomActQuant, return_quant_tensor=True)

		# Middle flow
		self.block4  = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation, start_with_relu=False, grow_first=True)
		self.block4relu = qnn.QuantReLU(inplace=True, bit_width=activationBitWidth, return_quant_tensor=True, act_quant=CustomActQuant) if weightBitWidth not in (1,2) else qnn.QuantIdentity(bit_width=activationBitWidth, act_quant=CustomActQuant, return_quant_tensor=True)
		self.block5  = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation, start_with_relu=False, grow_first=True)
		self.block5relu = qnn.QuantReLU(inplace=True, bit_width=activationBitWidth, return_quant_tensor=True, act_quant=CustomActQuant) if weightBitWidth not in (1,2) else qnn.QuantIdentity(bit_width=activationBitWidth, act_quant=CustomActQuant, return_quant_tensor=True)
		self.block6  = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation, start_with_relu=False, grow_first=True)
		self.block6relu = qnn.QuantReLU(inplace=True, bit_width=activationBitWidth, return_quant_tensor=True, act_quant=CustomActQuant) if weightBitWidth not in (1,2) else qnn.QuantIdentity(bit_width=activationBitWidth, act_quant=CustomActQuant, return_quant_tensor=True)
		self.block7  = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation, start_with_relu=False, grow_first=True)
		self.block7relu = qnn.QuantReLU(inplace=True, bit_width=activationBitWidth, return_quant_tensor=True, act_quant=CustomActQuant) if weightBitWidth not in (1,2) else qnn.QuantIdentity(bit_width=activationBitWidth, act_quant=CustomActQuant, return_quant_tensor=True)
		self.block8  = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation, start_with_relu=False, grow_first=True)
		self.block8relu = qnn.QuantReLU(inplace=True, bit_width=activationBitWidth, return_quant_tensor=True, act_quant=CustomActQuant) if weightBitWidth not in (1,2) else qnn.QuantIdentity(bit_width=activationBitWidth, act_quant=CustomActQuant, return_quant_tensor=True)
		self.block9  = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation, start_with_relu=False, grow_first=True)
		self.block9relu = qnn.QuantReLU(inplace=True, bit_width=activationBitWidth, return_quant_tensor=True, act_quant=CustomActQuant) if weightBitWidth not in (1,2) else qnn.QuantIdentity(bit_width=activationBitWidth, act_quant=CustomActQuant, return_quant_tensor=True)
		self.block10 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation, start_with_relu=False, grow_first=True)
		self.block10relu = qnn.QuantReLU(inplace=True, bit_width=activationBitWidth, return_quant_tensor=True, act_quant=CustomActQuant) if weightBitWidth not in (1,2) else qnn.QuantIdentity(bit_width=activationBitWidth, act_quant=CustomActQuant, return_quant_tensor=True)
		self.block11 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation, start_with_relu=False, grow_first=True)
		self.block11relu = qnn.QuantReLU(inplace=True, bit_width=activationBitWidth, return_quant_tensor=True, act_quant=CustomActQuant) if weightBitWidth not in (1,2) else qnn.QuantIdentity(bit_width=activationBitWidth, act_quant=CustomActQuant, return_quant_tensor=True)
		self.block12 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation, start_with_relu=False, grow_first=True)
		self.block12relu = qnn.QuantReLU(inplace=True, bit_width=activationBitWidth, return_quant_tensor=True, act_quant=CustomActQuant) if weightBitWidth not in (1,2) else qnn.QuantIdentity(bit_width=activationBitWidth, act_quant=CustomActQuant, return_quant_tensor=True)
		self.block13 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation, start_with_relu=False, grow_first=True)
		self.block13relu = qnn.QuantReLU(inplace=True, bit_width=activationBitWidth, return_quant_tensor=True, act_quant=CustomActQuant) if weightBitWidth not in (1,2) else qnn.QuantIdentity(bit_width=activationBitWidth, act_quant=CustomActQuant, return_quant_tensor=True)
		self.block14 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation, start_with_relu=False, grow_first=True)
		self.block14relu = qnn.QuantReLU(inplace=True, bit_width=activationBitWidth, return_quant_tensor=True, act_quant=CustomActQuant) if weightBitWidth not in (1,2) else qnn.QuantIdentity(bit_width=activationBitWidth, act_quant=CustomActQuant, return_quant_tensor=True)
		self.block15 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation, start_with_relu=False, grow_first=True)
		self.block15relu = qnn.QuantReLU(inplace=True, bit_width=activationBitWidth, return_quant_tensor=True, act_quant=CustomActQuant) if weightBitWidth not in (1,2) else qnn.QuantIdentity(bit_width=activationBitWidth, act_quant=CustomActQuant, return_quant_tensor=True)
		self.block16 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation, start_with_relu=False, grow_first=True)
		self.block16relu = qnn.QuantReLU(inplace=True, bit_width=activationBitWidth, return_quant_tensor=True, act_quant=CustomActQuant) if weightBitWidth not in (1,2) else qnn.QuantIdentity(bit_width=activationBitWidth, act_quant=CustomActQuant, return_quant_tensor=True)
		self.block17 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation, start_with_relu=False, grow_first=True)
		self.block17relu = qnn.QuantReLU(inplace=True, bit_width=activationBitWidth, return_quant_tensor=True, act_quant=CustomActQuant) if weightBitWidth not in (1,2) else qnn.QuantIdentity(bit_width=activationBitWidth, act_quant=CustomActQuant, return_quant_tensor=True)
		self.block18 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation, start_with_relu=False, grow_first=True)
		self.block18relu = qnn.QuantReLU(inplace=True, bit_width=activationBitWidth, return_quant_tensor=True, act_quant=CustomActQuant) if weightBitWidth not in (1,2) else qnn.QuantIdentity(bit_width=activationBitWidth, act_quant=CustomActQuant, return_quant_tensor=True)
		self.block19 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation, start_with_relu=False, grow_first=True)
		self.block19relu = qnn.QuantReLU(inplace=True, bit_width=activationBitWidth, return_quant_tensor=True, act_quant=CustomActQuant) if weightBitWidth not in (1,2) else qnn.QuantIdentity(bit_width=activationBitWidth, act_quant=CustomActQuant, return_quant_tensor=True)

		# Exit flow
		self.block20 = Block(728, 1024, reps=2, stride=1, dilation=exit_block_dilations[0], start_with_relu=False, grow_first=False, is_last=True) #skip
		self.block20skip = qnn.QuantConv2d(728, 1024, 1, stride=1, bias=False, weight_bit_width=weightBitWidth, bias_quant=BiasQuant, weight_quant=CustomWeightQuant, return_quant_tensor=True)
		self.block20skipbn = nn.BatchNorm2d(1024)
		self.block20relu = qnn.QuantReLU(inplace=True, bit_width=activationBitWidth, return_quant_tensor=True, act_quant=CustomActQuant) if weightBitWidth not in (1,2) else qnn.QuantIdentity(bit_width=activationBitWidth, act_quant=CustomActQuant, return_quant_tensor=True)
		self.block20identity = qnn.QuantIdentity( bit_width=activationBitWidth, return_quant_tensor=True, act_quant=CustomActQuant)

		self.conv3 = SeparableConv2d(1024, 1536, 3, stride=1, dilation=exit_block_dilations[1])
		self.bn3 = nn.BatchNorm2d(1536)
		self.relu3 = qnn.QuantReLU(inplace=True, bit_width=activationBitWidth, return_quant_tensor=True, act_quant=CustomActQuant) if weightBitWidth not in (1,2) else qnn.QuantIdentity(bit_width=activationBitWidth, act_quant=CustomActQuant, return_quant_tensor=True)

		self.conv4 = SeparableConv2d(1536, 1536, 3, stride=1, dilation=exit_block_dilations[1])
		self.bn4 = nn.BatchNorm2d(1536)
		self.relu4 = qnn.QuantReLU(inplace=True, bit_width=activationBitWidth, return_quant_tensor=True, act_quant=CustomActQuant) if weightBitWidth not in (1,2) else qnn.QuantIdentity(bit_width=activationBitWidth, act_quant=CustomActQuant, return_quant_tensor=True)

		self.conv5 = SeparableConv2d(1536, 2048, 3, stride=1, dilation=exit_block_dilations[1])
		self.bn5 = nn.BatchNorm2d(2048)
		self.relu5 = qnn.QuantReLU(inplace=True, bit_width=activationBitWidth, return_quant_tensor=True, act_quant=CustomActQuant) if weightBitWidth not in (1,2) else qnn.QuantIdentity(bit_width=activationBitWidth, act_quant=CustomActQuant, return_quant_tensor=True)

		#ASPP

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
		self.aspprelu1 = qnn.QuantReLU(bit_width=activationBitWidth, return_quant_tensor=True, act_quant=CustomActQuant) if weightBitWidth not in (1,2) else qnn.QuantIdentity(bit_width=activationBitWidth, act_quant=CustomActQuant, return_quant_tensor=True)

		self.global_avg_pool = nn.Sequential(
			nn.AdaptiveAvgPool2d((1, 1)),
			qnn.QuantIdentity( bit_width=activationBitWidth, return_quant_tensor=True, act_quant=CustomActQuant),
			qnn.QuantConv2d(inplanes, 256, 1, stride=1, bias=False, weight_bit_width=weightBitWidth, bias_quant=BiasQuant, weight_quant=CustomWeightQuant, return_quant_tensor=True),
			nn.BatchNorm2d(256)
			)
		self.avgpoolidentity = qnn.QuantReLU(bit_width=activationBitWidth, return_quant_tensor=True, act_quant=CustomActQuant) if weightBitWidth not in (1,2) else qnn.QuantIdentity(bit_width=activationBitWidth, act_quant=CustomActQuant, return_quant_tensor=True)
		self.interpolidentity1= self.aspprelu1

		self.asppconv1 = qnn.QuantConv2d(1280, 256, 1, bias=False, weight_bit_width=weightBitWidth, bias_quant=BiasQuant, weight_quant=CustomWeightQuant, return_quant_tensor=True)
		self.asppbn1 = nn.BatchNorm2d(256)
		self.aspprelu2 = qnn.QuantReLU(bit_width=activationBitWidth, return_quant_tensor=True, act_quant=CustomActQuant) if weightBitWidth not in (1,2) else qnn.QuantIdentity(bit_width=activationBitWidth, act_quant=CustomActQuant, return_quant_tensor=True)
		self.asppdropout = qnn.QuantDropout(0.5, return_quant_tensor=True)
		


		low_level_inplanes = 128


		self.decoderconv1 = qnn.QuantConv2d(low_level_inplanes, 48, 1, bias=False, weight_bit_width=weightBitWidth, bias_quant=BiasQuant, weight_quant=CustomWeightQuant, return_quant_tensor=True)
		self.decoderbn1 = nn.BatchNorm2d(48)
		self.decoderrelu = qnn.QuantReLU(bit_width=activationBitWidth, return_quant_tensor=True, act_quant=CustomActQuant) if weightBitWidth not in (1,2) else qnn.QuantIdentity(bit_width=activationBitWidth, act_quant=CustomActQuant, return_quant_tensor=True)

		self.decoderlast_conv = nn.Sequential(
			qnn.QuantConv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False, weight_bit_width=weightBitWidth, bias_quant=BiasQuant, weight_quant=CustomWeightQuant, return_quant_tensor=True),
			nn.BatchNorm2d(256),
			qnn.QuantReLU(bit_width=activationBitWidth, return_quant_tensor=True, act_quant=CustomActQuant) if weightBitWidth not in (1,2) else qnn.QuantIdentity(bit_width=activationBitWidth, act_quant=CustomActQuant, return_quant_tensor=True),
			qnn.QuantDropout(0.5, return_quant_tensor=True),
			qnn.QuantConv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False, weight_bit_width=weightBitWidth, bias_quant=BiasQuant, weight_quant=CustomWeightQuant, return_quant_tensor=True),
			nn.BatchNorm2d(256),
			qnn.QuantReLU(bit_width=activationBitWidth, return_quant_tensor=True, act_quant=CustomActQuant) if weightBitWidth not in (1,2) else qnn.QuantIdentity(bit_width=activationBitWidth, act_quant=CustomActQuant, return_quant_tensor=True),
			qnn.QuantDropout(0.1, return_quant_tensor=True),
			qnn.QuantConv2d(256, num_classes, kernel_size=1, stride=1, weight_bit_width=weightBitWidth, bias_quant=BiasQuant, weight_quant=CustomWeightQuant, return_quant_tensor=True)
			)
		self.decoderidentity = qnn.QuantIdentity( bit_width=activationBitWidth, return_quant_tensor=True, act_quant=CustomActQuant)
		self.interpolidentity3= qnn.QuantIdentity( bit_width=activationBitWidth, return_quant_tensor=False, act_quant=CustomActQuant)

	def setBitWidths(weight,activation):
		global weightBitWidth
		global activationBitWidth
		weightBitWidth=weight
		activationBitWidth=activation

	def forward(self, input):

		#BACKBONE

		 # Entry flow
		x = self.conv1(self.imageQuant(input))
		x = self.bn1(x)
		x = self.relu1(x)

		x = self.conv2(x)
		x = self.bn2(x)
		x = self.relu2(x)

		x_ = self.block1(x)
		x_ = self.block1relu(x_)
		x = self.block1skip(x)
		x = self.block1skipbn(x)
		x = self.block1relu(x)
		x = x+x_
		x = self.block1identity(x)




		# add relu here
		#x = self.relu3(x)
		low_level_feat = x

		x_ = self.block2(x)
		x_ = self.block2relu(x_)
		x = self.block2skip(x)
		x = self.block2skipbn(x)
		x = self.block2relu(x)
		x = x+x_
		x = self.block2identity(x)

		x_ = self.block3(x)
		x_ = self.block3relu1(x_)
		x = self.block3skip(x)
		x = self.block3skipbn(x)
		x = self.block3relu1(x)
		x = x+x_
		x = self.block3relu2(x)




		# Middle flow
		x_ = self.block4(x)
		x_ = self.block3relu2(x_)
		x = x+x_
		x = self.block4relu(x)

		x_ = self.block5(x)
		x_ = self.block4relu(x_)
		x = x+x_
		x = self.block5relu(x)

		x_ = self.block6(x)
		x_ = self.block5relu(x_)
		x = x+x_
		x = self.block6relu(x)

		x_ = self.block7(x)
		x_ = self.block6relu(x_)
		x = x+x_
		x = self.block7relu(x)

		x_ = self.block8(x)
		x_ = self.block7relu(x_)
		x = x+x_
		x = self.block8relu(x)

		x_ = self.block9(x)
		x_ = self.block8relu(x_)
		x = x+x_
		x = self.block9relu(x)

		x_ = self.block10(x)
		x_ = self.block9relu(x_)
		x = x+x_
		x = self.block10relu(x)

		x_ = self.block11(x)
		x_ = self.block10relu(x_)
		x = x+x_
		x = self.block11relu(x)

		x_ = self.block12(x)
		x_ = self.block11relu(x_)
		x = x+x_
		x = self.block12relu(x)

		x_ = self.block13(x)
		x_ = self.block12relu(x_)
		x = x+x_
		x = self.block13relu(x)

		x_ = self.block14(x)
		x_ = self.block13relu(x_)
		x = x+x_
		x = self.block14relu(x)

		x_ = self.block15(x)
		x_ = self.block14relu(x_)
		x = x+x_
		x = self.block15relu(x)

		x_ = self.block16(x)
		x_ = self.block15relu(x_)
		x = x+x_
		x = self.block16relu(x)

		x_ = self.block17(x)
		x_ = self.block16relu(x_)
		x = x+x_
		x = self.block17relu(x)

		x_ = self.block18(x)
		x_ = self.block17relu(x_)
		x = x+x_
		x = self.block18relu(x)

		x_ = self.block19(x)
		x_ = self.block18relu(x_)
		x = x+x_
		x = self.block19relu(x)

		# Exit flow
		x_ = self.block20(x)
		x_ = self.block20relu(x_)
		x = self.block20skip(x)
		x = self.block20skipbn(x)
		x = self.block20relu(x)
		x = x+x_
		x = self.block20identity(x)

		#x = self.relu4(x)
		x = self.conv3(x)
		x = self.bn3(x)
		x = self.relu3(x)

		x = self.conv4(x)
		x = self.bn4(x)
		x = self.relu4(x)

		x = self.conv5(x)
		x = self.bn5(x)
		x = self.relu5(x)

		#ASPP
		x1 = self.aspp1(x)
		x1 =self.aspprelu1(x1)
		x2 = self.aspp2(x)
		x2 =self.aspprelu1(x2)
		x3 = self.aspp3(x)
		x3 =self.aspprelu1(x3)
		x4 = self.aspp4(x)
		x4 =self.aspprelu1(x4)
		#print("before avg pool ="+ str(x.size()))
		x5 = self.global_avg_pool(x)
		x5 = self.avgpoolidentity(x5)

		#print("before size ="+ str(x5.size()))
		x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
	   #print("after size ="+ str(x5.size()))
		x5 =self.aspprelu1(x5)

		x = torch.cat((x1, x2, x3, x4, x5), dim=1)


		x = self.asppconv1(x)
		x = self.asppbn1(x)
		x = self.aspprelu2(x)
		x = self.asppdropout(x)


		#DECODER

		low_level_feat = self.decoderconv1(low_level_feat)
		low_level_feat = self.decoderbn1(low_level_feat)
		low_level_feat = self.decoderrelu(low_level_feat)
		#print("before size ="+ str(x.size()))
		x = F.interpolate(x, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)
		x =self.decoderrelu(x)
		#print("after size ="+ str(x.size()))
		x = torch.cat((x, low_level_feat), dim=1)
		x = self.decoderlast_conv(x)
		x = self.decoderidentity(x)

		#print("before size ="+ str(x.size()))
		x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)
		x = self.interpolidentity3(x)
		#print("after size ="+ str(x.size()))


		return x

	




