import torch.nn as nn

from model import MODEL



def conv3x3(in_planes, out_planes, stride = 1, groups = 1, dilation = 1):
	return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride = 1) -> nn.Conv2d:
	return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def deconv2x2(in_planes, out_planes, stride = 1, groups = 1, dilation = 1):
	return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=2, stride=stride, groups=groups, bias=False, dilation=dilation)


class DeBasicBlock(nn.Module):
	expansion = 1

	def __init__(self, inplanes, planes, stride = 1, upsample = None, groups = 1, base_width = 64,
	 dilation = 1, norm_layer = None):
		super(DeBasicBlock, self).__init__()
		if norm_layer is None:
			norm_layer = nn.BatchNorm2d
		if groups != 1 or base_width != 64:
			raise ValueError('BasicBlock only supports groups=1 and base_width=64')
		if dilation > 1:
			raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
  
		if stride == 2:
			self.conv1 = deconv2x2(inplanes, planes, stride)
		else:
			self.conv1 = conv3x3(inplanes, planes, stride)
		self.bn1 = norm_layer(planes)
		self.relu = nn.ReLU(inplace=True)
		self.conv2 = conv3x3(planes, planes)
		self.bn2 = norm_layer(planes)
		self.upsample = upsample
		self.stride = stride

	def forward(self, x):
		identity = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)

		if self.upsample is not None:
			identity = self.upsample(x)

		out += identity
		out = self.relu(out)

		return out


class DeBottleneck(nn.Module):
	expansion = 4

	def __init__(self, inplanes, planes, stride = 1, upsample = None, groups = 1, base_width = 64,
	 dilation = 1, norm_layer = None):
		super(DeBottleneck, self).__init__()
		if norm_layer is None:
			norm_layer = nn.BatchNorm2d
		width = int(planes * (base_width / 64.)) * groups
		
		self.conv1 = conv1x1(inplanes, width)
		self.bn1 = norm_layer(width)
		if stride == 2:
			self.conv2 = deconv2x2(width, width, stride, groups, dilation)
		else:
			self.conv2 = conv3x3(width, width, stride, groups, dilation)
		self.bn2 = norm_layer(width)
		self.conv3 = conv1x1(width, planes * self.expansion)
		self.bn3 = norm_layer(planes * self.expansion)
		self.relu = nn.ReLU(inplace=True)
		self.upsample = upsample
		self.stride = stride

	def forward(self, x):
		identity = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)
		out = self.relu(out)

		out = self.conv3(out)
		out = self.bn3(out)

		if self.upsample is not None:
			identity = self.upsample(x)

		out += identity
		out = self.relu(out)

		return out


class ResNet(nn.Module):

	def __init__(self, block, layers, num_classes = 1000,
	    zero_init_residual = False, groups = 1, width_per_group = 64, replace_stride_with_dilation = None,
	    norm_layer = None ):
		super(ResNet, self).__init__()
		if norm_layer is None:
			norm_layer = nn.BatchNorm2d
		self._norm_layer = norm_layer

		self.inplanes = 512 * block.expansion
		self.dilation = 1
		if replace_stride_with_dilation is None:
			replace_stride_with_dilation = [False, False, False]
		if len(replace_stride_with_dilation) != 3:
			raise ValueError("replace_stride_with_dilation should be None or a 3-element tuple, got {}".format(replace_stride_with_dilation))
		self.groups = groups
		self.base_width = width_per_group
		self.layer1 = self._make_layer(block, 256, layers[0], stride=2)
		self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
		self.layer3 = self._make_layer(block, 64, layers[2], stride=2, dilate=replace_stride_with_dilation[1])

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
			elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)
  
  
  
		if zero_init_residual:
			for m in self.modules():
				if isinstance(m, DeBottleneck):
					nn.init.constant_(m.bn3.weight, 0)  
				elif isinstance(m, DeBasicBlock):
					nn.init.constant_(m.bn2.weight, 0)  

	def _make_layer(self, block, planes, blocks, stride = 1, dilate = False):
		norm_layer = self._norm_layer
		upsample = None
		previous_dilation = self.dilation
		if dilate:
			self.dilation *= stride
			stride = 1
		if stride != 1 or self.inplanes != planes * block.expansion:
			upsample = nn.Sequential(deconv2x2(self.inplanes, planes * block.expansion, stride),
			       norm_layer(planes * block.expansion),)
		layers = []
		layers.append(block(self.inplanes, planes, stride, upsample, self.groups, self.base_width, previous_dilation, norm_layer))
		self.inplanes = planes * block.expansion
		for _ in range(1, blocks):
			layers.append(block(self.inplanes, planes, groups=self.groups, base_width=self.base_width, dilation=self.dilation, norm_layer=norm_layer))
		return nn.Sequential(*layers)

	def _forward_impl(self, x):
		feature_a = self.layer1(x)  
		feature_b = self.layer2(feature_a)  
		feature_c = self.layer3(feature_b)  
		return [feature_c, feature_b, feature_a]

	def forward(self, x):
		return self._forward_impl(x)


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
	model = ResNet(block, layers, **kwargs)
	
	 
	 
	return model

@MODEL.register_module
def de_resnet18(pretrained = False, progress = True, **kwargs):
	return _resnet('resnet18', DeBasicBlock, [2, 2, 2, 2], pretrained, progress, **kwargs)

@MODEL.register_module
def de_resnet34(pretrained = False, progress = True, **kwargs):
	return _resnet('resnet34', DeBasicBlock, [3, 4, 6, 3], pretrained, progress, **kwargs)

@MODEL.register_module
def de_resnet50(pretrained = False, progress = True, **kwargs):
	return _resnet('resnet50', DeBottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)

@MODEL.register_module
def de_wide_resnet50_2(pretrained = False, progress = True, **kwargs):
	kwargs['width_per_group'] = 64 * 2
	return _resnet('wide_resnet50_2', DeBottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)
