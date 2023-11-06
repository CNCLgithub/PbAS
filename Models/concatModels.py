import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ConcatModels(nn.Module):
	def __init__(self, model1, model2):
		super(ConcatModels, self).__init__()
		self.model1 = model1
		self.model2 = model2
		self.model = list(self.model1.children())
		self.model.extend(list(self.model2.children()))
		self.model = nn.Sequential(*self.model)
	
	def forward(self, x, y=None, z=None):
		if not isinstance(x, torch.Tensor):
			x = self.transformNumpyInputToTensor(x)

		if y is None or z is None:
			y = self.model(x)
			return y
		else:
			# to be used with triplet loss
			if not isinstance(y, torch.Tensor):
				y = self.transformNumpyInputToTensor(y)
			if not isinstance(z, torch.Tensor):
				z = self.transformNumpyInputToTensor(z)

			o1 = self.model(x)
			o2 = self.model(y)
			o3 = self.model(z)

			return o1, o2, o3

	def transformNumpyInputToTensor(self, x):
		x = torch.from_numpy(x)
		if next(self.model.parameters()).is_cuda:
			if len(x.shape) == 3:
				x = x.view(1, x.shape[0], x.shape[1], x.shape[2]).float().cuda()
			else:
				x = x.view(-1, x.shape[1], x.shape[2], x.shape[3]).float().cuda()
		else:
			if len(x.shape) == 3:
				x = x.view(1, x.shape[0], x.shape[1], x.shape[2]).float()
			else:
				x = x.view(-1, x.shape[1], x.shape[2], x.shape[3]).float()
		return x