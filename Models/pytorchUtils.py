import numpy as np

class PyTorchUtils(object):
	def __init__(self):
		pass

	def paramsExpressivity(self, model, onlyTrainableParams=False):
		# Returns the ratio of sum of absolute value of parameter vectors to the number of parameters
		numParams = self.numParams(model=model, onlyTrainableParams=onlyTrainableParams)
		if numParams == 0:
			raise Exception("==> Error: Your model does not have any [trainable] parameters")
		sumParams = 0
		for name, param in model.named_parameters():
			if not onlyTrainableParams or (onlyTrainableParams and param.requires_grad):
				sumParams += np.abs(param.data.clone().cpu().numpy()).sum()
		return sumParams/numParams

	def numParams(self, model, onlyTrainableParams=False):
		# Returns the number of parameters
		try:
			numParams = sum(p.numel() for p in model.parameters() if (not onlyTrainableParams or (onlyTrainableParams and p.requires_grad)))
		except:
			numParams = 0
		return numParams
