import matplotlib
matplotlib.use('Agg') # Run in the background
import matplotlib.pyplot as plt
matplotlib.rcParams.update({'figure.autolayout': True})
from copy import deepcopy

class plotClass(object):
	def __init__(self):
		self.figures = {'figObj': [], 'axObj': [], 'ax_kwargs': [], 'plotParams': [], 'fillParams': []}
		self.figIdx = -1
		self.ax_kwargsTemp = {'figTitle': '', 'figTitleFontSize': 8, 'xTicksMajor': '', 'yTicksMajor': '', 'xTickLabelsMajor': None, 'yTickLabelsMajor': None,
							'xLabel': '', 'yLabel': '', 'xLabelFontSize': 7, 'yLabelFontSize': 7, 'xLabelWeight': 'normal', 'yLabelWeight': 'normal',
							'xTicksMinor': None, 'yTicksMinor': None, 'xTickLabelsMinor': None, 'yTickLabelsMinor': None,
							'xTickRot': 0, 'yTickRot': 0, 'xTickFontSize': 7, 'yTickFontSize': 7, 'xTickWeight': 'normal', 'yTickWeight': 'normal',
							'xLimLow': None, 'xLimHigh': None, 'yLimLow': None, 'yLimHigh': None,
							'yGrid': True, 'xGrid': False, 'dpi': 320,  'diagLine': False, 'plotSavePath': ''}

	def createSubplot(self, figSize=8.0):
		figObj, axObj = plt.subplots(figsize=(figSize, figSize))

		self.figures['figObj'].append(figObj)
		self.figures['axObj'].append(axObj)
		self.figures['ax_kwargs'].append(deepcopy(self.ax_kwargsTemp))
		self.figures['plotParams'].append({})
		self.figures['fillParams'].append({})
		self.figIdx += 1

		del figObj
		del axObj

		return self.figIdx

	def plotPoints(self, figIdx, axType, axLegend=False, legend_labels=None, legend_kwargs=None, axAnnotate=False, annotate_kwargs=None, axChanceBar=False, chanceProbability=None):
		# axType could take either of the following values: 'hist', 'bar', 'errorbar', 'scatter'

		ax = self.figures['axObj'][figIdx]
		fillParams = self.figures['fillParams'][figIdx]
		if axType == 'plot':
			X = self.figures['plotParams'][figIdx]['x']
			Y = self.figures['plotParams'][figIdx]['y']
			lineProps = self.figures['plotParams'][figIdx]['lineProps']
			del self.figures['plotParams'][figIdx]['x']
			del self.figures['plotParams'][figIdx]['y']
			for i in range(len(X)):
				x = X[i]
				y = Y[i]
				ax.plot(x, y, **lineProps[i])
		# elif axType == 'errorbar':
		# 	ax.errorbar(**self.figures['plotParams'][figIdx])
		# elif axType == 'scatter':
		# 	ax.scatter(**self.figures['plotParams'][figIdx])
		# elif axType == 'bar':
		# 	ax.bar(**self.figures['plotParams'][figIdx])
		# elif axType == 'hist':
		# 	ax.hist(**self.figures['plotParams'][figIdx])
		# elif axType == 'boxplot':
		# 	ax.boxplot(**self.figures['plotParams'][figIdx])

		if fillParams:
			X = fillParams['x']
			Y = fillParams['y']
			del fillParams['x']
			del fillParams['y']
			for i in range(len(X)):
				fillColorParams = fillParams['fillColorParams'][i]
				x = X[i]
				y = Y[i]
				ax.fill(x, y, **fillColorParams)

		if axLegend:
			self.figures['axObj'][figIdx].legend(legend_labels, **legend_kwargs)

		if axAnnotate:
			self.figures['axObj'][figIdx].annotate(**annotate_kwargs)

		if axChanceBar:
			self.figures['axObj'][figIdx].axhline(chanceProbability, color="gray") # chance line

	def savePlot(self, figIdx):

		'''
		{'figTitle': '', 'figTitleFontSize': 12, 'xTicksMajor': '', 'yTicksMajor': '', 'xTickLabelsMajor': None, 'yTickLabelsMajor': None,
		'xLabel': '', 'yLabel': '', 'xTicksMinor': None, 'yTicksMinor': None, 'xTickLabelsMinor': None, 'yTickLabelsMinor': None,
		'xLabelRot': 0, 'yLabelRot': 0, 'xLabelFontSize': 7, 'yLabelFontSize': 7, 'xLimLow': None, 'xLimHigh': None, 'yLimLow': None, 'yLimHigh': None,
		'yGrid': True, 'xGrid': False, 'dpi': 320,  'diagLine': False, 'plotSavePath': ''}
		'''

		ax = self.figures['axObj'][figIdx]
		fig = self.figures['figObj'][figIdx]
		axParams = self.figures['ax_kwargs'][figIdx]
		ax.set_title(axParams['figTitle'], fontsize=axParams['figTitleFontSize'])


		ax.set_xticks(axParams['xTicksMajor'])
		ax.set_yticks(axParams['yTicksMajor'])
		ax.set_xlabel(axParams['xLabel'], fontsize=axParams['xLabelFontSize'], fontweight=axParams['xLabelWeight'], labelpad=10)
		ax.set_ylabel(axParams['yLabel'], fontsize=axParams['yLabelFontSize'], fontweight=axParams['yLabelWeight'], labelpad=10)

		if axParams['xTickLabelsMajor'] is not None:
			ax.set_xticklabels(axParams['xTickLabelsMajor'], rotation=axParams['xTickRot'], fontsize=axParams['xTickFontSize'], fontweight=axParams['xTickWeight'])
		else:
			plt.setp(ax.get_xmajorticklabels(), visible=False)
			ax.tick_params(axis='x', which='major', length=0)
		if axParams['yTickLabelsMajor'] is not None:
			ax.set_yticklabels(axParams['yTickLabelsMajor'], rotation=axParams['yTickRot'], fontsize=axParams['yTickFontSize'], fontweight=axParams['yTickWeight'])
		else:
			plt.setp(ax.get_ymajorticklabels(), visible=False)
			ax.tick_params(axis='y', which='major', length=0)
		
		if axParams['xTicksMinor'] is not None:
			ax.set_xticks(axParams['xTicksMinor'], minor=True)
		else:
			plt.setp(ax.get_xminorticklabels(), visible=False)
		if axParams['yTicksMinor'] is not None:
			ax.set_yticks(axParams['yTicksMinor'], minor=True)
		else:
			plt.setp(ax.get_yminorticklabels(), visible=False)

		if axParams['xTickLabelsMinor'] is not None:
			ax.set_xticklabels(axParams['xTickLabelsMinor'], minor=True, rotation=axParams['xLabelRot'], fontsize=axParams['xLabelFontSize'])
		if axParams['yTickLabelsMinor'] is not None:
			ax.set_yticklabels(axParams['yTickLabelsMinor'], minor=True, rotation=axParams['yLabelRot'], fontsize=axParams['yLabelFontSize'])


		if axParams['xGrid']:
			ax.xaxis.grid(axParams['xGrid'], which='major', color='black', linestyle='-', linewidth=0.5)
		if axParams['yGrid']:
			ax.yaxis.grid(axParams['yGrid'], which='major', color='black', linestyle='-', linewidth=0.5)
		if axParams['xLimLow'] is not None and axParams['xLimHigh'] is not None:
			ax.set_xlim(axParams['xLimLow'], axParams['xLimHigh'])
		if axParams['yLimLow'] is not None and axParams['yLimHigh'] is not None:
			ax.set_ylim(axParams['yLimLow'], axParams['yLimHigh'])
		if axParams['diagLine']:
			ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c='black', alpha=0.4)
		ax.autoscale(False)
		fig.gca().margins(x=0., y=0.)
		fig.savefig(axParams['plotSavePath'], dpi=axParams['dpi'])
		# plt.clf() # clears the entire current figure 
		plt.close()