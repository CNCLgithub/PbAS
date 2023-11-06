from common import fileExist, savePickle, loadPickle, mkdir, cp, computeNumShapes, pngToNumpy, maskDrapedShape
from prepareShapeNet import ShapeNet
from blender.blenderClass import Blender
from Models.pytorchModels import Model
from Utils.plotClass import plotClass
from copy import deepcopy
import torch.multiprocessing as mp
from torch.multiprocessing import Process
import PyFleX as flex
import os, math, timeit
import numpy as np

class prepStimuli():
	def __init__(self, opt):
		self.opts = opt
		self.cwd = os.getcwd() + "/"

		if opt.seedStimuli != 0:
			np.random.seed(opt.seedStimuli)

		self.shapeUncertainty = opt.shapeUncertainty
		self.shapeUncertaintyNumExtraShapes = opt.shapeUncertaintyNumExtraShapes
		self.shapeUncertaintyNNDistanceStartPercentage = opt.shapeUncertaintyNNDistanceStartPercentage
		self.shapeUncertaintyEmbeddingModelName = opt.shapeUncertaintyEmbeddingModelName
		self.shapeUncertaintyEmbeddingFeatureLayerName = opt.shapeUncertaintyEmbeddingFeatureLayerName
		self.shapeUncertaintyEmbeddingModelPath = opt.shapeUncertaintyEmbeddingModelPath
		self.embeddingMinDistance = 900 # empirically, we found that 900 is a good number (for AlexNet's FC1 layer) also for the distance in renderings, qualitatively, and not just the embeddings
		self.embeddingSimplifiedObjsPathTrain = self.cwd + opt.datasetStorePath + '/embeddingSimplifiedObjsTrain/'
		self.embeddingMeshPathsPkl = self.cwd + opt.datasetStorePath + '/embeddingMeshPaths.pkl'
		self.embeddingMeshRenderingMainPath = opt.stimuliResultsPath
		self.embeddingDistancePkl = self.cwd + opt.datasetStorePath + "/sortedEmbeddings-TestStimuli.pkl"
		self.embeddingPathPkl = self.cwd + opt.datasetStorePath + '/embeddingDistances-TestStimuli.pkl'
		self.nearestNeighborTrialsPklPath = self.cwd + opt.datasetStorePath + "/trialsNearestNeighbors-TestStimuli.pkl"
		self.renderingDistancePath = self.cwd + opt.datasetStorePath +'/rendering_distance/'
		self.plotsPath = self.cwd + opt.datasetStorePath +'/plots/nearest_neighbors'
		self.visualizationPath = self.cwd + opt.datasetStorePath +'/visualization/nearestNeighbors'
		if self.shapeUncertainty:
			self.model = Model(opt, modelName=self.shapeUncertaintyEmbeddingModelName, chopLayerAt=self.shapeUncertaintyEmbeddingFeatureLayerName, modelPath=self.shapeUncertaintyEmbeddingModelPath)
			self.model.eval()

		self.finetunePretrainedModel = opt.finetunePretrainedModel
		self.finetuneNumStimuliSets = opt.finetuneNumStimuliSets
		self.finetuneStimuliReady = opt.finetuneStimuliReady

		self.simplifyObjs = opt.simplifyObjs
		self.maskClothRendering = opt.maskClothRendering
		self.silhouetteStimuli = opt.silhouetteStimuli

		self.numTestStimuli = opt.numStimuli
		self.numTrainStimuli = opt.numTrainStimuli
		self.numStimuli = self.numTestStimuli
		self.numDistractorShapesPerTrial = opt.numDistractorShapesPerTrial
		self.withinClassPercentage = opt.withinClassPercentage
		self.allCategories = opt.allCategories
		self.category = opt.category
		self.testCategory = opt.testCategory

		self.simplifiedObjsPath = self.cwd + opt.datasetStorePath
		self.stimuliResultsPath = opt.stimuliResultsPath
		self.datasetPklPath = self.cwd + opt.datasetStorePath + '/dataset.pkl'
		self.trialsPklPath = self.cwd + opt.datasetStorePath + "/trials.pkl"
		self.stimuliSet = {}
		self.simplifiedStimuliSet = {}

		self.stimuliFlexConfigPath = opt.stimuliFlexConfigPath
		self.useQuat = opt.useQuat
		self.fixedRotation = opt.fixedRotation

		# self.lampEnergy = opt.lampEnergy
		self.lampPosePath = self.cwd + '/lampPosList.txt'
		self.camPosePath = self.cwd + '/camPosList.txt'
		self.lampEnergy = opt.lampEnergy
		self.resolutions = opt.resolutions

		# Read the pre-stored list of data from disk
		if fileExist(self.datasetPklPath):
			self.datasetTrainTest = loadPickle(self.datasetPklPath)
			print ("==> '" + 'dataset.pkl' + "' is loaded")
			# self.dataset = self.datasetTrainTest[1] # Indices 0 and 1 point to the train and test set respectively
		else:
			print ("==> 'dataset.pkl' file does not exist in '" + self.cwd + self.opts.datasetStorePath + "'. Running the 'prepareDataset' function to store the list of data set first")
			prepShapenet = ShapeNet(opt=self.opts)
			self.datasetTrainTest = prepShapenet.getDatasetList() #self.dataSets[0] and self.dataSets[1] contain the training set and test set respectively
			savePickle(self.datasetPklPath, self.datasetTrainTest)
			# self.dataset = self.datasetTrainTest[1]
		self.switchDataset()
	
	def makeStimuliList(self):
		self.numShapeFromGtCat, self.numShapeFromDistractorCats = computeNumShapes(numStimuli=self.numStimuli, testCategory=self.testCategory, numDistractorShapesPerTrial=self.numDistractorShapesPerTrial)
		withinClassChosenIdx = {} # Stores the indices of shapes chosen for each category to prevent having duplicates
		distractorChosenIdx = {} # Stores the indices of shapes chosen for each category to prevent having duplicates

		withinClassChosenShapes = {}
		distractorChosenShapes = {}

		# Choose shapes from the same category
		self.chosenShapeIdicesForCats = {} # A global dictionary to keep track of indices of shapes for each category that have been used so far
		gtCatsIndices = []
		for i, categoryData in enumerate(self.dataset):
			stopSearchFlag = False
			j = 0
			chosenShapesNo = 0
			if categoryData[1] in self.testCategory:
				self.chosenShapeIdicesForCats[categoryData[1]] = []
				gtCatsIndices.append(i)
				withinClassChosenIdx[categoryData[1]] = [[], []]
				withinClassChosenShapes[categoryData[1]] = [[], []]
				while not stopSearchFlag:
					# randIdx = np.random.choice(len(categoryData[2]), self.numShapeFromGtCat, replace=False).tolist()
					validIndices = self.manuallyCrossoutIndices(category=categoryData[1], indices=[categoryData[2][j][1]]) # We manually ignore some of the shapes that look weird
					if validIndices and categoryData[2][j][1] not in withinClassChosenIdx[categoryData[1]][1]:
						withinClassChosenIdx[categoryData[1]][0].append(i)
						withinClassChosenIdx[categoryData[1]][1].append(categoryData[2][j][1])

						withinClassChosenShapes[categoryData[1]][0].append(i)
						withinClassChosenShapes[categoryData[1]][1].append(categoryData[2][j])

						self.chosenShapeIdicesForCats[categoryData[1]].append(categoryData[2][j][1])
						chosenShapesNo += 1
					if chosenShapesNo == self.numShapeFromGtCat:
						stopSearchFlag = True
					j+=1
		
		# Choose shapes from distractor categories
		for targetCatIndex, categoryData in enumerate(self.dataset):
			if categoryData[1] in self.testCategory:
				foundValidCategoryIdices = False
				while not foundValidCategoryIdices:
					# distractorsCatsIdx = np.random.choice(len(self.dataset), self.numShapeFromDistractorCats).tolist() # Randomly choose distractor categories
					distractorsCatsIdx = np.random.choice(gtCatsIndices, self.numShapeFromDistractorCats).astype(np.int32).tolist() # Randomly choose distractor categories but make sure the categories are the same as the gt categories
					flag = False
					if self.numDistractorShapesPerTrial > 1:
						flag = self.constrainDistractorsCats(indices=distractorsCatsIdx, numNotAllowedConsecutiveIndices=2, minNumCats=len(self.testCategory)-4)
					if targetCatIndex not in distractorsCatsIdx and not flag: # Make sure that none of those categories are the same as the ground-truth category against which we are selecting stimuli in addition to other constraints
						'''
						The following lines of code make sure the chosen cageroties are guaranteed
						to have enough number of shapes to for the desired number of trials
						'''
						validIndices = 0
						tempSumOfCatShapes = {}
						for idx in distractorsCatsIdx:
							tempSumOfCatShapes[idx] = 0
						for idx in distractorsCatsIdx:
							tempSumOfCatShapes[idx] += self.numShapeFromDistractorCats
							# tempSumOfCatShapes[idx] += int(self.numStimuli/len(self.testCategory)/2)
						for idx in distractorsCatsIdx:
							categoryDataDistractor = self.dataset[idx]
							if len(self.chosenShapeIdicesForCats[categoryDataDistractor[1]])+tempSumOfCatShapes[idx] < len(categoryDataDistractor[2]):
								validIndices += 1
						if validIndices == self.numShapeFromDistractorCats:
						# if validIndices == int(self.numStimuli/len(self.testCategory)/2):
							distractorChosenIdx[categoryData[1]] = [distractorsCatsIdx, []]
							distractorChosenShapes[categoryData[1]] = [distractorsCatsIdx, []]
							foundValidCategoryIdices = True

				# Select distractor shape indices, one at a time, for each of the distractor categories
				for idx in distractorsCatsIdx:
					categoryDataDistractor = self.dataset[idx]
					correctIndex = False
					while not correctIndex:
						distractorShapeIndex = int(np.random.choice(len(categoryDataDistractor[2]), 1)) #Sample a shape ID
						validIndices = self.manuallyCrossoutIndices(category=self.allCategories[idx], indices=[categoryDataDistractor[2][distractorShapeIndex][1]])
						if validIndices and categoryDataDistractor[2][distractorShapeIndex][1] not in self.chosenShapeIdicesForCats[categoryDataDistractor[1]]:
							distractorChosenIdx[categoryData[1]][1].append(categoryDataDistractor[2][distractorShapeIndex])

							distractorChosenShapes[categoryData[1]][1].append(categoryDataDistractor[2][distractorShapeIndex])

							self.chosenShapeIdicesForCats[categoryDataDistractor[1]].append(categoryDataDistractor[2][distractorShapeIndex][1])
							correctIndex = True

				self.stimuliSet[categoryData[1]] = []
				self.stimuliSet[categoryData[1]].append(withinClassChosenShapes[categoryData[1]]) # Add the shape indices for the ground-truth category
				self.stimuliSet[categoryData[1]].append(distractorChosenShapes[categoryData[1]]) # Add a list containing indices of distractor categories and the selected shapes from those categories

		if self.trainOrTestData == 'train':
			trainStimuli = {}
			counter = 0
			for k, v in self.stimuliSet.items():
				print('here')
				counter += 1
				if counter <= (self.numTrainStimuli < len(self.testCategory) and self.numTrainStimuli or len(self.testCategory)):
					# Becuase we only want 10 trials in total
					trainStimuli[k] = v
			self.stimuliSet = trainStimuli

		return self.stimuliSet

	def makeTrialsList(self):

		if self.simplifyObjs:
			self.simplifyMainStimuliMeshes()
			stimuliSet = self.simplifiedStimuliSet
		else:
			stimuliSet = self.stimuliSet


		numTestCats = self.trainOrTestData == 'test' and len(self.testCategory) or (self.numTrainStimuli < len(self.testCategory) and self.numTrainStimuli or len(self.testCategory))
		
		self.trials = {}
		usedShapeIDCatPairs = {}

		for catCounter, _ in enumerate(self.allCategories):
			usedShapeIDCatPairs[str(catCounter)] = []

		for category in self.testCategory:
			if category in stimuliSet:
				self.trials[category] = {}
				self.trials[category]['CatIDGt'] = [] # Ground-truth -- And always from one category (e.g. car)
				self.trials[category]['ObjPathGt'] = [] # Ground-truth -- And always from one category (e.g. car)
				self.trials[category]['ShapeIDGt'] = [] # Ground-truth -- And always from one category (e.g. car)
				self.trials[category]['OccludedGt'] = [] 

				self.trials[category]['CatIDDistractor'] = [] # Could be from the same category
				self.trials[category]['ObjPathDistractor'] = [] # Could be from the same category
				self.trials[category]['ShapeIDDistractor'] = [] # Could be from the same category
				self.trials[category]['OccludedDistractor'] = []

				CatIDDistractorList = []
				ObjPathDistractorList = []
				ShapeIDDistractorList = []

				stimuli = stimuliSet[category]

				for occlusionType in range(3):
					# occlusionType = 0 is used for the Unoccluded, Unoccluded, Occluded task
					# occlusionType = 1 is used for the Occluded, Occluded, Unoccluded task
					# occlusionType = 2 is used for the Unoccluded, Unoccluded, Unoccluded task

					CatIDDistractorList = []
					ObjPathDistractorList = []
					ShapeIDDistractorList = []
					for i, stimuliList in enumerate(stimuli):
						distractorShapeIndicator = False
						trialNo = i != 0 and int(self.numStimuli/2/numTestCats) or 0
						for j in range(len(stimuliList[0])):
							if i == 0: # The distractor is chosen from the GT category
								# print ((self.numStimuli/2)/5*(self.numDistractorShapesPerTrial+1))
								# exit()
								if j <= (self.numStimuli/2)/numTestCats*(self.numDistractorShapesPerTrial+1):
									if j > 0 and j % (self.numDistractorShapesPerTrial+1) == 0:
										trialNo += 1
										distractorShapeIndicator = False

										self.trials[category]['CatIDDistractor'].append(CatIDDistractorList)
										self.trials[category]['ObjPathDistractor'].append(ObjPathDistractorList)
										self.trials[category]['ShapeIDDistractor'].append(ShapeIDDistractorList)
										self.trials[category]['OccludedDistractor'].append(occlusionType)

										CatIDDistractorList = []
										ObjPathDistractorList = []
										ShapeIDDistractorList = []
										# print ('')
									elif j > 0:
										distractorShapeIndicator = True
								else:
									distractorShapeIndicator = False
									trialNo += 1
							else: # The distractor is chosen from a different category than GT
								if j > 0 and j % self.numDistractorShapesPerTrial == 0:
									trialNo += 1
									self.trials[category]['CatIDDistractor'].append(CatIDDistractorList)
									self.trials[category]['ObjPathDistractor'].append(ObjPathDistractorList)
									self.trials[category]['ShapeIDDistractor'].append(ShapeIDDistractorList)
									self.trials[category]['OccludedDistractor'].append(occlusionType)

									CatIDDistractorList = []
									ObjPathDistractorList = []
									ShapeIDDistractorList = []
								distractorShapeIndicator = True
							objDirPath = '/' + category + '/trial' + str(trialNo) + '/' + (not distractorShapeIndicator and 'ground-truth_' or 'distractor_') + self.allCategories[int(stimuliList[0][j])] + '-gtIdx' + str(stimuliList[1][j][1])
							
							if not distractorShapeIndicator:
								self.trials[category]['CatIDGt'].append(stimuliList[0][j])
								self.trials[category]['ObjPathGt'].append(stimuliList[1][j][0])
								self.trials[category]['ShapeIDGt'].append(stimuliList[1][j][1])
								self.trials[category]['OccludedGt'].append(occlusionType)
							else:
								CatIDDistractorList.append(stimuliList[0][j])
								ObjPathDistractorList.append(stimuliList[1][j][0])
								ShapeIDDistractorList.append(stimuliList[1][j][1])

							# print (objDirPath)
					self.trials[category]['CatIDDistractor'].append(CatIDDistractorList)
					self.trials[category]['ObjPathDistractor'].append(ObjPathDistractorList)
					self.trials[category]['ShapeIDDistractor'].append(ShapeIDDistractorList)
					self.trials[category]['OccludedDistractor'].append(occlusionType)

				# 	print ('\n\n')
				# 	print (self.trials[category]['CatIDGt'])
				# 	print ('')
				# 	print (self.trials[category]['ShapeIDGt'])
				# 	print ('\n')
				# 	print (self.trials[category]['CatIDDistractor'])
				# 	print ('')
				# 	print (self.trials[category]['ShapeIDDistractor'])
				# 	print ('\n')
				# print ('\n\n')
		savePickle(self.trialsPklPath, self.trials)


	def renderStimuli(self, noSimAndRendering=False, fineTuningStimuli=False, fineTuningStimuliCounter=1, lastStimuliSetSignal=False):
		# Call this after calling makeTrialsList()

		polish = not self.simplifyObjs

		uuoTrialNo = 0
		oouTrialNo = 0
		uuuTrialNo = 0
		uuoOouRandomStateLock = False
		uuuRandomStateLock = False
		if self.numDistractorShapesPerTrial == 1 and not self.finetunePretrainedModel:
			uuoOouRandomState = np.random.get_state()
			np.random.seed(self.opts.seedStimuli+12)
			uuuRandomState = np.random.get_state()
		stimuliRenderingsMainDir = self.stimuliResultsPath + ((not self.maskClothRendering and not self.silhouetteStimuli and not fineTuningStimuli) and '/renderings' or (self.maskClothRendering and not fineTuningStimuli) and '/maskedStimuli' or self.silhouetteStimuli and '/silhouetteStimuli' or (not self.maskClothRendering and fineTuningStimuli) and ('/stimuliSet-' + str(fineTuningStimuliCounter)) or (self.maskClothRendering and fineTuningStimuli) and ('/maskedStimuliSet-' + str(fineTuningStimuliCounter)))

		lamps = np.loadtxt(self.lampPosePath)
		camPos = np.loadtxt(self.camPosePath)
		blenderInternal = Blender(resultsPath=stimuliRenderingsMainDir, onlyRgbRender=True, rotLimitDegree=self.opts.rotLimitDegree)
		blenderInternal.setupScene(lampPosList=lamps, camPosList=camPos, lampEnergy=self.lampEnergy, camIdx=4)

		if not fineTuningStimuli:
			self.trials['PathsUUO'] = [] #Stores the rendering paths (.png) for the unoccluded-unoccluded-occluded set of trials
			self.trials['PathsOOU'] = [] #Stores the rendering paths (.png) for the occluded-occluded-unoccluded set of trials
			self.trials['PathsUUU'] = [] #Stores the rendering paths (.png) for the unoccluded-unoccluded-unoccluded set of trials
			self.trials['PathsUUUCanonical'] = [] #Stores the rendering paths (.png) for the unoccluded-unoccluded-unoccluded set of trials with canonical pose
			self.trials['RotationsUUO'] = [] #Stores the rotation vectors for the unoccluded-unoccluded-occluded set of trials
			self.trials['RotationsOOU'] = [] #Stores the rotation vectors for the occluded-occluded-unoccluded set of trials
			self.trials['RotationsUUU'] = [] #Stores the rotation vectors for the unoccluded-unoccluded-unoccluded set of trials
		else:
			stimuliImgPaths = {'uuo': [], 'oou': [], 'uuu': []}


		startTime = timeit.default_timer()
		for category in self.testCategory:
			if category in self.trials:
			# if category == 'sofa':
				catTrials = self.trials[category]
				catTrials['trialID'] = []

				if not fineTuningStimuli:
					self.trials[category]['PathsUUO'] = [] #Stores the rendering paths (.png) for the unoccluded-unoccluded-occluded set of trials
					self.trials[category]['PathsOOU'] = [] #Stores the rendering paths (.png) for the occluded-occluded-unoccluded set of trials
					self.trials[category]['PathsUUU'] = [] #Stores the rendering paths (.png) for the unoccluded-unoccluded-unoccluded set of trials
					self.trials[category]['PathsUUUCanonical'] = [] #Stores the rendering paths (.png) for the unoccluded-unoccluded-unoccluded set of trials with canonical pose
					self.trials[category]['RotationsUUO'] = [] #Stores the rotation vectors for the unoccluded-unoccluded-occluded set of trials
					self.trials[category]['RotationsOOU'] = [] #Stores the rotation vectors for the occluded-occluded-unoccluded set of trials
					self.trials[category]['RotationsUUU'] = [] #Stores the rotation vectors for the unoccluded-unoccluded-unoccluded set of trials
					self.trials[category]['categoryListGt'] = []
					self.trials[category]['categoryListDis'] = []

				for trialNoWithinCat in range(len(catTrials['CatIDGt'])):
					trialImgPaths = {}
					trialRotVecs = {}
					trialImgPathsCanonical = {}

					gtCatID = catTrials['CatIDGt'][trialNoWithinCat]
					gtObjPath = catTrials['ObjPathGt'][trialNoWithinCat]
					gtShapeID = catTrials['ShapeIDGt'][trialNoWithinCat]
					gtOcclusion = catTrials['OccludedGt'][trialNoWithinCat]

					distractorCatID = catTrials['CatIDDistractor'][trialNoWithinCat]
					distractorObjPath = catTrials['ObjPathDistractor'][trialNoWithinCat]
					distractorShapeID = catTrials['ShapeIDDistractor'][trialNoWithinCat]
					distractorOcclusion = catTrials['OccludedDistractor'][trialNoWithinCat]


					occlusionSum = gtOcclusion + distractorOcclusion
					trialNo = occlusionSum == 0 and uuoTrialNo or occlusionSum == 2 and oouTrialNo or uuuTrialNo
					catTrials['trialID'].append(trialNo)

					for i in range(2): # Do the simulations/renderings twice: once without applying any rotation and once with random rotation to the meshes
						# optionalDirTextGt = category + '/' + (i == 0 and '/canonical' or '/rotation') + '/' + (gtOcclusion == 0 and 'unoccluded-unoccluded-occluded/' or 'occluded-occluded-unoccluded') + '/' + 'trial' + str(trialNo) + '/' + 'ground-truth-' + self.allCategories[gtCatID] + '-gtIdx' + str(gtShapeID)
						# optionalDirTextDistractor = category + '/' + (i == 0 and '/canonical' or '/rotation') + '/' + (gtOcclusion == 0 and 'unoccluded-unoccluded-occluded/' or 'occluded-occluded-unoccluded') + '/' + 'trial' + str(trialNo) + '/' + 'distractor-' + self.allCategories[distractorCatID] + '-gtIdx' + str(distractorShapeID)

						optionalDirTextGt = (occlusionSum == 0 and 'unoccluded-unoccluded-occluded/' or occlusionSum == 2 and 'occluded-occluded-unoccluded' or 'unoccluded-unoccluded-unoccluded/') + '/' + 'trial' + str(trialNo) + '/' + (i == 0 and '/canonical' or '/rotation') + '/' + 'ground-truth-' + self.allCategories[gtCatID] + '-gtIdx' + str(gtShapeID)
						optionalDirTextDistractor = []
						for k in range(len(distractorCatID)):
							optionalDirTextDistractor.append((occlusionSum == 0 and 'unoccluded-unoccluded-occluded/' or occlusionSum == 2 and 'occluded-occluded-unoccluded' or 'unoccluded-unoccluded-unoccluded/') + '/' + 'trial' + str(trialNo) + '/' + (i == 0 and '/canonical' or '/rotation') + '/' + 'distractor-' + self.allCategories[distractorCatID[k]] + '-gtIdx' + str(distractorShapeID[k]))

						copyOptionalDirTextGt = stimuliRenderingsMainDir + '/' + (gtOcclusion == 0 and 'unoccluded-unoccluded-occluded/' or gtOcclusion == 1 and 'occluded-occluded-unoccluded' or 'unoccluded-unoccluded-unoccluded') + '/' + 'trial' + str(trialNo) + '/' + (i == 0 and '/canonical' or '/rotation') + '/'
						copyOptionalDirTextDistractor = stimuliRenderingsMainDir + '/' + (distractorOcclusion == 0 and 'unoccluded-unoccluded-occluded/' or gtOcclusion == 1 and 'occluded-occluded-unoccluded' or 'unoccluded-unoccluded-unoccluded') + '/' + 'trial' + str(trialNo) + '/' + (i == 0 and '/canonical' or '/rotation') + '/'

						copyOptionalDirTextGt2 = stimuliRenderingsMainDir + '/' + (gtOcclusion == 0 and 'unoccluded-unoccluded-occluded/' or 'occluded-occluded-unoccluded') + '/'
						copyOptionalDirTextDistractor2 = stimuliRenderingsMainDir + '/' + (distractorOcclusion == 0 and 'unoccluded-unoccluded-occluded/' or 'occluded-occluded-unoccluded') + '/'

						copyOptionalDirTextGt4 = stimuliRenderingsMainDir + '/' + (gtOcclusion == 0 and 'unoccluded-unoccluded-occluded/' or gtOcclusion == 1 and 'occluded-occluded-unoccluded' or 'unoccluded-unoccluded-unoccluded') + '/'
						copyOptionalDirTextDistractor4 = stimuliRenderingsMainDir + '/' + (distractorOcclusion == 0 and 'unoccluded-unoccluded-occluded/' or distractorOcclusion == 1 and 'occluded-occluded-unoccluded' or 'unoccluded-unoccluded-unoccluded') + '/'

						flexArgs = {}
						flexArgs['configPath'] = self.stimuliFlexConfigPath
						flexArgs['verbose'] = False
						flexArgs['local'] = True
						flexArgs['useQuat'] = self.useQuat
						


						if occlusionSum == 0:
							# UUO
							if not uuoOouRandomStateLock and self.numDistractorShapesPerTrial == 1 and not self.finetunePretrainedModel:
								if uuuRandomStateLock:
									uuuRandomState = np.random.get_state()
									uuuRandomStateLock = False
								np.random.set_state(uuoOouRandomState)
								uuoOouRandomStateLock = True

							# Render both ground-truth and distractor shape
							rotVecGt = i == 0 and [0.0, 0.0, 0.0] or np.random.uniform(-self.opts.rotLimitDegree, self.opts.rotLimitDegree, 3)
							rotVecGt = np.array(rotVecGt).astype(dtype=np.float32)
							rotVecDis = []
							for k in range(len(distractorCatID)):
								if not self.fixedRotation:
									rotVecDis.append(i == 0 and [0.0, 0.0, 0.0] or np.random.uniform(-self.opts.rotLimitDegree, self.opts.rotLimitDegree, 3))
									rotVecDis[k] = np.array(rotVecDis[k]).astype(dtype=np.float32)
								else:
									rotVecDis.append(rotVecGt)

							gtNoClothRenderingPath = copyOptionalDirTextGt2 + 'trial_' + str(trialNo) + '_gtNoCloth.png'
							gtClothRenderingPath = copyOptionalDirTextGt2 + 'trial_' + str(trialNo) + '_gtCloth.png'
							distNoClothRenderingPath = []
							for k in range(len(distractorCatID)):
								distNoClothRenderingPath.append(copyOptionalDirTextDistractor2 + 'trial_' + str(trialNo) + '_disNoCloth' + str(k) + '.png')
							
							trialImgPaths['GtNoCloth'] = gtNoClothRenderingPath
							trialRotVecs['GtNoCloth'] = rotVecGt

							trialImgPaths['DisNoCloth'] = []
							trialRotVecs['DisNoCloth'] = []
							for k in range(len(distractorCatID)):
								trialImgPaths['DisNoCloth'].append(distNoClothRenderingPath[k])
								trialRotVecs['DisNoCloth'].append(rotVecDis[k])

							if i == 0:
								trialImgPaths['GtCloth'] = gtClothRenderingPath
								trialRotVecs['GtCloth']  = rotVecGt
								GtNoClothCanonical = gtNoClothRenderingPath
								GtCloth2 = copyOptionalDirTextGt + 'trial_' + str(trialNo) + '_gtCloth_' + self.allCategories[gtCatID] + '.png'

							elif self.numDistractorShapesPerTrial == 1 and not self.finetunePretrainedModel:
								np.random.randint(100000000) # These are necessary to compensate for a mistake that I made before so that we can reproduce the results for experiments with 2 shapes per trial
								np.random.randint(100000000)

							if not noSimAndRendering:
								# Do the rendering for the unoccluded GT shape
								blenderRenderJob1 = Process(target=blenderInternal.render, kwargs={'trainOrTest': 'test', 'objPath': gtObjPath, 'category': category, 'gtIdx': gtShapeID, 'resolutions': self.resolutions, 'smallScaleRendering': True, 'optionalText': optionalDirTextGt, 'copyRenderFile': True, 'newFilePath': copyOptionalDirTextGt + 'trial_' + str(trialNo) + '_gtNoCloth_' + self.allCategories[gtCatID] + '.png', 'newFilePath2': gtNoClothRenderingPath, 'numRotation': i != 0 and self.opts.numStimuliRotation or 0, 'externalRotVec': rotVecGt, 'simultaneousRotation': 0, 'removeMatAfterSimRotSteps': 10000, 'removeMats': True, 'numNewMatColor': 0, 'rgb': True, 'polish': polish, 'numpySeed':  None})
								blenderRenderJob1.start()

								# Do the rendering for the unoccluded distractor shape(s)
								blenderRenderJob2 = []
								for k in range(len(distractorCatID)):
									blenderRenderJob2.append(Process(target=blenderInternal.render, kwargs={'trainOrTest': 'test', 'objPath': distractorObjPath[k], 'category': category, 'gtIdx': distractorShapeID[k], 'resolutions': self.resolutions, 'smallScaleRendering': True, 'optionalText': optionalDirTextDistractor[k], 'copyRenderFile': True, 'newFilePath': copyOptionalDirTextDistractor + 'trial_' + str(trialNo) + '_disNoCloth_' + self.allCategories[distractorCatID[k]] + str(k) + '.png', 'newFilePath2': distNoClothRenderingPath[k], 'numRotation': i != 0 and self.opts.numStimuliRotation or 0, 'externalRotVec': rotVecDis[k], 'simultaneousRotation': 0, 'removeMatAfterSimRotSteps': 10000, 'removeMats': True, 'numNewMatColor': 0, 'rgb': True, 'polish': polish, 'numpySeed': None}))
									blenderRenderJob2[-1].start()

								if i == 0:
									# Simplify the mesh to speed up simulations
									simplifiedMeshDirGt = stimuliRenderingsMainDir + '/' + optionalDirTextGt + '/' + 'simplifiedMesh'
									mkdir(simplifiedMeshDirGt)
									blenderSimplifyJobGT = Process(target=blenderInternal.loadObj, kwargs={'objPath': gtObjPath, 'layerIdx': 1, 'polish': polish, 'harshPolish': True, 'save': True, 'objSavePath': simplifiedMeshDirGt})
									blenderSimplifyJobGT.start()
									
									# Simplify the distractor meshes to speed up simulations
									blenderSimplifyJobDistractor = []
									simplifiedMeshDirDistractor = []
									for k in range(len(distractorCatID)):
										simplifiedMeshDirDistractor.append(stimuliRenderingsMainDir + '/' + optionalDirTextDistractor[k] + '/' + 'simplifiedMesh')
										mkdir(simplifiedMeshDirDistractor[-1])
										blenderSimplifyJobDistractor.append(Process(target=blenderInternal.loadObj, kwargs={'objPath': distractorObjPath[k], 'layerIdx': 1, 'polish': polish, 'harshPolish': True, 'save': True, 'objSavePath': simplifiedMeshDirDistractor[k]}))
										blenderSimplifyJobDistractor[-1].start()

									blenderSimplifyJobGT.join()
									# Get occluded version of the gt shape
									simResDirGt = stimuliRenderingsMainDir + '/' + optionalDirTextGt + '/' + 'simulationResObjs/'
									mkdir(simResDirGt)
									meshPath = not self.simplifyObjs and simplifiedMeshDirGt + '/mesh.obj' or gtObjPath
									# flexArgs['rot'] = eulerToQuat(rotVecGt)
									flexArgs['rot'] = np.radians(rotVecGt)
									flexArgs['outObjBaseName'] = simResDirGt
									flexArgs['objPath'] = meshPath
									# Run simulation
									# flex.simulate(meshPath, self.cwd + flexConfig, verbose=True, **flexArgs)
									flexJobGt = Process(target=flex.simulate, kwargs={**flexArgs})
									flexJobGt.start()

									# Get occluded version of the distractor shape
									for k in range(len(distractorCatID)):
										blenderSimplifyJobDistractor[k].join()
									# flexArgs['rot'] = eulerToQuat(rotVecDis)
									flexJobDistractor = []
									simResDirDistractor = []
									for k in range(len(distractorCatID)):
										simResDirDistractor.append(stimuliRenderingsMainDir + '/' + optionalDirTextDistractor[k] + '/' + 'simulationResObjs')
										mkdir(simResDirDistractor[k])
										meshPath = not self.simplifyObjs and simplifiedMeshDirDistractor[k] + '/mesh.obj' or distractorObjPath[k]
										flexArgs['rot'] = np.radians(rotVecDis[k])
										flexArgs['outObjBaseName'] = simResDirDistractor[k]
										flexArgs['objPath'] = meshPath
										# Run simulation
										# flex.simulate(meshPath, self.cwd + flexConfig, verbose=True, **flexArgs)
										flexJobDistractor.append(Process(target=flex.simulate, kwargs={**flexArgs}))
										flexJobDistractor[-1].start()


								blenderRenderJob1.join()
								for k in range(len(distractorCatID)):
									blenderRenderJob2[k].join()

								if i == 0:
									# Render the stored cloth for the gt shape
									flexJobGt.join()
									blenderRenderJob3 = Process(target=blenderInternal.render, kwargs={'trainOrTest': 'test', 'objPath': simResDirGt + '/' + '_cloth.obj', 'category': category, 'gtIdx': gtShapeID, 'resolutions': self.resolutions, 'smallScaleRendering': True, 'optionalText': optionalDirTextGt, 'copyRenderFile': True, 'newFilePath': copyOptionalDirTextGt + 'trial_' + str(trialNo) + '_gtCloth_' + self.allCategories[gtCatID] + '.png', 'newFilePath2': gtClothRenderingPath, 'numRotation': 0, 'simultaneousRotation': 0, 'removeMatAfterSimRotSteps': 10000, 'removeMats': False, 'numNewMatColor': 0, 'rgb': True, 'cloth': True, 'numpySeed': None})
									blenderRenderJob3.start()

									# Render the stored cloth for the distractor shape
									blenderRenderJob4 = []
									for k in range(len(distractorCatID)):
										flexJobDistractor[k].join()
										blenderRenderJob4.append(Process(target=blenderInternal.render, kwargs={'trainOrTest': 'test', 'objPath': simResDirDistractor[k] + '/' + '_cloth.obj', 'category': category, 'gtIdx': distractorShapeID[k], 'resolutions': self.resolutions, 'smallScaleRendering': True, 'optionalText': optionalDirTextDistractor[k], 'numRotation': 0, 'simultaneousRotation': 0, 'removeMatAfterSimRotSteps': 10000, 'removeMats': False, 'numNewMatColor': 0, 'rgb': True, 'cloth': True, 'numpySeed': None}))
										# blenderRenderJob4.append(Process(target=blenderInternal.render, kwargs={'trainOrTest': 'test', 'objPath': simResDirDistractor[k] + '/' + '_cloth.obj', 'category': category, 'gtIdx': distractorShapeID[k], 'resolutions': self.resolutions, 'smallScaleRendering': True, 'optionalText': optionalDirTextDistractor[k], 'copyRenderFile': True, 'newFilePath': copyOptionalDirTextDistractor + 'trial_' + str(trialNo) + '_gtCloth_' + self.allCategories[distractorCatID] + '.png', 'newFilePath2': copyOptionalDirTextDistractor2 + 'trial_' + str(trialNo) + '_disCloth.png', 'numRotation': 0, 'simultaneousRotation': 0, 'removeMatAfterSimRotSteps': 10000, 'removeMats': False, 'numNewMatColor': 0, 'rgb': True, 'cloth': True, 'numpySeed': None}))
										blenderRenderJob4[-1].start()

									blenderRenderJob3.join()
									for k in range(len(distractorCatID)):
										blenderRenderJob4[k].join()
									
									if self.maskClothRendering or self.silhouetteStimuli:
										maskDrapedShape(drapedShapeRenderingPath=trialImgPaths['GtCloth'], maskRenderingPath=GtNoClothCanonical, resolution=self.resolutions[0], maskClothRendering=self.maskClothRendering, silhouetteStimuli=self.silhouetteStimuli)
										maskDrapedShape(drapedShapeRenderingPath=GtCloth2, maskRenderingPath=GtNoClothCanonical, resolution=self.resolutions[0], maskClothRendering=self.maskClothRendering, silhouetteStimuli=self.silhouetteStimuli)
					


						
						elif occlusionSum == 2:
							# OOU
							rotVecGt = i == 0 and [0.0, 0.0, 0.0] or np.random.uniform(-self.opts.rotLimitDegree, self.opts.rotLimitDegree, 3)
							rotVecGt = np.array(rotVecGt).astype(dtype=np.float32)
							rotVecDis = []
							for k in range(len(distractorCatID)):
								if not self.fixedRotation:
									rotVecDis.append(i == 0 and [0.0, 0.0, 0.0] or np.random.uniform(-self.opts.rotLimitDegree, self.opts.rotLimitDegree, 3))
									rotVecDis[k] = np.array(rotVecDis[k]).astype(dtype=np.float32)
								else:
									rotVecDis.append(rotVecGt)

							gtClothRenderingPath = copyOptionalDirTextGt2 + 'trial_' + str(trialNo) + '_gtCloth.png'
							gtNoClothRotatedShapeRenderingPath = copyOptionalDirTextGt + 'trial_' + str(trialNo) + '_gtNoCloth_' + self.allCategories[gtCatID] + '.png'
							gtNoClothRenderingPath = copyOptionalDirTextGt2 + 'trial_' + str(trialNo) + '_gtNoCloth.png'
							disClothRenderingPath = []
							disNoClothRotatedShapeRenderingPath = []
							for k in range(len(distractorCatID)):
								disClothRenderingPath.append(copyOptionalDirTextDistractor2 + 'trial_' + str(trialNo) + '_disCloth' + str(k) + '.png')
								disNoClothRotatedShapeRenderingPath.append(copyOptionalDirTextDistractor + 'trial_' + str(trialNo) + '_disNoCloth_' + self.allCategories[distractorCatID[k]] + str(k) + '.png')

							trialImgPaths['GtCloth'] = gtClothRenderingPath
							trialRotVecs['GtCloth'] = rotVecGt
							
							trialImgPaths['DisCloth'] = []
							trialRotVecs['DisCloth'] = []
							for k in range(len(distractorCatID)):
								trialImgPaths['DisCloth'].append(disClothRenderingPath[k])
								trialRotVecs['DisCloth'].append(rotVecDis[k])

							if i == 0:
								trialImgPaths['GtNoCloth'] = gtNoClothRenderingPath
								trialRotVecs['GtNoCloth'] = rotVecGt

							elif self.numDistractorShapesPerTrial == 1 and not self.finetunePretrainedModel:
								np.random.randint(100000000)

							if not noSimAndRendering:
								# Simplify the mesh to speed up simulations
								simplifiedMeshDirGt = stimuliRenderingsMainDir + '/' + optionalDirTextGt + '/' + 'simplifiedMesh'
								mkdir(simplifiedMeshDirGt)
								blenderSimplifyJobGT = Process(target=blenderInternal.loadObj, kwargs={'objPath': gtObjPath, 'layerIdx': 1, 'polish': polish, 'harshPolish': True, 'save': True, 'objSavePath': simplifiedMeshDirGt})
								blenderSimplifyJobGT.start()

								# Simplify the distractor meshes to speed up simulations
								blenderSimplifyJobDistractor = []
								simplifiedMeshDirDistractor = []
								for k in range(len(distractorCatID)):							
									simplifiedMeshDirDistractor.append(stimuliRenderingsMainDir + '/' + optionalDirTextDistractor[k] + '/' + 'simplifiedMesh')
									mkdir(simplifiedMeshDirDistractor[k])
									blenderSimplifyJobDistractor.append(Process(target=blenderInternal.loadObj, kwargs={'objPath': distractorObjPath[k], 'layerIdx': 1, 'polish': polish, 'harshPolish': True, 'save': True, 'objSavePath': simplifiedMeshDirDistractor[k]}))
									blenderSimplifyJobDistractor[k].start()

								blenderSimplifyJobGT.join()
								# Get occluded version of the gt shape
								simResDirGt = stimuliRenderingsMainDir + '/' + optionalDirTextGt + '/' + 'simulationResObjs/'
								mkdir(simResDirGt)
								meshPath = not self.simplifyObjs and simplifiedMeshDirGt + '/mesh.obj' or gtObjPath
								# flexArgs['rot'] = eulerToQuat(rotVecGt)
								flexArgs['rot'] = np.radians(rotVecGt) 
								flexArgs['outObjBaseName'] = simResDirGt
								flexArgs['objPath'] = meshPath
								flexJobGt = Process(target=flex.simulate, kwargs={**flexArgs})
								flexJobGt.start()

									
								for k in range(len(distractorCatID)):
									blenderSimplifyJobDistractor[k].join()
								# Get occluded version of the distractor shape
								simResDirDistractor = []
								flexJobDistractor = []
								for k in range(len(distractorCatID)):
									# Run simulation
									meshPath = not self.simplifyObjs and simplifiedMeshDirDistractor[k] + '/mesh.obj' or distractorObjPath[k]
									simResDirDistractor.append(stimuliRenderingsMainDir + '/' + optionalDirTextDistractor[k] + '/' + 'simulationResObjs/')
									mkdir(simResDirDistractor[k])
									# print ('')
									# print (rotVecDis[k])
									# print ('')
									# flexArgs['rot'] = eulerToQuat(rotVecDis[k]) 
									flexArgs['rot'] = np.radians(rotVecDis[k]) 
									flexArgs['outObjBaseName'] = simResDirDistractor[k]
									flexArgs['objPath'] = meshPath
									flexJobDistractor.append(Process(target=flex.simulate, kwargs={**flexArgs}))
									flexJobDistractor[k].start()


							if not noSimAndRendering:
								# Render the gt shape
								if i == 0:
									blenderRenderJob1 = Process(target=blenderInternal.render, kwargs={'trainOrTest': 'test', 'objPath': gtObjPath, 'category': category, 'gtIdx': gtShapeID, 'resolutions': self.resolutions, 'smallScaleRendering': True, 'optionalText': optionalDirTextGt, 'copyRenderFile': True, 'newFilePath': copyOptionalDirTextGt + 'trial_' + str(trialNo) + '_gtNoCloth_' + self.allCategories[gtCatID] + '.png', 'newFilePath2': gtNoClothRenderingPath, 'numRotation': i != 0 and self.opts.numStimuliRotation or 0, 'simultaneousRotation': 0, 'removeMatAfterSimRotSteps': 10000, 'removeMats': True, 'numNewMatColor': 0, 'rgb': True, 'polish': polish, 'numpySeed':  None})
								else:
									blenderRenderJob1 = Process(target=blenderInternal.render, kwargs={'trainOrTest': 'test', 'objPath': gtObjPath, 'category': category, 'gtIdx': gtShapeID, 'resolutions': self.resolutions, 'smallScaleRendering': True, 'optionalText': optionalDirTextGt, 'copyRenderFile': True, 'newFilePath': gtNoClothRotatedShapeRenderingPath, 'numRotation': i != 0 and self.opts.numStimuliRotation or 0, 'externalRotVec': rotVecGt, 'simultaneousRotation': 0, 'removeMatAfterSimRotSteps': 10000, 'removeMats': True, 'numNewMatColor': 0, 'rgb': True, 'polish': polish, 'numpySeed':  None})
								blenderRenderJob1.start()
								
								# Render the distractor shape
								blenderRenderJob2 = []
								for k in range(len(distractorCatID)):
									if i == 0:
										blenderRenderJob2.append(Process(target=blenderInternal.render, kwargs={'trainOrTest': 'test', 'objPath': distractorObjPath[k], 'category': category, 'gtIdx': distractorShapeID[k], 'resolutions': self.resolutions, 'smallScaleRendering': True, 'optionalText': optionalDirTextDistractor[k], 'copyRenderFile': True, 'newFilePath': copyOptionalDirTextDistractor + 'trial_' + str(trialNo) + '_disNoCloth_' + self.allCategories[distractorCatID[k]] + str(k) + '.png', 'numRotation': i != 0 and self.opts.numStimuliRotation or 0, 'simultaneousRotation': 0, 'removeMatAfterSimRotSteps': 10000, 'removeMats': True, 'numNewMatColor': 0, 'rgb': True, 'polish': polish, 'numpySeed': None}))
									else:
										blenderRenderJob2.append(Process(target=blenderInternal.render, kwargs={'trainOrTest': 'test', 'objPath': distractorObjPath[k], 'category': category, 'gtIdx': distractorShapeID[k], 'resolutions': self.resolutions, 'smallScaleRendering': True, 'optionalText': optionalDirTextDistractor[k], 'copyRenderFile': True, 'newFilePath': disNoClothRotatedShapeRenderingPath[k], 'externalRotVec': rotVecDis[k], 'numRotation': i != 0 and self.opts.numStimuliRotation or 0, 'simultaneousRotation': 0, 'removeMatAfterSimRotSteps': 10000, 'removeMats': True, 'numNewMatColor': 0, 'rgb': True, 'polish': polish, 'numpySeed': None}))
									blenderRenderJob2[-1].start()

								# Render the stored cloth mesh for the gt shape
								flexJobGt.join()
								blenderRenderJob3 = Process(target=blenderInternal.render, kwargs={'trainOrTest': 'test', 'objPath': simResDirGt + '/' + '_cloth.obj', 'category': category, 'gtIdx': gtShapeID, 'resolutions': self.resolutions, 'smallScaleRendering': True, 'optionalText': optionalDirTextGt, 'copyRenderFile': True, 'newFilePath': copyOptionalDirTextGt + 'trial_' + str(trialNo) + '_gtCloth_' + self.allCategories[gtCatID] + '.png', 'newFilePath2': gtClothRenderingPath, 'numRotation': 0, 'simultaneousRotation': 0, 'removeMatAfterSimRotSteps': 10000, 'removeMats': False, 'numNewMatColor': 0, 'rgb': True, 'cloth': True, 'externalRotVec': rotVecGt, 'numpySeed': None})
								blenderRenderJob3.start()

								# Render the stored cloth mesh for the distractor shape
								for k in range(len(distractorCatID)):
									flexJobDistractor[k].join()
								blenderRenderJob4 = []
								for k in range(len(distractorCatID)):
									blenderRenderJob4.append(Process(target=blenderInternal.render, kwargs={'trainOrTest': 'test', 'objPath': simResDirDistractor[k] + '/' + '_cloth.obj', 'category': category, 'gtIdx': distractorShapeID[k], 'resolutions': self.resolutions, 'smallScaleRendering': True, 'optionalText': optionalDirTextDistractor[k], 'copyRenderFile': True, 'newFilePath': copyOptionalDirTextDistractor + 'trial_' + str(trialNo) + '_disCloth_' + self.allCategories[distractorCatID[k]] + str(k) + '.png', 'newFilePath2': disClothRenderingPath[k], 'numRotation': 0, 'simultaneousRotation': 0, 'removeMatAfterSimRotSteps': 10000, 'removeMats': False, 'numNewMatColor': 0, 'rgb': True, 'cloth': True, 'externalRotVec': rotVecDis[k], 'numpySeed': None}))
									blenderRenderJob4[-1].start()


								blenderRenderJob1.join()

								for k in range(len(distractorCatID)):
									blenderRenderJob2[k].join()
									blenderRenderJob4[k].join()
								blenderRenderJob3.join()

								if i == 1:
									if self.maskClothRendering or self.silhouetteStimuli:
										maskDrapedShape(drapedShapeRenderingPath=trialImgPaths['GtCloth'], maskRenderingPath=gtNoClothRotatedShapeRenderingPath, resolution=self.resolutions[0], maskClothRendering=self.maskClothRendering, silhouetteStimuli=self.silhouetteStimuli)
										for k in range(len(distractorCatID)):
											maskDrapedShape(drapedShapeRenderingPath=trialImgPaths['DisCloth'][k], maskRenderingPath=disNoClothRotatedShapeRenderingPath[k], resolution=self.resolutions[0], maskClothRendering=self.maskClothRendering, silhouetteStimuli=self.silhouetteStimuli)




						elif occlusionSum == 4:
							if uuoOouRandomStateLock and self.numDistractorShapesPerTrial == 1 and not self.finetunePretrainedModel:
								uuoOouRandomState = np.random.get_state()
								np.random.set_state(uuuRandomState)
								uuoOouRandomStateLock = False
								uuuRandomStateLock = True
							# UUU
							# Render both ground-truth and distractor shape
							rotVecGt = i == 0 and [0.0, 0.0, 0.0] or np.random.uniform(-self.opts.rotLimitDegree, self.opts.rotLimitDegree, 3)
							rotVecGt = np.array(rotVecGt).astype(dtype=np.float32)
							rotVecDis = []
							for k in range(len(distractorCatID)):
								if not self.fixedRotation:
									rotVecDis.append(i == 0 and [0.0, 0.0, 0.0] or np.random.uniform(-self.opts.rotLimitDegree, self.opts.rotLimitDegree, 3))
									rotVecDis[k] = np.array(rotVecDis[k]).astype(dtype=np.float32)
								else:
									rotVecDis.append(rotVecGt)

							gtNoClothRenderingPath = copyOptionalDirTextGt4 + 'trial_' + str(trialNo) + '_gtNoCloth.png'
							gtTargetNoClothRenderingPath = copyOptionalDirTextGt4 + 'trial_' + str(trialNo) + '_gtTargetNoCloth.png'
							distNoClothRenderingPath = []
							for k in range(len(distractorCatID)):
								distNoClothRenderingPath.append(copyOptionalDirTextDistractor4 + 'trial_' + str(trialNo) + '_disNoCloth' + str(k) + '.png')
							
							trialImgPaths['GtNoCloth'] = gtNoClothRenderingPath
							trialRotVecs['GtNoCloth'] = rotVecGt
							# trialCategoryGt.append(gtCatID)
							trialImgPaths['DisNoCloth'] = []
							trialRotVecs['DisNoCloth'] = []
							for k in range(len(distractorCatID)):
								trialImgPaths['DisNoCloth'].append(distNoClothRenderingPath[k])
								trialRotVecs['DisNoCloth'].append(rotVecDis[k])
							
							if i == 0:
								trialImgPaths['GtTargetNoCloth'] = gtTargetNoClothRenderingPath
								trialRotVecs['GtTargetNoCloth']  = rotVecGt

								trialImgPathsCanonical['GtNoCloth'] = stimuliRenderingsMainDir + '/' + optionalDirTextGt + '/' + str(self.resolutions[0]) + '/rgb/X0.000_Y0.000_Z0.000/cam0.png'
								trialImgPathsCanonical['DisNoCloth'] = []
								for k in range(len(distractorCatID)):
									trialImgPathsCanonical['DisNoCloth'].append(stimuliRenderingsMainDir + '/' + optionalDirTextDistractor[k] + '/' + str(self.resolutions[0]) + '/rgb/X0.000_Y0.000_Z0.000/cam0.png')

							elif self.numDistractorShapesPerTrial == 1 and not self.finetunePretrainedModel:
								np.random.randint(100000000)
								np.random.randint(100000000)

							if not noSimAndRendering:
								blenderRenderJob1 = Process(target=blenderInternal.render, kwargs={'trainOrTest': 'test', 'objPath': gtObjPath, 'category': category, 'gtIdx': gtShapeID, 'resolutions': self.resolutions, 'smallScaleRendering': True, 'optionalText': optionalDirTextGt, 'copyRenderFile': True, 'newFilePath': copyOptionalDirTextGt + 'trial_' + str(trialNo) + '_gtNoCloth_' + self.allCategories[gtCatID] + '.png', 'newFilePath2': gtNoClothRenderingPath, 'numRotation': i != 0 and self.opts.numStimuliRotation or 0, 'externalRotVec': rotVecGt, 'simultaneousRotation': 0, 'removeMatAfterSimRotSteps': 10000, 'removeMats': True, 'numNewMatColor': 0, 'rgb': True, 'polish': polish, 'numpySeed': None})
								blenderRenderJob1.start()
								blenderRenderJob2 = []
								for k in range(len(distractorCatID)):
									blenderRenderJob2.append(Process(target=blenderInternal.render, kwargs={'trainOrTest': 'test', 'objPath': distractorObjPath[k], 'category': category, 'gtIdx': distractorShapeID[k], 'resolutions': self.resolutions, 'smallScaleRendering': True, 'optionalText': optionalDirTextDistractor[k], 'copyRenderFile': True, 'newFilePath': copyOptionalDirTextDistractor + 'trial_' + str(trialNo) + '_disNoCloth_' + self.allCategories[distractorCatID[k]] + str(k) + '.png', 'newFilePath2': distNoClothRenderingPath[k], 'numRotation': i != 0 and self.opts.numStimuliRotation or 0, 'externalRotVec': rotVecDis[k], 'simultaneousRotation': 0, 'removeMatAfterSimRotSteps': 10000, 'removeMats': True, 'numNewMatColor': 0, 'rgb': True, 'polish': polish, 'numpySeed': None}))
									blenderRenderJob2[k].start()

								if i == 0:
									# Get unoccluded version of the gt shape with canonical pose
									blenderRenderJob3 = Process(target=blenderInternal.render, kwargs={'trainOrTest': 'test', 'objPath': gtObjPath, 'category': category, 'gtIdx': gtShapeID, 'resolutions': self.resolutions, 'smallScaleRendering': True, 'optionalText': optionalDirTextGt, 'copyRenderFile': True, 'newFilePath': copyOptionalDirTextGt + 'trial_' + str(trialNo) + '_gtTargetNoCloth_' + self.allCategories[gtCatID] + '.png', 'newFilePath2': gtTargetNoClothRenderingPath, 'numRotation': i != 0 and self.opts.numStimuliRotation or 0, 'simultaneousRotation': 0, 'removeMatAfterSimRotSteps': 10000, 'removeMats': True, 'numNewMatColor': 0, 'rgb': True, 'polish': polish, 'numpySeed': None})
									blenderRenderJob3.start()
									blenderRenderJob3.join()
								
								blenderRenderJob1.join()
								for k in range(len(distractorCatID)):
									blenderRenderJob2[k].join()

					if occlusionSum == 0:
						self.trials['PathsUUO'].append(trialImgPaths)
						self.trials['RotationsUUO'].append(trialRotVecs)
						self.trials[category]['PathsUUO'].append(trialImgPaths)
						self.trials[category]['RotationsUUO'].append(trialRotVecs)
						if fineTuningStimuli:
							stimuliImgPaths['uuo'].append(trialImgPaths)
						# 	self.trials['stimuliSetNoUUO'].append(fineTuningStimuliCounter)
						# 	self.trials[category]['stimuliSetNoUUO'].append(fineTuningStimuliCounter)
					elif occlusionSum == 2:
						self.trials['PathsOOU'].append(trialImgPaths)
						self.trials['RotationsOOU'].append(trialRotVecs)
						self.trials[category]['PathsOOU'].append(trialImgPaths)
						self.trials[category]['RotationsOOU'].append(trialRotVecs)
						if fineTuningStimuli:
							stimuliImgPaths['oou'].append(trialImgPaths)
						# 	self.trials['stimuliSetNoOOU'].append(fineTuningStimuliCounter)
						# 	self.trials[category]['stimuliSetNoOOU'].append(fineTuningStimuliCounter)
					elif occlusionSum == 4:
						self.trials['PathsUUU'].append(trialImgPaths)
						self.trials['PathsUUUCanonical'].append(trialImgPathsCanonical)
						self.trials['RotationsUUU'].append(trialRotVecs)
						self.trials[category]['PathsUUU'].append(trialImgPaths)
						self.trials[category]['PathsUUUCanonical'].append(trialImgPathsCanonical)
						self.trials[category]['RotationsUUU'].append(trialRotVecs)

						self.trials[category]['categoryListGt'].append(gtCatID)
						self.trials[category]['categoryListDis'].append(distractorCatID)

						if fineTuningStimuli:
							stimuliImgPaths['uuu'].append(trialImgPaths)
							# self.trials['stimuliSetNoUUU'].append(fineTuningStimuliCounter)
						# 	self.trials[category]['stimuliSetNoUUU'].append(fineTuningStimuliCounter)

					# print (trialNo, category)
					if occlusionSum == 0:
						uuoTrialNo += 1
					elif occlusionSum == 2:
						oouTrialNo += 1
					elif occlusionSum == 4:
						uuuTrialNo += 1
				self.trials[category] = catTrials
		if fineTuningStimuli:
			self.trials['finetuneSeparateStimuli'].append(stimuliImgPaths)
		if not fineTuningStimuli or (fineTuningStimuli and lastStimuliSetSignal):
			savePickle(self.trialsPklPath, self.trials)

		if not noSimAndRendering:
			print ("==> It took {0:.2f} minutes to do the renderings for {1:d} trials and simulations\n".format((timeit.default_timer() - startTime)/60, self.numStimuli))

	
	def getShapeEmbeddings(self):
		numShapes = self.getNumberOfShapes()
		if not fileExist(self.embeddingMeshPathsPkl):
			print ('==> There are {0:d} objects in the data set and it will take a while to load, simplify and render them. Please stay tuned'.format(numShapes))
			self.simplifyShapesForShapeEmbedding()
		else:
			self.embeddingMeshPaths = loadPickle(self.embeddingMeshPathsPkl)
			print ('==> Loaded the paths for {0:d} objects in the data set to be used to obtain embeddings. The embeddings will be used to select nearest neighbor shapes for the stimuli'.format(numShapes))
			

		if not fileExist(self.embeddingPathPkl):
			lamps = np.loadtxt(self.lampPosePath)
			camPos = np.loadtxt(self.camPosePath)
			blenderInternal = Blender(onlyRgbRender=True, rotLimitDegree=self.opts.rotLimitDegree)
			blenderInternal.setupScene(lampPosList=lamps, camPosList=camPos, lampEnergy=self.lampEnergy, camIdx=4)
			
			rotVec = [0.0, 0.0, 0.0]
			polish = not self.simplifyObjs
			stimuliRenderingsMainDir = self.embeddingMeshRenderingMainPath + '/datasetShapeRenderings/'

			# Obtain embeddings of the dataset
			missingRenderingPaths = {}
			self.datasetShapeEmbeddingsAndRenderings = {}
			missingRenderCounter = 0

			self.datasetShapeEmbeddingsAndRenderings['stimuliEmbeddings'] = {}
			self.datasetShapeEmbeddingsAndRenderings['stimuliEmbeddings']['gt'] = np.zeros((self.numStimuli, 4096))
			self.datasetShapeEmbeddingsAndRenderings['stimuliEmbeddings']['dis'] = [np.zeros((self.numStimuli, 4096)) for k in range(self.numDistractorShapesPerTrial)]
			stimuliCounter = 0


			for category in self.testCategory:
				self.datasetShapeEmbeddingsAndRenderings[category] = {}
				self.datasetShapeEmbeddingsAndRenderings[category]['obj'] = []
				self.datasetShapeEmbeddingsAndRenderings[category]['rendering'] = []
				self.datasetShapeEmbeddingsAndRenderings[category]['stimuliEmbeddings'] = {}
				missingRenderingPaths[category] = []
				renderCatPath = stimuliRenderingsMainDir + '/' + category + '/'
				mkdir(renderCatPath)

				validRenderCounter = 0

				tempRenderPaths = []
				tempShapeGtIndices = []
				tempArrays = []
				successStatValueArray = []

				blenderRenderJob = []
				embeddingJob = []

				for i, meshPathAndGtIdx in enumerate(self.embeddingMeshPaths[category]):
					tempRenderPaths.append(renderCatPath + str(meshPathAndGtIdx[1]) + '.png')
					tempShapeGtIndices.append(meshPathAndGtIdx)
					blenderRenderJob.append(Process(target=blenderInternal.simpleRender, kwargs={'objPath': meshPathAndGtIdx[0] + '/simplifiedModel.obj', 'resolution': self.resolutions[0], 'renderPath': renderCatPath + str(meshPathAndGtIdx[1]) + '.png', 'removeMats': True, 'rotVec': rotVec, 'polish': polish}))
					if len(blenderRenderJob) == 30 or i == len(self.embeddingMeshPaths[category])-1:
						renderIndices = []
						for k in range(len(blenderRenderJob)):
							if not fileExist(tempRenderPaths[k]):
								renderIndices.append(k)
								blenderRenderJob[k].start()

						for k in renderIndices:
							if not fileExist(tempRenderPaths[k]) or blenderRenderJob[k].is_alive():
								blenderRenderJob[k].join()


						for k in range(len(blenderRenderJob)):
							if not fileExist(tempRenderPaths[k]):
								print('==> Error: Shape rendering not done properly', tempRenderPaths[k])
								missingRenderCounter += 1
								missingRenderingPaths[category].append([tempShapeGtIndices[k][0], tempShapeGtIndices[k][1], tempRenderPaths[k]])
							else:
								tempArrays.append(mp.RawArray('f', 4096))
								successStatValueArray.append(mp.Value('l', 0))
								self.datasetShapeEmbeddingsAndRenderings[category]['obj'].append(tempShapeGtIndices[k])
								self.datasetShapeEmbeddingsAndRenderings[category]['rendering'].append(tempRenderPaths[k])
								embeddingJob.append(Process(target=self.forwardPassProcess, kwargs={'renderingPath': tempRenderPaths[k], 'tempBuffer': tempArrays[validRenderCounter], 'successStatValue': successStatValueArray[validRenderCounter]}))
								embeddingJob[-1].start()
								validRenderCounter += 1

						blenderRenderJob = []
						tempRenderPaths = []
						tempShapeGtIndices = []
						# break # Uncomment this line to quickly pass the shape renderings for each category (only one batch will be rendered)

				for k, proc in enumerate(embeddingJob):
					proc.join()

				### Make sure the curropted renderings are fixed by re-rendering the mesh
				for k, val in enumerate(successStatValueArray):
					if val.value == 0:
						blenderRenderJob.append(Process(target=blenderInternal.simpleRender, kwargs={'objPath': self.datasetShapeEmbeddingsAndRenderings[category]['obj'][k][0] + '/simplifiedModel.obj', 'resolution': self.resolutions[0], 'renderPath': self.datasetShapeEmbeddingsAndRenderings[category]['rendering'][k], 'removeMats': True, 'rotVec': rotVec, 'polish': polish}))
						blenderRenderJob[-1].start()

				for k, val in enumerate(successStatValueArray):
					if val.value == 0:
						blenderRenderJob[k].join()

				tempEmbedProcess = []
				for k, val in enumerate(successStatValueArray):
					if val.value == 0:
						tempEmbedProcess.append(Process(target=self.forwardPassProcess, kwargs={'renderingPath': self.datasetShapeEmbeddingsAndRenderings[category]['rendering'][k], 'tempBuffer': tempArrays[k], 'successStatValue': successStatValueArray[k]}))
						tempEmbedProcess[-1].start()


				for k, val in enumerate(successStatValueArray):
					if val.value == 0:
						tempEmbedProcess[k].join()
				### END make sure the curropted renderings are fixed by re-rendering the mesh

				self.datasetShapeEmbeddingsAndRenderings[category]['embedding'] = np.zeros((len(tempArrays), 4096))
				for k, tempArray in enumerate(tempArrays):
					self.datasetShapeEmbeddingsAndRenderings[category]['embedding'][k][:] = np.reshape(np.frombuffer(tempArray, dtype=np.float32), 4096)
				print (category, ' -- sum of embeddings: ', self.datasetShapeEmbeddingsAndRenderings[category]['embedding'].sum())
				self.datasetShapeEmbeddingsAndRenderings[category]['embedding'] = self.datasetShapeEmbeddingsAndRenderings[category]['embedding'].astype(np.float16)

				# Get the embeddings for the ground-truth and distractor shapes in our stimuli, rendered from a canonical pose
				# self.datasetShapeEmbeddingsAndRenderings[category]['stimuliEmbeddings']['gt'] = np.zeros((self.numStimuli, 4096))
				# self.datasetShapeEmbeddingsAndRenderings[category]['stimuliEmbeddings']['dis'] = [np.zeros((self.numStimuli, 4096)) for k in range(self.numDistractorShapesPerTrial)]

				# embeddingJob = []
				# for i in range(self.numStimuli):
				# 	gtRenderingPath = self.trials['PathsUUUCanonical'][i]
				# 	print (gtRenderingPath)
				# 	embeddingJob.append(Process(target=self.forwardPassProcess, kwargs={'renderingPath': tempRenderPaths[k], 'tempBuffer': tempArrays[validRenderCounter], 'successStatValue': successStatValueArray[validRenderCounter]}))


				self.datasetShapeEmbeddingsAndRenderings[category]['stimuliEmbeddings']['gt'] = np.zeros((int(self.numStimuli/len(self.testCategory)), 4096))
				self.datasetShapeEmbeddingsAndRenderings[category]['stimuliEmbeddings']['dis'] = [np.zeros((int(self.numStimuli/len(self.testCategory)), 4096)) for k in range(self.numDistractorShapesPerTrial)]
				tempGtArrays = []
				tempDisArrays = [[] for i in range(self.numDistractorShapesPerTrial)]
				embeddingJobGt = []
				embeddingJobDis = [[] for i in range(self.numDistractorShapesPerTrial)]
				for i in range(int(self.numStimuli/len(self.testCategory))):
					tempGtArrays.append(mp.RawArray('f', 4096))
					embeddingJobGt.append(Process(target=self.forwardPassProcess, kwargs={'renderingPath': self.trials[category]['PathsUUUCanonical'][i]['GtNoCloth'], 'tempBuffer': tempGtArrays[i]}))
					embeddingJobGt[-1].start()
					embeddingJobGt[-1].join()
					for k in range(self.numDistractorShapesPerTrial):
						tempDisArrays[k].append(mp.RawArray('f', 4096))
						embeddingJobDis[k].append(Process(target=self.forwardPassProcess, kwargs={'renderingPath': self.trials[category]['PathsUUUCanonical'][i]['DisNoCloth'][k], 'tempBuffer': tempDisArrays[k][i]}))
						embeddingJobDis[k][-1].start()

					if len(embeddingJobGt) == 30 or i == int(self.numStimuli/len(self.testCategory))-1:
						for ii in range(len(embeddingJobGt)):
							embeddingJobGt[ii].join()
							for k in range(self.numDistractorShapesPerTrial):
								embeddingJobDis[k][ii].join()
						embeddingJobGt = []
						embeddingJobDis = [[] for i in range(self.numDistractorShapesPerTrial)]

				for i in range(int(self.numStimuli/len(self.testCategory))):
					self.datasetShapeEmbeddingsAndRenderings[category]['stimuliEmbeddings']['gt'][i, :] = np.reshape(np.frombuffer(tempGtArrays[i], dtype=np.float32), 4096)
					self.datasetShapeEmbeddingsAndRenderings['stimuliEmbeddings']['gt'][stimuliCounter, :] = self.datasetShapeEmbeddingsAndRenderings[category]['stimuliEmbeddings']['gt'][i]
					for k in range(self.numDistractorShapesPerTrial):
						self.datasetShapeEmbeddingsAndRenderings[category]['stimuliEmbeddings']['dis'][k][i, :] = np.reshape(np.frombuffer(tempDisArrays[k][i], dtype=np.float32), 4096)
						self.datasetShapeEmbeddingsAndRenderings['stimuliEmbeddings']['dis'][k][stimuliCounter, :] = self.datasetShapeEmbeddingsAndRenderings[category]['stimuliEmbeddings']['dis'][k][i, :]
					stimuliCounter += 1

				self.datasetShapeEmbeddingsAndRenderings[category]['stimuliEmbeddings']['gt'] = self.datasetShapeEmbeddingsAndRenderings[category]['stimuliEmbeddings']['gt'].astype(np.float16)
				for k in range(self.numDistractorShapesPerTrial):
					self.datasetShapeEmbeddingsAndRenderings[category]['stimuliEmbeddings']['dis'][k] = self.datasetShapeEmbeddingsAndRenderings[category]['stimuliEmbeddings']['dis'][k].astype(np.float16)
				
				tempArrays = None
				tempGtArrays = None
				tempDisArrays = None

			self.datasetShapeEmbeddingsAndRenderings['stimuliEmbeddings']['gt'] = self.datasetShapeEmbeddingsAndRenderings['stimuliEmbeddings']['gt'].astype(np.float16)
			self.datasetShapeEmbeddingsAndRenderings['stimuliEmbeddings']['dis'][k] = self.datasetShapeEmbeddingsAndRenderings['stimuliEmbeddings']['dis'][k].astype(np.float16)
			
			savePickle(self.embeddingPathPkl, self.datasetShapeEmbeddingsAndRenderings)
			blenderInternal.exitBlender()
		else:
			self.datasetShapeEmbeddingsAndRenderings = None


	def getNearestNeighborsToStimuli(self):
		# Get the nearest neighbor shapes to the stimuli shapes
		chosenShapeIdicesForCatsLocal = {}
		for category in self.testCategory:
			chosenShapeIdicesForCatsLocal[category] = []
			chosenShapeIdicesForCatsLocal[category] = self.chosenShapeIdicesForCats[category][:]
		if not fileExist(self.embeddingDistancePkl):

			if self.datasetShapeEmbeddingsAndRenderings is None:
				self.datasetShapeEmbeddingsAndRenderings = loadPickle(self.embeddingPathPkl)
			elif not fileExist(self.embeddingPathPkl):
				raise Exception('==> Error: You need to execute the function getShapeEmbeddings() first. Quiting')

			self.sortedDatasetShapesToSource = {}
			self.sortedDatasetShapesToSource['gt'] = {}
			self.sortedDatasetShapesToSource['dis'] = {}

			for category in self.testCategory:

				self.sortedDatasetShapesToSource['gt'][category] = []
				self.sortedDatasetShapesToSource['dis'][category] = []

				sourceShapeEmbeddingGt = self.datasetShapeEmbeddingsAndRenderings[category]['stimuliEmbeddings']['gt']
				sourceShapeEmbeddingDis = self.datasetShapeEmbeddingsAndRenderings[category]['stimuliEmbeddings']['dis']
				

				gtShapeCategories = self.trials[category]['categoryListGt']
				disShapeCategories = self.trials[category]['categoryListDis']

				sortedDatasetShapesToSourceGt = []
				sortedDatasetShapesToSourceDis = []
				
				for i in range(len(self.trials[category]['categoryListGt'])):
					gtCategory = self.allCategories[gtShapeCategories[i]]
					sourceShapeRenderingGt = self.trials[category]['PathsUUUCanonical'][i]['GtNoCloth']
					if gtCategory != 'bicycle':
						datasetEmbeddingsGt = self.datasetShapeEmbeddingsAndRenderings[gtCategory]['embedding']
						datasetShapesGt = [objData for objData in self.datasetShapeEmbeddingsAndRenderings[gtCategory]['obj']]
						datasetRenderingsGt = self.datasetShapeEmbeddingsAndRenderings[gtCategory]['rendering']
						catIndicesGt = [gtShapeCategories[i] for _ in range(len(datasetShapesGt))]
					else:
						# We don't have enough shapes in the bicycle category. Luckily, we have enough shapes in motorcycle category, so we combine the embeddings and obj shape data for both
						bicycleEmbeddingsGt = self.datasetShapeEmbeddingsAndRenderings[gtCategory]['embedding']
						bicycleShapesGt = [objData for objData in self.datasetShapeEmbeddingsAndRenderings[gtCategory]['obj']]
						bicycleRenderingsGt = self.datasetShapeEmbeddingsAndRenderings[gtCategory]['rendering']
						bicycleCatIndicesGt = [gtShapeCategories[i] for _ in range(len(bicycleShapesGt))]
						motorcycleEmbeddingsGt = self.datasetShapeEmbeddingsAndRenderings['motorcycle']['embedding']
						motorcycleEmbeddingsGt[:] = motorcycleEmbeddingsGt
						motorcycleEmbeddingsGt += 0.05
						motorcycleShapesGt = [objData for objData in self.datasetShapeEmbeddingsAndRenderings['motorcycle']['obj']]
						motorcycleRenderingsGt = self.datasetShapeEmbeddingsAndRenderings['motorcycle']['rendering']
						shapeIdx = 0
						for cat in self.allCategories:
							if cat == 'motorcycle':
								motorcycleCatIdx = shapeIdx
							shapeIdx += 1
						motorcycleCatIndicesGt = [motorcycleCatIdx for i in range(len(motorcycleShapesGt))]

						datasetEmbeddingsGt = np.concatenate((bicycleEmbeddingsGt, motorcycleEmbeddingsGt), axis=0)
						datasetShapesGt = bicycleShapesGt
						datasetRenderingsGt = bicycleRenderingsGt
						catIndicesGt = bicycleCatIndicesGt
						for l, shapeData in enumerate(motorcycleShapesGt):
							datasetShapesGt.append(shapeData)
							datasetRenderingsGt.append(motorcycleRenderingsGt[l])
							catIndicesGt.append(motorcycleCatIndicesGt[l])

					distancePklPath = self.renderingDistancePath + '/' + self.trainOrTestData + '/{0:s}-{1:s}_gt_{2:d}.pkl'.format(category, gtCategory, i)
					if not fileExist(distancePklPath):
						mkdir(self.renderingDistancePath + '/' + self.trainOrTestData)
						# Calculate source vs dataset rendering distance to augment distance calculation of embeddings and make the distance calculation more robust
						# This is because sometimes the dataset embeddings are pretty close to the source embeddings but the shapes look very different in reality
						# Vice versa is also true
						sourceVsDatasetRenderingDistanceGt = self.computeSourceVsDatasetRenderingDistance(path=distancePklPath, sourceShapeRendering=sourceShapeRenderingGt, datasetRenderings=datasetRenderingsGt)
					else:
						sourceVsDatasetRenderingDistanceGt = loadPickle(distancePklPath)
					sortedDatasetShapesToSourceGt.append(self.getSortedShapesByDistance(sourceShapeEmbedding=sourceShapeEmbeddingGt[i], datasetEmbeddings=datasetEmbeddingsGt, datasetShapeObjs=datasetShapesGt, sourceVsDatasetRenderingDistance=sourceVsDatasetRenderingDistanceGt, datasetRenderings=datasetRenderingsGt, catIndices=catIndicesGt))

					# Sort shape indices based on their distance in embedding space to the groud-truth shape
					datasetEmbeddingsDis = []
					datasetShapesDis = []
					sortedDatasetShapesToSourceDisTrial = []
					for k in range(self.numDistractorShapesPerTrial):
						disCategory = self.allCategories[disShapeCategories[i][k]]
						sourceShapeRenderingDis = self.trials[category]['PathsUUUCanonical'][i]['DisNoCloth'][k]
						if disCategory != 'bicycle':
							datasetEmbeddingsDis.append(self.datasetShapeEmbeddingsAndRenderings[disCategory]['embedding'])
							datasetShapesDis.append([objData for objData in self.datasetShapeEmbeddingsAndRenderings[disCategory]['obj']])
							datasetRenderingsDis = self.datasetShapeEmbeddingsAndRenderings[disCategory]['rendering']
							catIndicesDis = [disShapeCategories[i][k] for _ in range(len(datasetShapesDis[k]))]
						else:
							bicycleEmbeddingsDis = self.datasetShapeEmbeddingsAndRenderings[disCategory]['embedding']
							bicycleShapesDis = [objData for objData in self.datasetShapeEmbeddingsAndRenderings[disCategory]['obj']]
							bicycleCatIndicesDis = [disShapeCategories[i][k] for _ in range(len(bicycleShapesDis))]
							bicycleRenderingsDis = self.datasetShapeEmbeddingsAndRenderings[disCategory]['rendering']
							motorcycleEmbeddingsDis = self.datasetShapeEmbeddingsAndRenderings['motorcycle']['embedding']
							motorcycleEmbeddingsDis[:] = motorcycleEmbeddingsDis
							motorcycleEmbeddingsDis += 0.05
							motorcycleShapesDis = [objData for objData in self.datasetShapeEmbeddingsAndRenderings['motorcycle']['obj']]
							motorcycleRenderingsDis = self.datasetShapeEmbeddingsAndRenderings['motorcycle']['rendering']
							shapeIdx = 0
							for cat in self.allCategories:
								if cat == 'motorcycle':
									motorcycleCatIdx = shapeIdx
								shapeIdx += 1
							motorcycleCatIndicesDis = [motorcycleCatIdx for i in range(len(motorcycleShapesDis))]

							datasetEmbeddingsDis.append(np.concatenate((bicycleEmbeddingsDis, motorcycleEmbeddingsDis), axis=0))
							datasetShapesDis.append(bicycleShapesDis)
							datasetRenderingsDis = bicycleRenderingsDis
							catIndicesDis = bicycleCatIndicesDis
							for l, shapeData in enumerate(motorcycleShapesDis):
								datasetShapesDis[k].append(shapeData)
								datasetRenderingsDis.append(motorcycleRenderingsDis[l])
								catIndicesDis.append(motorcycleCatIndicesDis[l])

						# for k in range(self.numDistractorShapesPerTrial):
						distancePklPath = self.renderingDistancePath + '/' + self.trainOrTestData + '/{0:s}-{1:s}_dis_{2:d}-{3:d}.pkl'.format(category, disCategory, i, k)
						if not fileExist(distancePklPath):
							mkdir(self.renderingDistancePath + '/' + self.trainOrTestData)
							# Calculate source vs dataset rendering distance to augment distance calculation of embeddings and make the distance calculation more robust
							# This is because sometimes the dataset embeddings are pretty close to the source embeddings but the shapes look very different in reality
							# Vice versa is also true
							sourceVsDatasetRenderingDistanceDis = self.computeSourceVsDatasetRenderingDistance(path=distancePklPath, sourceShapeRendering=sourceShapeRenderingDis, datasetRenderings=datasetRenderingsDis)
						else:
							sourceVsDatasetRenderingDistanceDis = loadPickle(distancePklPath)
						sortedDatasetShapesToSourceDisTrial.append(self.getSortedShapesByDistance(sourceShapeEmbedding=sourceShapeEmbeddingDis[k][i], datasetEmbeddings=datasetEmbeddingsDis[k], datasetShapeObjs=datasetShapesDis[k], sourceVsDatasetRenderingDistance=sourceVsDatasetRenderingDistanceDis, datasetRenderings=datasetRenderingsDis, catIndices=catIndicesDis))

					sortedDatasetShapesToSourceDis.append(sortedDatasetShapesToSourceDisTrial)

				self.sortedDatasetShapesToSource['gt'][category] = sortedDatasetShapesToSourceGt
				self.sortedDatasetShapesToSource['dis'][category] = sortedDatasetShapesToSourceDis

			savePickle(self.embeddingDistancePkl, self.sortedDatasetShapesToSource)
		else:
			self.sortedDatasetShapesToSource = None

		self.datasetShapeEmbeddingsAndRenderings = None


		if not fileExist(self.nearestNeighborTrialsPklPath):
			if self.sortedDatasetShapesToSource is None:
				self.sortedDatasetShapesToSource = loadPickle(self.embeddingDistancePkl)
			"""
			Choose nearest neighbor shapes
			"""
			for category in self.testCategory:
				self.trials[category]['newShapes'] = {}
				self.trials[category]['newShapes']['SourceCatIDGt'] = []
				self.trials[category]['newShapes']['SourceCatGt'] = []
				self.trials[category]['newShapes']['CatIDGt'] = []
				self.trials[category]['newShapes']['ObjPathGt'] = []
				self.trials[category]['newShapes']['ShapeIDGt'] = []
				self.trials[category]['newShapes']['ShapeNormDistanceGt'] = []
				self.trials[category]['newShapes']['ShapeUnnormDistanceGt'] = []
				self.trials[category]['newShapes']['SourceEmbeddingGt'] = []
				self.trials[category]['newShapes']['NearestNeighborEmbeddingDistanceGt'] = []
				self.trials[category]['newShapes']['NearestNeighborEmbeddingGt'] = []
				self.trials[category]['newShapes']['SourceRenderingGt'] = []
				self.trials[category]['newShapes']['NearestNeighborRenderingDistanceGt'] = []
				self.trials[category]['newShapes']['NearestNeighborRenderingGt'] = []

				self.trials[category]['newShapes']['SourceCatIDDis'] = []
				self.trials[category]['newShapes']['SourceCatDis'] = []
				self.trials[category]['newShapes']['CatIDDis'] = []
				self.trials[category]['newShapes']['ObjPathDis'] = []
				self.trials[category]['newShapes']['ShapeIDDis'] = []
				self.trials[category]['newShapes']['ShapeNormDistanceDis'] = []
				self.trials[category]['newShapes']['ShapeUnnormDistanceDis'] = []
				self.trials[category]['newShapes']['SourceEmbeddingDis'] = []
				self.trials[category]['newShapes']['NearestNeighborEmbeddingDistanceDis'] = []
				self.trials[category]['newShapes']['NearestNeighborEmbeddingDis'] = []
				self.trials[category]['newShapes']['SourceRenderingDis'] = []
				self.trials[category]['newShapes']['NearestNeighborRenderingDistanceDis'] = []
				self.trials[category]['newShapes']['NearestNeighborRenderingDis'] = []

				gtShapeCategories = self.trials[category]['categoryListGt']
				disShapeCategories = self.trials[category]['categoryListDis']


				for i in range(len(self.trials[category]['categoryListGt'])):

					self.trials[category]['newShapes']['SourceCatIDGt'].append(gtShapeCategories[i])
					self.trials[category]['newShapes']['SourceCatGt'].append(self.allCategories[gtShapeCategories[i]])
					self.trials[category]['newShapes']['CatIDGt'].append([])
					self.trials[category]['newShapes']['ObjPathGt'].append([])
					self.trials[category]['newShapes']['ShapeIDGt'].append([])
					self.trials[category]['newShapes']['ShapeNormDistanceGt'].append([])
					self.trials[category]['newShapes']['ShapeUnnormDistanceGt'].append([])
					self.trials[category]['newShapes']['SourceEmbeddingGt'].append([])
					self.trials[category]['newShapes']['NearestNeighborEmbeddingDistanceGt'].append([])
					self.trials[category]['newShapes']['NearestNeighborEmbeddingGt'].append([])
					self.trials[category]['newShapes']['SourceRenderingGt'].append(self.trials[category]['PathsUUUCanonical'][i]['GtNoCloth'])
					self.trials[category]['newShapes']['NearestNeighborRenderingDistanceGt'].append([])
					self.trials[category]['newShapes']['NearestNeighborRenderingGt'].append([])


					self.trials[category]['newShapes']['SourceCatIDDis'].append([])
					self.trials[category]['newShapes']['SourceCatDis'].append([])
					self.trials[category]['newShapes']['SourceRenderingDis'].append([])
					for k in range(self.numDistractorShapesPerTrial):
						self.trials[category]['newShapes']['SourceCatIDDis'][i].append(disShapeCategories[i][k])
						self.trials[category]['newShapes']['SourceCatDis'][i].append(self.allCategories[disShapeCategories[i][k]])
						self.trials[category]['newShapes']['SourceRenderingDis'][i].append(self.trials[category]['PathsUUUCanonical'][i]['DisNoCloth'][k])
					self.trials[category]['newShapes']['CatIDDis'].append([[] for k in range(self.numDistractorShapesPerTrial)])
					self.trials[category]['newShapes']['ObjPathDis'].append([[] for k in range(self.numDistractorShapesPerTrial)])
					self.trials[category]['newShapes']['ShapeIDDis'].append([[] for k in range(self.numDistractorShapesPerTrial)])
					self.trials[category]['newShapes']['ShapeNormDistanceDis'].append([[] for k in range(self.numDistractorShapesPerTrial)])
					self.trials[category]['newShapes']['ShapeUnnormDistanceDis'].append([[] for k in range(self.numDistractorShapesPerTrial)])
					self.trials[category]['newShapes']['SourceEmbeddingDis'].append([[] for k in range(self.numDistractorShapesPerTrial)])
					self.trials[category]['newShapes']['NearestNeighborEmbeddingDistanceDis'].append([[] for k in range(self.numDistractorShapesPerTrial)])
					self.trials[category]['newShapes']['NearestNeighborEmbeddingDis'].append([[] for k in range(self.numDistractorShapesPerTrial)])
					# self.trials[category]['newShapes']['SourceRenderingDis'].append([[] for k in range(self.numDistractorShapesPerTrial)])
					self.trials[category]['newShapes']['NearestNeighborRenderingDistanceDis'].append([[] for k in range(self.numDistractorShapesPerTrial)])
					self.trials[category]['newShapes']['NearestNeighborRenderingDis'].append([[] for k in range(self.numDistractorShapesPerTrial)])

			for shapeNo in range(self.shapeUncertaintyNumExtraShapes):
				for i in range(len(self.trials[category]['categoryListGt'])):
					for category in self.testCategory:					
						gtShapesSorted = self.sortedDatasetShapesToSource['gt'][category][i]
						gtCategoryIndices = gtShapesSorted['catIndices']
						disShapesSorted = self.sortedDatasetShapesToSource['dis'][category][i]
						disCategoryIndices = []
						for k in range(self.numDistractorShapesPerTrial):
							disCategoryIndices.append(disShapesSorted[k]['catIndices'])

						searchFlagGt = True
						gtIndex = 0
						while searchFlagGt:
							validIdx = self.manuallyCrossoutIndices(category=self.allCategories[gtCategoryIndices[gtIndex]], indices=[gtShapesSorted['datasetShapes'][gtIndex][1]])
							if gtShapesSorted['datasetShapes'][gtIndex][1] not in chosenShapeIdicesForCatsLocal[self.allCategories[gtCategoryIndices[gtIndex]]] and validIdx:
								if shapeNo > 0:
									embeddingDistanceGt = 1000000
									renderingDistanceGt = 1000000
									for num in range(shapeNo):
										embeddingDistanceGt = min(embeddingDistanceGt, np.linalg.norm(self.trials[category]['newShapes']['NearestNeighborEmbeddingGt'][i][num].astype(np.float32) - gtShapesSorted['embeddings'][gtIndex].astype(np.float32), ord=1))
										sourceImg = pngToNumpy(pngPath=self.trials[category]['newShapes']['NearestNeighborRenderingGt'][i][num], renderType='rgb', resolution=self.resolutions[0], dtype='float32').flatten()
										targetImg = pngToNumpy(pngPath=gtShapesSorted['datasetRenderings'][gtIndex], renderType='rgb', resolution=self.resolutions[0], dtype='float32').flatten()
										renderingDistanceGt = min(renderingDistanceGt, np.linalg.norm(sourceImg - targetImg, ord=1))
								else:
									embeddingDistanceGt = np.linalg.norm(gtShapesSorted['sourceShapeEmbedding'].astype(np.float32) - gtShapesSorted['embeddings'][gtIndex].astype(np.float32), ord=1)
									sourceImg = pngToNumpy(pngPath=self.trials[category]['PathsUUUCanonical'][i]['GtNoCloth'], renderType='rgb', resolution=self.resolutions[0], dtype='float32').flatten()
									targetImg = pngToNumpy(pngPath=gtShapesSorted['datasetRenderings'][gtIndex], renderType='rgb', resolution=self.resolutions[0], dtype='float32').flatten()
									renderingDistanceGt = np.linalg.norm(sourceImg - targetImg, ord=1)

								if embeddingDistanceGt > self.embeddingMinDistance and renderingDistanceGt > self.embeddingMinDistance or self.allCategories[gtCategoryIndices[gtIndex]] == 'bicycle':
									self.trials[category]['newShapes']['CatIDGt'][i].append(gtCategoryIndices[gtIndex])
									self.trials[category]['newShapes']['ObjPathGt'][i].append(gtShapesSorted['datasetShapes'][gtIndex][0] + '/simplifiedModel.obj')
									self.trials[category]['newShapes']['ShapeIDGt'][i].append(gtShapesSorted['datasetShapes'][gtIndex][1])
									self.trials[category]['newShapes']['ShapeNormDistanceGt'][i].append(gtShapesSorted['normalizedDistances'][gtIndex])
									self.trials[category]['newShapes']['ShapeUnnormDistanceGt'][i].append(gtShapesSorted['unnormalizedDistances'][gtIndex])
									self.trials[category]['newShapes']['SourceEmbeddingGt'][i] = np.array(gtShapesSorted['sourceShapeEmbedding']).astype(np.float16)
									self.trials[category]['newShapes']['NearestNeighborEmbeddingGt'][i].append(gtShapesSorted['embeddings'][gtIndex])
									self.trials[category]['newShapes']['NearestNeighborEmbeddingDistanceGt'][i].append(embeddingDistanceGt)
									self.trials[category]['newShapes']['NearestNeighborRenderingGt'][i].append(gtShapesSorted['datasetRenderings'][gtIndex])
									self.trials[category]['newShapes']['NearestNeighborRenderingDistanceGt'][i].append(renderingDistanceGt)

									chosenShapeIdicesForCatsLocal[self.allCategories[gtCategoryIndices[gtIndex]]].append(gtShapesSorted['datasetShapes'][gtIndex][1])
									searchFlagGt = False
							gtIndex += 1

						for k in range(self.numDistractorShapesPerTrial):
							tempNewShapesDis = []
							searchFlagDis = True
							disIndex = 0
							while searchFlagDis:
								validIdx = self.manuallyCrossoutIndices(category=self.allCategories[disCategoryIndices[k][disIndex]], indices=[disShapesSorted[k]['datasetShapes'][disIndex][1]])
								if disShapesSorted[k]['datasetShapes'][disIndex][1] not in chosenShapeIdicesForCatsLocal[self.allCategories[disCategoryIndices[k][disIndex]]] and validIdx:

									if shapeNo > 0:
										embeddingDistanceDis = 1000000
										renderingDistanceDis = 1000000
										for num in range(shapeNo):
											embeddingDistanceDis = min(embeddingDistanceDis, np.linalg.norm(self.trials[category]['newShapes']['NearestNeighborEmbeddingDis'][i][k][num].astype(np.float32) - disShapesSorted[k]['embeddings'][disIndex].astype(np.float32), ord=1))
											sourceImg = pngToNumpy(pngPath=self.trials[category]['newShapes']['NearestNeighborRenderingDis'][i][k][num], renderType='rgb', resolution=self.resolutions[0], dtype='float32').flatten()
											targetImg = pngToNumpy(pngPath=disShapesSorted[k]['datasetRenderings'][disIndex], renderType='rgb', resolution=self.resolutions[0], dtype='float32').flatten()
											renderingDistanceDis = min(renderingDistanceDis, np.linalg.norm(sourceImg - targetImg, ord=1))
									else:
										embeddingDistanceDis = np.linalg.norm(disShapesSorted[k]['sourceShapeEmbedding'].astype(np.float32) - disShapesSorted[k]['embeddings'][disIndex].astype(np.float32), ord=1)
										sourceImg = pngToNumpy(pngPath=self.trials[category]['PathsUUUCanonical'][i]['DisNoCloth'][k], renderType='rgb', resolution=self.resolutions[0], dtype='float32').flatten()
										targetImg = pngToNumpy(pngPath=disShapesSorted[k]['datasetRenderings'][disIndex], renderType='rgb', resolution=self.resolutions[0], dtype='float32').flatten()
										renderingDistanceDis = np.linalg.norm(sourceImg - targetImg, ord=1)

									if embeddingDistanceDis > self.embeddingMinDistance and renderingDistanceDis > self.embeddingMinDistance or self.allCategories[disCategoryIndices[k][disIndex]] == 'bicycle':
										self.trials[category]['newShapes']['SourceCatIDDis'][i] = disShapeCategories[i][k]
										self.trials[category]['newShapes']['CatIDDis'][i][k].append(disCategoryIndices[k][disIndex])
										self.trials[category]['newShapes']['ObjPathDis'][i][k].append(disShapesSorted[k]['datasetShapes'][disIndex][0] + '/simplifiedModel.obj')
										self.trials[category]['newShapes']['ShapeIDDis'][i][k].append(disShapesSorted[k]['datasetShapes'][disIndex][1])
										self.trials[category]['newShapes']['ShapeNormDistanceDis'][i][k].append(disShapesSorted[k]['normalizedDistances'][disIndex])
										self.trials[category]['newShapes']['ShapeUnnormDistanceDis'][i][k].append(disShapesSorted[k]['unnormalizedDistances'][disIndex])
										self.trials[category]['newShapes']['SourceEmbeddingDis'][i][k] = np.array(disShapesSorted[k]['sourceShapeEmbedding']).astype(np.float16)
										self.trials[category]['newShapes']['NearestNeighborEmbeddingDis'][i][k].append(disShapesSorted[k]['embeddings'][disIndex])
										self.trials[category]['newShapes']['NearestNeighborEmbeddingDistanceDis'][i][k].append(embeddingDistanceDis)
										# self.trials[category]['newShapes']['SourceRenderingDis'][i][k].append(self.trials[category]['PathsUUUCanonical'][i]['DisNoCloth'][k])
										self.trials[category]['newShapes']['NearestNeighborRenderingDis'][i][k].append(disShapesSorted[k]['datasetRenderings'][disIndex])
										self.trials[category]['newShapes']['NearestNeighborRenderingDistanceDis'][i][k].append(renderingDistanceDis)

										chosenShapeIdicesForCatsLocal[self.allCategories[disCategoryIndices[k][disIndex]]].append(disShapesSorted[k]['datasetShapes'][disIndex][1])
										searchFlagDis = False
								disIndex += 1

			# for category in self.testCategory:
			# 	for i in range(len(self.trials[category]['categoryListGt'])):
			# 		unn = []
			# 		n = []
			# 		nnEmbedDistance = []
			# 		nnRenderDistance = []
			# 		for shapeNo in range(self.shapeUncertaintyNumExtraShapes):
			# 			unn.append(self.trials[category]['newShapes']['ShapeUnnormDistanceGt'][i][shapeNo])
			# 			n.append(self.trials[category]['newShapes']['ShapeNormDistanceGt'][i][shapeNo])
			# 			nnEmbedDistance.append(self.trials[category]['newShapes']['NearestNeighborEmbeddingDistanceGt'][i][shapeNo])
			# 			nnRenderDistance.append(self.trials[category]['newShapes']['NearestNeighborRenderingDistanceGt'][i][shapeNo])
			# 		print (category, i, unn, n, nnEmbedDistance, nnRenderDistance)
			# 	print ('\n')
			# # exit()


			savePickle(self.nearestNeighborTrialsPklPath, self.trials)

			"""
			Print some statistics on the distances of ground-truth shapes (both gt and its distractor) and their nearest neighbors
			"""
			# Print rendering path of the ground-truth shape + its nearest neighbors
				# 	print (self.trials[category]['PathsUUUCanonical'][i]['GtNoCloth'])
				# 	print (category, i, 'gtGt', self.trials[category]['newShapes']['ObjPathGt'][i][0], self.trials[category]['newShapes']['ShapeUnnormDistanceGt'][i][0])
				# 	print (category, i, 'gtGt', self.trials[category]['newShapes']['ObjPathGt'][i][1], self.trials[category]['newShapes']['ShapeUnnormDistanceGt'][i][1])
				# 	print (category, i, 'gtGt', self.trials[category]['newShapes']['ObjPathGt'][i][2], self.trials[category]['newShapes']['ShapeUnnormDistanceGt'][i][2])
				# 	print (category, i, 'gtGt', self.trials[category]['newShapes']['ObjPathGt'][i][3], self.trials[category]['newShapes']['ShapeUnnormDistanceGt'][i][3])
				# 	print ('')
				# print ('\n')

			# for category in self.testCategory:
			# 	# print (category)
			# 	catDistances = []
			# 	catGtGtDistances = []
			# 	catGtDifDistances = []
			# 	catDisGtDistances = []
			# 	catDisDifDistances = []
			# 	for i in range(len(self.trials[category]['categoryListGt'])):
			# 		# dist = np.array(self.trials[category]['newShapes']['ShapeUnnormDistanceGt'][i])
			# 		catDistances += self.trials[category]['newShapes']['ShapeUnnormDistanceGt'][i]
			# 		for k in range(self.numDistractorShapesPerTrial):
			# 			catDistances += self.trials[category]['newShapes']['ShapeUnnormDistanceDis'][i][k]
					
			# 		if i > 0 and i < 6:
			# 			catGtGtDistances += self.trials[category]['newShapes']['ShapeUnnormDistanceGt'][i]
			# 			catDisGtDistances = []
			# 			for k in range(self.numDistractorShapesPerTrial):
			# 				catDisGtDistances += self.trials[category]['newShapes']['ShapeUnnormDistanceDis'][i][k]
			# 		elif i >= 6:
			# 			catGtDifDistances += self.trials[category]['newShapes']['ShapeUnnormDistanceGt'][i]
			# 			catDisDifDistances = []
			# 			for k in range(self.numDistractorShapesPerTrial):
			# 				catDisDifDistances += self.trials[category]['newShapes']['ShapeUnnormDistanceDis'][i][k]
				
				# Print some statistics on the distance of of nearest neighbor shapes in the embedding space
				# catDistances = np.array(catDistances)
				# catGtGtDistances = np.array(catGtGtDistances)
				# catGtDifDistances = np.array(catGtDifDistances)
				# catDisGtDistances = np.array(catDisGtDistances)
				# catDisDifDistances = np.array(catDisDifDistances)
				# print ('catDistances', catDistances.mean(), catDistances.min(), catDistances.max(), catDistances.std())
				# print ('catGtGtDistances', catGtGtDistances.mean(), catGtGtDistances.min(), catGtGtDistances.max(), catGtGtDistances.std())
				# print ('catGtDifDistances', catGtDifDistances.mean(), catGtDifDistances.min(), catGtDifDistances.max(), catGtDifDistances.std())
				# print ('catDisGtDistances', catDisGtDistances.mean(), catDisGtDistances.min(), catDisGtDistances.max(), catDisGtDistances.std())
				# print ('catDisDifDistances', catDisDifDistances.mean(), catDisDifDistances.min(), catDisDifDistances.max(), catDisDifDistances.std())
				# print ('')
		else:
			self.trials = loadPickle(self.nearestNeighborTrialsPklPath)

		self.sortedDatasetShapesToSource = None

	def makeFineTuningTrialsList(self):
		self.newTrials = {}
		for i in range(2):
			self.newTrials[i] = {}
			self.newTrials[i]['finetuneSeparateStimuli'] = []
			for category in self.testCategory:
				self.newTrials[i][category] = {}
				self.newTrials[i][category]['CatIDGt'] = [] # Ground-truth -- And always from one category (e.g. car)
				self.newTrials[i][category]['ObjPathGt'] = [] # Ground-truth -- And always from one category (e.g. car)
				self.newTrials[i][category]['ShapeIDGt'] = [] # Ground-truth -- And always from one category (e.g. car)
				self.newTrials[i][category]['OccludedGt'] = [] 

				self.newTrials[i][category]['CatIDDistractor'] = [] # Could be from the same category
				self.newTrials[i][category]['ObjPathDistractor'] = [] # Could be from the same category
				self.newTrials[i][category]['ShapeIDDistractor'] = [] # Could be from the same category
				self.newTrials[i][category]['OccludedDistractor'] = []

				for occlusionType in range(3):
					# occlusionType = 0 is used for the Unoccluded, Unoccluded, Occluded task
					# occlusionType = 1 is used for the Occluded, Occluded, Unoccluded task
					# occlusionType = 2 is used for the Unoccluded, Unoccluded, Unoccluded task
					for withinCatTrialNo in range(self.numStimuli//len(self.testCategory)):
						for shapeNo in range(self.shapeUncertaintyNumExtraShapes+1):
							CatIDGt = self.trials[category]['CatIDGt'][withinCatTrialNo]
							ObjPathGt = self.trials[category]['ObjPathGt'][withinCatTrialNo]
							ShapeIDGt = self.trials[category]['ShapeIDGt'][withinCatTrialNo]

							for k in range(self.numDistractorShapesPerTrial):
								if shapeNo == 0:
									CatIDDis = self.trials[category]['CatIDDistractor'][withinCatTrialNo][k]
									ObjPathDis = self.trials[category]['ObjPathDistractor'][withinCatTrialNo][k]
									ShapeIDDis = self.trials[category]['ShapeIDDistractor'][withinCatTrialNo][k]
								else:
									CatIDDis = self.trials[category]['newShapes']['CatIDDis'][withinCatTrialNo][k][shapeNo-1]
									ObjPathDis = self.trials[category]['newShapes']['ObjPathDis'][withinCatTrialNo][k][shapeNo-1]
									ShapeIDDis = self.trials[category]['newShapes']['ShapeIDDis'][withinCatTrialNo][k][shapeNo-1]

								self.newTrials[i][category]['CatIDGt'].append(CatIDGt)
								self.newTrials[i][category]['ObjPathGt'].append(ObjPathGt)
								self.newTrials[i][category]['ShapeIDGt'].append(ShapeIDGt)
								self.newTrials[i][category]['OccludedGt'].append(occlusionType)

								self.newTrials[i][category]['CatIDDistractor'].append([CatIDDis])
								self.newTrials[i][category]['ObjPathDistractor'].append([ObjPathDis])
								self.newTrials[i][category]['ShapeIDDistractor'].append([ShapeIDDis])
								self.newTrials[i][category]['OccludedDistractor'].append(occlusionType)


					for withinCatTrialNo in range(self.numStimuli//len(self.testCategory)):
						for sourceShapeNo in range(self.shapeUncertaintyNumExtraShapes):
							for shapeNo in range(self.shapeUncertaintyNumExtraShapes):
								CatIDGt = self.trials[category]['newShapes']['CatIDGt'][withinCatTrialNo][sourceShapeNo]
								ObjPathGt = self.trials[category]['newShapes']['ObjPathGt'][withinCatTrialNo][sourceShapeNo]
								ShapeIDGt = self.trials[category]['newShapes']['ShapeIDGt'][withinCatTrialNo][sourceShapeNo]
								for k in range(self.numDistractorShapesPerTrial):
									if shapeNo == 0:
										CatIDDis = self.trials[category]['CatIDDistractor'][withinCatTrialNo][k]
										ObjPathDis = self.trials[category]['ObjPathDistractor'][withinCatTrialNo][k]
										ShapeIDDis = self.trials[category]['ShapeIDDistractor'][withinCatTrialNo][k]
										self.newTrials[i][category]['CatIDGt'].append(CatIDGt)
										self.newTrials[i][category]['ObjPathGt'].append(ObjPathGt)
										self.newTrials[i][category]['ShapeIDGt'].append(ShapeIDGt)
										self.newTrials[i][category]['OccludedGt'].append(occlusionType)

										self.newTrials[i][category]['CatIDDistractor'].append([CatIDDis])
										self.newTrials[i][category]['ObjPathDistractor'].append([ObjPathDis])
										self.newTrials[i][category]['ShapeIDDistractor'].append([ShapeIDDis])
										self.newTrials[i][category]['OccludedDistractor'].append(occlusionType)


									CatIDDis = self.trials[category]['newShapes']['CatIDDis'][withinCatTrialNo][k][shapeNo]
									ObjPathDis = self.trials[category]['newShapes']['ObjPathDis'][withinCatTrialNo][k][shapeNo]
									ShapeIDDis = self.trials[category]['newShapes']['ShapeIDDis'][withinCatTrialNo][k][shapeNo]

								self.newTrials[i][category]['CatIDGt'].append(CatIDGt)
								self.newTrials[i][category]['ObjPathGt'].append(ObjPathGt)
								self.newTrials[i][category]['ShapeIDGt'].append(ShapeIDGt)
								self.newTrials[i][category]['OccludedGt'].append(occlusionType)

								self.newTrials[i][category]['CatIDDistractor'].append([CatIDDis])
								self.newTrials[i][category]['ObjPathDistractor'].append([ObjPathDis])
								self.newTrials[i][category]['ShapeIDDistractor'].append([ShapeIDDis])
								self.newTrials[i][category]['OccludedDistractor'].append(occlusionType)
					# print (self.trials[category]['ShapeIDGt'])
					# print (self.trials[category]['ShapeIDDistractor'])
					# print (self.trials[category]['OccludedGt'])
					# print ('\n\n')
					# print (self.newTrials[i][category]['ShapeIDGt'])
					# print ('')
					# print (self.newTrials[i][category]['ShapeIDDistractor'])
					# print ('')
					# print (self.newTrials[i][category]['OccludedGt'])
					# print ('\n\n')
					# print (category, len(self.newTrials[i][category]['OccludedGt']))

				self.newTrials[i][category]['PathsUUO'] = [] #Stores the rendering paths (.png) for the unoccluded-unoccluded-occluded set of trials
				self.newTrials[i][category]['PathsOOU'] = [] #Stores the rendering paths (.png) for the occluded-occluded-unoccluded set of trials
				self.newTrials[i][category]['PathsUUU'] = [] #Stores the rendering paths (.png) for the unoccluded-unoccluded-unoccluded set of trials
				self.newTrials[i][category]['PathsUUUCanonical'] = [] #Stores the rendering paths (.png) for the unoccluded-unoccluded-unoccluded set of trials with canonical pose
				self.newTrials[i][category]['RotationsUUO'] = [] #Stores the rotation vectors for the unoccluded-unoccluded-occluded set of trials
				self.newTrials[i][category]['RotationsOOU'] = [] #Stores the rotation vectors for the occluded-occluded-unoccluded set of trials
				self.newTrials[i][category]['RotationsUUU'] = [] #Stores the rotation vectors for the unoccluded-unoccluded-unoccluded set of trials
				self.newTrials[i][category]['categoryListGt'] = []
				self.newTrials[i][category]['categoryListDis'] = []
				self.newTrials[i][category]['stimuliSetNoUUO'] = []
				self.newTrials[i][category]['stimuliSetNoOOU'] = []
				self.newTrials[i][category]['stimuliSetNoUUU'] = []

			self.newTrials[i]['PathsUUO'] = [] #Stores the rendering paths (.png) for the unoccluded-unoccluded-occluded set of trials
			self.newTrials[i]['PathsOOU'] = [] #Stores the rendering paths (.png) for the occluded-occluded-unoccluded set of trials
			self.newTrials[i]['PathsUUU'] = [] #Stores the rendering paths (.png) for the unoccluded-unoccluded-unoccluded set of trials
			self.newTrials[i]['PathsUUUCanonical'] = [] #Stores the rendering paths (.png) for the unoccluded-unoccluded-unoccluded set of trials with canonical pose
			self.newTrials[i]['RotationsUUO'] = [] #Stores the rotation vectors for the unoccluded-unoccluded-occluded set of trials
			self.newTrials[i]['RotationsOOU'] = [] #Stores the rotation vectors for the occluded-occluded-unoccluded set of trials
			self.newTrials[i]['RotationsUUU'] = [] #Stores the rotation vectors for the unoccluded-unoccluded-unoccluded set of trials
			self.newTrials[i]['stimuliSetNoUUO'] = []
			self.newTrials[i]['stimuliSetNoOOU'] = []
			self.newTrials[i]['stimuliSetNoUUU'] = []



	def plotNearestNeighborsStats(self):

		dataTemplateDict = {'allShapes': {}, 'gtShapes': {}, 'disShapes': {}}
		dataTemplateList = {'allShapes': [], 'gtShapes': [], 'disShapes': []}
		data = {'unnormalized': {}, 'normalized': {}}
		for i in range(2):
			plotPath = self.plotsPath + '/' + self.trainOrTestData + (i == 0 and '/normalized_distances/' or '/unnormalized_distances/')
			
			
			"""
			Make a placeholder dictionary
			"""
			normalized = i == 0 and 'normalized' or 'unnormalized'
			data[normalized] = {'allStimuli': {}, 'categoryStimuli': {}}
			for category in self.testCategory:
				data[normalized]['categoryStimuli'][category] = {}
				data[normalized]['categoryStimuli'][category]['trials'] = deepcopy(dataTemplateDict)
				data[normalized]['categoryStimuli'][category]['category'] = deepcopy(dataTemplateList)
				data[normalized]['categoryStimuli'][category]['nearestShape'] = deepcopy(dataTemplateDict)
				for shapeNo in range(self.shapeUncertaintyNumExtraShapes):
					data[normalized]['categoryStimuli'][category]['nearestShape']['allShapes'][shapeNo] = []
					data[normalized]['categoryStimuli'][category]['nearestShape']['gtShapes'][shapeNo] = []
					data[normalized]['categoryStimuli'][category]['nearestShape']['disShapes'][shapeNo] = []

			data[normalized]['allStimuli']['trials'] = deepcopy(dataTemplateDict)
			for trialNo in range(len(self.trials[category]['categoryListGt'])):
				# data[normalized]['allStimuli']['trials']['allShapes'][trialNo] = []
				data[normalized]['allStimuli']['trials']['allShapes'] = [[] for k in range(len(self.trials[category]['categoryListGt']) * 2)]
				data[normalized]['allStimuli']['trials']['gtShapes'][trialNo] = []
				data[normalized]['allStimuli']['trials']['disShapes'][trialNo] = []
				for category in self.testCategory:
					# data[normalized]['categoryStimuli'][category]['trials']['allShapes'][trialNo] = []
					data[normalized]['categoryStimuli'][category]['trials']['allShapes'] = []
					data[normalized]['categoryStimuli'][category]['trials']['gtShapes'][trialNo] = []
					data[normalized]['categoryStimuli'][category]['trials']['disShapes'][trialNo] = []

			data[normalized]['allStimuli']['nearestShape'] = deepcopy(dataTemplateDict)
			for shapeNo in range(self.shapeUncertaintyNumExtraShapes):
				data[normalized]['allStimuli']['nearestShape']['allShapes'][shapeNo] = []
				data[normalized]['allStimuli']['nearestShape']['gtShapes'][shapeNo] = []
				data[normalized]['allStimuli']['nearestShape']['disShapes'][shapeNo] = []


			"""
			Fill the dictionaries
			"""
			normalizedDistanceGt = i == 0 and 'ShapeNormDistanceGt' or 'ShapeUnnormDistanceGt'
			normalizedDistanceDis = i == 0 and 'ShapeNormDistanceDis' or 'ShapeUnnormDistanceDis'
			# for shapeNo in range(self.shapeUncertaintyNumExtraShapes):
			for category in self.testCategory:
				for trialNo in range(len(self.trials[category]['categoryListGt'])):
					# data[normalized]['categoryStimuli'][category]['trials']['allShapes'][trialNo] += self.trials[category]['newShapes'][normalizedDistanceGt][trialNo]
					data[normalized]['categoryStimuli'][category]['trials']['allShapes'].append(self.trials[category]['newShapes'][normalizedDistanceGt][trialNo])
					data[normalized]['categoryStimuli'][category]['trials']['gtShapes'][trialNo] += self.trials[category]['newShapes'][normalizedDistanceGt][trialNo]

					data[normalized]['categoryStimuli'][category]['category']['allShapes'] += self.trials[category]['newShapes'][normalizedDistanceGt][trialNo]
					data[normalized]['categoryStimuli'][category]['category']['gtShapes'] += self.trials[category]['newShapes'][normalizedDistanceGt][trialNo]

					# data[normalized]['allStimuli']['trials']['allShapes'][trialNo] += self.trials[category]['newShapes'][normalizedDistanceGt][trialNo]
					data[normalized]['allStimuli']['trials']['allShapes'][trialNo * 2] += self.trials[category]['newShapes'][normalizedDistanceGt][trialNo]
					data[normalized]['allStimuli']['trials']['gtShapes'][trialNo] += self.trials[category]['newShapes'][normalizedDistanceGt][trialNo]

					for shapeNo in range(self.shapeUncertaintyNumExtraShapes):
						data[normalized]['categoryStimuli'][category]['nearestShape']['allShapes'][shapeNo].append(self.trials[category]['newShapes'][normalizedDistanceGt][trialNo][shapeNo])
						data[normalized]['categoryStimuli'][category]['nearestShape']['gtShapes'][shapeNo].append(self.trials[category]['newShapes'][normalizedDistanceGt][trialNo][shapeNo])
						data[normalized]['allStimuli']['nearestShape']['allShapes'][shapeNo].append(self.trials[category]['newShapes'][normalizedDistanceGt][trialNo][shapeNo])
						data[normalized]['allStimuli']['nearestShape']['gtShapes'][shapeNo].append(self.trials[category]['newShapes'][normalizedDistanceGt][trialNo][shapeNo])

					tempDistanceListCategory = []
					tempDistanceListAllStimuli = []
					for k in range(self.numDistractorShapesPerTrial):
						# data[normalized]['categoryStimuli'][category]['trials']['allShapes'][trialNo] += self.trials[category]['newShapes'][normalizedDistanceDis][trialNo][k]
						tempDistanceListCategory += self.trials[category]['newShapes'][normalizedDistanceDis][trialNo][k]
						data[normalized]['categoryStimuli'][category]['trials']['disShapes'][trialNo] += self.trials[category]['newShapes'][normalizedDistanceDis][trialNo][k]

						data[normalized]['categoryStimuli'][category]['category']['allShapes'] += self.trials[category]['newShapes'][normalizedDistanceDis][trialNo][k]
						data[normalized]['categoryStimuli'][category]['category']['disShapes'] += self.trials[category]['newShapes'][normalizedDistanceDis][trialNo][k]
						
						# data[normalized]['allStimuli']['trials']['allShapes'][trialNo] += self.trials[category]['newShapes'][normalizedDistanceDis][trialNo][k]
						tempDistanceListAllStimuli += self.trials[category]['newShapes'][normalizedDistanceDis][trialNo][k]
						data[normalized]['allStimuli']['trials']['disShapes'][trialNo] += self.trials[category]['newShapes'][normalizedDistanceDis][trialNo][k]

						for shapeNo in range(self.shapeUncertaintyNumExtraShapes):
							data[normalized]['categoryStimuli'][category]['nearestShape']['allShapes'][shapeNo].append(self.trials[category]['newShapes'][normalizedDistanceDis][trialNo][k][shapeNo])
							data[normalized]['categoryStimuli'][category]['nearestShape']['disShapes'][shapeNo].append(self.trials[category]['newShapes'][normalizedDistanceDis][trialNo][k][shapeNo])
							data[normalized]['allStimuli']['nearestShape']['allShapes'][shapeNo].append(self.trials[category]['newShapes'][normalizedDistanceDis][trialNo][k][shapeNo])
							data[normalized]['allStimuli']['nearestShape']['disShapes'][shapeNo].append(self.trials[category]['newShapes'][normalizedDistanceDis][trialNo][k][shapeNo])

					data[normalized]['categoryStimuli'][category]['trials']['allShapes'].append(tempDistanceListCategory)
					data[normalized]['allStimuli']['trials']['allShapes'][trialNo * 2 + 1] += tempDistanceListAllStimuli
					

			# print (len(data[normalized]['allStimuli']['trials']['gtShapes'][11]))
			# print ('')
			# print (data[normalized]['allStimuli']['nearestShape']['gtShapes'][0])
			# print ('')
			# print (len(data[normalized]['categoryStimuli'][category]['category']['gtShapes']))
			# exit()


			"""
			Make the plots
			"""
			# Generate plots for 'categoryStimuli'
			print ("==> Making plots for statistics of the chosen nearest neighbor shapes (" + normalized + ")")
			for allStimuli in range(2):
				for plotTypeNo in (allStimuli == 0 and range(3) or [0, 2]): # plotTypeNo = 0 refers to "trial" plots, plotTypeNo = 1 refers to category plots, plotTypeNo = 2 refers to nearest neighbor plots
					plotType = plotTypeNo == 0 and 'trials' or plotTypeNo == 1 and 'category' or 'nearestShape'
					mkdir(plotPath + '/' + plotType + '/' + (allStimuli and 'allStimuli' or 'categoryStimuli'))
					plotTypeVerbose = plotTypeNo == 0 and 'Trials' or plotTypeNo == 1 and 'Each Category' or 'Nearest Neighbor Shapes'
					for shapeType in range(3): # allShapes, gtShapes, disShapes
						shapesType = shapeType == 0 and 'allShapes' or shapeType == 1 and 'gtShapes' or 'disShapes'
						shapesTypeVerbose = shapeType == 0 and 'GT and Distractor Shapes' or shapeType == 1 and 'Ground-Truth Shapes' or 'Distractor Shapes'
						for numCat, category in enumerate(self.testCategory):
							if allStimuli == 0:
								if plotTypeNo != 1:
									boxPlotRawData = data[normalized]['categoryStimuli'][category][plotType][shapesType]
								else:
									boxPlotRawData = []
									for cat in self.testCategory:
										boxPlotRawData.append(data[normalized]['categoryStimuli'][cat][plotType][shapesType])
							else:
								boxPlotRawData = data[normalized]['allStimuli'][plotType][shapesType]
							maxList = []
							npArrDataList = []
							numRange = (plotTypeNo == 0 and shapeType == 0) and len(self.trials[category]['categoryListGt']) * 2 or (plotTypeNo == 0 and shapeType > 0) and len(self.trials[category]['categoryListGt']) or plotTypeNo == 2 and self.shapeUncertaintyNumExtraShapes or len(self.testCategory)
							for num in range(numRange):
								npArrData = np.array(boxPlotRawData[num])
								npArrDataList.append(npArrData)
								maxList.append(npArrData.max())
							npArrDataList = npArrDataList[::-1]
							maxList = maxList[::-1]

							# xTicks = np.linspace(0, 1, 21) if i == 0 else np.linspace(0, math.ceil(max(maxList)/100.0)*100, 21)
							xTicks = np.linspace(0, 1, 21) if i == 0 else np.linspace(0, 6000, 21)
							# if i == 1:
							# 	xTicks = [math.floor(xTick/100.0) * 100 for xTick in xTicks]
							xTickLabels = [(i == 0 and '%.2f' or '%d') % xTick for xTick in xTicks]
							if plotTypeNo == 0:
								yTicks = []
								for kk in range(len(self.trials[category]['categoryListGt'])):
									yTicks.append(kk+1)
									if shapeType == 0:
										yTicks.append(kk+1)
								if shapeType == 0:
									yTickLabels = [(str(yTick) + (kk % 2 == 0 and '-gt' or '-dis')) for kk, yTick in enumerate(yTicks)]
								else:
									yTickLabels = [(str(yTick) + (shapeType == 1 and '-gt' or '-dis')) for kk, yTick in enumerate(yTicks)]
								yTicks = list(range(1, len(yTickLabels)+1))
							else:
								numYTicks = plotTypeNo == 1 and len(self.testCategory) or self.shapeUncertaintyNumExtraShapes
								yTicks = list(range(1, numYTicks+1))
								yTickLabels = [(plotTypeNo == 1 and self.testCategory[yTick-1].title() or ('NN-' + str(yTick))) for yTick in yTicks]
							yTickLabels = yTickLabels[::-1]
							yGap = 0.4
							xGap = max(xTicks) <= 1.0 and 0.01 or max(xTicks) > 1.0 and (max(xTicks)/100)
							ax_kwargs = {}
							plotObj = plotClass()
							figObj, axObj, ax_save_kwargs = plotObj.createSubplots()
							ax_kwargs = {'x': npArrDataList, 'vert': False, 'showfliers': False, 'whis': 2.0}
							axObj = plotObj.setupAx(axObj=axObj, axType='boxplot', ax_kwargs=ax_kwargs)
							
							ax_save_kwargs['figObj'] = figObj
							ax_save_kwargs['axObj'] = axObj
							ax_save_kwargs['figTitle'] = normalized.title() + ' Shape Distance for ' + plotTypeVerbose + ' - Embeddings from: ' + self.shapeUncertaintyEmbeddingModelName.upper() + ' ' + self.opts.BOResultsPath + (allStimuli == 0 and plotTypeNo != 1 and (' - Category: ' + category) or ' - All Categories') + ' - ' + shapesTypeVerbose
							ax_save_kwargs['xTicks'] = xTicks
							ax_save_kwargs['yTicks'] = yTicks
							ax_save_kwargs['xLabel'] = normalized.title() + ' Distance'
							ax_save_kwargs['yLabel'] = ''
							ax_save_kwargs['xTickLabels'] = xTickLabels
							ax_save_kwargs['yTickLabels'] = yTickLabels
							ax_save_kwargs['plotSavePath'] = plotPath + '/' + plotType + '/' + (allStimuli and 'allStimuli' or 'categoryStimuli') + '/' + normalized.title() + ' - ' + self.shapeUncertaintyEmbeddingModelName.upper() + ' ' + self.opts.BOResultsPath + (allStimuli == 0 and plotTypeNo != 1 and (' - Category -- ' + category + ' -- ') or ' - ') + shapesTypeVerbose + '.png'
							ax_save_kwargs['figTitleFontSize'] = 5
							ax_save_kwargs['xLimLow'] = min(xTicks) - xGap
							ax_save_kwargs['xLimHigh'] = max(xTicks) + xGap
							ax_save_kwargs['yLimLow'] = min(yTicks) - yGap
							ax_save_kwargs['yLimHigh'] = max(yTicks) + yGap
							# ax_save_kwargs['yGrid'] = False
							plotObj.savePlot(**ax_save_kwargs)

							if allStimuli == 1 or plotTypeNo == 1:
								break

			

	def visualizeNearestNeighbors(self):
		mkdir (self.visualizationPath + '/' + self.trainOrTestData)
		visualizationPath = self.visualizationPath + '/' + self.trainOrTestData + '/'
		trialCounter = 0
		for category in self.testCategory:
			for i in range(len(self.trials[category]['categoryListGt'])):
				cp(self.trials[category]['newShapes']['SourceRenderingGt'][i], visualizationPath + 'trial{0:d}_{1:s}_{2:d}_gt.png'.format(trialCounter, category, i))
				for k in range(self.numDistractorShapesPerTrial):
					cp(self.trials[category]['newShapes']['SourceRenderingDis'][i][k], visualizationPath + 'trial{0:d}_{1:s}_{2:d}_dis{3:d}.png'.format(trialCounter, category, i, k))
				for shapeNo in range(self.shapeUncertaintyNumExtraShapes):
					cp(self.trials[category]['newShapes']['NearestNeighborRenderingGt'][i][shapeNo], visualizationPath + 'trial{0:d}_{1:s}_{2:d}_gt_nearestNeighbor{3:d}.png'.format(trialCounter, category, i, shapeNo))
					for k in range(self.numDistractorShapesPerTrial):
						cp(self.trials[category]['newShapes']['NearestNeighborRenderingDis'][i][k][shapeNo], visualizationPath + 'trial{0:d}_{1:s}_{2:d}_dis{3:d}_nearestNeighbor{4:d}.png'.format(trialCounter, category, i, k, shapeNo))
				trialCounter += 1

	def renderFineTuningStimuli(self):
		trainDataDictLock = False
		testDataDictLock = False
		self.stimuliResultsPath = self.opts.stimuliResultsPath + '/fineTuningStimuli'
		for i in range(self.finetuneNumStimuliSets):
			if i < (self.finetuneNumStimuliSets - 2) and not trainDataDictLock:
				self.trialsPklPath = self.cwd + self.opts.datasetStorePath + '/trialsFineTuningTrain.pkl'
				self.trials = self.newTrials[0]
				trainDataDictLock = True
			elif i >= (self.finetuneNumStimuliSets - 2) and not testDataDictLock:
				self.trialsPklPath = self.cwd + self.opts.datasetStorePath + '/trialsFineTuningTest.pkl'
				self.trials = self.newTrials[1]
				testDataDictLock = True
			np.random.seed(self.opts.seedStimuli+982+i) # 982 is an arbitrary number
			if (trainDataDictLock and i == self.finetuneNumStimuliSets-3) or (testDataDictLock and i == self.finetuneNumStimuliSets-1):
				lastStimuliSetSignal = True
			else:
				lastStimuliSetSignal = False
			self.renderStimuli(noSimAndRendering=self.finetuneStimuliReady, fineTuningStimuli=True, fineTuningStimuliCounter=i, lastStimuliSetSignal=lastStimuliSetSignal)


	# Helper functions

	def getSortedShapesByDistance(self, sourceShapeEmbedding, datasetEmbeddings, datasetShapeObjs, sourceVsDatasetRenderingDistance, datasetRenderings, catIndices, shapeUncertaintyNNDistanceStartPercentage=0.0):
		sameIndices = []
		tempValidDatasetShapes = []
		tempValidCatIndices = []
		tempValidShapeEmbeddings = []
		tempValidDatasetRenderings = []
		distances = []
		datasetEmbeddings = datasetEmbeddings.astype(np.float32)
		sourceShapeEmbedding = sourceShapeEmbedding.astype(np.float32)
		sourceVsDatasetRenderingDistanceMax = sourceVsDatasetRenderingDistance.max()
		for i, shapeData in enumerate(datasetShapeObjs):
			embeddingDistance = np.linalg.norm(sourceShapeEmbedding - datasetEmbeddings[i], ord=1)
			renderingDistance = sourceVsDatasetRenderingDistance[i]
			if not np.isclose(embeddingDistance, 0, atol=90.0) and not np.isclose(renderingDistance, 0, atol=sourceVsDatasetRenderingDistanceMax/100*4):
				'''
				Numbers 90 and sourceVsDatasetRenderingDistanceMax/100*4 were chosen arbitrarily after some trial and error
				to find the sweet spot for choosing shapes that neither nearly identical nor very different to the source shape
				'''
				tempValidDatasetShapes.append(shapeData)
				tempValidCatIndices.append(catIndices[i])
				tempValidShapeEmbeddings.append(datasetEmbeddings[i].astype(np.float16))
				tempValidDatasetRenderings.append(datasetRenderings[i])
				distances.append(embeddingDistance)
		unnormalizedDistances = np.array(distances[:])
		distances = np.array(distances)
		distances -= distances.min()
		distances /= distances.max()

		validDatasetShapes = []
		validCatIndices = []
		validShapeEmbeddings = []
		validDatasetRenderings = []
		tempDistances = []
		tempUnnormalizedDistances = []
		for i, distance in enumerate(distances):
			if distance >= shapeUncertaintyNNDistanceStartPercentage:
				validDatasetShapes.append(tempValidDatasetShapes[i])
				validCatIndices.append(tempValidCatIndices[i])
				validShapeEmbeddings.append(tempValidShapeEmbeddings[i])
				validDatasetRenderings.append(tempValidDatasetRenderings[i])
				tempDistances.append(distance)
				tempUnnormalizedDistances.append(unnormalizedDistances[i])
		distances = np.array(tempDistances)
		unnormalizedDistances = np.array(tempUnnormalizedDistances)

		sortedIndices = np.argsort(distances)
		sortedDistances = distances[sortedIndices]
		sortedUnnormalizedDistances = unnormalizedDistances[sortedIndices]
		sortedDatasetShapes = []
		sortedDatasetRenderings = []
		sortedCatIndices = []
		sortedShapeEmbeddings = []
		for sortedIdx in sortedIndices:
			sortedDatasetShapes.append(validDatasetShapes[sortedIdx])
			sortedDatasetRenderings.append(validDatasetRenderings[sortedIdx])
			sortedCatIndices.append(validCatIndices[sortedIdx])
			
			# sortedReLUActivations = np.array(validShapeEmbeddings[sortedIdx])
			# sortedReLUActivations[sortedReLUActivations < 0] = 0.0
			# sortedShapeEmbeddings.append(sortedReLUActivations)

			sortedShapeEmbeddings.append(validShapeEmbeddings[sortedIdx])

		return {'normalizedDistances': sortedDistances, 'unnormalizedDistances': sortedUnnormalizedDistances, 'datasetShapes': sortedDatasetShapes, 'catIndices': sortedCatIndices, 'embeddings': sortedShapeEmbeddings, 'sourceShapeEmbedding': sourceShapeEmbedding, 'datasetRenderings': sortedDatasetRenderings}

	def computeSourceVsDatasetRenderingDistance(self, path, sourceShapeRendering, datasetRenderings):
		distances = []
		for i in range(len(datasetRenderings)):
			distances.append(np.linalg.norm(pngToNumpy(pngPath=sourceShapeRendering, renderType='rgb', resolution=self.resolutions[0], dtype='float32').flatten() - pngToNumpy(pngPath=datasetRenderings[i], renderType='rgb', resolution=self.resolutions[0], dtype='float32').flatten(), ord=1))
		distances = np.array(distances)
		savePickle(path, distances)
		return distances

	def simplifyMainStimuliMeshes(self):
		# Call this after calling makeStimuliList()
		simplifiedMeshePathsPkl = self.trainOrTestData == 'test' and self.simplifiedObjsPath + '/simplifiedMeshePaths.pkl' or self.simplifiedObjsPath + '/simplifiedMeshePathsTrain.pkl'
		if not fileExist(simplifiedMeshePathsPkl):
			blender = Blender()
			blender.setupScene(lampPosList=None, camPosList=None, lampEnergy=None, saveObjOnly=True)
			simplifiedShapesMainDir = self.trainOrTestData == 'test' and self.simplifiedObjsPath + '/simplifiedStimuliObjs' or self.simplifiedObjsPath + '/simplifiedStimuliObjsTrain'
			mkdir(simplifiedShapesMainDir)
			blenderJob = []
			for category in self.testCategory:
				if category in self.stimuliSet:
					# TODO fill simplifiedStimuliSet
					self.simplifiedStimuliSet[category] = [[[], []], [[], []]]
					catStimuli = self.stimuliSet[category]
					mkdir(simplifiedShapesMainDir + '/' + category)
					for i, stimuli in enumerate(catStimuli):
						distractorShapeIndicator = False
						trialNo = i != 0 and int(self.numStimuli/2/len(self.testCategory)) or 0
						for j in range(len(stimuli[0])):
							if i == 0: # The distractor is chosen from the GT category
								if j <= (self.numStimuli/2)/len(self.testCategory)*(self.numDistractorShapesPerTrial+1):
									if j > 0 and j % (self.numDistractorShapesPerTrial+1) == 0:
										trialNo += 1
										distractorShapeIndicator = False
									elif j > 0:
										distractorShapeIndicator = True
								else:
									distractorShapeIndicator = False
									trialNo += 1
							else: # The distractor is chosen from a different category than GT
								if j > 0 and j % self.numDistractorShapesPerTrial == 0:
									trialNo += 1
								distractorShapeIndicator = True
							objDirPath = simplifiedShapesMainDir + '/' + category + '/trial' + str(trialNo) + '/' + (not distractorShapeIndicator and 'ground-truth_' or 'distractor_') + self.allCategories[int(stimuli[0][j])] + '-gtIdx' + str(stimuli[1][j][1])
							mkdir(objDirPath)
							objSavePath = objDirPath + '/simplifiedModel.obj'
							self.simplifiedStimuliSet[category][i][0].append(stimuli[0][j])
							self.simplifiedStimuliSet[category][i][1].append([objSavePath, stimuli[1][j][1]])

							# Simplify the mesh
							if not fileExist(objSavePath):
								blenderJob.append(Process(target=blender.loadObj, kwargs={'objPath': stimuli[1][j][0], 'layerIdx': 1, 'harshPolish': True, 'save': True, 'objSavePath': objSavePath}))
							
							if len(blenderJob) == 12 or i == len(catStimuli)-1:
								for k in range(len(blenderJob)):
									blenderJob[k].start()

								for k in range(len(blenderJob)):
									blenderJob[k].join()
								blenderJob = []
			savePickle(simplifiedMeshePathsPkl, self.simplifiedStimuliSet)
		else:
			self.simplifiedStimuliSet = loadPickle(simplifiedMeshePathsPkl)

	def simplifyShapesForShapeEmbedding(self):
		blender = Blender()
		blender.setupScene(lampPosList=None, camPosList=None, lampEnergy=None, saveObjOnly=True)
		mkdir(self.embeddingSimplifiedObjsPathTrain)
		blenderJob = []
		self.embeddingMeshPaths = {}

		startTime = timeit.default_timer()

		for catData in self.datasetTrainTest[0]:
			if catData[1] in self.testCategory:
				category = catData[1]
				self.embeddingMeshPaths[catData[1]] = []
				for i, meshPathAndGtIdx in enumerate(catData[2]):
					simplifiedObjDirPath = self.embeddingSimplifiedObjsPathTrain + '/' + category + '/' + category + '-gtIdx' + str(meshPathAndGtIdx[1])
					mkdir(simplifiedObjDirPath)
					if not self.simplifyObjs:
						self.embeddingMeshPaths[category].append(meshPathAndGtIdx)

					# if not fileExist(simplifiedObjDirPath + '/simplifiedModel.obj') and self.simplifyObjs:
					if not fileExist(simplifiedObjDirPath + '/simplifiedModel.obj') and self.simplifyObjs:
						blenderJob.append(Process(target=blender.loadObj, kwargs={'objPath': meshPathAndGtIdx[0], 'layerIdx': 1, 'harshPolish': True, 'save': True, 'objSavePath': simplifiedObjDirPath + '/simplifiedModel.obj'}))
						blenderJob[-1].start()
					if len(blenderJob) == 40 or i == len(catData[2])-1:
						for k in range(len(blenderJob)):
							blenderJob[k].join()
						blenderJob = []

				for i, meshPathAndGtIdx in enumerate(catData[2]):
					simplifiedObjDirPath = self.embeddingSimplifiedObjsPathTrain + '/' + category + '/' + category + '-gtIdx' + str(meshPathAndGtIdx[1])
					# if fileExist(simplifiedObjDirPath + '/simplifiedModel.obj') and self.simplifyObjs:
					if fileExist(simplifiedObjDirPath) and self.simplifyObjs:
						self.embeddingMeshPaths[category].append([simplifiedObjDirPath, meshPathAndGtIdx[1]])
		savePickle(self.embeddingMeshPathsPkl, self.embeddingMeshPaths)
		print ("==> It took {0:.2f} minutes to do the mesh simplification\n".format((timeit.default_timer() - startTime)/60, self.numStimuli))

				


	def manuallyCrossoutIndices(self, category, indices):
		if category == 'bus':
			ignoreIndices = [104, 152, 165, 549]
		elif category == 'guitar':
			ignoreIndices = [31, 100, 642]
		elif category == 'rifle':
			ignoreIndices = [43, 101, 111, 136, 146, 184, 197, 1249, 1287, 1813, 2014, 2065]
		elif category == 'pistol':
			ignoreIndices = [91, 134]
		elif category == 'motorcycle':
			ignoreIndices = [67, 105, 208, 311]
		elif category == 'chair':
			ignoreIndices = [51, 67, 88, 109, 183, 2843, 5773]
		elif category == 'airplane':
			ignoreIndices = [128, 206, 1822, 2050, 2372, 2856, 2995, 3129]
		elif category == 'table':
			ignoreIndices = [1, 51, 56, 160, 1605, 3044]
		elif category == 'bicycle':
			ignoreIndices = [1, 4, 11, 13, 19, 27, 34, 36, 38, 45]
		elif category == 'car':
			ignoreIndices = [37]
		else:
			ignoreIndices = []
		
		if any(idx in indices for idx in ignoreIndices):
			return False
		else:
			return True

	def constrainDistractorsCats(self, indices, numNotAllowedConsecutiveIndices, minNumCats):
		
		if len(set(indices)) < minNumCats:
			return True

		flag = False
		for i in range(len(indices)-numNotAllowedConsecutiveIndices):
			consecutiveIndices = indices[i] == indices[i+1]
			if numNotAllowedConsecutiveIndices == 2:
				consecutiveIndices = consecutiveIndices and (indices[i] == indices[i+2])
			if consecutiveIndices:
				flag = True
				break
		return flag

	def switchDataset(self, train=False, test=True):
		if not train and not test:
			print ('==> Error: Please specify whether to use the trainint or test dataset')
			exit()
		if train:
			self.trainOrTestData = 'train'
			self.numStimuli = self.numTrainStimuli
			self.trialsPklPath = self.cwd + self.opts.datasetStorePath + "/trialsTrain.pkl"
			self.stimuliResultsPath = self.stimuliResultsPath + '/trainHumanSubjects'
			self.dataset = self.datasetTrainTest[0]

			self.embeddingDistancePkl = self.cwd + self.opts.datasetStorePath + "/sortedEmbeddings-TrainStimuli.pkl"
			self.embeddingPathPkl = self.cwd + self.opts.datasetStorePath + '/embeddingDistances-TrainStimuli.pkl'
			self.nearestNeighborTrialsPklPath = self.cwd + self.opts.datasetStorePath + "/trialsNearestNeighbors-TrainStimuli.pkl"
		else:
			self.trainOrTestData = 'test'
			self.numStimuli = self.numTestStimuli
			self.dataset = self.datasetTrainTest[1]

	def getNumberOfShapes(self):
		numOfShapes=0
		for catData in self.datasetTrainTest[0]:
			if catData[1] in self.testCategory:
				numOfShapes += len(catData[2])
		return numOfShapes

	def forwardPassProcess(self, renderingPath, tempBuffer, successStatValue=None):
		renderingNumpy = pngToNumpy(pngPath=renderingPath, renderType='rgb', resolution=self.resolutions[0], dtype='float32')
		# self.model.cuda()
		embedding = self.model(renderingNumpy)
		var = np.reshape(np.frombuffer(tempBuffer, dtype=np.float32), embedding.clone().data.cpu().numpy().shape)
		var[:] = embedding.clone().data.cpu().numpy()
		del embedding
		if successStatValue is not None:
			successStatValue.value = 1
