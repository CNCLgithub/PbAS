from common import fileExist, getFilesList, savePickle, loadPickle, computeNumShapes
import csv, math
from collections import Counter
import os
from numpy import random as rand

class ShapeNet(object):
	def __init__(self, opt):
		self.cwd = os.getcwd() + "/"

		self.shapenetObjsPklPath = self.cwd + opt.datasetStorePath + '/shapenetObjPaths.pkl'

		numStimuli = opt.numStimuli
		self.testCategory = opt.testCategory
		self.numDistractorShapesPerTrial = opt.numDistractorShapesPerTrial
		self.shapeUncertainty = opt.shapeUncertainty

		self.datasetPath = opt.datasetRawPath 
		self.trainPercent = opt.pTrain

		self.numShapeFromGtCat, self.numShapeFromDistractorCats = computeNumShapes(numStimuli=numStimuli, testCategory=self.testCategory, numDistractorShapesPerTrial=self.numDistractorShapesPerTrial)

	def getDatasetList(self):
		# Returns a List containing with the following structure:
			#[synsetId, categoryName, [objFilePaths(in order of appearance in the directory)]
		self.getMainDirectories()
		if not fileExist(self.shapenetObjsPklPath):
			self.getCategoryFilesPath()
		else:
			self.catFilesPaths = loadPickle(self.shapenetObjsPklPath)
		self.getTrainTestSets()
		# if self.numDistractorShapesPerTrial > 1 and not self.shapeUncertainty:
		# 	# The following function must be called even if self.numDistractorShapesPerTrial == 1
		# 	# However, we did not expect to run experiments with more than 2 shapes for each trial.
		# 	# So we made a data set without making sure that each test category has enough samples for trials
		# 	# that have more than 2 shapes in them.
		# 	self.balanceTestSet()

		# The format of the output Lists is as follow:
			#[synsetId, catName, [catFilesPath, indexOfEachFileOnDisk]] -- There are 55 of such list in both self.trainSet and self.testSet, each for one category
		if self.trainPercent < 1:
			return (self.trainSet, self.testSet)
		else:
			return (self.trainSet)

	def getMainDirectories(self):
		self.categoryDirs = getFilesList(self.datasetPath, onlyDir=True)

	def getCategoryFilesPath(self):
		tempNames = {}
		synsetIdList = []
		self.catFilesPaths = []
		for catDir in self.categoryDirs:
			synsetId = catDir.split('/')[len(catDir.split('/')) - 1]
			if synsetId not in tempNames:
				synsetIdList.append(synsetId)
				tempNames[synsetId] = []
				synsetCsvPath = getFilesList(self.datasetPath, fileType='csv', lookupStr=synsetId)
				with open(synsetCsvPath[0], 'r') as csvFile:
					reader = csv.DictReader(csvFile)
					for row in reader:
						splitNames = ', '.join(row['wnlemmas'].split(' '))
						tempNames[synsetId].append(splitNames)
					tempNames[synsetId] = ', '.join(tempNames[synsetId]).split(',')
					tempNames[synsetId] = [name.strip().lower() for name in tempNames[synsetId]]
		
		chosenCatNames = []
		for synsetId in synsetIdList:
			if chosenCatNames:
				tempNames[synsetId] = [name for name in tempNames[synsetId] if name not in chosenCatNames]
			catName = self.getMostCommonName(tempNames[synsetId])
			chosenCatNames.append(catName)
			catFilesPath = self.getFileLists(synsetId)
			self.catFilesPaths.append([synsetId, catName, catFilesPath])
		savePickle(self.shapenetObjsPklPath, self.catFilesPaths)

	def getTrainTestSets(self):
		self.trainSet = []
		self.testSet = []
		counterr = 0
		for categorySpecificObjFiles in self.catFilesPaths:
			counterr += len(categorySpecificObjFiles[2])
			numTestSamples = len(categorySpecificObjFiles[2]) - math.floor(len(categorySpecificObjFiles[2]) * self.trainPercent)

			# Make sure there is at least 28 samples in each category in the test set
			if numTestSamples < 24 and self.trainPercent < 1:
				numTestSamples = 24
				numTrainSamples = len(categorySpecificObjFiles[2]) - 24
			elif self.trainPercent == 1:
				numTrainSamples = len(categorySpecificObjFiles[2])
			else:
				numTrainSamples = math.floor(len(categorySpecificObjFiles[2]) * self.trainPercent)
			trainSamplesIdx = rand.choice(len(categorySpecificObjFiles[2]), numTrainSamples, replace=False).tolist()
			if self.trainPercent < 1:
				testSamplesIdx = [num for num in range(len(categorySpecificObjFiles[2])) if num not in trainSamplesIdx]

			# Uncomment for not randomizing the file list
			# trainSamplesIdx = [i for i in range(numTrainSamples)]
			# testSamplesIdx = [i for i in range(len(categorySpecificObjFiles[2])) if i not in trainSamplesIdx]
			
			self.trainSet.append([categorySpecificObjFiles[0], categorySpecificObjFiles[1], [[categorySpecificObjFiles[2][i], i] for i in trainSamplesIdx if 'om/data/public/ShapeNet/ShapeNetCore.v1/03001627/c5c4e6110fbbf5d3d83578ca09f86027' not in categorySpecificObjFiles[2][i]]])
			if self.trainPercent < 1:
				self.testSet.append([categorySpecificObjFiles[0], categorySpecificObjFiles[1], [[categorySpecificObjFiles[2][i], i] for i in testSamplesIdx if 'om/data/public/ShapeNet/ShapeNetCore.v1/03001627/c5c4e6110fbbf5d3d83578ca09f86027' not in categorySpecificObjFiles[2][i]]])

	
	# Helper functions
	def getMostCommonName(self, wordsList):
		c = Counter(wordsList)
		chosen = c.most_common()[0][0].strip()
		if len(c.most_common()) > 1 and c.most_common()[0][1] == c.most_common()[1][1]:
			for i in range(1, min(4, len(c.most_common()))):
				if len(chosen) < len(c.most_common()[i][0].strip()):
					chosen = c.most_common()[i][0].strip()
		if chosen == 'shooting' or chosen == 'handgun':
			chosen = 'pistol'
		if chosen == 'bike':
			chosen = 'motorcycle'
		if chosen == 'computer' or chosen == 'keypad':
			chosen = 'keyboard'
		if chosen == 'plane':
			chosen = 'airplane'
		if chosen == 'couch':
			chosen = 'sofa'
		return chosen

	def getFileLists(self, synsetId):
		dirList = []
		objDirs = getFilesList(self.datasetPath + synsetId)
		for dirr in objDirs:
			if os.path.isfile(dirr + '/model.obj'):
				dirList.append(dirr + '/model.obj')
		return dirList

	# def balanceTestSet(self):
	# 	# In case 
	# 	for i in range(len(self.testSet)):
	# 		trainCategoryData = self.trainSet[i]
	# 		testCategoryData = self.testSet[i]
	# 		if testCategoryData[1] in self.testCategory:
	# 			if len(testCategoryData[2]) < (self.numShapeFromGtCat + self.numShapeFromDistractorCats):
	# 				while len(testCategoryData[2]) < (self.numShapeFromGtCat + self.numShapeFromDistractorCats):
	# 					testCategoryData[2].append(trainCategoryData[2].pop())
	# 				self.testSet[i][2] = testCategoryData[2]