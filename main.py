from args import opts
from common import fileExist, savePickle, loadPickle
from copy import deepcopy
import numpy as np
import random, subprocess
# from renderData import prepDataSet
from prepareStimuli import prepStimuli
import torch

opt = opts().parse()


if opt.seed != 0:
	np.random.seed(opt.seed) # default seed is 14
	random.seed(opt.seed)
	torch.manual_seed(opt.seed)
	torch.cuda.manual_seed(opt.seed)


if opt.generateStimuli:
	if fileExist(opt.stimuliResultsPath + '/renderings'):
		print('==> Generating stimuli: The directory ' + opt.stimuliResultsPath + '/renderings' + ' exists. You need to remove it if you want to create a new set of stimuli and rerun the code. Exiting now')
		exit()
	else:
		print("==> Generating stimuli and storing them in the directory '" + opt.stimuliResultsPath + '/renderings' + "'")
		print("==> About to generate stimuli for the computational model and testing human subjects")
		stimuli = prepStimuli(opt)
		stimuli.makeStimuliList()
		stimuli.makeTrialsList()
		stimuli.renderStimuli()

		# Make a small stimuli set for training human subjects
		stimuli.switchDataset(train=True)
		stimuli.makeStimuliList()
		stimuli.makeTrialsList()
		stimuli.renderStimuli()

if not fileExist(opt.stimuliResultsPath + '/renderings') and not opt.generateStimuli:
	print("==> Error: The directory " + opt.stimuliResultsPath + " does not exist. Make sure to either create it manually and copy all files and directories (stimuli etc) into it before running the code again \
or, if you're willing to create new set of stimuli, set generateStimuli to 1 in runMain.sh before program execution. Exiting")
	exit()

