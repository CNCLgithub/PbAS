import subprocess, sys
from argparse import ArgumentParser

categories = ['airplane', 'trash', 'bag', 'basket', 'tub', 'bunk', 'bench', 'bicycle', 'birdhouse', 'boat', 'bookshelf', 'bottle', 'bowl', 'bus', 'cabinet', 'camera', 'can', 'cap', 'car', 'phone', 'chair', 'clock', 'keyboard', 'washer', 'screen', 'headphone', 'faucet', 'file', 'guitar', 'helmet', 'vase', 'knife', 'lamp', 'laptop', 'speaker', 'mailbox', 'microphone', 'microwave', 'motorcycle', 'mug', 'piano', 'pillow', 'pistol', 'flowerpot', 'printer', 'remote', 'rifle', 'missile', 'skateboard', 'sofa', 'stove', 'table', 'telephone', 'tower', 'train', 'ship', 'machine']
testCategory = ['airplane', 'bicycle', 'bus', 'car', 'chair', 'guitar', 'motorcycle', 'pistol', 'rifle', 'table']
parser = ArgumentParser()
parser.add_argument('--datasetRawPath', default='/om/data/public/ShapeNet/ShapeNetCore.v1', type=str)
parser.add_argument('--category', type=str, default=None, nargs="+", 
	help="Choose category from the followings: 'airplane', 'trash', 'bag', 'basket', 'tub', 'bunk', 'bench', 'bicycle', 'birdhouse', 'boat', 'bookshelf', 'bottle', 'bowl', 'bus', 'cabinet', 'camera', 'can', 'cap', 'car', 'phone', 'chair', 'clock', 'keyboard', 'washer', 'screen', 'headphone', 'faucet', 'file', 'guitar', 'helmet', 'vase', 'knife', 'lamp', 'laptop', 'speaker', 'mailbox', 'microphone', 'microwave', 'motorcycle', 'mug', 'piano', 'pillow', 'pistol', 'flowerpot', 'printer', 'remote', 'rifle', 'missile', 'skateboard', 'sofa', 'stove', 'table', 'telephone', 'tower', 'train', 'ship', 'machine'",
	choices=['airplane', 'trash', 'bag', 'basket', 'tub', 'bunk', 'bench', 'bicycle', 'birdhouse', 'boat', 'bookshelf', 'bottle', 'bowl', 'bus', 'cabinet', 'camera', 'can', 'cap', 'car', 'phone', 'chair', 'clock', 'keyboard', 'washer', 'screen', 'headphone', 'faucet', 'file', 'guitar', 'helmet', 'vase', 'knife', 'lamp', 'laptop', 'speaker', 'mailbox', 'microphone', 'microwave', 'motorcycle', 'mug', 'piano', 'pillow', 'pistol', 'flowerpot', 'printer', 'remote', 'rifle', 'missile', 'skateboard', 'sofa', 'stove', 'table', 'telephone', 'tower', 'train', 'ship', 'machine'])
parser.add_argument('--testCategory', type=str, default=['airplane', 'bicycle', 'bus', 'car', 'chair', 'guitar', 'motorcycle', 'pistol', 'rifle', 'table'], 
	choices=['airplane', 'bicycle', 'bus', 'car', 'chair', 'guitar', 'motorcycle', 'pistol', 'rifle', 'table'], 
	nargs="+", 
	help='The test categories from which shapes will be choosen')
parser.add_argument('--categoryID', default=[-1], type=int, nargs="+", help='Category ID: set to negative numbers to ignore the rendering phase')
parser.add_argument('--numShape', default=2, type=int)
parser.add_argument('--numRotation', default=10, type=int)
parser.add_argument('--resolutions', default=[224], type=int, nargs="+")
parser.add_argument('--pTrain', default=0.9, type=float)
parser.add_argument('--generateStimuli', default=0, type=int, choices=[0, 1])
parser.add_argument('--sortedDataset', default=0, type=int, choices=[0, 1])
parser.add_argument('--simultaneousRotation', default=0, type=int, choices=[0, 1])
parser.add_argument('--runBayesianOpt', default=0, type=int, choices=[0, 1])
parser.add_argument('--genStats_Plots', default=0, type=int, choices=[0, 1])
parser.add_argument('--simplifyObjs', default=0, type=int, choices=[0, 1])
parser.add_argument('--finetunePretrainedModel', type=int, default=0, help='Fine-tunes a pre-trained model')
parser.add_argument('--BoNumRunStart', type=int, default=-1, help='Starting BO run number')
parser.add_argument('--BoNumRunEnd', type=int, default=-1, help='Ending BO run number')
parser.add_argument('--Boxi', type=int, default=330, help='Ending BO run number')
parser.add_argument('--removeMatAfterSimRotSteps', default=7, type=int)
parser.add_argument('--BOLossModel', default='alexnet', choices=['baseline', 'alexnet', 'vgg', 'vggbn', 'densenet', 'resnet', 'cornet_s', 'midas'], help='Name of the model to be loaded and used for computing Bayesian Optimization loss function')
parser.add_argument('--BOLossModelFeatureLayerName', default='fc1', choices=['lastPool', 'fc1', 'fc2', 'conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'layer1', 'layer2', 'layer3', 'layer4', 'fc11', 'fc12', 'v1Linear', 'v1Conv', 'v2Linear', 'v2Conv', 'v4Linear', 'v4Conv', 'itLinear', 'itConv', 'decoder'], help='Name of the layer where we want to chop the model at')
parser.add_argument('--BOUnifiedInferencePipelineWith5Shapes', type=int, default=1, choices=[0, 1], help='Determines whether the computational model should have access to either 5 or 10 shapes. You may exclude the ground truth shapes by setting BOExcludeGTShapes to 1 to allow the inference pipeline to have access to either 4 nearest neighbors or 8 nearest neighbors (assuming BOUnifiedInferencePipelineWith5Shapes=1)')
parser.add_argument('--BOExcludeGTShapes', type=int, default=1, choices=[0, 1], help='Set to 1 to exclude the ground-truth shapes for both the target shape and the distractor during second phase of inference')
parser.add_argument('--BOUseEmbeddingNearestNeighborPrior', type=int, default=3, choices=[0, 1, 2, 3, 4, 5], help='Didcates how prior for nearest neighbors should be constructed. Ideally, this should be set to 1 but since loss function embeddings do not have a fixed reference points, the obtained numbers are intuitively wrong, hence set this to 3 or 4 for ranked priors with different increments')
parser.add_argument('--BOResultsPathNote', default='', help='an arbitrary note to be appended to the end of BOResults Path directory')
parser.add_argument('--visMakeGPPlots', type=int, default=1, choices=[0, 1], help='Set to 1 to make Gaussian process plots after getting the posterior over mean and covariance of the GP')
parser.add_argument('--expType', type=str, default='', help='Type of experiment')
parser.add_argument('--forwardPassType', type=str, default='', help='Type of forward pass')
parser.add_argument('--featureNetwork', type=str, default='', help='Type of forward pass')
parser.add_argument('--featLayerNo', type=int, default=1, help='Determines the layer number from which features will be extracted and stored on disk')
args = parser.parse_args()

if args.runBayesianOpt == 1:
	args.simplifyObjs = 1

if args.expType == 0:
	args.expType = '0'

if args.forwardPassType == 0:
	args.forwardPassType = '0'


if len(args.categoryID) == 1 and args.categoryID[0] <= -1:
	args.categoryID = None

if args.categoryID is not None:
	# Override the category argument
	catList = []
	for catID in args.categoryID:
		catList.append((not args.simplifyObjs or args.runBayesianOpt) and categories[catID] or testCategory[catID])
	args.category = catList
	args.testCategory = testCategory
elif args.categoryID is None and args.category is None:
	args.category = categories
elif args.categoryID is None and args.category is None or (args.generateStimuli == 0 and args.simplifyObjs == 0):
	print ('==> Error: Make sure you have specified the value(s) for either of categoryID, category and have specified the value for either of generateStimuli or simplifyObjs')
	exit()

if args.categoryID is None and args.testCategory is None:
	args.testCategory = testCategory
elif args.categoryID is None and args.testCategory is None and args.simplifyObjs == 1:
	print ('==> Error: You need to specify either of categoryID, category to generate data')
	exit()

if args.simultaneousRotation == 1 and args.numRotation == 0 and args.simplifyObjs == 0:
	print ('==> Error: You need to specify a number of rotations more than 0 if you want to get simultaneous rotation renderings alongside multi-view renderings')
	exit()

if args.removeMatAfterSimRotSteps > args.numRotation and args.simplifyObjs == 0:
	print ('==> Error: You cannot have removeMatAfterSimRotSteps greater or equal to numRotation')
	exit()



dataGenSubProc = subprocess.Popen('python3.6 main.py --datasetRawPath {0:s} --category {1:s} --testCategory {2:s} --numShape {3:s} --numRotation {4:s} --resolutions {5:s} --pTrain {6:s} --simultaneousRotation {7:s} --removeMatAfterSimRotSteps {8:s} --simplifyObjs {9:s} --generateStimuli {10:s} --bayesOpt {11:s} --genStats_Plots {12:s} --finetunePretrainedModel {13:s} --BoNumRunStart {14:s} --BoNumRunEnd {15:s} --BOxi {16:s} --BOLossModel {17:s} --BOLossModelFeatureLayerName {18:s} --BOUnifiedInferencePipelineWith5Shapes {19:s} --BOExcludeGTShapes {20:s} --BOUseEmbeddingNearestNeighborPrior {21:s} --BOResultsPathNote {22:s} --visMakeGPPlots {23:s} 2>&1'.format(args.datasetRawPath, ' '.join(args.category), ' '.join(args.testCategory), str(args.numShape), str(args.numRotation), ''.join(str(args.resolutions)[1:-1].split(',')), str(args.pTrain), str(args.simultaneousRotation), str(args.removeMatAfterSimRotSteps), str(args.simplifyObjs), str(args.generateStimuli), str(args.runBayesianOpt), str(args.genStats_Plots), str(args.finetunePretrainedModel), str(args.BoNumRunStart), str(args.BoNumRunEnd), str(args.Boxi), str(args.BOLossModel), str(args.BOLossModelFeatureLayerName), str(args.BOUnifiedInferencePipelineWith5Shapes), str(args.BOExcludeGTShapes), str(args.BOUseEmbeddingNearestNeighborPrior), str(args.BOResultsPathNote), str(args.visMakeGPPlots)), shell=True).wait()
exit()


if args.expType != '0':
	if args.expType == 'extractFeatures':
		expParams = '--datasetRawPath {0:s} --category {1:s} --numShape {2:s} --numRotation {3:s} --resolutions {4:s} --simultaneousRotation {5:s} --removeMatAfterSimRotSteps {6:s} --sortedDataset {7:s} --expType {8:s} --featureNetwork {9:s} --featLayerNo {10:s} 2>&1'.format(args.datasetRawPath, args.category is not None and ' '.join(args.category) or None, str(args.numShape), str(args.numRotation), ''.join(str(args.resolutions)[1:-1].split(',')), str(args.simultaneousRotation), str(args.removeMatAfterSimRotSteps), str(args.sortedDataset), args.expType, args.featureNetwork, str(args.featLayerNo))
	elif args.expType == 'forwardPass':
		if args.forwardPassType != '0':
			expParams = '--datasetRawPath {0:s} --category {1:s} --numShape {2:s} --numRotation {3:s} --resolutions {4:s} --simultaneousRotation {5:s} --removeMatAfterSimRotSteps {6:s} --sortedDataset {7:s} --expType {8:s} --forwardPassType {9:s} 2>&1'.format(args.datasetRawPath, args.category is not None and ' '.join(args.category) or None, str(args.numShape), str(args.numRotation), ''.join(str(args.resolutions)[1:-1].split(',')), str(args.simultaneousRotation), str(args.removeMatAfterSimRotSteps), str(args.sortedDataset), args.expType, args.forwardPassType)
		else:
			print ('==> Error: Please specify the forwardPassType argument')
			exit()
	else:
		print ("==> Error: Please specify a valid value for the expType argument from either of 'forwardPass', 'randomSamples' and 'extractFeatures'")
		exit()
	experimentProc = subprocess.Popen('python3.6 main.py ' + expParams, shell=True).wait()