import argparse, sys, os
from common import mkdirs, savePickle, fileExist, downloadFile

class opts():
	def __init__(self):
		self.parser = argparse.ArgumentParser()
		self.initialized = False

	def initialize(self):
		#Global
		self.parser.add_argument('--datasetRawPath', default='/om/data/public/ShapeNet/ShapeNetCore.v1', help='path to ShapeNet Core v1')
		self.parser.add_argument('--datasetStorePath', default='results/ShapeNetCore.v1', help='path to store all results')
		self.parser.add_argument('--featurePath', default='extractedFeatures/', help='path to store rendering results')
		self.parser.add_argument('--argsDir', type=str, default="experiments", help='A directory name for the storing the arguments')
		self.parser.add_argument('--optionalText', type=str, default="", help='A optional text to be concatenated with the experiment directory path name')
		self.parser.add_argument('--seed', type=int, default=14, help='Fix the random seed. Setting to 0 means using the default random seed')
		self.parser.add_argument('--train', type=int, default=1, help='Whether start training a model or not; 0 means we are going to use the model for test')
		self.parser.add_argument('--flexVerbose', type=int, default=0, choices=[0, 1], help='Verbose FleX output. Useful for debugging FleX')

		# Dataset and rendering
		self.parser.add_argument('--numVPs', type=int, default=20, help='the number of rendered view points')
		self.parser.add_argument('--numLamps', type=int, default=14, help='the number of lamps to be used to shed light on an object during rendering')
		self.parser.add_argument('--lampEnergy', type=float, default=0.5, help='the number of lamps to be used to shed light on an object during rendering')
		self.parser.add_argument('--regularSolids', type=int, default=1, choices=[0, 1], help="Method of proposing camera angles. Set to '1' to use platonic solids. Set to '0' for generating camera angles from a sphere using Fibonacci lattice")
		self.parser.add_argument('--radius', type=float, default=1.22, help='Determines the radius of a sphere on which the cameras are going to be placed')
		self.parser.add_argument('--depthMaxValue', type=float, default=1.725, help='The maximum value of depth range after obtaining raw values in exr files. Will be used for clipping')
		self.parser.add_argument('--depthMinValue', type=float, default=0.69, help='The minimum value of depth range after obtaining raw values in exr files. Will be used for clipping')
		self.parser.add_argument('--findNewMinMax', type=int, default=0, choices=[0, 1], help='Set this to 1 to find new maximum and minimum values for depth renderings')
		self.parser.add_argument('--maxMemory', type=int, default=8000, help='In MBs: maximum amount of memory to be used to chunk the data set and save data files on disk')
		self.parser.add_argument('--pTrain', type=float, default=.9, help='The percentage of the data (3D objects) to be used for training; the rest for test/validation')
		self.parser.add_argument('--fromScratch', type=int, default=1, help='Determines whether the data preparation process should be done from scratch or not')
		self.parser.add_argument('--resolutions', type=int, default=[224], nargs="+", help='The rendering resolutions. Usage: --resolutions 64 124 256 etc')
		self.parser.add_argument('--removeMats', type=int, default=1, help='Determines whether or remove the materials when rendering RGBs or not')
		self.parser.add_argument('--numNewMatColor', type=int, default=0, help='Determines how many times the material colors of a shape should be changed to get new renderings. Set to 0 and each 3D shape will be rendered once, with the original material colors')
		self.parser.add_argument('--depthAndNormalRenderFormat', type=str, default='exr', choices=['png', 'exr'], help='Set the format to save the rendering results')
		self.parser.add_argument('--renderAccuracy', type=int, default=16, choices=[16, 32], help='Set the number of bits per image channel. This is only used when storing the rendering results in EXR format')
		self.parser.add_argument('--renderer', type=str, default="blender", choices=['cycles', 'blender'], help='The rendering engine to be used in Blender [blender|cycles]')
		self.parser.add_argument('--rotLimitDegree', type=float, default=89.9, help='Rotate the 3D shapes withing the range [-rotLimitDegree, +rotLimitDegree] when generating the data. Do not set this argument to more than 180')
		self.parser.add_argument('--allCategories', type=str, default=None, nargs="+", help='All categories: rendering, simulation, Bayesian optimization')
		self.parser.add_argument('--category', type=str, default=None, 
			choices=['airplane', 'trash', 'bag', 'basket', 'tub', 'bunk', 'bench', 'bicycle', 'birdhouse', 'boat', 'bookshelf', 'bottle', 'bowl', 'bus', 'cabinet', 'camera', 'can', 'cap', 'car', 'phone', 'chair', 'clock', 'keyboard', 'washer', 'screen', 'headphone', 'faucet', 'file', 'guitar', 'helmet', 'vase', 'knife', 'lamp', 'laptop', 'speaker', 'mailbox', 'microphone', 'microwave', 'motorcycle', 'mug', 'piano', 'pillow', 'pistol', 'flowerpot', 'printer', 'remote', 'rifle', 'missile', 'skateboard', 'sofa', 'stove', 'table', 'telephone', 'tower', 'train', 'ship', 'machine'], 
			nargs="+", 
			help='The categories to be chosen during experiments: rendering, simulation, Bayesian optimization')
		self.parser.add_argument('--testCategory', type=str, default=['airplane', 'bicycle', 'bus', 'car', 'chair', 'guitar', 'motorcycle', 'pistol', 'rifle', 'table'], 
			choices=['airplane', 'bicycle', 'bus', 'car', 'chair', 'guitar', 'motorcycle', 'pistol', 'rifle', 'table'], 
			nargs="+", 
			help='The test categories from which shapes will be choosen')
		self.parser.add_argument('--numShape', type=int, default=0, help='The number of shapes to be used during rendering from each category')
		self.parser.add_argument('--numRotation', type=int, default=12, help='The number of random rotations of each shape during rendering')
		# self.parser.add_argument('--combineNPArrays', type=int, default=0, choices=[0, 1], help='A flag which determines whether or not stored Numpy arrays, containing rendering results, should be combined together to make a chunks of Numpy arrays and make them ready for reading in LuaTorch')
		self.parser.add_argument('--sortedDataset', type=int, default=0, choices=[0, 1], help='A flag which determines whether or not the order of category data should be kept when making chunks of Numpy arrays')
		self.parser.add_argument('--simultaneousRotation', default=1, type=int, choices=[0, 1], help="A flag which determines whether, in addition to normal rendering with numVPs=12, 20 etc, the shapes should be also rendered with random rotations")
		self.parser.add_argument('--removeMatAfterSimRotSteps', default=8, type=int, help="The probability that defines with what percentage the material of a shape should be removed while rendering with simultaneousRotation=1. The value of this parameter will be set to 10000 if removeMats is set to 1")
		self.parser.add_argument('--simplifyObjs', type=int, default=1, choices=[0, 1], help='Determines whether the mesh simplification algorithm should be run for test shapes and store them on disk')

		#Stimuli
		self.parser.add_argument('--shapeUncertainty', type=int, default=1, choices=[0, 1], help='Whether or not the additional shapes must be chosen using nearest neighbor distance on some neural network embedding space from ShapeNet data set')
		self.parser.add_argument('--shapeUncertaintyFromScratch', type=int, default=0, choices=[0, 1], help='Determines whether the unoccluded uncertainty experiment must be done from scratch or not')
		self.parser.add_argument('--shapeUncertaintyNumExtraShapes', type=int, default=4, choices=[1, 2, 3, 4], help='The number of shapes to be added to the list of ground-truth and distractor shapes for each trial')
		self.parser.add_argument('--shapeUncertaintyNumBinsRotation', type=int, default=-1, help='The number of bins to for discretization of the posterior variables for rotation. Set to -1 to only use the inferred shape posteriors')
		self.parser.add_argument('--shapeUncertaintyNumShapeInferenceBOChains', type=int, default=20, choices=list(range(5, 21)), help='The number of BO chains to be run to infer the unoccluded shape')
		self.parser.add_argument('--shapeUncertaintyNumShapeInferenceBOSteps', type=int, default=40, choices=list(range(20, 81)), help='The number of BO iterations to be run for each BO chain to infer the unoccluded shape')
		self.parser.add_argument('--shapeUncertaintyNNDistanceStartPercentage', type=int, default=0, choices=list(range(40)), help='Determines the percentage of closeness of selected shapes to ground-truth and distractor shapes. Higher values of this parameter results in having easier trials')
		self.parser.add_argument('--shapeUncertaintyEmbeddingModelName', default='alexnet', choices=['vgg', 'vggbn', 'alexnet', 'densenet', 'resnet'], help='The neural network model to be used to obtain shape embeddings. Note that vgg and alexnet get a 2D image as input. In case multiview is set to 1, 3dvae can directly accept an input tensor of with numVPs rgb renderings whereas for other models the embeddings will get concatenated on dimension 0')
		self.parser.add_argument('--shapeUncertaintyEmbeddingFeatureLayerName', default='fc1', choices=['fc1'], help='Name of the layer where we want to chop the model at')
		self.parser.add_argument('--shapeUncertaintyEmbeddingModelPath', default='', help='path to the neural network model to to be used to get nearest neighbor shapes')
		self.parser.add_argument('--maskClothRendering', type=int, default=0, choices=[0, 1], help='Determines whether the rendering of unoccluded ground-truth shape should be used to mask the rendering of draped ground-truth mesh')
		self.parser.add_argument('--silhouetteStimuli', type=int, default=0, choices=[0, 1], help='Determines whether the draped rendering of trials should be silhouettes or not. If 1, the three conditions would be SUU, SSU, UUU')
		self.parser.add_argument('--multiview', type=int, default=0, choices=[0, 1], help='The neural network model to be used to obtain shape embeddings. Note that vgg and alexnet get a 2D image as input but 3dvae takes a tensor of with numVPs rgb renderings as input')
		self.parser.add_argument('--stimuliResultsPath', default='', help='path for the stimuli')
		self.parser.add_argument('--generateStimuli', type=int, default=0, choices=[0, 1], help='Determines whether stimuli set should be generated or not')
		self.parser.add_argument('--numStimuli', type=int, default=120, help='Total number of trials in the stimuli')
		self.parser.add_argument('--numTrainStimuli', type=int, default=20, help='Total number of trials to be generated to train human subjects')
		self.parser.add_argument('--numDistractorShapesPerTrial', type=int, default=1, choices=[1], help='Number of distractor shapes for each trial. For each trial, one shape is always the ground-truth shape. So the total number of shapes for each trial is numDistractorShapesPerTrial + 1. Currently, some of the test categories such as bicycle do not allow to have more than 3 distractor shapes. Note that all experiments in the paper have been run with the numDistractorShapesPerTrial set to 1 and the code depends on the value set to 1')
		self.parser.add_argument('--withinClassPercentage', type=float, default=0.75, help='Determines the percentage of stimuli shapes to be chosen from the test categories')
		self.parser.add_argument('--numStimuliRotation', type=int, default=1, help='The number of random rotations of each shape during rendering for the stimuli set')
		self.parser.add_argument('--seedStimuli', type=int, default=0, help="Fix the random seed for generating the stimuli set. Setting to 0 will not change the random seed and will instead use the global random seed set by the 'seed' parameter")
		self.parser.add_argument('--stimuliFlexConfigPathID', type=int, default=2, choices=[0, 1, 2, 3, 4, 5, 6, 7, 8], help='The number for this argument will be used to determine the path to FleX configuration file when generating the stimuli. See below on how this is done')
		self.parser.add_argument('--flexConfigPathID', type=int, default=2, choices=[0, 1, 2, 3, 4, 5, 6, 7, 8], help='The number for this argument will be used to determine the path to FleX configuration file to be used when doing inference. See below on how this is done')
		self.parser.add_argument('--flexConfigPath', default='', help='Determines the path to a FleX configuration file to be used when doing inference')
		self.parser.add_argument('--stimuliFlexConfigPath', default='', help='Determines the path to a FleX configuration file to be used when generating the stimuli')
		self.parser.add_argument('--fixedRotation', type=int, default=0, choices=[0, 1], help='Whether or not the distractor shapes should have the same rotations')
		self.parser.add_argument('--useQuat', type=int, default=0, choices=[0, 1], help='Dictates whether or not use Quaternions')

		# Model performance correlation with human performance data
		self.parser.add_argument('--behavioralDataPath', default='behavioralData', help='path to the results obtained from the experiments done on mechanical turk on humans')
		self.parser.add_argument('--stimulousPerformanceComparisonCriterion', default='accuracy', choices=['accuracy', 'l1', 'corr'], help='The criterion to be used to compare model performance data for each BO Iteration (no BO Iteration for pretrained models). accuracy will refer to the classification accuracy of each trial across all runs')


		#Modeling
		#Bayesian Optimization
		self.parser.add_argument('--bayesOpt', type=int, default=1, choices=[0, 1], help='Setting to 1 will run inference using Bayesian optimization to do shape/pose estimation')
		self.parser.add_argument('--saveGPState', type=int, default=1, choices=[0, 1], help='Determines whether the GP state should be saved on disk or not. Once saved, will be read next times unless physically removed from disk')
		self.parser.add_argument('--BOUnifiedInferencePipeline', type=int, default=1, choices=[0, 1], help='Set to 1 to have a similar occluded shape inference pipeline for all 3 tasks. Setting to 0 will cause OOU and UUU to have the same occluded shape inference pipeline and UUO a different one')
		self.parser.add_argument('--BOUnifiedInferencePipelineWith5Shapes', type=int, default=1, choices=[0, 1], help='Determines whether the computational model should have access to either 5 or 10 shapes. You may exclude the ground truth shapes by setting BOExcludeGTShapes to 1 to allow the inference pipeline to have access to either 4 nearest neighbors or 8 nearest neighbors (assuming BOUnifiedInferencePipelineWith5Shapes=1)')
		self.parser.add_argument('--BOExemptOOUUnifiedInferencePipeline', type=int, default=1, choices=[0, 1], help='Set to 1 to exempt OOU from unified inference pipeline')
		self.parser.add_argument('--BOFeedbackInferencePipeline', type=int, default=0, choices=[0, 1], help='Set to 1 to incorporate the feedback that the model gets by explaining the study items with the target item')
		self.parser.add_argument('--BOUseEmbeddingShapePosterior', type=int, default=1, choices=[0, 1], help='This parameter determines how to compute shape posterior. It determines whether to use the distance of the nearest neighbor shape embeddings to the source shape or infer shape posterior using BO')
		self.parser.add_argument('--BOUseEmbeddingNearestNeighborPrior', type=int, default=3, choices=[0, 1, 2, 3, 4, 5], help='Didcates how prior for nearest neighbors should be constructed. Ideally, this should be set to 1 but since loss function embeddings do not have a fixed reference points, the obtained numbers are intuitively wrong, hence set this to 3 or 4 for ranked priors with different increments')
		self.parser.add_argument('--BOExcludeGTShapes', type=int, default=1, choices=[0, 1], help='Set to 1 to exclude the ground-truth shapes for both the target shape and the distractor during second phase of inference')
		self.parser.add_argument('--BOOnlyGTShapes', type=int, default=0, choices=[0, 1], help='this overrides what BOExcludeGTShapes does by only using the GT shapes in the experiments')
		self.parser.add_argument('--BOResultsPathNote', default='Ranked_Prior_200', help='an arbitrary note to be appended to the end of BOResults Path directory')
		self.parser.add_argument('--BOResultsPath', default='', help='path to the results obtained from the Bayesian optimization process')
		self.parser.add_argument('--BOAcqFunc', type=str, default='ei', choices=['ei', 'ucb', 'poi'], help='Determines the choice of acquisition function for Bayesian Optimization')
		self.parser.add_argument('--BOxi', type=float, default=330.0, choices=[330.0, 600.0, 4200.0], help='Trades-off exploration and exploitation for BOAcqFunc=ei. The higher the number, the more exploration; 0.0 is more exploitation. Set to 330.0, 600.0 and 4200.0 if BOLossModel is set to either AlexNet fc1, baseline (pixel loss) or AlexNet conv3 respectively')
		self.parser.add_argument('--BOlength_scale', type=float, default=1.0, help='The scikit-learn Matern/RBF Gaussian process kernel parameter. Higher values means the observed points have a higher rich to explain other observations hence lower prediction variance')
		self.parser.add_argument('--BOnu', type=float, default=1.5, choices=[1.5, 2.5], help='The scikit-learn Matern Gaussian process kernel parameter. 1.5 for once-differentiable functions and 2.5 for twice-differentiable. Setting to higher values will result in more uncertainty')
		self.parser.add_argument('--BoOptimizerRestarts', type=int, default=55, help='The scikit-learn Gaussian process optimizer parameter to determine the number of steps of the optimizer')
		self.parser.add_argument('--BoNumRuns', type=int, default=32, help='The total number of BO runs for each trial. This could be interpreted as the number of independent subjects who do the experiment')
		self.parser.add_argument('--BONumRunCustome', type=int, default=1, choices=[0, 1], help='Determines whether BO runs should start from a desired number that user defines through BoNumRunStart and BoNumRunEnd or not. Setting it to 0 will start BO runs from 0 up to BoNumRuns')
		self.parser.add_argument('--BoNumRunStart', type=int, default=-1, help='The starting BO run number')
		self.parser.add_argument('--BoNumRunEnd', type=int, default=-1, help='The ending BO run number')
		self.parser.add_argument('--BOIters', type=int, default=200, help='The total number of BO iterations/proposals for each BO run')
		self.parser.add_argument('--BOInitPriorRuns', type=int, default=2, help='The number of samples to be draw to initialize BO prior')
		self.parser.add_argument('--BOLossFunction', default='l1', choices=['l1, l2', 'corr'], help='Determines the loss function to be used when comparing the results [L1, L2, Pearson Correlation]')
		self.parser.add_argument('--genStats_Plots', type=int, default=0, choices=[0, 1], help='Flag to calculate of the statistics for the results of BO and generate plots')

		#Post-PNAS Occluded Image Recognition
		self.parser.add_argument('--OTaskSpecialProcess', type=int, default=0, choices=[0, 1], help='Set to 1 to use a different pipeline for processing O stimuli (OTask refers to using the rendering of GT object draped with cloth from the OOU trials)')
		self.parser.add_argument('--OTaskPriorType', type=int, default=2, choices=[0, 1, 2, 3], help='Set this to 0 to use 72 nearest neighbor shapes for the ground-truth object in each category without any structure (see 1). Set it to 1 to sort the shape prior in 0 with distance to the occluded GT object. Set to 2 to use uniform prior. Set to 3 to do optimization in PCA space to find the shape')
		
		# BO likelihood
		self.parser.add_argument('--BOLossModel', default='alexnet', choices=['baseline', 'alexnet', 'vgg', 'vggbn', 'densenet', 'resnet', 'cornet_s', 'midas'], help='Name of the model to be loaded and used for computing Bayesian Optimization loss function')
		self.parser.add_argument('--BOLossModelSecond', default='none', choices=['none', 'baseline', 'alexnet', 'vgg', 'vggbn', 'densenet', 'resnet', 'cornet_s', 'midas'], help='Name of the model to be loaded and used for computing Bayesian Optimization loss function')
		self.parser.add_argument('--BOLossModelPath', default='', help='path to the neural network model to be used for computing loss function for BO')
		self.parser.add_argument('--BOLossModelFeature', type=int, default=1, choices=[0, 1], help='Determine the layer number from which features are going to be extracted')
		self.parser.add_argument('--BOLossModelNormalize', type=int, default=0, choices=[0, 1], help='Whether or not the input images should be normalized by ImageNet mean and std')
		self.parser.add_argument('--BOLossModelFeatureLayerName', default='fc1', choices=['lastPool', 'fc1', 'fc2', 'conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'layer1', 'layer2', 'layer3', 'layer4', 'fc11', 'fc12', 'v1Linear', 'v1Conv', 'v2Linear', 'v2Conv', 'v4Linear', 'v4Conv', 'itLinear', 'itConv', 'decoder'], help='Name of the layer where we want to chop the model at')
		self.parser.add_argument('--BOLossModelFeatureLayerNameSecond', default='conv3', choices=['lastPool', 'fc1', 'fc2', 'conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'layer1', 'layer2', 'layer3', 'layer4', 'fc11', 'fc12', 'v1Linear', 'v1Conv', 'v2Linear', 'v2Conv', 'v4Linear', 'v4Conv', 'itLinear', 'itConv', 'decoder'], help='Name of the layer where we want to chop the model at')
		self.parser.add_argument('--BOLossModelSecondDivider', type=int, default=60, help='Divide the loss coming from the second model by this value')
		self.parser.add_argument('--BOLossModelRandomNetwork', type=int, default=0, choices=[0, 1], help='Set to 1 for random initialization of models')

		# Training pretrained neural networks
		self.parser.add_argument('--finetunePretrainedModel', type=int, default=0, choices=[0, 1], help='Set to 1 to fine-tune a pre-trained model')
		self.parser.add_argument('--finetuneForceTraining', type=int, default=0, choices=[0, 1], help='Set to 1 to start fine-tuning a pre-trained model from scratch')
		self.parser.add_argument('--finetuneTrainLastFCLayer', type=int, default=1, choices=[0, 1], help='Set 1 to train the weights of the last FC layer (currently working for AlexNet only)')
		self.parser.add_argument('--finetuneNormalizeInput', type=int, default=0, choices=[0, 1], help='Whether or not normalize the input images')
		self.parser.add_argument('--finetuneNumModels', type=int, default=32, choices=list(range(1, 37)), help='Number of times that the random seed changes and the model starts training -- To be used to get error bars in the final result')
		self.parser.add_argument('--finetuneRunForwardPassAfterTraining', type=int, default=1, choices=[0, 1], help='Set to 1 to start running experiments with the fine-tuned model')
		self.parser.add_argument('--finetunedModelPath', type=str, default='', help='The fine-tuned model path')
		self.parser.add_argument('--finetunePretrainedModelName', type=str, default='alexnet', choices=['alexnet', 'vgg', 'resnet', 'resnet50-sin-in', 'resnet50-sin-in_in', 'cornet_s', 'midas'], help='Name of the model to be fine-tuned on training data and used for experiments')
		self.parser.add_argument('--finetunePretrainedModelLayer', type=str, default='fc1', choices=['all', 'conv5', 'fc1', 'fc2', 'layer1', 'layer2', 'layer3', 'layer4', 'fc11', 'fc12', 'v1Linear', 'v1Conv', 'v2Linear', 'v2Conv', 'v4Linear', 'v4Conv', 'itLinear', 'itConv', 'decoder'], help='The layer of the pretrained model to be used for adding the decoding layer. layer1, layer2, layer3, layer4, fc11, fc12 are all used for the ResNet model. decoder is used for CORnet-S')
		self.parser.add_argument('--finetuneNumStimuliSets', type=int, default=52, choices=list(range(3, 53)), help='The number of stimuli set to be generated for fine-tuning neural networks. The last two sets are reserved for testing. Two of these sets belong to a test set that is going to be used during training')
		self.parser.add_argument('--finetuneNumTrainStimuliSets', type=int, default=8, choices=list(range(1, 39)), help='The number of stimuli set to be generated for fine-tuning neural networks. The last two sets are reserved for testing')
		self.parser.add_argument('--finetuneStimuliReady', type=int, default=1, choices=[0, 1], help='Set to 1 if you know that the fine-tuning stimuli has been generated already; 0 otherwise. Setting to 0 will generate the stimuli again')
		self.parser.add_argument('--finetuneBatchSize', type=int, default=8, choices=list(range(1, 32)), help='The number of stimuli set to be generated for fine-tuning neural networks. The last two batches are used for testing')
		self.parser.add_argument('--finetuneDecoderDimension', type=int, default=120, help='Dimension of decoding layer output')
		self.parser.add_argument('--finetuneTripletLossMargin', type=float, default=2., help='The loss margin to be use for the triplet loss')
		self.parser.add_argument('--finetuneNumConcurrentBatchReads', type=int, default=2, choices=list(range(1, 4)), help='The number of batches to be loaded concurrent to training')
		self.parser.add_argument('--finetuneLR', type=float, default=0.0000012, help='Learning rate for fine-tuning')
		self.parser.add_argument('--finetuneNumEpochs', type=int, default=200, help='Number of epochs to fine-tune the model')
		self.parser.add_argument('--finetuneFromEpoch', type=int, default=-1, choices=[-1]+list(range(1, 200)), help='The epoch from which the saved model will be loaded')
		self.parser.add_argument('--finetuneSaveModelOnLastEpochOnly', type=int, default=1, choices=[0, 1], help='Set to 1 to only save the model for the last epoch')

		# Metric Learning
		self.parser.add_argument('--metricLearning', type=int, default=0, choices=[0, 1], help='Set to 1 to perform metric learning')

		# Visualization
		self.parser.add_argument('--visGtOrDis', type=str, default='gt', choices=['gt', 'dis', 'both'], help='Determines whether the visualizations (e.g. cloth draping video and GP plots) should be created for either ground-truth or distractor shape')
		self.parser.add_argument('--visTrialNum', type=int, default=40, choices=[-1]+list(range(120)), help='The trial number for which all steps of the simulation will be saved. Set to -1 to do this for all trials')
		self.parser.add_argument('--visTrialCondition', type=str, default='uuo', choices=['uuo, oou', 'both'], help='The trial number for which all steps of the simulation will be saved. Set to -1 to do this for all trials')
		self.parser.add_argument('--visSaveClothPerSimStep', type=int, default=0, choices=[0, 1], help='Dictates whether the cloth should be saved to disk for each iteration of the solver')
		self.parser.add_argument('--visVideoMaxNumSimSteps', type=int, default=155, help='The number of steps the simulation is done for')
		self.parser.add_argument('--visVideoResolution', type=int, default=768, help='The number of steps the simulation is done for')
		self.parser.add_argument('--visVideoFixRotX', type=float, default=0.5, help='Fix the value of rotation on axis X. BO will not propose any values for that axis if the number if in the range [0, 1] corresponding to [-rotLimitDegree, +rotLimitDegree] and if visSaveClothPerSimStep is set to 1')
		self.parser.add_argument('--visVideoFixRotY', type=float, default=0.5, help='Fix the value of rotation on axis Y. BO will not propose any values for that axis if the number if in the range [0, 1] corresponding to [-rotLimitDegree, +rotLimitDegree] and if visSaveClothPerSimStep is set to 1')
		self.parser.add_argument('--visVideoFixRotZ', type=float, default=0.5, help='Fix the value of rotation on axis Z. BO will not propose any values for that axis if the number if in the range [0, 1] corresponding to [-rotLimitDegree, +rotLimitDegree] and if visSaveClothPerSimStep is set to 1')
		self.parser.add_argument('--visMakeGPPlots', type=int, default=0, choices=[0, 1], help='Set to 1 to make Gaussian process plots after getting the posterior over mean and covariance of the GP')
		self.parser.add_argument('--visMakeGPPlotsBestParamsOnly', type=int, default=0, choices=[0, 1], help='Set to 1 to only generate plots for the solution that achieves the best explanation for a stimulous up to observation X')
		self.parser.add_argument('--visMakeGPPlotsOneVariableAtATime', type=int, default=0, choices=[0, 1], help='Set to 1 to make Gaussian process plots for each variable while fixing the others to the GT value')
		self.parser.add_argument('--visMakeGPPlotsOnlyPrediction', type=int, default=0, choices=[0, 1], help='Set to 1 to only use the predictions of Gaussian processes and not evaluate the true cost function')
		self.parser.add_argument('--visMakeGPPlotsOnlyGTLossCurve', type=int, default=0, choices=[0, 1], help='Set to 1 to only show the ground truth loss curve')
		self.parser.add_argument('--visMakeGPPlotsNumObs', type=list, default=[15, 30, 60, 90, 120], help='The maximum number of observations to fit the GP. Set to -1 to generate plots for all number of observations (BOIters)')
		self.parser.add_argument('--visMakeGPPlotsNumObsOneVar', type=list, default=[2, 5, 10, 15], help='The maximum number of observations to fit the GP when creating plots for one-variable case. Set to -1 to generate plots for all number of observations (BOIters)')
		self.parser.add_argument('--visMakeGPPlotsNumUnobservedPoints', type=int, default=200, help='The number of points for which we query the Guassian Processes for, for each parameter')
		self.parser.add_argument('--visMakeGPPlotsNumLikelihoodEvals', type=int, default=4, help='The number of times the model evaluates the likelihood function for unobserved parameter sets. This is only done for UUO and OOU as the physics simulation is not deterministic')
		self.parser.add_argument('--visMakeGPPlotsOneVarGPFittingNumObs', type=int, default=15, choices=list(range(1, 200)), help='The maximum number of observations to consider for generating GP plots fitted one-variable-at-a-time. Set to 0 to disable making these plots')

		#Experiments/Visualization
		self.parser.add_argument('--expType', type=str, default='0', choices=['0', 'forwardPass', 'randomSamples', 'extractFeatures'], help='Type of experiment')
		self.parser.add_argument('--forwardPassType', type=str, default='0', choices=['0', 'userData', 'reconstruction'], help='Type of forward pass')
		self.parser.add_argument('--featureNetwork', type=str, default='3dvae', choices=['3dvae', 'alexnet'], help='Type of forward pass')
		self.parser.add_argument('--featLayerNo', type=int, default=1, help='Determines the layer number from which features will be extracted and stored on disk')

		
		self.initialized = True

	def parse(self):
		if not self.initialized:
			self.initialize()
		self.opt = self.parser.parse_args()

		# Some manual intervention!
		if self.opt.datasetRawPath[len(self.opt.datasetRawPath)-1] != '/':
			self.opt.datasetRawPath = self.opt.datasetRawPath + '/'

		if self.opt.datasetStorePath[len(self.opt.datasetStorePath)-1] != '/':
			self.opt.datasetStorePath = self.opt.datasetStorePath + '/'

		if self.opt.featurePath[len(self.opt.featurePath)-1] != '/':
			self.opt.featurePath = self.opt.featurePath + '/'

		if self.opt.behavioralDataPath[len(self.opt.behavioralDataPath)-1] != '/':
			self.opt.behavioralDataPath = self.opt.behavioralDataPath + '/'

		if self.opt.category is None:
			self.opt.category = ['airplane', 'trash', 'bag', 'basket', 'tub', 'bunk', 'bench', 'bicycle', 'birdhouse', 'boat', 'bookshelf', 'bottle', 'bowl', 'bus', 'cabinet', 'camera', 'can', 'cap', 'car', 'phone', 'chair', 'clock', 'keyboard', 'washer', 'screen', 'headphone', 'faucet', 'file', 'guitar', 'helmet', 'vase', 'knife', 'lamp', 'laptop', 'speaker', 'mailbox', 'microphone', 'microwave', 'motorcycle', 'mug', 'piano', 'pillow', 'pistol', 'flowerpot', 'printer', 'remote', 'rifle', 'missile', 'skateboard', 'sofa', 'stove', 'table', 'telephone', 'tower', 'train', 'ship', 'machine']

		if self.opt.allCategories is None:
			self.opt.allCategories = ['airplane', 'trash', 'bag', 'basket', 'tub', 'bunk', 'bench', 'bicycle', 'birdhouse', 'boat', 'bookshelf', 'bottle', 'bowl', 'bus', 'cabinet', 'camera', 'can', 'cap', 'car', 'phone', 'chair', 'clock', 'keyboard', 'washer', 'screen', 'headphone', 'faucet', 'file', 'guitar', 'helmet', 'vase', 'knife', 'lamp', 'laptop', 'speaker', 'mailbox', 'microphone', 'microwave', 'motorcycle', 'mug', 'piano', 'pillow', 'pistol', 'flowerpot', 'printer', 'remote', 'rifle', 'missile', 'skateboard', 'sofa', 'stove', 'table', 'telephone', 'tower', 'train', 'ship', 'machine']

		if self.opt.numShape < 0:
			print ('==> Please choose a number greater than or equal to 0 for numShape. Setting numShapes to 0 will result in rendering all shapes in a specific category')
			sys.exit()

		if self.opt.numRotation < 0:
			print ('==> Please choose a number greater or equal to 0 for numRotations') 
			sys.exit()

		if self.opt.radius != 1.22:
			print ("\n\n==> IMPORTANT MESSAGE: You have changed the diameter of surface of the sphere to the origin on which cameras will be located. Make sure you run a small experiment with findNewMinMax equal to 'True' to adjust the maximum and minimum depth values. \nThen set depthMinValue and depthMaxValue to the new numbers before rendering your entire data set\n\n")
			sys.exit()

		if self.opt.simultaneousRotation == 1 and self.opt.numRotation == 0:
			print ('==> Error: You need to specify a number of rotations more than 0 if you want to get simultaneous rotation renderings alongside multi-view renderings')
			exit()

		if self.opt.rotLimitDegree > 89.9 or self.opt.rotLimitDegree < 0:
			print ('==> Error: The rotLimitDegree parameter must be in the range [0, 89.9]')
			exit()

		if self.opt.numDistractorShapesPerTrial <= 0:
			print ("==> Error: The value of the argument 'numDistractorShapesPerTrial' cannot be less than 1")
			exit()

		self.opt.shapeUncertaintyNNDistanceStartPercentage = self.opt.shapeUncertaintyNNDistanceStartPercentage/100

		# if self.opt.shapeUncertainty:
		# 	print ('==> Shape uncertainty specific parameters: BOIters = 200, BOxi = 0.5 instead of 140 and 3.5 respectively')
		# 	self.opt.BOIters = 180
		# 	self.opt.BOxi = 1.0

		flexConfigPaths = ['FleXConfigs/flexConfig.yml', \
							'FleXConfigs/flexConfig-Silk.yml', \
							'FleXConfigs/flexConfig-UnnormalizedShapes.yml', \
							'FleXConfigs/flexConfig-UnnormalizedShapes-Silk.yml', \
							'FleXConfigs/flexConfig-UnnormalizedShapes-OldParameters.yml', \
							'FleXConfigs/flexConfig-UnnormalizedShapes-LowQuality.yml', \
							'FleXConfigs/flexConfig-UnnormalizedShapes-LowClothRes.yml', \
							'FleXConfigs/flexConfig-UnnormalizedShapes-LowQualityLowClothRes.yml', \
							'FleXConfigs/flexConfig-UnnormalizedShapes-RandomSimQualRandomClothRes.yml']
		
		self.opt.flexConfigPath = os.getcwd() + "/" + flexConfigPaths[self.opt.flexConfigPathID]
		self.opt.stimuliFlexConfigPath = os.getcwd() + "/" + flexConfigPaths[self.opt.stimuliFlexConfigPathID]

		# Get list of more model URLs from here:
		# https://github.com/Cadene/pretrained-models.pytorch
		modelUrls = {
	    'densenet201': 'https://download.pytorch.org/models/densenet201-c1103571.pth',
	    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
	    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
	    'resnet50-sin-in': 'https://download.pytorch.org/models/resnet50-sin-in.pth', # https://openreview.net/forum?id=Bygh9j09KX
		'resnet50-sin-in_in': 'https://bitbucket.org/robert_geirhos/texture-vs-shape-pretrained-models/raw/60b770e128fffcbd8562a3ab3546c1a735432d03/resnet50_finetune_60_epochs_lr_decay_after_30_start_resnet50_train_45_epochs_combined_IN_SF-ca06340c.pth.tar',
	    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
	    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
	    'cornet_s': 'https://s3.amazonaws.com/cornet-models/cornet_s-1d3f7974.pth',
	    'midas': 'https://github.com/intel-isl/MiDaS/releases/download/v2/model.pt'}
		self.opt.modelUrls = modelUrls


		# self.opt.datasetStorePath = self.opt.datasetStorePath + (self.opt.numDistractorShapesPerTrial != 1 and (str(self.opt.numDistractorShapesPerTrial+1) + 'ShapesPerTrial') or '')
		# self.opt.datasetStorePath = self.opt.datasetStorePath + (not self.opt.shapeUncertainty and ((str(self.opt.numDistractorShapesPerTrial+1) + 'ShapesPerTrial/')) or 'ShapeUncertainty')
		# self.opt.datasetStorePath = self.opt.datasetStorePath + str(self.opt.numDistractorShapesPerTrial+1) + 'ShapesPerTrial/'
		mkdirs(self.opt.datasetStorePath)

		self.opt.depthAndNormalRenderFormat = self.opt.depthAndNormalRenderFormat.lower()
		self.opt.BOLossFunction = self.opt.BOLossFunction.lower()
		self.opt.renderer = self.opt.renderer.lower()

		# self.opt.dataResideOnOM2 = self.opt.dataResideOnOM2 == 1 and True or False
		self.opt.regularSolids = self.opt.regularSolids == 1 and True or False
		# TODO remove the line below and remove self.opt.blenderExrBug argument after exr reading/storing has been fixed
		self.opt.blenderExrBug = self.opt.depthAndNormalRenderFormat == 'exr' and True or False
		self.opt.flexVerbose = self.opt.flexVerbose == 1 and True or False
		self.opt.removeMats = self.opt.removeMats == 1 and True or False
		self.opt.findNewMinMax = self.opt.findNewMinMax == 1 and True or False
		# self.opt.combineNPArrays = self.opt.combineNPArrays == 1 and True or False
		self.opt.sortedDataset = self.opt.sortedDataset == 1 and True or False
		self.opt.simultaneousRotation = self.opt.simultaneousRotation == 1 and True or False
		self.opt.simplifyObjs = self.opt.simplifyObjs == 1 and True or False
		self.opt.bayesOpt = self.opt.bayesOpt == 1 and True or False
		self.opt.saveGPState = self.opt.saveGPState == 1 and True or False
		self.opt.genStats_Plots = self.opt.genStats_Plots == 1 and True or False
		self.opt.generateStimuli = self.opt.generateStimuli == 1 and True or False
		self.opt.train = self.opt.train == 1 and True or False
		self.opt.useQuat = self.opt.useQuat == 1 and True or False
		self.opt.BOLossModelFeature = self.opt.BOLossModelFeature == 1 and True or False
		self.opt.BOLossModelNormalize = self.opt.BOLossModelNormalize == 1 and True or False
		self.opt.BOLossModelRandomNetwork = self.opt.BOLossModelRandomNetwork == 1 and True or False
		self.opt.BOUnifiedInferencePipeline = self.opt.BOUnifiedInferencePipeline == 1 and True or False
		self.opt.BOUnifiedInferencePipelineWith5Shapes = self.opt.BOUnifiedInferencePipelineWith5Shapes == 1 and True or False
		self.opt.BOExemptOOUUnifiedInferencePipeline = self.opt.BOExemptOOUUnifiedInferencePipeline == 1 and True or False
		self.opt.BOFeedbackInferencePipeline = self.opt.BOFeedbackInferencePipeline == 1 and True or False
		self.opt.BOUseEmbeddingShapePosterior = self.opt.BOUseEmbeddingShapePosterior == 1 and True or False
		self.opt.BOExcludeGTShapes = self.opt.BOExcludeGTShapes == 1 and True or False
		self.opt.BOOnlyGTShapes = self.opt.BOOnlyGTShapes == 1 and True or False
		self.opt.shapeUncertainty = self.opt.shapeUncertainty == 1 and True or False
		self.opt.multiview = self.opt.multiview == 1 and True or False
		self.opt.silhouetteStimuli = self.opt.silhouetteStimuli == 1 and True or False
		self.opt.maskClothRendering = self.opt.maskClothRendering == 1 and True or False
		self.opt.finetunePretrainedModel = self.opt.finetunePretrainedModel == 1 and True or False
		self.opt.finetuneTrainLastFCLayer = self.opt.finetuneTrainLastFCLayer == 1 and True or False
		self.opt.finetuneNormalizeInput = self.opt.finetuneNormalizeInput = 1 and True or False
		self.opt.finetuneForceTraining = self.opt.finetuneForceTraining == 1 and True or False
		self.opt.finetuneRunForwardPassAfterTraining = self.opt.finetuneRunForwardPassAfterTraining == 1 and True or False
		self.opt.finetuneStimuliReady = self.opt.finetuneStimuliReady == 1 and True or False
		self.opt.finetuneSaveModelOnLastEpochOnly = self.opt.finetuneSaveModelOnLastEpochOnly == 1 and True or False
		self.opt.metricLearning == self.opt.metricLearning == 1 and True or False
		self.opt.visSaveClothPerSimStep = self.opt.visSaveClothPerSimStep == 1 and True or False
		self.opt.visMakeGPPlots = self.opt.visMakeGPPlots == 1 and True or False
		self.opt.visMakeGPPlotsBestParamsOnly = self.opt.visMakeGPPlotsBestParamsOnly == 1 and True or False
		self.opt.visMakeGPPlotsOneVariableAtATime = self.opt.visMakeGPPlotsOneVariableAtATime == 1 and True or False
		self.opt.visMakeGPPlotsOnlyPrediction = self.opt.visMakeGPPlotsOnlyPrediction == 1 and True or False
		self.opt.visMakeGPPlotsOnlyGTLossCurve = self.opt.visMakeGPPlotsOnlyGTLossCurve == 1 and True or False

		if (self.opt.BoNumRunStart < 0 and self.opt.BoNumRunEnd >= 0) or (self.opt.BoNumRunStart >= 0 and self.opt.BoNumRunEnd < 1) or self.opt.BoNumRunStart > self.opt.BoNumRunEnd:
			print ("==> Error: You cannot have BoNumRunStart and BoNumRunEnd set to incorrect values. Either set both to -1 or set BoNumRunStart to 0 or higher and BoNumRunStart to 1 or higher")
			exit()
		elif self.opt.BoNumRunStart < self.opt.BoNumRunEnd:
			self.opt.BONumRunCustome = True
		else:
			self.opt.BONumRunCustome = False

		if self.opt.finetunePretrainedModel:
			# self.opt.finetuneNumTrainStimuliSets = self.opt.finetuneNumStimuliSets-2 # The number of stimuli sets to be used for fine-tuning neural networks
			self.opt.finetuneWeightDecay = self.opt.finetuneNumTrainStimuliSets <= 2 and 0.002 or self.opt.finetuneNumTrainStimuliSets == 8 and 0.0018 or self.opt.finetuneNumTrainStimuliSets == 18 and 0.0015 or self.opt.finetuneNumTrainStimuliSets >= 28 and 0.001
			if 'resnet' in self.opt.finetunePretrainedModelName or 'cornet' in self.opt.finetunePretrainedModelName:
				self.opt.finetuneWeightDecay = self.opt.finetuneWeightDecay*10
			self.opt.shapeUncertainty = True
			self.opt.simplifyObjs = True
			# self.opt.finetunedModelPath = self.opt.datasetStorePath + '/fineTunedModels/stimuliSetsUsed-' + str(self.opt.finetuneNumTrainStimuliSets) + '/batchSize-' + str(self.opt.finetuneBatchSize) + '/' + self.opt.finetunePretrainedModelName + '/'
			self.opt.finetunedModelPath = self.opt.datasetStorePath + '/fineTunedModels/stimuliSetsUsed-' + str(self.opt.finetuneNumTrainStimuliSets) + '/batchSize-' + str(self.opt.finetuneBatchSize) + '/weightDecay{0:.4f}'.format(self.opt.finetuneWeightDecay) + '/LR{0:.9f}'.format(self.opt.finetuneLR) + '/' + self.opt.finetunePretrainedModelName + '/trainLastLayerWeights-{0:s}'.format(str(int(self.opt.finetuneTrainLastFCLayer))) + '/'
			self.opt.bayesOpt = self.opt.finetuneRunForwardPassAfterTraining
			self.opt.finetuneWarmupModel = 'resnet' in self.opt.finetunePretrainedModelName and 1 or 'vggbn' in self.opt.finetunePretrainedModelName and 1 or 0 # Set to 1 for models with batch normalization


		if self.opt.visSaveClothPerSimStep:
			self.opt.shapeUncertainty = False

		if self.opt.visSaveClothPerSimStep and self.opt.visMakeGPPlots:
			print ("==> You cannot have both visSaveClothPerSimStep and visMakeGPPlots arguments set to 1. Please set either of them to 0 and re-run the code\n")
			exit()

		if self.opt.shapeUncertainty or self.opt.bayesOpt or self.opt.genStats_Plots:
			self.opt.simplifyObjs = True

		if self.opt.shapeUncertainty and self.opt.bayesOpt:
			import time
			print ('\n\n\n\n\n\n==> ERROR: Fix the bug related to simplifyObjs=True argument when having bayesOpt and shapeUncertainty arguments set to True. You have 10 seconds to exit\n\n\n\n\n\n')
			time.sleep(1.2)
			# exit()

		if self.opt.maskClothRendering and self.opt.silhouetteStimuli:
			print ("==> Error: You cannot have both maskClothRendering and silhouetteStimuli set to 'True'")
			exit()

		if self.opt.generateStimuli and self.opt.finetunePretrainedModel:
			print ('==> You cannot have both generateStimuli and finetunePretrainedModel set to True. Please set one to False')
			exit()

		self.opt.stimuliResultsPath = os.getcwd() + "/" + self.opt.datasetStorePath + '/stimuli-rotLimitDegree-' + str(self.opt.rotLimitDegree) + '-' + str(self.opt.BOIters) + 'BOIterations' + (self.opt.fixedRotation and '-Fixed_Rotation' or '')
		self.opt.stimuliResultsPath += self.opt.maskClothRendering and '-ConvexHull' or ''
		self.opt.stimuliResultsPath += self.opt.silhouetteStimuli and '-Silhouette' or ''
		self.opt.stimuliResultsPath += (self.opt.flexConfigPathID == 5 and '-LowSimQual' or self.opt.flexConfigPathID == 6 and '-LowClothRes' or self.opt.flexConfigPathID == 7 and '-LowSimQualLowClothRes' or self.opt.flexConfigPathID == 8 and '-RandomSimQualRandomClothRes' or '')
		# self.opt.stimuliResultsPath += '-AlexNet-fc1-EmbeddingShapePosterior-UnifiedPipeline-4Shapes-Ranked_Prior_200' # any arbitrary ending of the experiment directory name
		arbitraryExpDirEnding = ""
		if self.opt.BOLossModel.lower() == 'baseline':
			arbitraryExpDirEnding += "-Pixel_loss"
		elif self.opt.BOLossModel.lower() != 'baseline':
			if self.opt.BOLossModel.lower() == 'alexnet':
				arbitraryExpDirEnding += "-AlexNet"
			elif self.opt.BOLossModel.lower() == 'vgg':
				arbitraryExpDirEnding += "-VGG16"
			elif self.opt.BOLossModel.lower() == 'vggbn':
				arbitraryExpDirEnding += "-VGG16_BN"
			elif self.opt.BOLossModel.lower() == 'resnet':
				arbitraryExpDirEnding += "-ResNet50"
			elif self.opt.BOLossModel.lower() == 'cornet_s':
				arbitraryExpDirEnding += "-CORNet_S"
			arbitraryExpDirEnding += "-" + self.opt.BOLossModelFeatureLayerName
		
		if self.opt.BOUseEmbeddingShapePosterior:
			arbitraryExpDirEnding += "-EmbeddingShapePosterior"
		else:
			arbitraryExpDirEnding += "-ProbabilisticShapePosterior"
			print("==> Error: Haven't run any experiments with the probabilisitc inference pipeline for a long time for nearest neighbors. Make sure everything works first and then resume. Exiting")
			exit()

		if self.opt.BOUnifiedInferencePipeline:
			arbitraryExpDirEnding += "-UnifiedPipeline"
			if self.opt.BOUnifiedInferencePipelineWith5Shapes:
				arbitraryExpDirEnding += "-4Shapes" if self.opt.BOExcludeGTShapes and not self.opt.BOOnlyGTShapes else "-5Shapes" if not self.opt.BOExcludeGTShapes and not self.opt.BOOnlyGTShapes else '-1Shape'
			else:
				arbitraryExpDirEnding += "-8Shapes" if self.opt.BOExcludeGTShapes and not self.opt.BOOnlyGTShapes else "-10Shapes" if not self.opt.BOExcludeGTShapes and not self.opt.BOOnlyGTShapes else '-2Shapes'
		else:
			arbitraryExpDirEnding += "-ClassicPipeline"
			print("==> Error: Haven't run any experiments without the unified pipeline for a long time. Make sure everything works first and then resume. Exiting")
			exit()

		if self.opt.BOResultsPathNote != "":
			arbitraryExpDirEnding += "-" + self.opt.BOResultsPathNote
		self.opt.stimuliResultsPath += arbitraryExpDirEnding
		

		# self.parser.add_argument('--BOUnifiedInferencePipeline', type=int, default=1, choices=[0, 1], help='Set to 1 to have a similar occluded shape inference pipeline for all 3 tasks. Setting to 0 will cause OOU and UUU to have the same occluded shape inference pipeline and UUO a different one')
		# self.parser.add_argument('--BOUnifiedInferencePipelineWith5Shapes', type=int, default=1, choices=[0, 1], help='Set to 1 to have a similar occluded shape inference pipeline for all 3 tasks. Setting to 0 will cause OOU and UUU to have the same occluded shape inference pipeline and UUO a different one')
		# self.parser.add_argument('--BOExemptOOUUnifiedInferencePipeline', type=int, default=1, choices=[0, 1], help='Set to 1 to exempt OOU from unified inference pipeline')
		# self.parser.add_argument('--BOFeedbackInferencePipeline', type=int, default=0, choices=[0, 1], help='Set to 1 to incorporate the feedback that the model gets by explaining the study items with the target item')
		# self.parser.add_argument('--BOUseEmbeddingShapePosterior', type=int, default=1, choices=[0, 1], help='This parameter determines how to compute shape posterior. It determines whether to use the distance of the nearest neighbor shape embeddings to the source shape or infer shape posterior using BO')
		# self.parser.add_argument('--BOExcludeGTShapes'


		if self.opt.BOLossModelRandomNetwork:
			self.opt.stimuliResultsPath += "-Random"

		self.opt.BOResultsPath = 'pixelLoss' if self.opt.BOLossModel == 'baseline' \
								else '1stFCLayer' if self.opt.BOLossModelFeatureLayerName == 'fc1' \
								else '2ndFCLayer' if self.opt.BOLossModelFeatureLayerName == 'fc2' \
								else 'conv1' if self.opt.BOLossModelFeatureLayerName == 'conv1' \
								else 'conv2' if self.opt.BOLossModelFeatureLayerName == 'conv2' \
								else 'conv3' if self.opt.BOLossModelFeatureLayerName == 'conv3' \
								else 'conv5' if self.opt.BOLossModelFeatureLayerName == 'conv5' \
								else 'lastPool' if self.opt.BOLossModelFeatureLayerName == 'lastPool' \
								else 'layer4' if self.opt.BOLossModelFeatureLayerName == 'layer4' \
								else 'v1Linear' if self.opt.BOLossModelFeatureLayerName == 'v1Linear' \
								else 'encoder' if self.opt.BOLossModelFeatureLayerName == 'encoder' else False
		if not self.opt.BOResultsPath:
			print ("==> Error: Please make sure to specify the layer name/type in the neural network model correctly")
			raise ValueError

		if self.opt.bayesOpt or self.opt.genStats_Plots:
			if self.opt.BOLossModel != 'baseline' and self.opt.BOLossModel != '3dvae':
				cwd = os.getcwd() + '/'
				pretrainedPath = cwd + 'pretrainedModels'
				if self.opt.BOLossModel == 'midas':
					os.environ["TORCH_HOME"] = cwd + "pretrainedModels/midas"
					pretrainedPath += '/midas'
					mkdirs(pretrainedPath)
				self.opt.BOLossModelPath = pretrainedPath + '/' + self.opt.BOLossModel + '.pth'
				urlModelName = self.opt.BOLossModel == 'alexnet' and 'alexnet' or self.opt.BOLossModel == 'vgg' and 'vgg19' \
								or self.opt.BOLossModel == 'vggbn' and 'vgg19_bn' or self.opt.BOLossModel == 'resnet' and 'resnet101' \
								or self.opt.BOLossModel == 'densenet' and 'densenet201' or self.opt.BOLossModel == 'cornet_s' and 'cornet_s' \
								or self.opt.BOLossModel == 'midas' and 'midas'
				if not fileExist(self.opt.BOLossModelPath):
					mkdirs(pretrainedPath)
					print ("==> Downloading '" + self.opt.BOLossModel + "' pre-trained model of ImageNet")
					downloadFile(url=modelUrls[urlModelName], savePath=self.opt.BOLossModelPath)

		
		if self.opt.finetunePretrainedModel:
			cwd = os.getcwd() + '/'
			pretrainedPath = cwd + 'pretrainedModels'
			if self.opt.finetunePretrainedModelName == 'midas':
					os.environ["TORCH_HOME"] = cwd + "pretrainedModels/midas"
					pretrainedPath += '/midas'
					mkdirs(pretrainedPath)
			self.opt.finetuneModelPath = pretrainedPath + '/' + (self.opt.finetunePretrainedModelName + '.pth' if self.opt.finetunePretrainedModelName != 'resnet50-sin-in_in' else '')
			urlModelName = self.opt.finetunePretrainedModelName == 'alexnet' and 'alexnet' \
							or self.opt.finetunePretrainedModelName == 'vgg' and 'vgg19' \
							or self.opt.finetunePretrainedModelName == 'vggbn' and 'vgg19_bn' \
							or self.opt.finetunePretrainedModelName == 'resnet' and 'resnet101' \
							or self.opt.finetunePretrainedModelName == 'resnet50-sin-in' and 'resnet50-sin-in' \
							or self.opt.finetunePretrainedModelName == 'resnet50-sin-in_in' and 'resnet50-sin-in_in' \
							or self.opt.finetunePretrainedModelName == 'densenet' and 'densenet201' \
							or self.opt.finetunePretrainedModelName == 'cornet_s' and 'cornet_s' \
							or self.opt.finetunePretrainedModelName == 'midas' and 'midas'
			if not fileExist(self.opt.finetuneModelPath):
				mkdirs(pretrainedPath)
				print ("==> Downloading '" + self.opt.finetunePretrainedModelName + "' pre-trained model of ImageNet")
				downloadFile(url=modelUrls[urlModelName], savePath=self.opt.finetuneModelPath)
			self.opt.finetuneWarmedupModelPath = self.opt.finetuneModelPath[:-4] + '-warmedup.pth'


		if self.opt.generateStimuli or self.opt.shapeUncertainty:
			cwd = os.getcwd() + '/'
			pretrainedPath = cwd + 'pretrainedModels'
			self.opt.shapeUncertaintyEmbeddingModelPath = pretrainedPath + '/' + self.opt.shapeUncertaintyEmbeddingModelName + '.pth'
			urlModelName = self.opt.shapeUncertaintyEmbeddingModelName == 'alexnet' and 'alexnet' or self.opt.shapeUncertaintyEmbeddingModelName == 'vgg' and 'vgg19' or self.opt.shapeUncertaintyEmbeddingModelName == 'vggbn' and 'vgg19_bn' or self.opt.shapeUncertaintyEmbeddingModelName == 'resnet' and 'resnet101' or self.opt.shapeUncertaintyEmbeddingModelName == 'densenet' and 'densenet201'
			if not fileExist(self.opt.shapeUncertaintyEmbeddingModelPath):
				mkdirs(pretrainedPath)
				print ("==> Downloading '" + self.opt.shapeUncertaintyEmbeddingModelName + "' pre-trained model of ImageNet")
				downloadFile(url=modelUrls[urlModelName], savePath=self.opt.shapeUncertaintyEmbeddingModelPath)

		args = vars(self.opt)

		print('------------ Options -------------')
		for k, v in sorted(args.items()):
			print('%s: %s' % (str(k), str(v)))
		print('-------------- End ----------------')

		# self.opt.argsDir = os.path.join(self.opt.argsDir, 'epoch' + str(self.opt.fromEpoch))
		if self.opt.train:
			if self.opt.fromScratch == 1:
				self.opt.fromEpoch = 1
			mkdirs(self.opt.argsDir)
			savePickle(filePath=os.path.join(self.opt.argsDir, 'opt.pkl'), data=self.opt)

		    # save to the disk
			file_name = os.path.join(self.opt.argsDir, 'opt_train.txt')
			with open(file_name, 'wt') as opt_file:
				opt_file.write('------------ Options -------------\n')
				for k, v in sorted(args.items()):
					opt_file.write('%s: %s\n' % (str(k), str(v)))
				opt_file.write('-------------- End ----------------\n')
		else:
			self.opt.argsDir = os.path.join(self.opt.argsDir, 'experiments')
			mkdirs(self.opt.argsDir)
		return self.opt