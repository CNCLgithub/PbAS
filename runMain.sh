#!/bin/sh

rm -rf /tmp/* 2> /dev/null
module load openmind/singularity/3.6.3

numShape="0"
removeMatsAfterSimStep="7"

catID="${1:--1}"
resolution="${2:-224}"
generateStimuli="${3:-0}" 
numRot="${4:-12}"
simRot="${5:-1}"
runBO="${6:-0}"
genStats_Plots="${7:-0}"
simplifyObjs="${8:-1}"
finetunePretrainedModel="${9:-0}"
BoNumRunStart="${10:--0}"
BoNumRunEnd="${11:--1}"
Boxi="${12:-330}"
BOLossModel="${13:-alexnet}"
BOLossModelFeatureLayerName="${14:-fc1}"
BOUnifiedInferencePipelineWith5Shapes="${15:-1}"
BOExcludeGTShapes="${16:-1}"
BOUseEmbeddingNearestNeighborPrior="${17:-3}"
BOResultsPathNote="${18:-Ranked_Prior_200}"
makeGPPlots="${19:-0}"

expType="${20:-0}" # Options: forwardPass, randomSamples or extractFeatures
forwardPassType="${21:-0}" # Options: userData, reconstruction
featureNetwork="${22:-alexnet}" # Options, alexnet
featLayerNo="${23:-1}" # Determines the layer number from which features will be extracted and stored on disk

singularity exec --nv -B /om:/om -B /om2:/om2 /om/user/arsalans/containers/pytorchBlenderLatest1.2New.simg python3.6 main_init.py --categoryID $catID --resolutions $resolution --numShape $numShape --numRotation $numRot --simultaneousRotation $simRot --generateStimuli $generateStimuli --removeMatAfterSimRotSteps $removeMatsAfterSimStep --runBayesianOpt $runBO --genStats_Plots $genStats_Plots --simplifyObjs $simplifyObjs --finetunePretrainedModel $finetunePretrainedModel --BoNumRunStart $BoNumRunStart --BoNumRunEnd $BoNumRunEnd --Boxi $Boxi --BOLossModel $BOLossModel --BOLossModelFeatureLayerName $BOLossModelFeatureLayerName --BOUnifiedInferencePipelineWith5Shapes $BOUnifiedInferencePipelineWith5Shapes --BOExcludeGTShapes $BOExcludeGTShapes --BOUseEmbeddingNearestNeighborPrior $BOUseEmbeddingNearestNeighborPrior --BOResultsPathNote $BOResultsPathNote --visMakeGPPlots $makeGPPlots --expType $expType --forwardPassType $forwardPassType --featureNetwork $featureNetwork --featLayerNo $featLayerNo
# singularity exec --nv -B /om:/om -B /nobackup:/nobackup /nobackup/to_delete/user/arsalans/containers/pytorchBlenderLatest1.2New.simg python3.6 main_init.py --categoryID $catID --resolutions $resolution --numShape $numShape --numRotation $numRot --simultaneousRotation $simRot --generateStimuli $generateStimuli --removeMatAfterSimRotSteps $removeMatsAfterSimStep --runBayesianOpt $runBO --genStats_Plots $genStats_Plots --simplifyObjs $simplifyObjs --finetunePretrainedModel $finetunePretrainedModel --BoNumRunStart $BoNumRunStart --BoNumRunEnd $BoNumRunEnd --Boxi $Boxi --BOLossModel $BOLossModel --BOLossModelFeatureLayerName $BOLossModelFeatureLayerName --BOUnifiedInferencePipelineWith5Shapes $BOUnifiedInferencePipelineWith5Shapes --BOExcludeGTShapes $BOExcludeGTShapes --BOUseEmbeddingNearestNeighborPrior $BOUseEmbeddingNearestNeighborPrior --BOResultsPathNote $BOResultsPathNote --visMakeGPPlots $makeGPPlots --expType $expType --forwardPassType $forwardPassType --featureNetwork $featureNetwork --featLayerNo $featLayerNo