import os, re, torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
from torch.utils import model_zoo
torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=False

class Model(nn.Module):

    def __init__(self, opt, modelName='', modelPath='', secondLossModel=False, finetuneModel=False, chopLayerAt=None, extraModule=False, runID=None):
        super(Model, self).__init__()
        self.cwd = os.getcwd() + '/'
        self.opts = opt

        self.runID = runID # runID could be interpreted as simulated subject number
        
        self.BOLossModel = modelName == '' and (opt.BOLossModel if not secondLossModel else opt.BOLossModelSecond) or modelName
        self.BOLossModelPath = modelPath == '' and opt.BOLossModelPath or modelPath
        self.BOLossModelFeature = opt.BOLossModelFeature
        self.BOLossModelFeatureLayerName = (opt.BOLossModelFeatureLayerName if not secondLossModel else opt.BOLossModelFeatureLayerNameSecond)
        self.BOLossModelRandomNetwork = opt.BOLossModelRandomNetwork

        # ImageNet Means and Stds
        # self.mean = [0.485, 0.456, 0.406]
        # self.std = [0.229, 0.224, 0.225]

        '''
        # Means and Stds for all renderings of the test set shapes of ShapeNet Core v1
        I used all renderings in .../ShapeNetCore.v1/2ShapesPerTrial/stimuli-rotLimitDegree-89.9-140BOIterations/datasetShapeRenderings/
        and also all images in .../ShapeNetCore.v1/2ShapesPerTrial/stimuli-rotLimitDegree-89.9-140BOIterations/renderings/
        For the second path, I did not use the images in 'unoccluded-unoccluded-unoccluded/' directory

        computed means and stds:
        self.mean = [0.11808602511882782, 0.11808602511882782, 0.11808602511882782]
        self.std = [0.2507171332836151, 0.2507171332836151, 0.2507171332836151]
        '''

        '''
        Means and Stds for all renderings for stimuli with occlusion:
        /om/user/arsalans/Occluded-object-detector/dataset/ShapeNetCore.v1/2ShapesPerTrial/stimuli-rotLimitDegree-89.9-140BOIterations/renderings/unoccluded-unoccluded-occluded/
        and
        /om/user/arsalans/Occluded-object-detector/dataset/ShapeNetCore.v1/2ShapesPerTrial/stimuli-rotLimitDegree-89.9-140BOIterations/renderings/occluded-occluded-unoccluded/
        '''
        # computed means and stds:
        self.mean = [0.1903279572725296, 0.1903279572725296, 0.1903279572725296]
        self.std = [0.273924857378006, 0.273924857378006, 0.273924857378006]
        
        
        # Fine tuning
        self.finetunePretrainedModelName = opt.finetunePretrainedModelName
        self.finetuneDecoderDimension = opt.finetuneDecoderDimension
        self.chopLayerAt = chopLayerAt
        self.extraModule = extraModule

        self.lossFunction = opt.BOLossFunction
        self.finetuneModel = None
        if finetuneModel:
            self.finetuneModel = finetuneModel
            self.finetuneModelPath = opt.finetuneModelPath
            self.finetuneTrainLastFCLayer = opt.finetuneTrainLastFCLayer
            self.modelName = self.finetunePretrainedModelName
            self.modelPath = self.finetuneModelPath
        else:
            self.modelName = self.BOLossModel
            self.modelPath = self.BOLossModelPath
        self.loss = self.lossFunction == 'l1' and nn.L1Loss(reduction='sum') or self.lossFunction == 'l2' and nn.MSELoss(reduction='sum') or self.lossFunction == 'corr' and np.corrcoef
        self.loadModel()

    def forward(self, x, y=None, z=None, normalize=False, finalOutputOnly=None):
        if not isinstance(x, torch.Tensor):
            x = self.transformNumpyInputToTensor(x=x, normalize=normalize)

        if y is None or z is None:
            y = self.model(x) if finalOutputOnly is None else self.model(x, finalOutputOnly=finalOutputOnly)
            return y
        else:
            # to be used with triplet loss
            if not isinstance(y, torch.Tensor):
                y = self.transformNumpyInputToTensor(x=y, normalize=normalize)
            if not isinstance(z, torch.Tensor):
                z = self.transformNumpyInputToTensor(x=z, normalize=normalize)
            
            o1 = self.model(x) if finalOutputOnly is None else self.model(x, finalOutputOnly=finalOutputOnly)
            o2 = self.model(y) if finalOutputOnly is None else self.model(y, finalOutputOnly=finalOutputOnly)
            o3 = self.model(z) if finalOutputOnly is None else self.model(z, finalOutputOnly=finalOutputOnly)

            return (o1, o2, o3)

    def loss_fn(self, x, target, inputEmbeddingProvided=False, targetEmbeddingProvided=False, normalize=False):
        if not inputEmbeddingProvided:
            x = self.transformNumpyInputToTensor(x=x, normalize=normalize)
            predFeats = self.forward(x)
        else:
            predFeats = x
        if not targetEmbeddingProvided:
            target = self.transformNumpyInputToTensor(x=target, normalize=normalize)
            targetFeats = self.forward(target)
        else:
            targetFeats = target
        loss = self.lossFunction == 'corr' and -torch.tensor(self.loss(predFeats.detach().cpu().numpy(), targetFeats.detach().cpu().numpy())[0][1]) or self.loss(predFeats, targetFeats)
        loss = -loss
        return loss

    def loadModel(self):
        if self.extraModule:
            self.model = ResNetExtraModules(chopLayerAt=(self.chopLayerAt))
        elif self.modelName == 'alexnet':
            self.model = AlexNet()
        elif self.modelName == 'vggbn':
            self.model = VGG(batch_norm=True)
        elif self.modelName == 'vgg':
            self.model = VGG(batch_norm=False)
        elif self.modelName == 'resnet':
            self.model = ResNet(resnetNumLayers=101)
        elif self.modelName == 'resnet50-sin-in' or self.modelName == 'resnet50-sin-in_in':
            self.model = ResNet(resnetNumLayers=50)
        elif self.modelName == 'cornet_s':
            self.model = CORnet_S()
        elif self.modelName == 'midas':
            self.model = MidasNet()
        elif self.modelName == 'densenet':
            self.model = DenseNet()
            pattern = re.compile(r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
            denseNetStateDict = torch.load(self.modelPath, map_location=torch.device('cpu'))
            for key in list(denseNetStateDict.keys()):
                res = pattern.match(key)
                if res:
                    new_key = res.group(1) + res.group(2)
                    denseNetStateDict[new_key] = denseNetStateDict[key]
                    del denseNetStateDict[key]
        elif self.modelName == 'baseline':
            self.model = Identity()


        if self.modelName != 'baseline' and self.modelName != '3dvae' and self.modelName != 'densenet' and not self.extraModule and not self.BOLossModelRandomNetwork:
            if self.modelName != 'cornet_s' and self.modelName != 'resnet50-sin-in_in':
                self.model.load_state_dict(torch.load(self.modelPath, map_location=torch.device('cpu')), strict=True)
            elif self.modelName == 'cornet_s':
                state_dict = torch.load(self.modelPath, map_location=torch.device('cpu'))['state_dict']
                new_state_dict = {}
                for k, v in state_dict.items():
                    name = k[7:] # remove `module.`
                    new_state_dict[name] = v
                self.model.load_state_dict(new_state_dict, strict=True)
            elif self.modelName == 'resnet50-sin-in_in':
                # uncomment the line below for PyTorch 1.5 and newer
                # state_dict = torch.hub.load_state_dict_from_url(self.opts.modelUrls['resnet50-sin-in_in'], model_dir=self.modelPath, file_name=self.modelName, map_location='cpu')['state_dict']
                state_dict = torch.hub.load_state_dict_from_url(self.opts.modelUrls['resnet50-sin-in_in'], model_dir=self.modelPath, map_location='cpu')['state_dict']
                new_state_dict = {}
                for k, v in state_dict.items():
                    name = k[7:] # remove `module.`
                    new_state_dict[name] = v
                self.model.load_state_dict(new_state_dict, strict=True)
        elif self.modelName == 'densenet' and not self.BOLossModelRandomNetwork:
            self.model.load_state_dict(denseNetStateDict)
        
        m = self.modelName
        if (self.BOLossModelFeature or self.finetuneModel) and self.modelName != 'densenet' and not self.extraModule:
            if self.chopLayerAt is not None and self.chopLayerAt != 'all':
                # if self.modelName == 'alexnet' or (self.modelName == 'resnet' and 'fc' not in self.chopLayerAt):
                # if m == 'resnet':
                #     print (self.model)
                #     for name, param in self.model.named_parameters():
                #         print(name, param.numel())
                #     exit()
                self.model = m == 'alexnet' and AlexNetFinetune(oldModel=self.model, chopLayerAt=self.chopLayerAt) or m == 'vgg' and VGGFinetune(oldModel=self.model, chopLayerAt=self.chopLayerAt) or 'resnet' in m and ResNetFinetune(oldModel=self.model, chopLayerAt=self.chopLayerAt) or m == 'cornet_s' and CORnet_SFinetune(oldModel=self.model, chopLayerAt=self.chopLayerAt)
                # if m == 'cornet_s':
                #     print (self.model)
                #     aa = self.model(torch.rand((1, 3, 224, 224)))
                #     print (aa.data.clone().cpu().numpy().shape)
                #     exit()
                for name, param in self.model.named_parameters():
                    if not self.finetuneModel or not self.finetuneTrainLastFCLayer or (self.finetuneTrainLastFCLayer and 'classifier' not in name):
                        param.requires_grad = False
                self.model = AddDecodingLayerAlexNetVGG(oldModel=self.model, decoderDimension=self.finetuneDecoderDimension, chopLayerAt=self.chopLayerAt) if ('resnet' not in m and 'cornet' not in m) else AddDecodingLayerResNet(oldModel=self.model, decoderDimension=self.finetuneDecoderDimension, chopLayerAt=self.chopLayerAt) if ('resnet' in m) else AddDecodingLayerCORnet_S(oldModel=self.model, decoderDimension=self.finetuneDecoderDimension, chopLayerAt=self.chopLayerAt)
                # if m == 'resnet':
                #     print(self.model)
                #     for name, param in self.model.named_parameters():
                #         print(name, param.requires_grad, param.numel())
                #     aa = self.model(torch.rand((1, 3, 224, 224)), finalOutputOnly=True)
                #     print (aa.data.clone().cpu().numpy().shape)
                #     exit()
                # if m == 'cornet_s':
                    # print (self.model)
                    # for name, param in self.model.named_parameters():
                    #     print(name, param.requires_grad, param.numel())
                    # aa = self.model(torch.rand((1, 3, 224, 224)), finalOutputOnly=True)
                    # print (aa.data.clone().cpu().numpy().shape)
                #     exit()
            else:
                # if self.modelName == 'midas':
                #     # self.model
                #     from torchviz import make_dot
                #     aa = self.model(torch.rand((1, 3, 224, 224)))
                #     print (aa.data.clone().cpu().numpy().shape)
                #     exit()
                #     dot = make_dot(aa)
                #     dot.format = 'png'
                #     dot.render(self.cwd + '/viz')
                #     print ('done')
                #     exit()
                self.model = AlexNetFeatures(oldModel=self.model, chopLayerAt=self.BOLossModelFeatureLayerName) if m == 'alexnet' \
                else VGGFeatures(self.model) if m == 'vgg' \
                else ResNetFeatures(self.model, chopLayerAt=self.BOLossModelFeatureLayerName) if m == 'resnet' \
                else CORnet_SFeatures(self.model, chopLayerAt=self.BOLossModelFeatureLayerName) if m == 'cornet_s' \
                else self.model
                # or m == 'midas' and MiDaSFeatures(oldModel=self.model, chopLayerAt=self.BOLossModelFeatureLayerName) \
                # or self.model
            # if self.modelName == 'midas':
            #     print ('\n\n\nafter\n')
            #     print (self.model)
            #     exit()

    def transformNumpyInputToTensor(self, x, normalize=False):
        x = torch.from_numpy(x)
        if self.modelName != 'baseline' and next(self.model.parameters()).is_cuda:
            if len(x.shape) == 3:
                x = x.view(1, x.shape[0], x.shape[1], x.shape[2]).float().cuda(non_blocking=True)
            else:
                x = x.view(-1, x.shape[1], x.shape[2], x.shape[3]).float().cuda(non_blocking=True)
        else:
            if len(x.shape) == 3:
                x = x.view(1, x.shape[0], x.shape[1], x.shape[2]).float()
            else:
                x = x.view(-1, x.shape[1], x.shape[2], x.shape[3]).float()
        
        if normalize:
            x = self.normalize(x)
        return x

    def normalize(self, x):
        dtype = x.dtype
        mean = torch.as_tensor(self.mean, dtype=dtype, device=x.device)
        std = torch.as_tensor(self.std, dtype=dtype, device=x.device)
        if mean.ndim == 1:
            mean = mean[:, None, None]
        if std.ndim == 1:
            std = std[:, None, None]
        x.sub_(mean).div_(std)
        return x


class AlexNetFeatures(nn.Module):
    def __init__(self, oldModel, chopLayerAt):
        super(AlexNetFeatures, self).__init__()
        self.chopLayerAt = chopLayerAt

        if 'fc' in self.chopLayerAt or 'lastPool' in self.chopLayerAt:
            self.features = nn.Sequential(
                *list(oldModel.features.children())[:] #-2, -4, -6, -9, -12 for the 5th, 4th, 3rd, 2nd and 1st conv layers
            )
        elif 'conv1' in self.chopLayerAt:
            self.features = nn.Sequential(
                *list(oldModel.features.children())[:-12] #-2, -4, -6, -9, -12 for the 5th, 4th, 3rd, 2nd and 1st conv layers
            )
        elif 'conv2' in self.chopLayerAt:
            self.features = nn.Sequential(
                *list(oldModel.features.children())[:-9] #-2, -4, -6, -9, -12 for the 5th, 4th, 3rd, 2nd and 1st conv layers
            )
        elif 'conv3' in self.chopLayerAt:
            self.features = nn.Sequential(
                *list(oldModel.features.children())[:-6] #-2, -4, -6, -9, -12 for the 5th, 4th, 3rd, 2nd and 1st conv layers
            )
        elif 'conv4' in self.chopLayerAt:
            self.features = nn.Sequential(
                *list(oldModel.features.children())[:-4] #-2, -4, -6, -9, -12 for the 5th, 4th, 3rd, 2nd and 1st conv layers
            )
        elif 'conv5' in self.chopLayerAt:
            self.features = nn.Sequential(
                *list(oldModel.features.children())[:-2] #-2, -4, -6, -9, -12 for the 5th, 4th, 3rd, 2nd and 1st conv layers
            )

        if 'fc1' in self.chopLayerAt:
            self.classifier = nn.Sequential(
                *list(oldModel.classifier.children())[:-5] #-5 1st FC layer of AlexNet (i.e. before applying ReLU to the activations)
            )
        elif 'fc2' in self.chopLayerAt:
            self.classifier = nn.Sequential(
                *list(oldModel.classifier.children())[:-2] #-5 1st FC layer of AlexNet (i.e. before applying ReLU to the activations)
            )

    def forward(self, x):
        y = self.features(x)
        y = y.view(y.size(0), y.size(1) * y.size(2) * y.size(3))
        if 'fc' in self.chopLayerAt:
            y = self.classifier(y)
        return y

class VGGFeatures(nn.Module):
    def __init__(self, oldModel):
        super(VGGFeatures, self).__init__()
        self.features = nn.Sequential(
            # *list(oldModel.features.children())[0:-3]
            *list(oldModel.features.children())
        )
        self.classifier = nn.Sequential(
            *list(oldModel.classifier.children())[:-6] #-6 for 1st FC layer
        )
    def forward(self, x):
        y = self.features(x)
        y = y.view(y.size(0), y.size(1) * y.size(2) * y.size(3))
        y = self.classifier(y)
        return y

class ResNetFeatures(nn.Module):
    def __init__(self, oldModel, chopLayerAt):
        super(ResNetFeatures, self).__init__()
        self.chopLayerAt = chopLayerAt
        chopLayerAtNum = chopLayerAt == 'layer4' and -2 or chopLayerAt == 'layer3' and -3 or chopLayerAt == 'layer2' and -4 or chopLayerAt == 'layer1' and -5
        if chopLayerAtNum:
            self.features = nn.Sequential(*list(oldModel.children())[:chopLayerAtNum])
        else:
            self.features = nn.Sequential(*list(oldModel.children())[:-2])
            self.AdaptiveAvgPool = nn.Sequential(*list(oldModel.children())[-2:-1]
            )
            self.classifier = nn.Sequential(*list(oldModel.children())[-1:]
            )
    
    def forward(self, x):
        y = self.features(x)
        if 'fc' in self.chopLayerAt:
            y = self.AdaptiveAvgPool(y)
            y = self.classifier(y)
        else:
            y = y.view(y.size(0), y.size(1) * y.size(2) * y.size(3))
        return y

class CORnet_SFeatures(nn.Module):
    def __init__(self, oldModel, chopLayerAt):
        super(CORnet_SFeatures, self).__init__()
        self.chopLayerAtNum = 'it' in chopLayerAt and -1 or 'v4' in chopLayerAt and -2 or 'v2' in chopLayerAt and -3 or 'v1' in chopLayerAt and -4
        if self.chopLayerAtNum:
            self.features = nn.Sequential(*list(oldModel.children())[:self.chopLayerAtNum])
        else:
            self.features = nn.Sequential(*list(oldModel.children())[:-1])
            self.AdaptiveAvgPool = nn.Sequential(*list(oldModel.decoder.children())[:-2]
            )
    def forward(self, x):
        y = self.features(x)
        if not self.chopLayerAtNum:
            y = self.AdaptiveAvgPool(y)
        else:
            y = y.view(y.size(0), y.size(1) * y.size(2) * y.size(3))
        return y


class MiDaSFeatures(nn.Module):
    def __init__(self, oldModel, chopLayerAt):
        super(MiDaSFeatures, self).__init__()
        # chopLayerAtNum = -5 if chopLayerAt == 'encoder' else False
        self.chopLayerAtNum = -5

        # self.features1 = nn.Sequential(
        #     *list(oldModel.children())[:-1]
        # )
        self.features1 = nn.Sequential(*list(oldModel.pretrained.children())[:])
        if self.chopLayerAtNum:
            self.features2 = nn.Sequential(
                *list(oldModel.scratch.children())[:-5]
            )
        else:
            self.features2 = nn.Sequential(
                *list(oldModel.scratch.children())[:]
            )

    def forward(self, x):
        y = self.features1(x)
        y = self.features2(y)
        return y


class AlexNetFinetune(nn.Module):
    def __init__(self, oldModel, chopLayerAt=None):
        super(AlexNetFinetune, self).__init__()
        self.fixedNet = nn.Sequential()

        self.noFC = False
        chopFCAt = -5 # default value
        chopConvAt = 0 # default value
        if chopLayerAt is not None:
            if chopLayerAt == 'conv5':
                self.noFC = True
                # chopConvAt = -2 # chop the model after the conv5 operations
                chopConvAt = 0 # after pooling
            elif chopLayerAt == 'conv1':
                self.noFC = True
                chopConvAt = -12
            else:
                chopConvAt = 0
                chopFCAt = chopLayerAt == 'fc1' and -5 or chopLayerAt == 'fc2' and -2
        
        if chopConvAt != 0:
            self.features = nn.Sequential(
                *list(oldModel.features.children())[:chopConvAt] #-2, -4, -6, -9, -12 for the 5th, 4th, 3rd, 2nd and 1st conv layers
            )
        else:
            self.features = nn.Sequential(
                *list(oldModel.features.children())[:]
            )
        if not self.noFC:
            self.classifier = nn.Sequential(
                *list(oldModel.classifier.children())[:chopFCAt] #-5 1st FC layer of AlexNet (i.e. before applying ReLU to the activations)
            )

class VGGFinetune(nn.Module):
    def __init__(self, oldModel, chopLayerAt=None):
        super(VGGFinetune, self).__init__()
        self.fixedNet = nn.Sequential()

        self.noFC = False
        chopFCAt = -5 # default value
        chopConvAt = 0 # default value
        if chopLayerAt is not None:
            chopConvAt = 0
            chopFCAt = chopLayerAt == 'fc1' and -6 or chopLayerAt == 'fc2' and -3
        
        self.features = nn.Sequential(
            *list(oldModel.features.children())[:]
        )
        if not self.noFC:
            self.classifier = nn.Sequential(
                *list(oldModel.classifier.children())[:chopFCAt] #-5 1st FC layer of AlexNet (i.e. before applying ReLU to the activations)
            )

class ResNetFinetune(nn.Module):
    def __init__(self, oldModel, chopLayerAt):
        super(ResNetFinetune, self).__init__()
        chopLayerAtNum = chopLayerAt == 'layer4' and -2 or chopLayerAt == 'layer3' and -3 or chopLayerAt == 'layer2' and -4 or chopLayerAt == 'layer1' and -5
        if chopLayerAtNum:
            self.features = nn.Sequential(*list(oldModel.children())[:chopLayerAtNum])
        else:
            self.features = nn.Sequential(*list(oldModel.children())[:-2])
            self.AdaptiveAvgPool = nn.Sequential(*list(oldModel.children())[-2:-1]
            )
            self.classifier = nn.Sequential(*list(oldModel.children())[-1:]
            )

class CORnet_SFinetune(nn.Module):
    def __init__(self, oldModel, chopLayerAt):
        super(CORnet_SFinetune, self).__init__()
        self.chopLayerAtNum = 'it' in chopLayerAt and -1 or 'v4' in chopLayerAt and -2 or 'v2' in chopLayerAt and -3 or 'v1' in chopLayerAt and -4
        if self.chopLayerAtNum:
            self.features = nn.Sequential(*list(oldModel.children())[:self.chopLayerAtNum])
        else:
            self.features = nn.Sequential(*list(oldModel.children())[:-1])
            self.AdaptiveAvgPool = nn.Sequential(*list(oldModel.decoder.children())[:-2]
            )
            # self.decoder = nn.Sequential(*list(oldModel.decoder.children())
            # )
    # def forward(self, x):
    #     y = self.features(x)
    #     if not self.chopLayerAtNum:
    #         y = self.AdaptiveAvgPool(y)
    #     return y






class AddDecodingLayerAlexNetVGG(nn.Module):
    def __init__(self, oldModel, decoderDimension=120, chopLayerAt=None):
        super(AddDecodingLayerAlexNetVGG, self).__init__()
        if hasattr(oldModel, 'features'):
            self.features = nn.Sequential(
                *list(oldModel.features.children())[:]
            )
        if hasattr(oldModel, 'classifier'):
            self.noFC = False
            self.classifier = nn.Sequential(
                *list(oldModel.classifier.children())[:]
            )
        else:
            self.noFC = True

        self.decodingLayer = nn.Sequential(
            # nn.Linear(chopLayerAt == 'conv5' and 9216 or 4096, 500, bias=False) # Default
            nn.Linear(chopLayerAt == 'conv5' and 9216 or 4096, decoderDimension, bias=False)
        )

    def forward(self, x, finalOutputOnly=False):
        y = self.features(x)
        y = y.view(y.size(0), -1)
        if not self.noFC:
            y = self.classifier(y)
        penultimateLayer = y.detach()
        y = self.decodingLayer(y)
        if finalOutputOnly:
            return y
        else:
            return y, penultimateLayer













'''
Temporary fix
'''
# class AddDecodingLayerAlexNet(nn.Module):
#     def __init__(self, oldModel, chopLayerAt=None):
#         super(AddDecodingLayerAlexNetVGG, self).__init__()
#         if hasattr(oldModel, 'features'):
#             self.features = nn.Sequential(
#                 *list(oldModel.features.children())[:]
#             )
#         if hasattr(oldModel, 'classifier'):
#             self.noFC = False
#             self.classifier = nn.Sequential(
#                 *list(oldModel.classifier.children())[:]
#             )
#         else:
#             self.noFC = True

#         self.decodingLayer = nn.Sequential(
#             nn.Linear(chopLayerAt == 'conv5' and 9216 or 4096, 500)
#         )

#     def forward(self, x):
#         y = self.features(x)
#         y = y.view(y.size(0), y.size(1) * y.size(2) * y.size(3))
#         if not self.noFC:
#             y = self.classifier(y)
#         y = self.decodingLayer(y)
#         return y
'''
End of temporary fix
'''

















class AddDecodingLayerCORnet_S(nn.Module):
    def __init__(self, oldModel, decoderDimension, chopLayerAt):
        super(AddDecodingLayerCORnet_S, self).__init__()

        self.decoderDimension = decoderDimension
        self.features = nn.Sequential(*list(oldModel.features.children())[:]
        )
        if hasattr(oldModel, 'AdaptiveAvgPool') and 'decoder' in chopLayerAt:
            self.AdaptiveAvgPool = nn.Sequential(*list(oldModel.AdaptiveAvgPool.children())[:]
            )

        # if hasattr(oldModel, 'decoder') and 'decoder' in chopLayerAt:
        #     self.decoder = nn.Sequential(*list(oldModel.decoder.children())[:]
        #     )

        
        self.View = False
        self.chopLayerAt = chopLayerAt        
        if 'v1' in self.chopLayerAt:
            if self.chopLayerAt == 'v1Linear':
                numUnits = 200704
                self.View = True
                self.decodingLayer = nn.Sequential(
                    nn.Linear(numUnits, 50, bias=False)
                )

            if self.chopLayerAt == 'v1Conv':
                outFeats=50 # 50 is a better number but I did not try it
                self.decodingLayer = nn.Sequential(
                    SmallConvModule(inFeats=64, outFeats=outFeats, kernelSize=2, stride=2, padding=0, batchNorm=False, ReLU=True, view=True),
                    nn.Linear(outFeats*27*27, self.decoderDimension, bias=False)
                )

                # outFeats=144
                # self.decodingLayer = nn.Sequential(
                #     SmallConvModule(inFeats=64, outFeats=outFeats, kernelSize=2, stride=2, padding=0, batchNorm=False, ReLU=True, view=False),
                #     SmallConvModule(inFeats=outFeats, outFeats=outFeats//2, kernelSize=2, stride=2, padding=0, batchNorm=False, ReLU=False, view=True),
                #     nn.Linear(outFeats//2*13*13, 100, bias=False)
                # )

        elif 'v2' in self.chopLayerAt:
            if self.chopLayerAt == 'v2Linear':
                numUnits = 100352
                self.View = True
                self.decodingLayer = nn.Sequential(
                    nn.Linear(numUnits, 50, bias=False)
                )

            if self.chopLayerAt == 'v2Conv':
                outFeats=80
                self.decodingLayer = nn.Sequential(
                    SmallConvModule(inFeats=128, outFeats=outFeats, kernelSize=2, stride=2, padding=0, batchNorm=False, ReLU=True, view=True),
                    nn.Linear(outFeats*13*13, self.decoderDimension, bias=False)
                )

                # outFeats=80
                # self.decodingLayer = nn.Sequential(
                #     SmallConvModule(inFeats=128, outFeats=outFeats, kernelSize=2, stride=2, padding=0, batchNorm=False, ReLU=True, view=False),
                #     SmallConvModule(inFeats=outFeats, outFeats=outFeats//2, kernelSize=2, stride=2, padding=0, batchNorm=False, ReLU=False, view=True),
                #     nn.Linear(outFeats//2*6*6, 50, bias=False)
                # )

        elif 'v4' in self.chopLayerAt:
            if self.chopLayerAt == 'v4Linear':
                numUnits = 50176
                self.View = True
                self.decodingLayer = nn.Sequential(
                    nn.Linear(numUnits, 50, bias=False)
                )

            if self.chopLayerAt == 'v4Conv':
                outFeats=200 # 200 is a better number but I did not try it
                self.decodingLayer = nn.Sequential(
                    SmallConvModule(inFeats=256, outFeats=outFeats, kernelSize=2, stride=2, padding=0, batchNorm=False, ReLU=True, view=True),
                    nn.Linear(outFeats*6*6, self.decoderDimension, bias=False)
                )

                # outFeats=200
                # self.decodingLayer = nn.Sequential(
                #     SmallConvModule(inFeats=256, outFeats=outFeats, kernelSize=2, stride=2, padding=0, batchNorm=False, ReLU=False, view=False),
                #     SmallConvModule(inFeats=outFeats, outFeats=outFeats//2, kernelSize=2, stride=2, padding=0, batchNorm=False, ReLU=False, view=True),
                #     nn.Linear(outFeats//2*2*2, 50, bias=False)
                # )
        elif 'it' in self.chopLayerAt:
            if self.chopLayerAt == 'itLinear':
                # Uncomment in case you are adding the linear decoding layer after the IT layer and before applying the average pool
                numUnits = 25088
                self.View = True
                self.decodingLayer = nn.Sequential(
                    nn.Linear(numUnits, 50, bias=False)
                )

            if self.chopLayerAt == 'itConv':
                outFeats=300
                self.decodingLayer = nn.Sequential(
                    SmallConvModule(inFeats=512, outFeats=outFeats, kernelSize=2, stride=2, padding=0, batchNorm=False, ReLU=True, view=True),
                    nn.Linear(outFeats*3*3, self.decoderDimension, bias=False)
                )

        elif self.chopLayerAt == 'decoder':
            numUnits = 512
            self.decodingLayer = nn.Sequential(
                nn.Linear(numUnits, self.decoderDimension, bias=False)
            )

            # Uncomment in case you are adding the decoder layer after the classification layer
            # numUnits = 1000
            # self.decodingLayer = nn.Sequential(
            #     nn.Linear(numUnits, 150, bias=False)
            # )

        for name, param in self.decodingLayer.named_parameters():
            torch.nn.init.orthogonal_(param)
        
    def forward(self, x, finalOutputOnly=False):
        y = self.features(x)
        if self.View:
            y = y.view(y.size(0), -1)
        elif not self.View and self.chopLayerAt == 'decoder':
            y = self.AdaptiveAvgPool(y)
        penultimateLayer = y.detach()

        y = self.decodingLayer(y)
        if finalOutputOnly:
            return y
        else:
            return y, penultimateLayer






class AddDecodingLayerResNet(nn.Module):
    def __init__(self, oldModel, decoderDimension, chopLayerAt):
        super(AddDecodingLayerResNet, self).__init__()
        self.chopLayerAt = chopLayerAt

        self.features = nn.Sequential(*list(oldModel.features.children())[:]
        )

        if hasattr(oldModel, 'AdaptiveAvgPool') and 'fc' in chopLayerAt:
            self.AdaptiveAvgPool = nn.Sequential(*list(oldModel.AdaptiveAvgPool.children())[:]
            )

        if hasattr(oldModel, 'classifier') and 'fc12' in chopLayerAt:
            self.classifier = nn.Sequential(*list(oldModel.classifier.children())[:]
            )

        
        self.View = False
        self.ConvLayer = False
        if chopLayerAt == 'layer1':
            pass
        elif chopLayerAt == 'layer2':
            pass
        elif chopLayerAt == 'layer3':
            pass
        elif chopLayerAt == 'layer4':
            # self.decodingLayer = nn.AdaptiveAvgPool2d((1, 1))
            outFeats = 300
            self.ConvLayer = True
            self.decodingLayer = nn.Sequential(
                SmallConvModule(inFeats=2048, outFeats=outFeats, kernelSize=2, stride=2, padding=0, batchNorm=False, ReLU=True),
                nn.Linear(outFeats*3*3, decoderDimension, bias=False)
            )
        elif chopLayerAt == 'fc11':
            numFeats = 2048
            self.View = True
            self.decodingLayer = nn.Sequential(
                nn.Linear(numFeats, decoderDimension, bias=False)
            )

            # outFeats=400
            # self.decodingLayer = nn.Sequential(
            #     SmallConvModule(inFeats=2048, outFeats=outFeats, kernelSize=2, stride=2, padding=0, batchNorm=False, ReLU=False, view=True),
            #     nn.Linear(outFeats*3*3, 70, bias=False)
            # )

        elif chopLayerAt == 'fc12':
            numUnits = 1000
            self.View = True
            self.decodingLayer = nn.Sequential(
                self.classifier,
                nn.Linear(numUnits, decoderDimension, bias=False)
            )
        
    def forward(self, x, finalOutputOnly=False):
        y = self.features(x)
        if self.View:
            y = self.AdaptiveAvgPool(y)
            y = y.view(y.size(0), -1)
        
        if self.ConvLayer:
            penultimateLayer = y.view(y.size(0), -1).detach()
        else:
            penultimateLayer = y.detach()
        y = self.decodingLayer(y)

        if finalOutputOnly:
            return y
        else:
            return y, penultimateLayer







# class AddDecodingLayerResNet(nn.Module):
#     def __init__(self, oldModel, chopLayerAt):
#         super(AddDecodingLayerResNet, self).__init__()

#         self.features = nn.Sequential(*list(oldModel.features.children())[:]
#         )
#         if hasattr(oldModel, 'AdaptiveAvgPool') and 'fc' in chopLayerAt:
#             self.AdaptiveAvgPool = nn.Sequential(*list(oldModel.AdaptiveAvgPool.children())[:]
#             )
#             self.addView = True
#         else:
#             self.addView = False
#         if hasattr(oldModel, 'classifier') and chopLayerAt == 'fc12':
#             self.classifier = nn.Sequential(*list(oldModel.classifier.children())[:]
#             )
#             self.noFC = False
#         else:
#             self.noFC = True

        
#         if chopLayerAt == 'layer1':
#             pass
#         elif chopLayerAt == 'layer2':
#             pass
#         elif chopLayerAt == 'layer3':
#             pass
#         elif chopLayerAt == 'layer4':
#             outFeats = 400
#             numUnits = outFeats*2*2
#             self.decodingLayer = nn.Sequential(
#                 SmallConvModule(inFeats=2048, outFeats=outFeats),
#                 nn.Linear(numUnits, 500, bias=False)
#             )
#         elif chopLayerAt == 'fc11':
#             numUnits = 2048
#             self.decodingLayer = nn.Sequential(
#                 nn.Linear(numUnits, 500, bias=False)
#             )
#         elif chopLayerAt == 'fc12':
#             numUnits = 1000
#             self.decodingLayer = nn.Sequential(
#                 nn.Linear(numUnits, 500, bias=False)
#             )
        
#     def forward(self, x):
#         y = self.features(x)
#         if self.addView:
#             y = self.AdaptiveAvgPool(y)
#             y = y.view(y.size(0), -1)
#         if not self.noFC:
#             y = self.classifier(y)
#         y = self.decodingLayer(y)
#         return y



class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        self.model = nn.Sequential()

    def forward(self, x):
        y = self.model(x)
        return y

class SmallConvModule(nn.Module):
    def __init__(self, inFeats, outFeats, kernelSize=4, stride=2, padding=1, dilation=2, batchNorm=True, ReLU=True, view=True):
        super(SmallConvModule, self).__init__()
        self.view = view
        self.modules = [nn.Conv2d(inFeats, outFeats, kernelSize, stride, padding, dilation, bias=False)]
        if batchNorm:
            self.modules.append(nn.BatchNorm2d(outFeats))
        if ReLU:
            self.modules.append(nn.ReLU(inplace=True))

        self.model = nn.Sequential(*self.modules)

    def forward(self, x):
        y = self.model(x)
        if self.view:
            y = y.view(y.size(0), -1)
        return y

class SmallLinearModule(nn.Module):
    def __init__(self, inUnits, outUnits):
        super(SmallLinearModule, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(2048, outFeats, 4, 2, 1, 2),
            nn.BatchNorm2d(outFeats),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        y = self.model(x)
        y = y.view(y.size(0), -1)
        return y

class ResNetExtraModules(nn.Module):
    def __init__(self, chopLayerAt):
        super(ResNetExtraModules, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(1000, 500)
        )

    def forward(self, x):
        y = self.model(x)
        return y

# AlexNet class copied from https://github.com/pytorch/vision/blob/master/torchvision/models/alexnet.py
class AlexNet(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        y = self.features(x)
        y = y.view(y.size(0), 256 * 6 * 6)
        y = self.classifier(y)
        return y


# VGG class copied from https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py and modified
class VGG(nn.Module):

    def __init__(self, batch_norm, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = self.make_layers(batch_norm=batch_norm)
        self.classifier = nn.Sequential(
        nn.Linear(512 * 7 * 7, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        y = self.features(x)
        y = y.view(y.size(0), -1)
        y = self.classifier(y)
        return y

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def make_layers(self, batch_norm=True):
        cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)


# ResNet classes copied from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py and modified
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
        base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = self.conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = self.conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = self.conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
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

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

    def conv3x3(self, in_planes, out_planes, stride=1, groups=1, dilation=1):
        """3x3 convolution with padding"""
        return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
        padding=dilation, groups=groups, bias=False, dilation=dilation)


    def conv1x1(self, in_planes, out_planes, stride=1):
        """1x1 convolution"""
        return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class ResNet(nn.Module):

    def __init__(self, resnetNumLayers=101, num_classes=1000, zero_init_residual=False, groups=1, width_per_group=64, replace_stride_with_dilation=None, norm_layer=None):
        super(ResNet, self).__init__()
        block = Bottleneck
        if resnetNumLayers == 101:
            layers = [3, 4, 23, 3]
        elif resnetNumLayers == 50:
            layers = [3, 4, 6, 3]
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                                bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                        dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                        dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                        dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                self.conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def conv1x1(self, in_planes, out_planes, stride=1):
        """1x1 convolution"""
        return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


# DenseNet classes copied from https://github.com/pytorch/vision/blob/master/torchvision/models/densenet.py and modified
class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                                growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
    growth_rate (int) - how many filters to add each layer (`k` in paper)
    block_config (list of 4 ints) - how many layers in each pooling block
    num_init_features (int) - the number of filters to learn in the first convolution layer
    bn_size (int) - multiplicative factor for number of bottle neck layers
    (i.e. bn_size * k features in the bottleneck layer)
    drop_rate (float) - dropout rate after each dense layer
    num_classes (int) - number of classification classes
    """

    def __init__(self, growth_rate=32, block_config=(6, 12, 48, 32),
                num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000):

        super(DenseNet, self).__init__()

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.avg_pool2d(out, kernel_size=7, stride=1).view(features.size(0), -1)
        out = self.classifier(out)
        return out



# CORnet-S classes copied from https://github.com/dicarlolab/CORnet/blob/master/cornet/cornet_s.py
class Flatten(nn.Module):

    """
    Helper module for flattening input tensor to 1-D for the use in Linear modules
    """

    def forward(self, x):
        return x.view(x.size(0), -1)


class Identity(nn.Module):

    """
    Helper module that stores the current tensor. Useful for accessing by name
    """

    def forward(self, x):
        return x


class CORblock_S(nn.Module):

    scale = 4  # scale of the bottleneck convolution channels

    def __init__(self, in_channels, out_channels, times=1):
        super().__init__()

        self.times = times

        self.conv_input = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.skip = nn.Conv2d(out_channels, out_channels,
                              kernel_size=1, stride=2, bias=False)
        self.norm_skip = nn.BatchNorm2d(out_channels)

        self.conv1 = nn.Conv2d(out_channels, out_channels * self.scale,
                               kernel_size=1, bias=False)
        self.nonlin1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels * self.scale, out_channels * self.scale,
                               kernel_size=3, stride=2, padding=1, bias=False)
        self.nonlin2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(out_channels * self.scale, out_channels,
                               kernel_size=1, bias=False)
        self.nonlin3 = nn.ReLU(inplace=True)

        self.output = Identity()  # for an easy access to this block's output

        # need BatchNorm for each time step for training to work well
        for t in range(self.times):
            setattr(self, f'norm1_{t}', nn.BatchNorm2d(out_channels * self.scale))
            setattr(self, f'norm2_{t}', nn.BatchNorm2d(out_channels * self.scale))
            setattr(self, f'norm3_{t}', nn.BatchNorm2d(out_channels))

    def forward(self, inp):
        x = self.conv_input(inp)

        for t in range(self.times):
            if t == 0:
                skip = self.norm_skip(self.skip(x))
                self.conv2.stride = (2, 2)
            else:
                skip = x
                self.conv2.stride = (1, 1)

            x = self.conv1(x)
            x = getattr(self, f'norm1_{t}')(x)
            x = self.nonlin1(x)

            x = self.conv2(x)
            x = getattr(self, f'norm2_{t}')(x)
            x = self.nonlin2(x)

            x = self.conv3(x)
            x = getattr(self, f'norm3_{t}')(x)

            x += skip
            x = self.nonlin3(x)
            output = self.output(x)

        return output


def CORnet_S():
    model = nn.Sequential(OrderedDict([
        ('V1', nn.Sequential(OrderedDict([  # this one is custom to save GPU memory
            ('conv1', nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                            bias=False)),
            ('norm1', nn.BatchNorm2d(64)),
            ('nonlin1', nn.ReLU(inplace=True)),
            ('pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
            ('conv2', nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1,
                            bias=False)),
            ('norm2', nn.BatchNorm2d(64)),
            ('nonlin2', nn.ReLU(inplace=True)),
            ('output', Identity())
        ]))),
        ('V2', CORblock_S(64, 128, times=2)),
        ('V4', CORblock_S(128, 256, times=4)),
        ('IT', CORblock_S(256, 512, times=2)),
        ('decoder', nn.Sequential(OrderedDict([
            ('avgpool', nn.AdaptiveAvgPool2d(1)),
            ('flatten', Flatten()),
            ('linear', nn.Linear(512, 1000)),
            ('output', Identity())
        ])))
    ]))

    # weight initialization
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, np.sqrt(2. / n))
        # nn.Linear is missing here because I originally forgot 
        # to add it during the training of this network
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

    return model











# MiDaS model definition obtained from https://github.com/intel-isl/MiDaS/tree/master/models and modified:

def _make_encoder(features):
    torch.hub.set_dir(str(os.environ["TORCH_HOME"]))
    pretrained = _make_pretrained_resnext101_wsl()
    scratch = _make_scratch([256, 512, 1024, 2048], features)

    return pretrained, scratch


def _make_resnet_backbone(resnet):
    pretrained = nn.Module()
    pretrained.layer1 = nn.Sequential(
        resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1
    )

    pretrained.layer2 = resnet.layer2
    pretrained.layer3 = resnet.layer3
    pretrained.layer4 = resnet.layer4

    return pretrained


def _make_pretrained_resnext101_wsl():
    resnet = torch.hub.load("facebookresearch/WSL-Images", "resnext101_32x8d_wsl")
    return _make_resnet_backbone(resnet)


def _make_scratch(in_shape, out_shape):
    scratch = nn.Module()

    scratch.layer1_rn = nn.Conv2d(
        in_shape[0], out_shape, kernel_size=3, stride=1, padding=1, bias=False
    )
    scratch.layer2_rn = nn.Conv2d(
        in_shape[1], out_shape, kernel_size=3, stride=1, padding=1, bias=False
    )
    scratch.layer3_rn = nn.Conv2d(
        in_shape[2], out_shape, kernel_size=3, stride=1, padding=1, bias=False
    )
    scratch.layer4_rn = nn.Conv2d(
        in_shape[3], out_shape, kernel_size=3, stride=1, padding=1, bias=False
    )
    return scratch


class Interpolate(nn.Module):
    """Interpolation module.
    """

    def __init__(self, scale_factor, mode):
        """Init.

        Args:
            scale_factor (float): scaling
            mode (str): interpolation mode
        """
        super(Interpolate, self).__init__()

        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: interpolated data
        """

        x = self.interp(
            x, scale_factor=self.scale_factor, mode=self.mode, align_corners=False
        )

        return x


class ResidualConvUnit(nn.Module):
    """Residual convolution module.
    """

    def __init__(self, features):
        """Init.

        Args:
            features (int): number of features
        """
        super().__init__()

        self.conv1 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True
        )

        self.conv2 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: output
        """
        out = self.relu(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)

        return out + x


class FeatureFusionBlock(nn.Module):
    """Feature fusion block.
    """

    def __init__(self, features):
        """Init.

        Args:
            features (int): number of features
        """
        super(FeatureFusionBlock, self).__init__()

        self.resConfUnit1 = ResidualConvUnit(features)
        self.resConfUnit2 = ResidualConvUnit(features)

    def forward(self, *xs):
        """Forward pass.

        Returns:
            tensor: output
        """
        output = xs[0]

        if len(xs) == 2:
            output += self.resConfUnit1(xs[1])

        output = self.resConfUnit2(output)

        output = nn.functional.interpolate(
            output, scale_factor=2, mode="bilinear", align_corners=True
        )

        return output


class MidasNet(nn.Module):
    """Network for monocular depth estimation.
    """

    def __init__(self, features=256, non_negative=True):
        """Init.

        Args:
            features (int, optional): Number of features. Defaults to 256.
            backbone (str, optional): Backbone network for encoder. Defaults to resnet50
        """

        super(MidasNet, self).__init__()

        self.pretrained, self.scratch = _make_encoder(features)

        self.scratch.refinenet4 = FeatureFusionBlock(features)
        self.scratch.refinenet3 = FeatureFusionBlock(features)
        self.scratch.refinenet2 = FeatureFusionBlock(features)
        self.scratch.refinenet1 = FeatureFusionBlock(features)

        self.scratch.output_conv = nn.Sequential(
            nn.Conv2d(features, 128, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear"),
            nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True) if non_negative else nn.Identity(),
        )


    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input data (image)

        Returns:
            tensor: depth
        """

        layer_1 = self.pretrained.layer1(x)
        layer_2 = self.pretrained.layer2(layer_1)
        layer_3 = self.pretrained.layer3(layer_2)
        layer_4 = self.pretrained.layer4(layer_3)

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        return layer_3_rn

        # path_4 = self.scratch.refinenet4(layer_4_rn)
        # return path_4
        # path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
        # path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        # path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        # out = self.scratch.output_conv(path_1)

        # return torch.squeeze(out, dim=1)