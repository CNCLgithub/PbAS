from common import silence, mkdirs, fileExist, exrToNumpy, pngToNumpy, saveRotation, convertTxtToNpy, appendSaveTxt, loadTxt, rm
import importlib, math, timeit
import numpy as np
import os, subprocess
from multiprocessing import Process, Value
from shutil import copyfile



class Blender(object):
	def __init__(self, resultsPath='', renderer='blender', depthAndNormalRenderFormat = 'exr', renderAccuracy=16, objCount=None, maxDepth=0, minDepth=0, onlyRgbRender=False, rotLimitDegree=180):

		self.cwd = os.getcwd() + "/"

		self.renderResultPath = resultsPath
		self.depthAndNormalRenderFormat = depthAndNormalRenderFormat
		self.renderAccuracy = renderAccuracy
		self.materials = []


		if objCount is not None:
			self.objCount = objCount
			self.objCount.value = 0
		else:
			self.objCount = 0

		self.importBlender()
		self.scene = self.bpy.context.scene

		self.rotLimitDegree = rotLimitDegree

		# Blender Internal renderig engine
		self.renderEngs = ['blender', 'cycles']
		self.renderingEngine(renderer)
		self.scene.render.image_settings.color_mode = "RGB"
		self.scene.render.image_settings.file_format='PNG'
		self.scene.render.resolution_percentage = 100
		self.scene.render.image_settings.compression = 90
		self.scene.render.image_settings.color_depth = '16'
		self.scene.render.use_antialiasing = True
		self.scene.render.antialiasing_samples = '5'
		self.scene.render.use_sequencer = False
		self.scene.render.layers[0].use_pass_normal = True
		self.scene.display_settings.display_device = 'sRGB'
		self.scene.view_settings.view_transform = 'Raw'
		self.scene.sequencer_colorspace_settings.name = 'Raw'


		# Cycles settings
		self.scene.cycles.samples = 400
		self.scene.cycles.preview_samples = 12
		self.scene.cycles.min_bounces = 8
		self.scene.cycles.max_bounces = 20
		self.scene.cycles.transparent_min_bounces = 6
		self.scene.cycles.transparent_max_bounces = 20
		self.scene.cycles.diffuse_bounces = 12
		self.scene.cycles.glossy_bounces = 12
		self.scene.cycles.caustics_reflective = True
		self.scene.cycles.caustics_refractive = True
		self.scene.cycles.use_transparent_shadows = True
		if not onlyRgbRender:
			self.scene.use_nodes = True
			self.setupRenderNodes(depthAndNormalRenderFormat=depthAndNormalRenderFormat, renderAccuracy=renderAccuracy)
			self.scene.use_nodes = False
		self.scene.cycles.device = 'CPU'
		# self.bpy.context.user_preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'
		# self.bpy.context.user_preferences.addons['cycles'].preferences.devices[0].use = True # Uncomment if you have multiple GPUs


		# Some memory management
		self.scene.render.use_free_image_textures = True
		self.bpy.context.user_preferences.edit.undo_steps = 0
		self.bpy.context.user_preferences.edit.undo_memory_limit = 60
		self.bpy.context.user_preferences.edit.use_global_undo = False

		# Multiprocessing
		# self.processesCount = Value('l', 0)
		# self.removeBlocksJobList = []

		# Subprocess
		self.renderer = renderer
		self.renderAccuracy = renderAccuracy
		self.maxDepth = maxDepth
		self.minDepth = minDepth


	# @silence
	def importBlender(self):
		'''
		You need to have a compiled Blender 2.79 as a Python module in advance.
		During development of this project, it wasn't impossible to render objects with Blender versions of 2.8 and above in the background
		'''
		self.bpy = importlib.import_module("bpy")
		global Vector
		from mathutils import Vector

	def setupScene(self, lampPosList, camPosList, lampEnergy, solidName='', saveObjOnly=False, camIdx=None):


		self.activateLayer([0])
		self.cleanScene()
		self.nLamps = not saveObjOnly and len(lampPosList) or 0
		self.scene.world.light_settings.environment_energy = 0
		self.scene.world.light_settings.environment_color = 'PLAIN'
		self.scene.world.horizon_color = (0.0, 0.0, 0.0)

		if not saveObjOnly:
			# Add lamps
			self.addSun(location=tuple((0.0, 0.0, 0.082)))
			for i, coords in enumerate(lampPosList):
				self.bpy.ops.object.lamp_add(type='SPOT', location=tuple(coords))
				self.activeObj = self.scene.objects.active
				self.activeObj.name = 'lamp.0' + str(i)
				self.activeObj.data.name = 'lamp.0' + str(i)
				self.activeObj.data.shadow_method = 'NOSHADOW'
				self.activeObj.data.distance = 5.
				self.activeObj.data.energy = lampEnergy
				self.activeObj.data.spot_size = math.radians(30)
				self.activeObj.data.spot_blend = 0.
				self.pointObjTo(self.activeObj, (0.0, 0.0, 0.0))

			
			# Add cameras

			if camIdx is not None:
				if isinstance(camIdx, list):
					newCamList = []
					for idx in camIdx:
						newCamList.append(idx)
					camPosList = np.asarray(newCamList)
				else:
					camPosList = [camPosList[camIdx]]

			# Add a fake camera to resolve the issue of not being able to get renderings when the rendering function is called for the first time
			self.bpy.ops.object.camera_add(location=tuple(camPosList[0]))
			self.activeObj = self.scene.objects.active
			self.activeObj.name = 'camera.0' + str(0)
			self.activeObj.data.name = 'camera.0' + str(0)
			self.pointObjTo(self.activeObj, (0.0, 0.0, 0.0))


			rotVectorPath = self.cwd + '/' + 'camsRotVector' + solidName + '.txt'
			self.nCams = len(camPosList)
			rotVectors = []
			for i, coords in enumerate(camPosList):
				self.bpy.ops.object.camera_add(location=tuple(coords))
				self.activeObj = self.scene.objects.active
				self.activeObj.name = 'camera.0' + str(i+1)
				self.activeObj.data.name = 'camera.0' + str(i+1)
				rotMat = self.pointObjTo(self.activeObj, (0.0, 0.0, 0.0))
				rotVectors.append(rotMat)

			rotVectors = np.asarray(rotVectors)
			if fileExist(rotVectorPath) and camIdx is None:
				vects = np.loadtxt(rotVectorPath)
				if vects.size == 3:
					vects = vects.reshape(1, 3)
				if not np.isclose(rotVectors, vects).all(1).all():
					print ('==> The rotation vectors are different than the one you had on your machine. The new rotation vectors are saved on disk in camsRotVector.txt')
					np.savetxt(rotVectorPath, rotVectors, delimiter=' ')
			elif camIdx is None:
				np.savetxt(rotVectorPath, rotVectors, delimiter=' ')

			
			# Add a fake camera to resolve the issue of not being able to get renderings when the rendering function is called for the first time
			self.bpy.ops.object.camera_add(location=tuple(coords))
			self.activeObj = self.scene.objects.active
			self.activeObj.name = 'camera.0' + str(i+2)
			self.activeObj.data.name = 'camera.0' + str(i+2)
			self.pointObjTo(self.activeObj, (0.0, 0.0, 0.0))

	def simpleRender(self, objPath, resolution, renderPath, removeMats=False, rotVec=None, cloth=False, polish=False, optionalObjPath=None):
		# Renders an rgb image
		self.loadObj(objPath = objPath, layerIdx = 1, removeMaterials=removeMats, cloth=cloth, polish=polish)
		# if optionalObjPath is not None:
		# 	self.loadObj(objPath = optionalObjPath, layerIdx = 1, removeMaterials=removeMats, cloth=False, polish=polish)
		self.activateCamera(camNo = 1) # the 5th camera (No. 4) seems to be a good choice when global numpy random seed is 14
		self.setRenderingSettings(idx=0, resolution=resolution)
		if rotVec is not None:
			rotVec = np.array(rotVec).astype(dtype=np.float32)
			self.rotateObject(rotVec=rotVec, meshName='theMeshBlender')
		self.scene.render.filepath = renderPath
		self.saveRendering()


	# @silence
	def render(self, trainOrTest, objPath, category, gtIdx, resolutions, smallScaleRendering=False, optionalText='', copyRenderFile=False, newFilePath='', newFilePath2='', numRotation=0, removeMats=False, simultaneousRotation=0, removeMatAfterSimRotSteps=10000, numNewMatColor=0, rgb=False, cloth=False, polish=True, externalRotVec=None, depthAndNormal=False, numpySeed=None):
		
		# Note: The current code does not generate single-view renderings with the shapes rotated when rendering depth maps and surface Normals due to author requirements
		startTime = timeit.default_timer()
		
		if removeMats:
			removeMatAfterSimRotSteps = 10000

		if rgb:
			i = 0
			self.loadObj(objPath = objPath, layerIdx = 1, removeMaterials=removeMats, cloth=cloth, polish=polish)
		elif depthAndNormal:
			i = 1
			removeMats = True # Always add a void material for Cycles renderings
			self.loadObj(objPath = objPath, layerIdx = 2, removeMaterials=removeMats, polish=polish)
		else:
			print ('==> You can either render RGB, depth map or surface Normals')
			exit()
		if numpySeed is not None and numRotation == 0 or smallScaleRendering:
			np.random.seed(numpySeed)
		self.originalMatIDColor  = [[], []] # Stores the original materials before changing their colors

		
		mainCategoryPath = self.renderResultPath + (not smallScaleRendering and ('/' + trainOrTest + '/' + category) or '')
		for simultaneousRot in range(i == 0 and simultaneousRotation+1 or 1):
			renderingFileName = optionalText == '' and (simultaneousRot == 0 and '-RGBRenderPaths' or '-RGBSimultaneousRotRenderPaths') or (numRotation != 0 and '-RGBRotationRenderPaths' or '-RGBRenderPaths')
			for resolution in resolutions:
				internalRenderCounter = 0
				for internalRender in range(max(1, numNewMatColor+1)):
					for rotNum in range(simultaneousRotation == 0 and max(1, numRotation+1) or simultaneousRot == 1 and numRotation or 1):
						rotEnabled = simultaneousRotation == 0 and numRotation > 0 and rotNum > 0 or simultaneousRot == 1
						if rotEnabled:
							rot = externalRotVec is None and self.genRandomRotation() or externalRotVec
							rot = np.array(rot).astype(dtype=np.float32)
							if i == 0 and rotNum == removeMatAfterSimRotSteps:
								self.activateMesh(meshName='theMeshBlender', layerIdx=1)
								self.removeMaterials()
								self.addVoidBlenderMaterial()
							self.rotateObject(rotVec=rot, meshName=i == 0 and 'theMeshBlender' or 'theMeshCycles')
							# if simultaneousRot == 1:
							# 	self.scaleObject(window=0.15, meshName=i == 0 and 'theMeshBlender' or 'theMeshCycles')
						self.changeMaterialColor(idx=i, counter=internalRenderCounter, maxNumMatsToChangeColor=numNewMatColor) #Does not change the materials color if this is the first rendering
						self.setRenderingSettings(idx=i, resolution=resolution)
						for camNo in range(simultaneousRot == 0 and self.nCams or 1):
							if loadTxt(mainCategoryPath + '/corruptMeshes.txt', lookupStr=str(gtIdx)) == 1:
								rm(self.renderResultPath + (not smallScaleRendering and ('/' + trainOrTest + '/' + category + '/' + category + '-' + str(gtIdx)) or + '/' + optionalText))
								break # Do not continue rendering if one of the mesh is corrupted
							self.activateCamera(camNo = simultaneousRot == 0 and camNo+1 or 4+1) # the 5th camera (No. 4) seems to be a good choice when global numpy random seed is 14
							storagePath = self.renderResultPath + (not smallScaleRendering and ('/' + trainOrTest + '/' + category + '/' + category + '-' + str(gtIdx)) or '/' + optionalText) + '/' + str(resolution)
							if rotEnabled:
								rotationPath = storagePath
								saveRotation(storagePath, rot, simultaneousRotation==1)
							if i == 0:
								#render RGB using Blender Internal
								storagePath += (not cloth and '/rgb' or '/cloth') + (rotEnabled and '/X{0:.3f}_Y{1:.3f}_Z{2:.3f}'.format(rot[0], rot[1], rot[2]) or numRotation > 0 and '/X0.0_Y0.0_Z0.0' or '') + (externalRotVec is not None and '/X{0:.3f}_Y{1:.3f}_Z{2:.3f}'.format(externalRotVec[0], externalRotVec[1], externalRotVec[2]) or '') + (numNewMatColor > 0 and ('/color' + str(internalRenderCounter)) or '')
								mkdirs(storagePath)
								renderPath = storagePath + '/cam' + str(camNo) + '.png'
								self.scene.render.filepath =  renderPath
								self.saveRendering()
								if copyRenderFile:
									copyfile(storagePath + '/cam' + str(camNo) + '.png', newFilePath)
									if newFilePath2 != '':
										copyfile(storagePath + '/cam' + str(camNo) + '.png', newFilePath2)
								if camNo == 0 and internalRenderCounter == 0 and simultaneousRot == 0:
									
									# This is to prevent obtaining invalid renderings as the materials of some shapes might 
									# sometimes result in obtaining RGB, depth or surface Normal maps that are completely black
									if not smallScaleRendering:
										self.validateRenderResults(renderPath1=renderPath, resolution=resolution, numNewMatColor=numNewMatColor, category=category, gtIdx=gtIdx, trainOrTest=trainOrTest, objPath=objPath, mainCategoryPath=mainCategoryPath)

								appendSaveTxt(mainCategoryPath + '/' + str(resolution) + renderingFileName + '.txt' , renderPath, noDuplicate=True)
							else:
								# Render depth maps and surface Normals
								renderDirDepth = storagePath + '/depth' + (rotEnabled and '/X{0:.3f}_Y{1:.3f}_Z{2:.3f}'.format(rot[0], rot[1], rot[2]) or simultaneousRotation == 0 and numRotation > 0 and '/X0.0_Y0.0_Z0.0' or '')
								renderDirNormal = storagePath + '/normal' + (rotEnabled and '/X{0:.3f}_Y{1:.3f}_Z{2:.3f}'.format(rot[0], rot[1], rot[2]) or simultaneousRotation == 0 and numRotation > 0 and '/X0.0_Y0.0_Z0.0' or '')
								outputPaths = [renderDirDepth, renderDirNormal]
								mkdirs(outputPaths)
								self.depthOutputNode.base_path = outputPaths[0]
								self.depthOutputNode.file_slots[0].path = 'cam' + str(camNo) + "#"
								self.normalOutputNode.base_path = outputPaths[1]
								self.normalOutputNode.file_slots[0].path = 'cam' + str(camNo) + "#"
								self.saveRendering()
								if rotNum > 0 and camNo == 0 and simultaneousRot == 0:
									# This is to prevent obtaining invalid renderings as the materials of some shapes might 
									# sometimes result in obtaining RGB, depth or surface Normal maps that are completely black
									self.validateRenderResults(renderPath1=outputPaths[0] + '/cam' + str(camNo) + '1.' + self.depthAndNormalRenderFormat, renderPath2=outputPaths[1] + '/cam' + str(camNo) + '1.' + self.depthAndNormalRenderFormat, resolution=resolution , category=category, gtIdx=gtIdx, trainOrTest=trainOrTest, objPath=objPath, mainCategoryPath=mainCategoryPath)
								renderPathDepth = renderDirDepth + '/cam' + str(camNo) + "1.exr" # 1 becuase we are rendering from frame 1
								renderPathNormal = renderDirNormal + '/cam' + str(camNo) + "1.exr" # 1 becuase we are rendering from frame 1
								appendSaveTxt(mainCategoryPath + '/' + str(resolution) + (simultaneousRot == 0 and '-DepthRenderPaths' or '-DepthSimultaneousRotRenderPaths') + '.txt', renderPathDepth, noDuplicate=True)
								appendSaveTxt(mainCategoryPath + '/' + str(resolution) + (simultaneousRot == 0 and '-NormalRenderPaths' or '/-NormalSimultaneousRotRenderPaths') + '.txt', renderPathNormal, noDuplicate=True)
					if rotEnabled:
						convertTxtToNpy(rotationPath, 'rotations')
					internalRenderCounter += 1

		self.killMeshes(layer = i == 0 and 1 or 2)

		if isinstance(self.objCount, int):
			self.objCount += 1
			objCount = self.objCount
		else:
			self.objCount.value += 1
			objCount = self.objCount.value

		# Remove unlinked data blocks to prevent memory leakage 			
		# removeJob = Process(target=self.removeDataBlocks, kwargs={'counter': self.processesCount})
		# self.removeBlocksJobList.append(removeJob)
		# removeJob.start()
		# removeJob.join()


	@silence
	def saveRendering(self, saveOnDisk=True):
		self.bpy.ops.render.render(write_still=saveOnDisk)





	# Utility functions
	def setRenderingSettings(self, idx, resolution):
		self.renderingEngine(self.renderEngs[idx])
		self.setResolution(resolution)
		self.activateLayer(idx == 0 and [0, 1] or [2])
		if idx == 0: # Blender Internal
			self.scene.use_nodes = False
			self.scene.render.tile_x = 16
			self.scene.render.tile_y = 16
		else: # Cycles
			self.scene.use_nodes = True
			self.scene.render.tile_x = 16
			self.scene.render.tile_y = 16


	def setupRenderNodes(self, depthAndNormalRenderFormat, renderAccuracy):
		# Render Layer node
		for node in self.scene.node_tree.nodes:
			self.scene.node_tree.nodes.remove(node)

		renderNode = self.scene.node_tree.nodes.new('CompositorNodeRLayers')


		# Depth map nodes
		normalizeNode = self.scene.node_tree.nodes.new('CompositorNodeNormalize')
		invertNode = self.scene.node_tree.nodes.new('CompositorNodeInvert')
		self.depthOutputNode = self.scene.node_tree.nodes.new('CompositorNodeOutputFile')

		self.depthOutputNode.format.file_format = depthAndNormalRenderFormat == 'png' and 'PNG' or 'OPEN_EXR'
		self.depthOutputNode.format.color_depth = renderAccuracy == 16 and '16' or '32'
		self.depthOutputNode.format.color_mode = 'RGB'
		if depthAndNormalRenderFormat == 'exr':
			self.depthOutputNode.format.exr_codec = 'ZIP'
			# self.depthOutputNode.format.use_zbuffer = True


		# Normal map nodes
		self.normalOutputNode = self.scene.node_tree.nodes.new('CompositorNodeOutputFile')
		self.normalOutputNode.format.file_format = depthAndNormalRenderFormat == 'png' and 'PNG' or 'OPEN_EXR'
		self.normalOutputNode.format.color_depth = renderAccuracy == 16 and '16' or '32'
		self.normalOutputNode.format.color_mode = 'RGB'
		if depthAndNormalRenderFormat == 'exr':
			self.normalOutputNode.format.exr_codec = 'ZIP'


		# Links
		# Depth map
		if depthAndNormalRenderFormat == 'png':
			# Note that this transforms the values. To store the non-transformed values use exr format
			self.scene.node_tree.links.new(renderNode.outputs[2], invertNode.inputs[1])
			self.scene.node_tree.links.new(invertNode.outputs[0], normalizeNode.inputs[0])
			self.scene.node_tree.links.new(normalizeNode.outputs[0], self.depthOutputNode.inputs[0])
		else:
			self.scene.node_tree.links.new(renderNode.outputs[2], self.depthOutputNode.inputs[0])

		# Surface Normal map
		self.scene.node_tree.links.new(renderNode.outputs[3], self.normalOutputNode.inputs[0])


	def validateRenderResults(self, renderPath1, resolution, category, gtIdx, trainOrTest, objPath, mainCategoryPath, renderPath2='', numNewMatColor=0):
		if self.renderer == 'CYCLES':
			if self.depthAndNormalRenderFormat == 'exr':
				npArr1 = exrToNumpy(exrPaths=renderPath1, renderType='depth', resolution=resolution, renderAccuracy=self.renderAccuracy, maxDepth=self.maxDepth, minDepth=self.minDepth)
				npArr2 = exrToNumpy(exrPaths=renderPath2, renderType='normal', resolution=resolution, renderAccuracy=self.renderAccuracy, maxDepth=self.maxDepth, minDepth=self.minDepth)
			else:
				npArr1 = pngToNumpy(pngPath=renderPath1, renderType='depth', resolution=resolution)
				npArr2 = pngToNumpy(pngPath=renderPath2, renderType='normal', resolution=resolution)
			if npArr1.max() == npArr1.min() or npArr2.max() == npArr2.min() or npArr1.size != resolution**2 or npArr2.size != resolution**2*3:
				rm(renderPath1)
				rm(renderPath2)
				self.quickRerender(meshName='theMeshCycles', layerIdx=2)

				if self.depthAndNormalRenderFormat == 'exr':
					npArr1 = exrToNumpy(exrPaths=renderPath1, renderType='depth', resolution=resolution, renderAccuracy=self.renderAccuracy, maxDepth=self.maxDepth, minDepth=self.minDepth)
					npArr2 = exrToNumpy(exrPaths=renderPath2, renderType='normal', resolution=resolution, renderAccuracy=self.renderAccuracy, maxDepth=self.maxDepth, minDepth=self.minDepth)
				else:
					npArr1 = pngToNumpy(pngPath=renderPath1, renderType='depth', resolution=resolution)
					npArr2 = pngToNumpy(pngPath=renderPath2, renderType='normal', resolution=resolution)
				if npArr1.max() == npArr1.min() or npArr2.max() == npArr2.min() or npArr1.size != resolution**2 or npArr2.size != resolution**2*3:
					rm(renderPath1)
					rm(renderPath2)
					self.killMeshes(layer = 2)
					self.loadObj(objPath = objPath, layerIdx = 2, recenter=True)
					self.quickRerender(meshName='theMeshCycles', layerIdx=1)
					if self.depthAndNormalRenderFormat == 'exr':
						npArr1 = exrToNumpy(exrPaths=renderPath1, renderType='depth', resolution=resolution, renderAccuracy=self.renderAccuracy, maxDepth=self.maxDepth, minDepth=self.minDepth)
						npArr2 = exrToNumpy(exrPaths=renderPath2, renderType='normal', resolution=resolution, renderAccuracy=self.renderAccuracy, maxDepth=self.maxDepth, minDepth=self.minDepth)
					else:
						npArr1 = pngToNumpy(pngPath=renderPath1, renderType='depth', resolution=resolution)
						npArr2 = pngToNumpy(pngPath=renderPath2, renderType='normal', resolution=resolution)
					if npArr1.max() == npArr1.min() or npArr2.max() == npArr2.min() or npArr1.size != resolution**2 or npArr2.size != resolution**2*3:
						rm(mainCategoryPath + '/' + category + '-' + str(gtIdx))
						print ('==> Err: It looks like that Cycles object with gtIdx ' + str(gtIdx) + " with path '" + objPath +  "' from category " + category + ' cannot be rendered properly, even after replacing its materials. The maximum and minimum value of the rendering pixels are equal')
						appendSaveTxt(mainCategoryPath + '/corruptMeshes.txt', ',,'.join([trainOrTest, 'Cycles', str(gtIdx), objPath]))
						self.exitBlender()
		else:
			npArr = pngToNumpy(pngPath=renderPath1, renderType='rgb', resolution=resolution)
			if npArr.max() == npArr.min() or npArr.size != resolution**2*3:
				self.quickRerender(meshName='theMeshBlender', layerIdx=1)
				npArr = pngToNumpy(pngPath=renderPath1, renderType='rgb', resolution=resolution)
				if npArr.max() == npArr.min() or npArr.size != resolution**2*3:
					self.killMeshes(layer = 1)
					self.loadObj(objPath = objPath, layerIdx = 1, recenter=True)
					self.quickRerender(meshName='theMeshBlender', layerIdx=1)
					npArr = pngToNumpy(pngPath=renderPath1, renderType='rgb', resolution=resolution)
					if npArr.max() == npArr.min() or npArr.size != resolution**2*3:
						rm(mainCategoryPath + '/' + category + '-' + str(gtIdx))
						print ('==> Err: It looks like that Blender internal object with gtIdx ' + str(gtIdx) + " with path '" + objPath +  "' from category " + category + ' cannot be rendered properly, even after replacing its materials. The maximum and minimum value of the rendering pixels are equal')
						appendSaveTxt(mainCategoryPath + '/corruptMeshes.txt', ',,'.join([trainOrTest, 'Blender', str(gtIdx), objPath]))
						self.exitBlender()

	def quickRerender(self, meshName, layerIdx, numNewMatColor=0):
		self.activateMesh(meshName=meshName, layerIdx=layerIdx)
		self.removeMaterials()
		self.addVoidBlenderMaterial()
		if meshName != 'theMeshCycles':
			self.originalMatIDColor  = [[], []]
			self.changeMaterialColor(idx=layerIdx, counter=0, maxNumMatsToChangeColor=numNewMatColor)
		self.saveRendering()

	def loadBlendFile(self, path='test.blend'):
		self.bpy.ops.wm.open_mainfile(filepath=path)

	def saveBlendFile(self, path='test.blend'):
		self.bpy.ops.wm.save_as_mainfile(filepath=path)

	@silence
	def exitBlender(self):
		self.bpy.ops.wm.quit_blender()

	def setResolution(self, newRes):
		self.scene.render.resolution_x = newRes
		self.scene.render.resolution_y = newRes

	def removeDataBlocks(self, removeAll=False, counter=None):
		# Removes unlinked data blocks and prevents memory leakage
		if counter is not None:
			counter.value += 1

		toRemove = [block for block in self.bpy.data.meshes if block.users == 0]
		for block in toRemove:
			self.bpy.data.meshes.remove(block, do_unlink=True)

		toRemove = [block for block in self.bpy.data.materials if block.users == 0]
		for block in toRemove:
			self.bpy.data.materials.remove(block, do_unlink=True)

		toRemove = [block for block in self.bpy.data.textures if block.users == 0]
		for block in toRemove:
			self.bpy.data.textures.remove(block, do_unlink=True)

		toRemove = [block for block in self.bpy.data.images if block.users == 0]
		for block in toRemove:
			self.bpy.data.images.remove(block, do_unlink=True)

		if removeAll:
			toRemove = [block for block in self.bpy.data.cameras if block.users == 0]
			for block in toRemove:
				self.bpy.data.cameras.remove(block, do_unlink=True)

			toRemove = [block for block in self.bpy.data.lamps if block.users == 0]
			for block in toRemove:
				self.bpy.data.lamps.remove(block, do_unlink=True)
		
		if counter is not None:
			counter.value -= 1

	def addSun(self, location=(0.0, 0.0, 0.0)):
		self.bpy.ops.object.lamp_add(type='SUN', location=location)
		self.activeObj = self.scene.objects.active
		self.activeObj.name = 'interiorIlluminator'
		self.activeObj.data.name = 'interiorIlluminator'
		self.activeObj.data.shadow_method = 'NOSHADOW'
		if self.renderer == "BLENDER_RENDER":
			self.activeObj.data.energy = 0.1
		else:
			self.activeObj.data.use_nodes = True


	# @silence
	def cleanScene(self):
		self.killLamps()
		self.killMeshes()
		self.killCameras()
		self.removeDataBlocks(removeAll=True)

	# @silence
	def killMeshes(self, layer = -1):
		for obj in self.scene.objects:
			if obj.type == 'MESH' and obj.layers[layer != -1 and layer or self.activeLayer[0]]:
				obj.select = True
			else:
				obj.select = False
		self.bpy.ops.object.delete()

	# @silence
	def killCameras(self):
		for obj in self.scene.objects:
			if obj.type == 'CAMERA' and obj.layers[self.activeLayer[0]]:
				obj.select = True
			else:
				obj.select = False
		self.bpy.ops.object.delete()

	# @silence
	def killLamps(self):
		for obj in self.scene.objects:
			if obj.type == 'LAMP' and obj.layers[self.activeLayer[0]]:
				obj.select = True
			else:
				obj.select = False
		self.bpy.ops.object.delete()

	# @silence
	def joinMeshes(self, meshName):

		selectedObjs = []
		for obj in self.scene.objects:
			if obj.type == 'MESH' and obj.layers[self.activeLayer[0]]:
				obj.select = True
				selectedObjs.append(obj.name) # Just filling this up
				self.scene.objects.active = obj
			else:
				obj.select = False
		if len (selectedObjs) > 1:
			self.bpy.ops.object.join()
		self.scene.objects.active.name = meshName
		self.scene.objects.active.data.name = meshName

	# @silence
	def polishShape(self, harsh):
		# The main purpose of this function is to fix the normal maps but it also does some mesh simplification
		# Expects a shape to have been selected in the scene
		# set harsh to True if to simplify the mesh. Mesh simplification will result in detailedness loss of the shapes

		self.bpy.ops.object.mode_set(mode='EDIT')
		self.bpy.ops.mesh.normals_make_consistent(inside=False)
		if not harsh:
			self.bpy.ops.mesh.remove_doubles(threshold=0.0004)
			self.bpy.ops.mesh.dissolve_limited(angle_limit=0.000227) # 0.013 degrees
		else:
			self.bpy.ops.mesh.dissolve_limited(angle_limit=0.04712389) # ~2.7 degrees
			self.triangulateFaces()
			self.bpy.ops.mesh.dissolve_limited(angle_limit=0.04712389) # ~2.7 degrees
			# self.bpy.ops.mesh.fill_holes(sides=3)
		self.triangulateFaces()
		self.bpy.ops.mesh.normals_make_consistent(inside=False)
		self.bpy.ops.object.mode_set(mode='OBJECT')
		self.activeObj.data.use_auto_smooth = False

	def fixNormals(self):
		self.bpy.ops.object.mode_set(mode='EDIT')
		self.bpy.ops.mesh.normals_make_consistent(inside=False)
		self.bpy.ops.object.mode_set(mode='OBJECT')

	# @silence
	def triangulateFaces(self):
		# Expects a shape to have been selected in the scene
		self.bpy.ops.mesh.quads_convert_to_tris(quad_method='BEAUTY', ngon_method='BEAUTY')

	def applyTransformation(self, scale, rotation, location=False):
		self.bpy.ops.object.transform_apply(location = location, scale = scale, rotation = rotation)

	@silence
	def loadObj(self, objPath, layerIdx, removeMaterials=False, cloth=False, polish=True, harshPolish=False, recenter=False, save=False, rotVec=None, objSavePath=''):

		if layerIdx == 1:
			self.renderingEngine('blender')
			if not cloth:
				meshName = 'theMeshBlender'
			else:
				meshName = 'clothMesh'
		elif layerIdx == 2:
			self.renderingEngine('cycles')
			meshName = 'theMeshCycles'

		self.activateLayer([layerIdx]) # For rendering RGBs (layerIdx == 1). For rendering depth maps or Normals using Cycles (layerIdx == 2)
		self.bpy.ops.import_scene.obj(filepath=objPath, split_mode="OFF", use_smooth_groups=False, use_edges=False, axis_forward='-Z', axis_up='Y')
		self.joinMeshes(meshName=meshName) #Useful only if split_mode is equated to 'ON' otherwise objects are already joined by the time they are imported
		self.activeCurrentObj()
		if not cloth:
			if polish:
				if save:
					self.fixNormals()
					self.activeObj.modifiers.new('Solidify', 'SOLIDIFY')
					self.activeObj.modifiers[0].thickness = -0.0001
					self.activeObj.modifiers[0].use_quality_normals = True
					self.activeObj.modifiers[0].use_rim = True
					# self.activeObj.modifiers[0].use_rim_only = True
				self.polishShape(harsh=harshPolish)
		self.bpy.ops.object.shade_smooth()

		if removeMaterials and not save:
			self.activateMesh(meshName=meshName, layerIdx=layerIdx)
			self.removeMaterials()
			self.addVoidMaterial(layerIdx=layerIdx)
		elif not removeMaterials:
			# self.modifyTransparentMaterials()
			pass
		
		if cloth:
			self.activateMesh(meshName=meshName, layerIdx=layerIdx)
			self.removeMaterials()
			self.addClothMaterial()
			# self.activeObj.scale = (0.4, 0.4, 0.4)
			clothLocation = self.activeObj.location
			self.activateCamera(camNo=1)
			# self.scene.camera.location = self.scene.camera.location * 1.2

		if recenter:
			self.bpy.ops.object.origin_set(type='GEOMETRY_ORIGIN', center='BOUNDS')
			self.bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY')
			self.activeObj.location = Vector((0, 0, 0))
			self.centroid = self.computeCentroid()
			self.focus()

		self.applyTransformation(scale=False, rotation=True)
		if rotVec is not None:
			self.rotateObject(rotVec=rotVec)
		self.scene.update()

		if save:
			if objSavePath == '':
				print ('==> Err: You need to speficy the path where the Obj file is going to be saved')
				exit()
			self.saveObj(objSavePath=objSavePath)

	def rotateObject(self, rotVec, meshName='theMeshBlender'):
		rot = np.radians(rotVec)
		self.activateMesh(meshName=meshName)
		self.activeObj.rotation_euler = Vector(rot)
		self.scene.update()

	def scaleObject(self, window, meshName='theMeshBlender'):
		self.activateMesh(meshName=meshName)
		scaleFactor = np.random.uniform(low=max(-0.05, -window), high=min(0.2, window), size=1)[0]
		self.activeObj.scale = Vector((1+scaleFactor, 1+scaleFactor, 1+scaleFactor))
		self.scene.update()

	# @silence
	def focusCamera(self):
		#Use self.nCams, self.centroid
		pass

	# @silence
	def focusLight(self):
		#Use self.nLamps, self.centroid
		pass

	# @silence
	def focus(self):
		self.focusLight()
		self.focusCamera()


	@silence
	def saveObj(self, objSavePath):
		self.bpy.ops.export_scene.obj(
                filepath=objSavePath, 
                check_existing=False, 
                axis_forward='-Z', 
                axis_up='Y', 
                use_selection=True, 
                use_animation=False, 
                use_mesh_modifiers=True, 
                use_edges=False, 
                use_smooth_groups=False, 
                use_smooth_groups_bitflags=False, 
                use_normals=False, 
                use_uvs=False, 
                use_materials=False, 
                use_triangles=True, 
                use_nurbs=False, 
                use_vertex_groups=False, 
                use_blen_objects=True, 
                group_by_object=False, 
                group_by_material=False, 
                keep_vertex_order=True, 
                global_scale=1, 
                path_mode='AUTO')


	def pointObjTo(self, obj, xyzTarget, justDirection=False):
		# This function operates directly on the input object (obj)
		xyzTarget = Vector(xyzTarget)
		direction = xyzTarget - obj.location
		rot_quat = direction.to_track_quat('-Z', 'Y')
		obj.rotation_euler = rot_quat.to_euler()
		return rot_quat.to_euler()

	def activateLayer(self, layersToSwitchTo):
		#layersToSwitchTo is a List containing integers in the range [0, 19]
		for i in range(20):
			if i in layersToSwitchTo:
				self.scene.layers[i] = True
			
		for i in range(20):
			if i not in layersToSwitchTo:
				self.scene.layers[i] = False
		self.activeLayer = layersToSwitchTo

	def activateCamera(self, camNo):
		self.scene.camera = self.scene.objects['camera.0' + str(camNo)]
		self.scene.update()

	def activeCurrentObj(self):
		self.activeObj = self.scene.objects.active

	def activateMesh(self, meshName, layerIdx=-1):
		for obj in self.scene.objects:
			if ((obj.type == 'MESH' or obj.type == 'CAMERA') and obj.name == meshName and obj.layers[layerIdx != -1 and layerIdx or self.activeLayer[0]]) or ((obj.type == 'MESH' or obj.type == 'CAMERA') and obj.name == meshName):
				obj.select = True
				self.scene.objects.active = obj
			else:
				obj.select = False
		self.activeCurrentObj()

	def renderingEngine(self, renderEng):
		self.renderer = renderEng == 'blender' and 'BLENDER_RENDER' or 'CYCLES'
		self.scene.render.engine = self.renderer




	# Material-related functions
	def removeMaterials(self):
		# Removes the materials for an active mesh object
		mats = self.activeObj.data.materials
		if len(mats) > 0:
			for mat in mats:
				mat.user_clear()

			self.removeDataBlocks()
			for i in range(len(mats)):
				mats.pop(0, update_data=True)
			mats.clear(1) # Do not execute this line. It might sometimes cause the object not be shown in final rendering

	def addVoidMaterial(self, layerIdx):
		if layerIdx == 1:
			self.addVoidBlenderMaterial()
		elif layerIdx == 2:
			self.addVoidCyclesMaterial()

	def addVoidBlenderMaterial(self):
		# Applies a void material on an active mesh object
		mats = self.activeObj.data.materials
		newMat = self.bpy.data.materials.new(name='voidMat')
		newMat.diffuse_color = (0.75, 0.75, 0.75)
		newMat.diffuse_intensity = 0.75
		newMat.specular_intensity = 0.07
		mats.append(newMat)

	def addVoidCyclesMaterial(self):
		newMat = self.bpy.data.materials.new('voidCyclesMaterial')
		newMat.use_nodes = True
		self.activeObj.active_material = newMat

	def addClothMaterial(self):
		# Applies a void material on an active mesh object
		mats = self.activeObj.data.materials
		newMat = self.bpy.data.materials.new(name='clothMat')
		# newMat.diffuse_color = (0.55, 0.25, 0.8)
		newMat.diffuse_color = (0.8, 0.8, 0.8)
		newMat.diffuse_intensity = 0.8
		newMat.specular_intensity = 0.05
		mats.append(newMat)

	def modifyTransparentMaterials(self):
		# Modify the transparent materials so that they are more transparent while rendering
		# RGB images and allow capturing inside of the 3D shapes (e.g. through 
		# the windshield of cars) when rendering depth maps or surface Normals

		mats = self.activeObj.data.materials
		if len(mats) > 0:
			for mat in mats:
				transParentFound = True
				if self.renderer == 'BLENDER_RENDER':
					if mat.use_transparency:
						mat.alpha = mat.alpha > 0.3 and mat.alpha < 0.95 and 0.3 or mat.alpha
						mat.transparency_method = 'Z_TRANSPARENCY'
				else:
					# Add shader nodes to the transparent materials in Cycles to let depth maps and Normal map rendering rays go through glass-like materials
					for node in mat.node_tree.nodes:
						if node.label == "Mix Color/Alpha":
							if node.inputs[1].default_value[0] < 1:
								mixColorAlphaNode = node
								for n in mat.node_tree.nodes:
									if n.label == "Alpha BSDF":
										alphaBSDFNode = n
									elif n.label == "Shader Add Refl":
										shaderAddReflNode = n
									elif n.label == "Material Out":
										materialOutNode = n

								# Math node
								mat.node_tree.nodes.new('ShaderNodeMath')
								mathNode = mat.node_tree.nodes[len(mat.node_tree.nodes)-1] # Last added node
								mathNode.operation = 'LESS_THAN'
								mathNode.inputs[1].default_value = 1.0

								# Mix shader node
								mat.node_tree.nodes.new('ShaderNodeMixShader')
								mixShaderNode = mat.node_tree.nodes[len(mat.node_tree.nodes)-1] # Last added node

								# Add links
								mat.node_tree.links.new(mathNode.outputs['Value'], mixShaderNode.inputs[0])
								mat.node_tree.links.new(mixColorAlphaNode.outputs[0], mathNode.inputs[0])
								mat.node_tree.links.new(mixColorAlphaNode.outputs[0], mathNode.inputs[0])
								mat.node_tree.links.new(shaderAddReflNode.outputs[0], mixShaderNode.inputs[1])
								mat.node_tree.links.new(alphaBSDFNode.outputs[0], mixShaderNode.inputs[2])
								mat.node_tree.links.new(mixShaderNode.outputs[0], materialOutNode.inputs[0])
			if not transParentFound:
				# TODO remove shader nodes for fast rendering in Cycles
				pass

	def changeMaterialColor(self, idx, counter, maxNumMatsToChangeColor=2):

		if idx == 0 and maxNumMatsToChangeColor > 0: #idx = 0 is used for doing rendering using Blender's internal renderer
			polys = self.activeObj.data.polygons
			mats = self.activeObj.data.materials

			if len(mats) > 0:
				totalArea = self.computeTotalArea(polys)

				if len(self.originalMatIDColor[0]) == 0:
					coloredMatCount = 0
					
					# Get the indices of faces assigned to materials
					materialPolys = { ms.material.name : [] for ms in self.activeObj.material_slots }
					for idx, poly in enumerate( self.activeObj.data.polygons ):
					    materialPolys[ self.activeObj.material_slots[ poly.material_index ].name ].append( idx )


				    # Compute the area of faces to which each material has been assigned
					for key, val in materialPolys.items():
						faceArea = 0
						for poly in val:
							faceArea += self.areaOfPoly(polys[poly])
						materialPolys[key] = faceArea/totalArea # Normalize the area to get probabilities

					materialPolys = sorted(materialPolys.items(), key=lambda x: x[1], reverse=True) # Sort by area covered for each material
				
					for keyValue in materialPolys:
						# keyValue[1] holds the percentage of the entire area covered by each material
						matID = self.getMatID(mats, keyValue[0])
						colorVec = np.asarray(mats[matID].diffuse_color)
						if coloredMatCount <= maxNumMatsToChangeColor:
							if self.getMatID(mats, mats[keyValue[0]].name) not in self.originalMatIDColor[0] and ((keyValue[1] > 0.08 and colorVec.sum() > 0.65) or keyValue[1] > 0.2): # The material should cover 8% of the object, at least. This prevents tires or rims change color
								self.originalMatIDColor[0].append(self.getMatID(mats, mats[keyValue[0]].name))
								self.originalMatIDColor[1].append(np.asarray(mats[keyValue[0]].diffuse_color).copy()) # Use copy() so that the contents of self.originalMatIDColor will not be changed
								coloredMatCount += 1

				# Change the material color
				for i in range(len(self.originalMatIDColor[0])):
					matID = self.originalMatIDColor[0][i]
					colorVec = self.originalMatIDColor[1][i]
					mat = mats[matID]
					mat.diffuse_color = colorVec.copy().tolist()
					if counter > 0:
						(newColor, newColorIntensity) = self.changeMatColor(mat)
						mats[matID].diffuse_color = newColor
						mats[matID].diffuse_intensity = newColorIntensity
					else:
						# Replace the original colors when starting to render for new resolutions
						mats[matID].diffuse_color = mat.diffuse_color


	def getMatID(self, materials, matName):
		# Finds material ID using its matName
		matID = -1
		for idx, mat in enumerate(materials):
			if mat.name == matName:
				matID = idx
				break

		return matID

	def colorDifference(self, colorVec):
		return colorVec.std()/(colorVec.mean() + 0.0001) #Avoid division by zero

	def changeMatColor(self, mat, changeRate=0.13):
		# mat is a Blender Internal material object

		colorVec = np.asarray(mat.diffuse_color)
		colorIntensity = mat.diffuse_intensity
		minColor = 0.06
		if mat.use_transparency and mat.alpha < 0.95:
			colorIntensity = np.random.uniform(low=max(0.1, colorIntensity - 0.05), high=min(1, colorIntensity + 0.05))
			# For transparent materials, change color with much less variacne
			changeRate -= max(0, 0.1)
			# colorVec = np.asarray(mat.diffuse_color)
			colorVec[0] = np.random.uniform(low=max(minColor, colorVec[0]-changeRate), high=min(1, max(minColor + changeRate, colorVec[0]+changeRate)), size=1)[0]
			colorVec[1] = np.random.uniform(low=max(minColor, colorVec[1]-changeRate), high=min(1, max(minColor + changeRate, colorVec[0]+changeRate)), size=1)[0]
			colorVec[2] = np.random.uniform(low=max(minColor, colorVec[2]-changeRate), high=min(1, max(minColor + changeRate, colorVec[0]+changeRate)), size=1)[0]
		else:
			colorIntensity = np.random.uniform(low=max(0.1, colorIntensity - 0.1), high=min(1, colorIntensity + 0.1))
			if colorVec.sum() <= 0.65 and self.colorDifference(colorVec) <= 0.12:
				# Small change of the colors if the material color is black (or close to dark black)
				changeRate -= max(0, 0.09)
				colorVec[0] = np.random.uniform(low=max(minColor, colorVec[0]-changeRate), high=min(1, max(minColor + changeRate, colorVec[0]+changeRate)), size=1)[0]
				colorVec[1] = np.random.uniform(low=max(minColor, colorVec[1]-changeRate), high=min(1, max(minColor + changeRate, colorVec[0]+changeRate)), size=1)[0]
				colorVec[2] = np.random.uniform(low=max(minColor, colorVec[2]-changeRate), high=min(1, max(minColor + changeRate, colorVec[0]+changeRate)), size=1)[0]
			elif colorVec.sum() >= 2.2 and self.colorDifference(colorVec) <= 0.12:
				# Small change of the colors if the material color is gray, towards white
				changeRate -= max(0, 0.09)
				colorVec[0] = np.random.uniform(low=max(minColor, colorVec[0]-changeRate), high=min(1, max(minColor + changeRate, colorVec[0]+changeRate)), size=1)[0]
				colorVec[1] = np.random.uniform(low=max(minColor, colorVec[1]-changeRate), high=min(1, max(minColor + changeRate, colorVec[0]+changeRate)), size=1)[0]
				colorVec[2] = np.random.uniform(low=max(minColor, colorVec[2]-changeRate), high=min(1, max(minColor + changeRate, colorVec[0]+changeRate)), size=1)[0]
			else:
				colorVec[0] = np.random.uniform(low=max(minColor, colorVec[0]-changeRate), high=min(1, max(minColor + changeRate, colorVec[0]+changeRate)), size=1)[0]
				colorVec[1] = np.random.uniform(low=max(minColor, colorVec[1]-changeRate), high=min(1, max(minColor + changeRate, colorVec[0]+changeRate)), size=1)[0]
				colorVec[2] = np.random.uniform(low=max(minColor, colorVec[2]-changeRate), high=min(1, max(minColor + changeRate, colorVec[0]+changeRate)), size=1)[0]
		return (colorVec, colorIntensity)


	def computeTotalArea(self, polygons):
		# The assumption is that the polygons are triangles (3 vertices)
		totalArea = 0
		for poly in polygons:
			totalArea += self.areaOfPoly(poly)
		return totalArea

	def getNumpyVerts(self, polygon):
		verts = []
		for i in range(len(polygon.vertices)):
			verts.append(np.asarray((self.activeObj.matrix_world * self.activeObj.data.vertices[polygon.vertices[i].co]).to_tuple()))
		return verts

	def centerOfPoly(self, polygon):
		verts = self.getNumpyVerts(polygon)
		summ = np.sum(verts, axis=0)
		return summ/len(polygon.vertices)

	def areaOfPoly(self, polygon):
		# The assumption is the polygon is a triangle (3 vertices)
		verts = self.getNumpyVerts(polygon)
		area = 0.5 * np.sqrt(np.linalg.norm(np.cross(np.subtract(verts[1], verts[0]), np.subtract(verts[2], verts[0]))))
		return area

	# @silence
	def computeCentroid(self):
		totalArea = 0.0
		centroid = np.asarray((0.0, 0.0, 0.0))
		for poly in self.activeObj.data.polygons:
			area = self.areaOfPoly(poly)
			polyCenter = self.centerOfPoly(poly)
			centroid += area*polyCenter
			totalArea += area
		centroid /= totalArea
		return centroid

	def genRandomRotation(self):
		# pi180 = np.pi / 180
		# rot = np.array([np.random.randint(-self.rotLimitDegree, self.rotLimitDegree) * pi180, np.random.randint(-self.rotLimitDegree, self.rotLimitDegree) * pi180, np.random.randint(-self.rotLimitDegree, self.rotLimitDegree) * pi180], dtype=np.float32)
		rot = np.random.uniform(-self.rotLimitDegree, self.rotLimitDegree, 3).astype(dtype=np.float32)
		return rot.tolist()
