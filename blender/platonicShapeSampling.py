from common import silence, fileExist, numLinesInFile
import numpy as np
import os, importlib, math, sys

class PlatonicShapeSampling(object):
	def __init__(self, fileName, fromScratch, numCameras):
		self.importBlender()
		self.scene = self.bpy.context.scene
		self.fileName = fileName
		self.camPosPath = os.getcwd() + "/" + self.fileName
		self.fromScratch = fromScratch
		self.numCameras = numCameras

	# @silence
	def importBlender(self):
		self.bpy = importlib.import_module("bpy")

	def equidistantPlatonicPoints(self, fromScratch=0, radius=1, numVertices=8):
		#TODO store on disk
		if numVertices == 4:
			# TODO: implement tetrahedron
			# https://blender.stackexchange.com/questions/102908/making-regular-polyhedra-in-blender-using-python/102999#102999
			sys.exit()
		elif numVertices == 6:
			# TODO: implement octahedron
			# https://blender.stackexchange.com/questions/102908/making-regular-polyhedra-in-blender-using-python/102999#102999
			sys.exit()
		elif numVertices == 8:
			self.cube(radius=radius)
		elif numVertices == 12:
			self.killMeshes()
			self.icosahedron(radius=radius)
		elif numVertices ==  20:
			self.killMeshes()
			self.dodecahedron(radius=radius)
		else:
			print ('==> You need to choose either 4, 6, 8, 12 or 20 for the number of vertices (opts.numVPs) to be able to generate camera positions using a regular solid')
			sys.exit()
		
		if fileExist(self.camPosPath):
			points = np.loadtxt(self.camPosPath)
			if not np.array_equal(self.coords, points):
				answer = input("==> It seems that you have changed the random seed or number of views. Are you sure you want to continue?\n==> Answering 'y' or 'yes' will result in doing the re-rendering process. It will also generate a new '" + self.fileName + "' file. So make sure you back up the old '" + self.fileName + "' file before you continue\n")
				if answer[0].lower() == 'y' or answer[0].lower() == 'yes':
					points = np.copy(self.coords)
					np.savetxt(self.camPosPath, self.coords, delimiter=' ')
					self.fromScratch = 1
				else:
					self.coords = points
					self.numCameras = numLinesInFile(self.camPosPath)
					print ("==> The currently available '" + self.fileName + "' has been loaded. opt.numVPs is going to be set to the number of lines in '" + self.fileName + "' which is " + str(self.numCameras))
		else:
			np.savetxt(self.camPosPath, self.coords, delimiter=' ')



	# Shape functions
	def cube(self, radius):
		self.shapeName = 'cube'
		self.selectObj(objName='Cube')
		self.scaleShape(radius=radius)
		self.rotateShape()
		self.getCoords()

	def icosahedron(self, radius):
		self.shapeName = 'icosahedron'
		self.addIcosahedron()
		self.scaleShape(radius=radius)
		self.rotateShape()
		self.getCoords()

	def dodecahedron(self, radius):
		self.shapeName = 'dodecahedron'
		self.addDodecahedron()
		self.scaleShape(radius=radius)
		self.rotateShape()
		self.getCoords()

	def addIcosahedron(self):
		self.bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=1)

	def addDodecahedron(self):
		self.bpy.ops.wm.addon_enable(module="add_mesh_extra_objects")
		self.bpy.ops.mesh.primitive_solid_add(source='12') # Add a dodecahedron



	# Utils
	def rotateShape(self):
		rotX = math.radians(np.random.random() * 360)
		rotY = math.radians(np.random.random() * 360)
		rotZ = math.radians(np.random.random() * 360)
		self.scene.objects.active.rotation_euler = (rotX, rotY, rotZ)
		self.scene.update()


	def getCoords(self):
		self.coords = []
		for vertex in self.scene.objects.active.data.vertices:
			self.coords.append(list((self.scene.objects.active.matrix_world * vertex.co).to_tuple()))
		self.coords = np.asarray(self.coords)

	def selectObj(self, objName):
		self.scene.objects.active = None
		for obj in self.scene.objects:
			if obj.name == objName:
				obj.select = True
				self.scene.objects.active = obj
			else:
				obj.select = False

	def scaleShape(self, radius):
		# This function assumes that the shape is regular and convex
		# radius refers to the radius of a sphere
		scaleFactor = self.shapeName == 'cube' and radius * (math.sqrt(3)/3) or radius
		self.scene.objects.active.scale[0] = self.scene.objects.active.scale[1] = self.scene.objects.active.scale[2] = scaleFactor
		self.scene.update()

	def killMeshes(self):
		for obj in self.scene.objects:
			if obj.type == 'MESH':
				obj.select = True
			else:
				obj.select = False
		self.bpy.ops.object.delete()
