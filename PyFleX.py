import os, subprocess, re
from common import runCmd

# ----- Setup environment ----- #

DEFAULT_FLEX_PATH = "/om/user/arsalans/Programs/FleX"
if "FLEX_PATH" not in os.environ:
	#sys.exit("Missing $FLEX_PATH environment variable")
	os.environ["FLEX_PATH"] = DEFAULT_FLEX_PATH
	print("$FLEX_PATH not set, using {}".format(DEFAULT_FLEX_PATH))

FLEX_PATH = os.environ["FLEX_PATH"]
FLEX_BIN_PATH = os.path.join(FLEX_PATH, "bin", "linux64", "NvFlexDemoReleaseCUDA_x64")

if not os.path.exists(FLEX_BIN_PATH):
	# TODO: include precise help message
	print("No FleX binary found.")

# For libGLEW.so.1.10
os.environ["LD_LIBRARY_PATH"] += ":{}".format(os.path.join(FLEX_PATH, "external"))

# ----------------------------- #

# see flex/demo/globals.h
COLOR_MAP = {
	"cyan"	 : 0,
	"gold"	 : 1,
	"green"  : 3,
	"yellow" : 4,
	"blue"	 : 5,
	"orange" : 6,
	"violet" : 7
}

#def _run(cmd, extra_vars={}, verbose=False):
#
#	if verbose:
#		print("Running {}".format(cmd))
#
#	env = os.environ.copy()
#	env.update(extra_vars)
#
#	proc = subprocess.Popen(
#		cmd,
#		shell = True,
#		#executable = '/bin/bash',
#		stdin = None,
#		stdout = subprocess.PIPE,
#		stderr = subprocess.PIPE,
#		env = env)
#
#	lines = []
#
#	while proc.poll() is None:
#		line = proc.stdout.readline().decode("ascii").rstrip()
#		lines.append(line)
#		if verbose and line != "":
#			print(line)
#
#	return lines
#
#	# if show_err:
#	#	  print(" ---- stderr ---- ")
#	#	  for line in proc.stderr.readlines():
#	#		  print(line.decode("ascii").rstrip())
#	#	  print(" ---------------- ")
#
# TODO: Break this up into different use cases
def simulate(
	objPath,
	configPath,
	rot=(0,0,0),
	local=True,
	clothColor="violet",
	floor=False,
	offline=True,
	occluded=True,
	useQuat=False,
	verbose=False,
	outImg=None,
	outObjBaseName=None,
	visSaveClothPerSimStep=False):

	'''
	Params:
		rot: Euler angles (in degrees) or quaternion (max unit norm)
	'''
	clothColor = clothColor.lower()
	assert(clothColor in COLOR_MAP)

	if not occluded:
		n_frames = 1
		offline = False # object mesh not rendered if offline=True
		outObjBaseName = None # No cloth data for unoccluded rendering

	sim_cmd = [FLEX_BIN_PATH,
		"-obj={}".format(os.path.abspath(objPath)),
		"-config={}".format(os.path.abspath(configPath)),

		#"-iters={}".format(n_frames),
		#"-ss={}".format(n_substeps_per_frame),

		# rotate
		not useQuat and "-rx={} -ry={} -rz={}".format(*rot) or "-rx={} -ry={} -rz={} -rw={}".format(*rot),

		# cloth properties
		#"-stiff_scale={}".format(stiffness),
		"-ccolor_id={}".format(COLOR_MAP[clothColor]),
	]

	if not floor:
		sim_cmd.append("-nofloor");

	if offline:
		sim_cmd.append("-offline");

	if not occluded:
		sim_cmd.append("-clothsize=1")
		# sim_cmd.append("-cam_dist={}".format(4.5))

	if useQuat:
		sim_cmd.append("-use_quat")
	else:
		sim_cmd.append("-use_euler")

	if outImg:
		sim_cmd.append("-img={}".format(os.path.abspath(outImg)))

	if outObjBaseName:
		sim_cmd.append("-export={}".format(outObjBaseName))
		
	if visSaveClothPerSimStep:
		sim_cmd.append("-saveClothPerSimStep")

	env = {}

	if local:
		# render on server display
		env["DISPLAY"] = ":0"
	else:
		# render using VirtualGL
		sim_cmd.insert(0, "vglrun")

	stdout = runCmd(" ".join(sim_cmd), extra_vars=env, verbose=verbose)
	return parse_sim_output(stdout)


def parse_sim_output(lines):
	s = "\n".join(lines)
	sim_out = {}

	def check_and_update(regex, *keys):
		m = re.search(regex, s)
		if m:
			for k, v in zip(keys, m.groups()):
				sim_out[k] = v

	check_and_update("Max inter-particle distance at rest: (\d*\.?\d+)", "max_ppd")
	check_and_update("Exporting cloth to (.+_cloth\.obj)", "cloth_file")
	check_and_update("Exporting mesh to (.+_mesh\.obj)", "mesh_file")
	check_and_update("Tear detected: (.+)", "torn")

	return sim_out
