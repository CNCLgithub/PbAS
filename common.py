import os, sys, shutil, subprocess, re, pickle, OpenEXR, Imath, array, fcntl, time, random, time, requests, cv2
import numpy as np
from glob import iglob
from pathlib import Path
from PIL import Image

def numericalSort(value):
    numbers = re.compile(r'(\d+)')
    # Only use this function for reading list of rendering files
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def getFilesList(path, fileType='', subDirs=False, lookupStr = '', onlyDir=False, forceSort=False):
    # Example of an acceptable fileType: png, zip etc -- So not put a dot '.' at the beginning
    # onlyDir=True returns directory names only
    filesList = sorted(iglob(path + (fileType == '' and '/**' or '/**/*.' + fileType), recursive=True) if subDirs else iglob(path + (fileType == '' and '/*' or '/*.' + fileType)), key=subDirs and numericalSort or None)
    if isinstance(lookupStr, list):
        filesList = [filePath for filePath in filesList if all([luStr in filePath for luStr in lookupStr])]
    elif lookupStr != '':
        filesList = [filePath for filePath in filesList if lookupStr in filePath]
    if onlyDir:
        dirs = []
        for f in filesList:
            if os.path.isdir(f):
                dirs.append(f)
        filesList = dirs
    if forceSort:
        filesList.sort(key=lambda f: int(re.sub('\D+', '', f)))
    return filesList

def ls(dir_):
    return iglob(os.path.join(dir_, "*"))

def fileExist(path):
    if path != '/':
        if os.path.isdir(path):
            return True
        else:
            temp = Path(path)
            return temp.is_file()
    else:
        return False

def mkdirs(paths):
    try:
        if isinstance(paths, list) and not isinstance(paths, str):
            for path in paths:
                mkdir(path)
        else:
            mkdir(paths)
    except:
        time.sleep(random.random()/8)
        if isinstance(paths, list) and not isinstance(paths, str):
            for path in paths:
                mkdir(path)
        else:
            mkdir(paths)

def mkdir(path):
    if not fileExist(path):
        os.makedirs(path)

def mv(src, dest):
    shutil.move(src, dest)

def cp(src, dest):
    shutil.copyfile(src, dest)

def rm(path):
    if fileExist(path):
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)

def loadPickle(filePath):
    data = None
    with open(filePath, 'rb') as f:
        data = pickle.load(f)
    return data

def savePickle(filePath, data, protocol=pickle.HIGHEST_PROTOCOL):
    with open(filePath, 'wb') as f:
        pickle.dump(data, f, protocol=protocol)

def safePickleLoad(filePath):
    try:
        if fileExist(filePath):
            loadedPkl = loadPickle(filePath)
        else:
            loadedPkl = False
    except:
        loadedPkl = False
    return loadedPkl

def downloadFile(url, savePath, redownload=False):
    if redownload:
        rm(savePath)
    else:
        if not fileExist(savePath):
            r = requests.get(url, stream=True)
            with open(savePath, 'wb') as f:
                for chunk in r.iter_content(chunk_size=2048): 
                    if chunk: # filter out keep-alive new chunks
                        f.write(chunk)
                        #f.flush() commented by recommendation from J.F.Sebastian
            r.close()
    return True

def loadTxt(filePath, splitLines=False, splitText='', lookupStr=None):
    if fileExist(filePath):
        with open(filePath, 'r') as f:
            if not splitLines:
                allLines = f.read()
            else:
                # allLines = f.read().splitlines()
                allLines = []
                for line in f.read().splitlines():
                    if line != '':
                        allLines.append(line)
        if lookupStr is not None:
            lookupRes = False
            if splitLines:
                for line in allLines:
                    if splitText != '':
                        line = line.split(splitText)
                    lookupRes = lookupStr in line
                    if lookupRes:
                        break
            else:
                lookupRes = lookupStr in allLines
            return lookupRes == True and 1 or 0

        else:
            return allLines
    else:
        return -1

def numLinesInFile(filePath):
    return sum(1 for line in open(filePath))

def exrToNumpy(exrPaths, renderType, resolution, renderAccuracy, maxDepth=10000, minDepth=0, rawArray=False):
    # Read the .exr file
    dtype = np.float16
    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)

    oneInstance = False
    if isinstance(exrPaths[0], list):
        allNpArrays = []
        for i in range(len(exrPaths)):
            allNpArrays.append(np.zeros((len(exrPaths[i]), renderType == 'depth' and 1 or 3, resolution, resolution), dtype=dtype))
    elif isinstance(exrPaths, list):
        allNpArrays = np.zeros((len(exrPaths), renderType == 'depth' and 1 or 3, resolution, resolution), dtype=dtype)
    else:
        oneExrPath = exrPaths
        oneInstance = True

def maskDrapedShape(drapedShapeRenderingPath, maskRenderingPath, resolution, maskClothRendering=False, silhouetteStimuli=False):
    # Uncomment the following lines to get a convex hall-masked version of the draped rendering of the shape
    if maskClothRendering:
        drapedImgNumpy = cv2.imread(drapedShapeRenderingPath)
        maskImgNumpy = cv2.imread(maskRenderingPath)

        # maskImgNumpy = cv2.cvtColor(maskImgNumpy, cv2.COLOR_BGR2RGB)

        maskImgNumpyGray = cv2.cvtColor(maskImgNumpy, cv2.COLOR_BGR2GRAY)
        _, thresholdImg = cv2.threshold(maskImgNumpyGray, 0, 255, cv2.THRESH_BINARY)
        contours, hier = cv2.findContours(thresholdImg, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])
        convexHall = cv2.convexHull(contours[0])
        blankImg = np.zeros_like(maskImgNumpy)

        # blankImg = cv2.cvtColor(blankImg, cv2.COLOR_BGR2RGB)

        cv2.drawContours(blankImg, [convexHall], -1, (255, 255, 255), -1)
        grayFilledBlankImg = cv2.cvtColor(blankImg, cv2.COLOR_BGR2GRAY)
        _, thresholdImg = cv2.threshold(grayFilledBlankImg, 2, 255, cv2.THRESH_BINARY)
        drapedImgNumpy = cv2.bitwise_and(drapedImgNumpy, drapedImgNumpy, mask = thresholdImg)
        cv2.imwrite(drapedShapeRenderingPath, drapedImgNumpy)

        # Uncomment the following lines to get a pixel-pwise masked version of the draped rendering of the shape
        # drapedImgNumpy = pngToNumpy(pngPath=drapedShapeRenderingPath, renderType='rgb', resolution=resolution, dtype='float32')
        # maskImgNumpy = pngToNumpy(pngPath=maskRenderingPath, renderType='rgb', resolution=resolution, dtype='float32')
        # drapedImgNumpy[maskImgNumpy == 0.] = 0.0
        # numpyToImg(npArr=drapedImgNumpy, imgPath=drapedShapeRenderingPath)
    elif silhouetteStimuli:
        drapedImgNumpy = pngToNumpy(pngPath=maskRenderingPath, renderType='rgb', resolution=resolution, dtype='float32')
        drapedImgNumpy[drapedImgNumpy > 0.001] = 1.0
        numpyToImg(npArr=drapedImgNumpy, imgPath=drapedShapeRenderingPath)

def numpyToImg(npArr, imgPath, imgName='', format='png'):
    # For surface Normals, npArr contains elements within the range [-1, 1]
    # TODO: store a 16-bit grayscale image for depth maps instead of 8-bit images
        # https://stackoverflow.com/questions/25696615/can-i-save-a-numpy-array-as-a-16-bit-image-using-normal-enthought-python
    if npArr.shape[0] == 1:
        # Depth map
        npArr = npArr[0]
        mode = 'L'
        # npArr *= 2**16-1 # TODO Figure out how to store 16 bit (grayscale) image
        npArr *= 2**8-1
    else:
        # Surface Normal or RGB image
        npArr[npArr < 0] = 0
        mode = 'RGB'
        npArr = npArr.transpose(1, 2, 0)
        npArr *= 2**8-1
    npArr = npArr.astype(np.uint8)
    cv2.imwrite(imgPath + (imgName != '' and ('/' + imgName + '.' + format) or ''), npArr)
    # im = Image.fromarray(npArr, mode=mode)
    # im.save(imgPath + (imgName != '' and ('/' + imgName + '.' + format) or ''))

def pngToNumpy(pngPath, renderType, resolution, normalize=True, dtype='float16'):
    dtype = dtype == 'float16' and np.float16 or dtype == 'float32' and np.float32
    hasList = isinstance(pngPath, list)
    if hasList:
        allNpArrays = np.zeros((len(pngPath), renderType == 'depth' and 1 or 3, resolution, resolution), dtype=dtype)
    for i in range(hasList and len(pngPath) or 1):
        try:
            img = Image.open(hasList and pngPath[i] or pngPath)
            if npArr.size != (resolution**2*(renderType == 'depth' and 1 or 3)):
                img = Image.open(hasList and pngPath[i] or pngPath)
        except:
            img = Image.open(hasList and pngPath[i] or pngPath)
        npArr = np.asarray(img).astype(dtype)
        npArr = npArr.transpose(2, 0, 1)
        if normalize:
            npArr /= 255.
        if renderType == 'depth':
            npArr = npArr[0]
            npArr.reshape((1, resolution, resolution))
        else:
            npArr.reshape((3, resolution, resolution))
        if hasList:
            if renderType == 'depth':
                allNpArrays[i][0] = npArr.astype(dtype)[:]
            else:
                allNpArrays[i] = npArr.astype(dtype)[:]
        else:
            allNpArrays = npArr.astype(dtype)[:]
    return allNpArrays

def getMinMaxDepth(path, minn, maxx, numVPs, resolution, renderAccuracy, renderType='depth'):
    for i in range(numVPs):
        fp = path[0:len(path)-6] + str(i) + '1.exr'
        if not fileExist(fp):
            fp = path[0:len(path)-7] + str(i) + '1.exr'
        npArr = exrToNumpy(fp, renderType=renderType, resolution=resolution, renderAccuracy=renderAccuracy, rawArray=True)
        if maxx < npArr[npArr<npArr.max()].max():
            maxx = npArr[npArr<npArr.max()].max()
        if minn > npArr[npArr<npArr.max()].min():
            minn = npArr[npArr<npArr.max()].min()
    return (minn, maxx)

def maxNumRenderingsSetsInMem(resolution, numVPs, numRotation, maxMemInMB, numBitsPerPixel, numSets=4):
    # "Set" refers to RGB, depth, Normal and simultaneous random renderings
    totalMBs = 0
    numSetsInChunk = 0
    while True:
        for i in range(numSets):
            if i == 0 or i == 2:
                #RGBs
                #Surface Normals
                totalMBs += (numVPs*resolution**2*3*numBitsPerPixel/8/1024/1024)
            elif i == 1:
                # Depth maps
                totalMBs += (numVPs*resolution**2*numBitsPerPixel/8/1024/1024)
            else:
                # Simultaneous random renderings
                totalMBs += (numRotation*resolution**2*3*numBitsPerPixel/8/1024/1024)
        if totalMBs + 20 < maxMemInMB: # Plus 20MB for some overhead
            numSetsInChunk+=1
        else:
            break
    return numSetsInChunk

def saveRotation(txtDir, rot, simultaneousRot):
    addToList = True
    newFile = False
    if fileExist(txtDir + '/rotations.txt'):
        rots = np.loadtxt(txtDir + '/rotations.txt')
        if rots.ndim == 1:
            rots = rots.reshape(1, rots.shape[0])
        if any(np.isclose(rots, rot.reshape(1, 3)).all(1)):
            addToList = False
    else:
        newFile = True
    with open(txtDir + '/rotations.txt', 'a') as txtFile:
        if addToList:
            rot = np.array_str(rot)[1:-1].strip()
            rot = newFile and not simultaneousRot and ('0.0 0.0 0.0\n' + rot + '\n') or rot + '\n'
            txtFile.write(rot)

def convertTxtToNpy(filePath, fileName):
    txtNumpy = np.loadtxt(filePath + '/' + fileName + '.txt')
    np.save(filePath + '/' + fileName + '.npy', txtNumpy)

def saveObjPath(txtDir, objPath):
    mkdir(txtDir)
    with open(txtDir + '/objPath.txt', 'w') as txtFile:
        objPath += '\n'
        txtFile.write(objPath)

def getPrestoredRenderPaths(renderResultsPath, category, trainOrTest, renderType, resolution, simultaneousRot):
    mainCategoryPath = renderResultsPath + '/' + trainOrTest + '/' + category
    renderPath = mainCategoryPath + '/' + str(resolution) + (renderType == 'depth' and '-DepthRenderPaths' or renderType == 'normal' and '-NormalRenderPaths' or (simultaneousRot == 0 and '-RGBRenderPaths' or '-RGBSimultaneousRotRenderPaths')) + '.txt'
    paths = sorted(loadTxt(renderPath, splitLines=True), key=numericalSort)
    return paths

def saveDataRow(filePath, textList, resolution, simultaneousRot):
    # Use fcntl to resolve the issues that might arise during concurrent read
    writeFlag = False
    # Start writing the data when the flag is equal to True
    while not writeFlag:
        if fileExist(filePath + '/noWriteFlag.txt'):
            writeFlag = False
            time.sleep(0.15)
        else:
            with open(filePath + '/noWriteFlag.txt', 'w') as f:
                f.write('')
            writeFlag = True

def readNpyPaths(path, resolution, sortedDataset, simultaneousRot):
    npyFiles = [path + '/' + '_{0:s}AllRgbNpyPaths.txt'.format(str(resolution)), path + '/' +'_{0:s}AllDepthNpyPaths.txt'.format(str(resolution)), path + '/' +'_{0:s}AllNormalNpyPaths.txt'.format(str(resolution)), path + '/' +'_{0:s}AllRgbNpySimultaneousRotPaths.txt'.format(str(resolution)), path + '/' +'_{0:s}AllRgbNpySimultaneousRotRotationVecsPaths.txt'.format(str(resolution))]
    npyPaths = []
    objNpy = []
    loopCount = simultaneousRot==1 and len(npyFiles) or len(npyFiles)-1
    for i in range(loopCount):
        npyPaths.append(sorted(list(set(loadTxt(npyFiles[i], splitLines=True))), key=numericalSort))
        objNpy.append([',,'.join(path.split(',,')[1:]) for path in npyPaths[i]])

    # Obtain the list of paths for unique objects
    validNpys = set(objNpy[0])
    for i in range(1, loopCount):
        validNpys = validNpys & set(objNpy[i])
    validNpys = list(validNpys)

    for i in range(len(npyPaths)):
        npyPaths[i] = [npyPath for npyPath in npyPaths[i] if ',,'.join(npyPath.split(',,')[1:]) in validNpys]

    if not sortedDataset:
        permutationIndices = list(range(len(validNpys)))
        random.shuffle(permutationIndices)
        for i in range(loopCount):
            tempList = []
            for idx in permutationIndices:
                tempList.append(npyPaths[i][idx])
            npyPaths[i] = tempList
        
    return npyPaths

def chunkNpyPaths(npyPaths, resolution, numVPs, numRotation, numBitsPerPixel, maxMemory):
    samplesPerChunk = maxNumRenderingsSetsInMem(resolution=resolution, numVPs=numVPs, numRotation=numRotation, maxMemInMB=maxMemory, numBitsPerPixel=numBitsPerPixel, numSets=len(npyPaths))
    chunkedSamples = []
    sampleCounter = 0

    for i in range(len(npyPaths)):
        chunkedSamples.append([])
        for j in range(len(npyPaths[i])):
            if j % samplesPerChunk == 0:
                chunkedSamples[i].append([])
            chunkedSamples[i][len(chunkedSamples[i])-1].append(npyPaths[i][j])
    return chunkedSamples

def appendSaveTxt(filePath, text, beginWith='', endWith='\n', noDuplicate=False, writeLockFlagPath=None):

    if writeLockFlagPath is not None:
        # Use fcntl to resolve the issues that might arise during concurrent read
        writeFlag = False
        # Start writing the data when the flag is equal to True
        while not writeFlag:
            if fileExist(writeLockFlagPath + '/noWriteFlag.txt'):
                writeFlag = False
                time.sleep(0.1)
            else:
                with open(writeLockFlagPath + '/noWriteFlag.txt', 'w') as f:
                    f.write('')
                writeFlag = True

    if noDuplicate:
        readRes = loadTxt(filePath, lookupStr=text)
        if readRes == 1:
            return 1

    if fileExist(filePath) and beginWith=='':
        beginWith='\n'

    with open(filePath, 'a') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(beginWith + text + endWith)
        fcntl.flock(f, fcntl.LOCK_UN)

    # with open(filePath, 'a') as f:
    #     f.write(text + endWith)

    if writeLockFlagPath is not None:
        rm(writeLockFlagPath + '/noWriteFlag.txt')

    return 1

def saveCombinedNpys(combinedNpysPath, chunkedSamples, resolution, numVPs, numRotation, numBitsPerPixel):

    dtype = numBitsPerPixel == 16 and np.float16 or np.float32
    catNames = []
    catIDs = []
    for i in range(len(chunkedSamples)-1):
        renderType = i == 0 and 'npyRgb' or i == 1 and 'npyDepth' or i == 2 and 'npyNormal' or i == 3 and 'npySimRot'
        mkdir(combinedNpysPath + '/' + renderType)
        mkdir(combinedNpysPath + '/catName')
        mkdir(combinedNpysPath + '/catID')
        mkdir(combinedNpysPath + '/gtIdx')
        mkdir(combinedNpysPath + '/npyRotVec')
        for j in range(len(chunkedSamples[i])):
            numNps = len(chunkedSamples[i][j])
            if numNps > 1:
                if i == 0 or i == 2:
                    arraySize=(numNps, numVPs, 3, resolution, resolution)
                elif i == 1:
                    arraySize=(numNps, numVPs, 1, resolution, resolution)
                else:
                    arraySize=(numNps, numRotation, 3, resolution, resolution)
                    arraySizeRotVecs=(numNps, numRotation, 3)
                    tempNpArrRotVecs = np.zeros(arraySizeRotVecs, dtype=np.float32)
                tempNpArr = np.zeros(arraySize, dtype=np.float32)

                currentNpySum = 0
                npyGtIdx = np.empty(0, dtype=np.float32)
                npyCatID = np.empty(0, dtype=np.float32)
                npyRotationVecs = np.empty(0, dtype=np.float32)
                for k, dataRow in enumerate(chunkedSamples[i][j]):
                    dataRow = dataRow.split(',,')
                    if i == 3:
                        dataRowRotVecs = chunkedSamples[i+1][j][k]
                        dataRowRotVecs = dataRowRotVecs.split(',,')
                        npyRotVecsPath = dataRowRotVecs[0]
                    npyPath = dataRow[0]
                    labels = dataRow[1:-1]

                    catName = labels[0]
                    if i == 0:
                        npyCatID = np.append(npyCatID, labels[1])
                        npyGtIdx = np.append(npyGtIdx, labels[2])

                    # Unique cat names and IDs
                    if (labels[1] + ',,' + labels[0]) not in catNames and loadTxt(combinedNpysPath + '/allCatNames.txt', splitLines=True, lookupStr=labels[0]) != 1:
                        catNames.append(labels[1] + ',,' + labels[0])
                    if labels[1] not in catIDs and loadTxt(combinedNpysPath + '/allCatIDs.txt', splitLines=True, lookupStr=labels[1]) != 1:
                        catIDs.append(labels[1])

                    try:
                        currentNpy = np.load(npyPath)
                        if i == 3:
                            currentNpyRotVecs = np.load(npyRotVecsPath)
                    except:
                        time.sleep(200) # To hopefully pass the IO traffic peak
                        currentNpy = np.load(npyPath)
                        if i == 3:
                            currentNpyRotVecs = np.load(npyRotVecsPath)

                    currentNpy = currentNpy.astype(np.float32)
                    tempNpArr[k] = currentNpy
                    if i == 3:
                        tempNpArrRotVecs[k] = currentNpyRotVecs
                        del currentNpyRotVecs
                    currentNpySum += currentNpy.sum()
                    del currentNpy
                    if i == 0:
                        # Stop adding more labels after i > 0 as there is no other unique labels (the function readNpyPaths takes care of obtaining unique labels)
                        # appendSaveTxt(combinedNpysPath + '/labels' + str(j) + '.txt', labels, noDuplicate=True)
                        appendSaveTxt(combinedNpysPath + '/catName' + '/' + str(j) + '.txt', catName)
                if i == 0:
                    np.save(combinedNpysPath + '/catID' + '/' + str(j) + '.npy', npyCatID.reshape(npyCatID.size, 1).astype(np.float32))
                    np.save(combinedNpysPath + '/gtIdx' + '/' + str(j) + '.npy', npyGtIdx.reshape(npyGtIdx.size, 1).astype(np.float32))
                np.save(combinedNpysPath + '/' + renderType + '/' + str(j) + '.npy', tempNpArr)
                if i == 3:
                    np.save(combinedNpysPath + '/npyRotVec' + '/' + str(j) + '.npy', tempNpArrRotVecs)
            if (j+1) % 20 == 0:
                print ('==> Done creating ' + str(j+1) + '/' + str(len(chunkedSamples[i])) + ' files for ' + renderType)
            if j == len(chunkedSamples[i])-1:
                print ('==> Done creating all npy files for ' + renderType)
        print ('')

        if i == 0:
            catNames = sorted(catNames, key=numericalSort)
            catNames = [name.split(',,')[1] for name in catNames]
            appendSaveTxt(combinedNpysPath + '/allCatNames.txt', '\n'.join(catNames), noDuplicate=True)
            appendSaveTxt(combinedNpysPath + '/allCatIDs.txt', '\n'.join(sorted(catIDs, key=numericalSort)), noDuplicate=True)

def computeNumShapes(numStimuli, testCategory, numDistractorShapesPerTrial):
    # This function computes the number of shapes to be chosen from the ground-truth and distractor categories
    while True:
        if numStimuli % len(testCategory) == 0 and (numStimuli/2) % len(testCategory) == 0:
            break
        else:
            numStimuli += 1
    numStimuli = numStimuli

    numShapeFromGtCat = (numStimuli/2)/len(testCategory)*numDistractorShapesPerTrial + numStimuli/len(testCategory)
    numShapeFromDistractorCats = (numStimuli * (numDistractorShapesPerTrial+1) - numShapeFromGtCat*len(testCategory))/len(testCategory)
    
    return (int(numShapeFromGtCat), int(numShapeFromDistractorCats))

def getSolidName(numCameras):
    if numCameras == 4:
        # TODO: implement tetrahedron
        print ('==> Error: You need to implement tetrahedron')
        sys.exit()
    elif numCameras == 6:
        # TODO: implement octahedron
        print ('==> Error: You need to implement octahedron')
        sys.exit()
    elif numCameras == 8:
        return 'Cube'
    elif numCameras == 12:
        return 'Icosahedron'
    elif numCameras == 20:
        return 'Dodecahedron'
    else:
        # Sampling points on a sphere for camera locations
        return 'Sphere'
    
def silence(f):
    def _f(*args, **kwargs):
        # save original fd's
        out_fd = os.dup(1)
        err_fd = os.dup(2)

        # close and reopen to replace original stdout/err
        os.close(1)
        os.open("/dev/null", os.O_WRONLY) # /dev/null is new stdout
        os.close(2)
        os.open("/dev/null", os.O_WRONLY) # and new stderr

        out = f(*args, **kwargs)

        # close and reopen to reinstate original fd's
        os.close(1)
        os.dup(out_fd) # restore original stdout (order matters)
        os.close(2)
        os.dup(err_fd) # restore original stderr (order matters)

        # delete fd copies
        os.close(out_fd)
        os.close(err_fd)

        return out

#   def _f(*args, **kwargs):
#       stdout = io.StringIO()
#       with cl.redirect_stdout(stdout):
#           f(*args, **kwargs)
#
    return _f

def runCmd(cmd, extra_vars={}, verbose=False):

    if verbose:
        print("Running {}".format(cmd))

    env = os.environ.copy()
    env.update(extra_vars)

    proc = subprocess.Popen(
        cmd,
        shell = True,
        #executable = '/bin/bash',
        stdin = None,
        stdout = subprocess.PIPE,
        stderr = subprocess.PIPE,
        env = env)

    lines = []

    while proc.poll() is None:
        line = proc.stdout.readline().decode("ascii").rstrip()
        lines.append(line)
        if verbose and line != "":
            print(line)

    return lines