#!/usr/bin/env python2
#
# Copyright 2015-2016 Carnegie Mellon University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#argparse模块的作用是用于解析命令行参数，首先导入该模块
import argparse
import cv2
import numpy as np
import os
import random
import shutil

import openface
import openface.helper
from openface.data import iterImgs
#os.path.dirname(path) #返回文件路径
#os.path.realpath(path) #返回path的真实路径
#os.path.join(path1[, path2[, ...]]) #把目录和文件名合成一个路径
fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join(fileDir, '..', 'models')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')


def write(vals, fName):
    if os.path.isfile(fName):
        print("{} exists. Backing up.".format(fName))
        os.rename(fName, "{}.bak".format(fName))
    with open(fName, 'w') as f:
        for p in vals:
            f.write(",".join(str(x) for x in p))
            f.write("\n")


def computeMeanMain(args):
    align = openface.AlignDlib(args.dlibFacePredictor)

    imgs = list(iterImgs(args.inputDir))
    if args.numImages > 0:
        imgs = random.sample(imgs, args.numImages)

    facePoints = []
    for img in imgs:
        rgb = img.getRGB()
        bb = align.getLargestFaceBoundingBox(rgb)
        alignedPoints = align.align(rgb, bb)
        if alignedPoints:
            facePoints.append(alignedPoints)

    facePointsNp = np.array(facePoints)
    mean = np.mean(facePointsNp, axis=0)
    std = np.std(facePointsNp, axis=0)

    write(mean, "{}/mean.csv".format(args.modelDir))
    write(std, "{}/std.csv".format(args.modelDir))

    # Only import in this mode.
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.scatter(mean[:, 0], -mean[:, 1], color='k')
    ax.axis('equal')
    for i, p in enumerate(mean):
        ax.annotate(str(i), (p[0] + 0.005, -p[1] + 0.005), fontsize=8)
    plt.savefig("{}/mean.png".format(args.modelDir))


def alignMain(args):
    openface.helper.mkdirP(args.outputDir)

    imgs = list(iterImgs(args.inputDir))

    # Shuffle so multiple versions can be run at once.
    random.shuffle(imgs)

    landmarkMap = {
        'outerEyesAndNose': openface.AlignDlib.OUTER_EYES_AND_NOSE,
        'innerEyesAndBottomLip': openface.AlignDlib.INNER_EYES_AND_BOTTOM_LIP
    }
    if args.landmarks not in landmarkMap:
        raise Exception("Landmarks unrecognized: {}".format(args.landmarks))

    landmarkIndices = landmarkMap[args.landmarks]

    align = openface.AlignDlib(args.dlibFacePredictor)

    nFallbacks = 0
    for imgObject in imgs:
        print("=== {} ===".format(imgObject.path))
        outDir = os.path.join(args.outputDir, imgObject.cls)
        openface.helper.mkdirP(outDir)
        outputPrefix = os.path.join(outDir, imgObject.name)
        imgName = outputPrefix + ".png"

        if os.path.isfile(imgName):
            if args.verbose:
                print("  + Already found, skipping.")
        else:
            rgb = imgObject.getRGB()
            if rgb is None:
                if args.verbose:
                    print("  + Unable to load.")
                outRgb = None
            else:
                outRgb = align.align(args.size, rgb,
                                     landmarkIndices=landmarkIndices,
                                     skipMulti=args.skipMulti)
                if outRgb is None and args.verbose:
                    print("  + Unable to align.")

            if args.fallbackLfw and outRgb is None:
                nFallbacks += 1
                deepFunneled = "{}/{}.jpg".format(os.path.join(args.fallbackLfw,
                                                               imgObject.cls),
                                                  imgObject.name)
                shutil.copy(deepFunneled, "{}/{}.jpg".format(os.path.join(args.outputDir,
                                                                          imgObject.cls),
                                                             imgObject.name))

            if outRgb is not None:
                if args.verbose:
                    print("  + Writing aligned file to disk.")
                outBgr = cv2.cvtColor(outRgb, cv2.COLOR_RGB2BGR)
                cv2.imwrite(imgName, outBgr)

    if args.fallbackLfw:
        print('nFallbacks:', nFallbacks)

if __name__ == '__main__':
	#创建一个argparse模块的解析对象
    parser = argparse.ArgumentParser()
    #parser.add_argument()该对象中添加你要关注的命令行参数和选项，每一个add_argument方法对应一个你要关注的参数或选项
    #参数名之前有‘--’表示可选参数
    parser.add_argument('inputDir', type=str, help="Input image directory.")
    parser.add_argument('--dlibFacePredictor', type=str, help="Path to dlib's face predictor.",
                        default=os.path.join(dlibModelDir, "shape_predictor_68_face_landmarks.dat"))
    #parser.add_subparsers()该对象中添加子参数模块，此处因为命令行参数为align，因此直接运行alignmentParser = subparsers.add_parser(
    #'align', help='Align a directory of images.')
    subparsers = parser.add_subparsers(dest='mode', help="Mode")
    computeMeanParser = subparsers.add_parser(
        'computeMean', help='Compute the image mean of a directory of images.')
    computeMeanParser.add_argument('--numImages', type=int, help="The number of images. '0' for all images.",
                                   default=0)  # <= 0 ===> all imgs
    alignmentParser = subparsers.add_parser(
        'align', help='Align a directory of images.')
    alignmentParser.add_argument('landmarks', type=str,
                                 choices=['outerEyesAndNose',
                                          'innerEyesAndBottomLip',
                                          'eyes_1'],
                                 help='The landmarks to align to.')
    alignmentParser.add_argument(
        'outputDir', type=str, help="Output directory of aligned images.")
    alignmentParser.add_argument('--size', type=int, help="Default image size.",
                                 default=96)
    alignmentParser.add_argument('--fallbackLfw', type=str,
                                 help="If alignment doesn't work, fallback to copying the deep funneled version from this directory..")
    alignmentParser.add_argument(
        '--skipMulti', action='store_true', help="Skip images with more than one face.")
    alignmentParser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()

    if args.mode == 'computeMean':
        computeMeanMain(args)
    else:
        alignMain(args)
