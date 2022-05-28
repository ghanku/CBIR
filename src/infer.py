# -*- coding: utf-8 -*-

from __future__ import print_function

from evaluate import infer
from DB import Database

import sys
from color import Color
from daisy import Daisy
from edge import Edge
from gabor import Gabor
from HOG import HOG
from vggnet import VGGNetFeat
from resnet import ResNetFeat
from ccv2 import CCV
from dcd import DCD
from efd import EFD
from gvf import GVF
from lbp import LBP
from slantlet import SLANT
from wavelet import WAVELET
import imageio
import matplotlib.pyplot as plt

depth = 5
d_type = 'd1'
query_idx = 0
feature_type = "color"

if (len(sys.argv) > 1):
    depth = int(sys.argv[1])
    d_type = sys.argv[2]
    query_idx = int(sys.argv[3])
    feature_type = sys.argv[4]


if __name__ == '__main__':
    db = Database()

    if (feature_type == "color"):
        # retrieve by color
        method = Color()
    elif(feature_type == "daisy"):
        # retrieve by daisy
        method = Daisy()
    elif(feature_type == "edge"):
        # retrieve by edge
        method = Edge()
    elif(feature_type == "gabor"):
        # retrieve by gabor
        method = Gabor()
    elif(feature_type == "HOG"):
        # retrieve by HOG
        method = HOG()
    elif(feature_type == "vggnet"):
        # retrieve by VGG
        method = VGGNetFeat()
    elif(feature_type == "resnet"):
        # retrieve by resnet
        method = ResNetFeat()
    elif(feature_type == "ccv2"):
        # retrieve by ccv
        method = CCV()
    elif(feature_type == "dcd"):
        # retrieve by dcd
        method = DCD()
    elif(feature_type == "efd"):
        # retrieve by efd
        method = EFD()
    elif(feature_type == "gvf"):
        # retrieve by gvf
        method = GVF()
    elif(feature_type == "lbp"):
        # retrieve by lbp
        method = LBP()
    elif(feature_type == "slantlet"):
        # retrieve by slantlet
        method = SLANT()
    elif(feature_type == "wavelet"):
        # retrieve by wavelet
        method = WAVELET()

    samples = method.make_samples(db)
    query = samples[query_idx]
    ap, result = infer(query, samples=samples, depth=depth, d_type=d_type)
    query_img = imageio.imread(query['img'], pilmode='RGB')
    plt.axis('off')
    plt.title("Query", fontsize=20)
    plt.imshow(query_img)
    plt.show()
    axes = []
    fig = plt.figure(figsize=(16, 8))

    for a in range(depth):
        score = result[a]['dis']
        cls = result[a]['cls']
        axes.append(fig.add_subplot(depth // 5, 5, a + 1))
        subplot_title = "{:.4f} ".format(score) + cls
        axes[-1].set_title(subplot_title)
        plt.axis('off')
        img = imageio.imread(result[a]['img'], pilmode='RGB')
        plt.imshow(img)
    fig.tight_layout()
    plt.show()
