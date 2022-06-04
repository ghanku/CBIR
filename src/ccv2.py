#!env python
# -*- coding: utf-8 -*-
# Color Coherence Vector of 2D image.
# file created by A.Chabira
# original implementaion by https://github.com/tamanobi
# License: Public Domain
#
# reference:
# Pass, G., Zabih, R., & Miller, J. (n.d.). Comparing Images Using Color Coherence Vectors. http://www.cs.cornell.edu/home/rdz/ccv.html


from __future__ import print_function

from evaluate import distance, evaluate_class
from DB import Database

import sys
import os
from six.moves import cPickle
import numpy as np
import imageio
import skimage.color
import skimage.data
import skimage.transform
import skimage.filters
import itertools
from tqdm import tqdm
import cv2

# configs for CCV
n = 2       # indicating n discretized colors
tau = 'default'  # threshold for connectet pixel to be classified as coherent or incoherent

d_type = 'd1'      # distance type (similarity measure)
depth = 3         # retrieved depth, set to None will count the ap for whole database

if (len(sys.argv) > 1):
    if (sys.argv[1] == "None"):
        depth = None
    else:
        depth = int(sys.argv[1])

    d_type = sys.argv[2]

# cache dir
cache_dir = 'cache'
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)


class CCV(object):
    def color_coherence_vector(self, input, p_n, p_tau, normalize=False, resize=True, flatten=False):
        ''' Calculates the Color Coherence Vector of an image

            arguments
              input  : image or 2D array either multi channel or grayscale
              p_n :  number of discretized colors bins
              p_tau : threshold for connectet pixel to be classified as coherent or incoherent, either 'defaut' or an integer

            return
                  a numpy array with len
        '''

        # read image
        if isinstance(input, np.ndarray):  # check if input is path to image or an array
            img = input.copy()
        else:
            img = imageio.imread(input, pilmode='RGB')

        # resize images into 200x200
        if resize:
            img = skimage.transform.resize(img, (200, 200))

        row, col, channels = img.shape

        # blur
        img = cv2.GaussianBlur(img, (3, 3), 0)
        # quantize color
        img = self.QuantizeColor(img, p_n)
        bgr = cv2.split(img)
        #bgr = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))

        if p_tau == 'default':
            # Classify as coherent for area is >= 1% of input image area
            p_tau = int(img.shape[0] * img.shape[1] * 0.01)

        alpha = np.zeros(p_n)
        beta = np.zeros(p_n)

        # labeling
        for i, ch in enumerate(bgr):
            ret, th = cv2.threshold(ch, 127, 255, 0)
            ret, labeled, stat, centroids = cv2.connectedComponentsWithStats(
                th, None, cv2.CC_STAT_AREA, None, connectivity=8)
            #!see https://github.com/atinfinity/lab/wiki/OpenCV%E3%82%92%E4%BD%BF%E3%81%A3%E3%81%9F%E3%83%A9%E3%83%99%E3%83%AA%E3%83%B3%E3%82%B0#samplecode
            #!see http://docs.opencv.org/3.0.0/d3/dc0/group__imgproc__shape.html#gac7099124c0390051c6970a987e7dc5c5

            # generate ccv
            areas = [[v[4], label_idx] for label_idx, v in enumerate(stat)]
            coord = [[v[0], v[1]] for label_idx, v in enumerate(stat)]
            # Counting
            for a, c in zip(areas, coord):
                area_size = a[0]
                x, y = c[0], c[1]
                if (x < ch.shape[1]) and (y < ch.shape[0]):
                    bin_idx = int(ch[y, x] // (256 // p_n))
                    if area_size >= p_tau:
                        # classify as coherent
                        alpha[bin_idx] = alpha[bin_idx] + area_size
                    else:
                        # classify as incoherent
                        beta[bin_idx] = beta[bin_idx] + area_size

        if normalize:
            alpha /= np.sum(alpha)
            beta /= np.sum(beta)

        return np.array(list(zip(alpha, beta))).flatten()

    def QuantizeColor(self, img, p_n=64):
        ''' Qauntazies an input image or 2D array into n color bins

            arguments
              img  : image or 2D array either multi channel or grayscale
              p_n :  number of discretized colors bins


            return
                  a quantized image or 2D array of same shape as input
        '''
        div = 256 // p_n
        rgb = cv2.split(img)
        q = []
        for ch in rgb:
            vf = np.vectorize(lambda x, div: int(x // div) * div)
            quantized = vf(ch, div)
            q.append(quantized.astype(np.uint8))
        d_img = cv2.merge(q)

        return d_img

    def make_samples(self, db, verbose=True):
        sample_cache = "ccv_cache"
        try:
            samples = cPickle.load(
                open(os.path.join(cache_dir, sample_cache), "rb", True))
            if verbose:
                print("Using cache..., config=%s, distance=%s, depth=%s" %
                      (sample_cache, d_type, depth))
        except:
            if verbose:
                print("Counting ccv..., config=%s, distance=%s, depth=%s" % (
                    sample_cache, d_type, depth))
            samples = []
            data = db.get_data()
            for d in tqdm(data.itertuples(), total=len(data)):
                d_img, d_cls = getattr(d, "img"), getattr(d, "cls")
                d_ccv = self.color_coherence_vector(
                    d_img, p_n=n, p_tau=tau, resize=True, flatten=False)
                samples.append({
                    'img':  d_img,
                    'cls':  d_cls,
                    'hist': d_ccv
                })
            cPickle.dump(samples, open(os.path.join(
                cache_dir, sample_cache), "wb", True))
        return samples


if __name__ == '__main__':

    db = Database()
    data = db.get_data()
    ccv = CCV()

    # test DCD on one instance
    Desciptor = ccv.color_coherence_vector(
        data.iloc[0, 0], p_n=n, p_tau=tau, resize=True, flatten=False)
    print('done instance')
    APs = evaluate_class(db, f_class=CCV, d_type=d_type, depth=depth)
    cls_MAPs = []
    for cls, cls_APs in sorted(APs.items()):
        MAP = np.mean(cls_APs)
        print("Class {}, MAP {}".format(cls, MAP))
        cls_MAPs.append(MAP)
    print("MMAP", np.mean(cls_MAPs))
