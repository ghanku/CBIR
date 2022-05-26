#!env python
# -*- coding: utf-8 -*-
# Dominant Color Descriptor of 2D image.
# file created by A.Chabira
# original matlab implementaion by https://github.com/Molen1945
# License: Public Domain
#
# reference:
# Shao, H., Wu, Y., Cui, W., Zhang, J. (2008). Image retrieval based on MPEG-7 dominant color descriptor. Proceedings of the 9th International Conference for Young Computer Scientists, ICYCS 2008, 753–757. https://doi.org/10.1109/ICYCS.2008.89


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
import itertools
from tqdm import tqdm


# configs for DCD
n = 8        # order of DCD

d_type = 'd1'      # distance type (similarity measure)
depth = 3         # retrieved depth, set to None will count the ap for whole database

if (len(sys.argv) > 1):
    depth = int(sys.argv[1])
    d_type = sys.argv[2]

# cache dir
cache_dir = 'cache'
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)


class DCD(object):
    def dominant_color_descriptor(self, input, img_type='RGB', order=8, normalize=False, resize=True, flatten=False):
        ''' Calculates the Dominant Color Descriptor described in the MPEG-7 standard

            arguments
              img  : either HSV image, or RGB image
              img_type : input image type either 'HSV' or 'RGB'
              order : n first dominant colors

            return
                  2 numpy arrays, one  with size equal to input image, and second ###72
        '''

        # read image
        if isinstance(input, np.ndarray):  # check if input is path to image or an array
            img = input.copy()
        else:
            img = imageio.imread(input, pilmode='RGB')

        if resize:
            img = skimage.transform.resize(img, (200, 200))

        if img_type == 'RGB':
            img = skimage.color.rgb2hsv(img)

        # HSV component sorting
        H = 360 * img[:, :, 0]
        S = img[:, :, 1]
        V = img[:, :, 2]

        # HSV Component Quantization
        Hq = self.QunatizeH(H)
        Sq = self.QunatizeSV(S)
        Vq = self.QunatizeSV(V)

        # HSV matrix generation
        C = np.round(9 * Hq + 3 * Sq + Vq)

        m, n = C.shape
        color, _ = np.histogram(C, bins=72, range=(0, 72))
        Pi = color / (m * n)

        M = order

        # Pi values ​​are sorted in descending order and stored in Qj
        # I : the index of the Pi values ​​that have been sorted in descending order

        I = np.argsort(Pi)[::-1]  # indices of sorted elements
        Qj = np.sort(Pi)[::-1]

        # Take the first 8 values ​​of Qj
        Qj = Qj[0:M]

        Pi1 = np.zeros(72)
        I = I[0:M]
        Pi1[I] = Qj

        P = Pi1 / sum(Qj)

        if normalize:
            C /= np.sum(C)

        if flatten:
            return np.array(P, C).flatten()

        return np.array((P, C))

    def QunatizeH(self, H):
        ''' Calculates the Quantization of the Hue channel of an image

                arguments
                  H  : Hue channel of HSV image

                return
                      numpy array with size equal to input image
              '''

        bins = np.array([20, 40, 75, 155, 190, 270, 295, 316])
        ix = np.digitize(H, bins=bins)

        return ix

    def QunatizeSV(self, SV):
        ''' Calculates the Quantization of either the Saturation or Value channels of an image

                arguments
                  SV  : either S or V channels of an image

                return
                      numpy array with size equal to input image
              '''

        bins = np.array([1, 0.7, 0.2])
        ix = np.digitize(SV, bins=bins, right=True)

        return ix

    def make_samples(self, db, verbose=True):
        sample_cache = "dcd_cache"
        try:
            samples = cPickle.load(
                open(os.path.join(cache_dir, sample_cache), "rb", True))
            if verbose:
                print("Using cache..., config=%s, distance=%s, depth=%s" %
                      (sample_cache, d_type, depth))
        except:
            if verbose:
                print("Counting dcd..., config=%s, distance=%s, depth=%s" % (
                    sample_cache, d_type, depth))
        samples = []
        data = db.get_data()
        for d in tqdm(data.itertuples(), total=len(data)):
            d_img, d_cls = getattr(d, "img"), getattr(d, "cls")
            d_dcd = self.dominant_color_descriptor(
                d_img, img_type='RGB', order=n, resize=True, flatten=False)
            samples.append({
                'img':  d_img,
                'cls':  d_cls,
                'hist': d_dcd
            })
        cPickle.dump(samples, open(os.path.join(
            cache_dir, sample_cache), "wb", True))
        return samples


if __name__ == '__main__':

    db = Database()
    data = db.get_data()
    dcd = DCD()

    # test DCD on one instance
    Desciptor = dcd.dominant_color_descriptor(
        data.iloc[0, 0], img_type='RGB', order=n, resize=True, flatten=False)

    APs = evaluate_class(db, f_class=DCD, d_type=d_type, depth=depth)
    cls_MAPs = []
    for cls, cls_APs in APs.items():
        MAP = np.mean(cls_APs)
        print("Class {}, MAP {}".format(cls, MAP))
        cls_MAPs.append(MAP)
    print("MMAP", np.mean(cls_MAPs))
