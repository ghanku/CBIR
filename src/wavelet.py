#!env python
# -*- coding: utf-8 -*-
# Wavelet transform of 2D image.
# file created by A.Chabira
# original class structure made by Po-Chih Huang
# License: Public Domain


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
import skimage.filters as skimage_filter
import itertools
from tqdm import tqdm
import pywt


# configs for wavelet
level = 4        # order of slantlet matrix
wavelet = 'bior1.5'

d_type = 'd1'      # distance type (similarity measure)
depth = 3         # retrieved depth, set to None will count the ap for whole database

if (len(sys.argv) > 1):
    depth = int(sys.argv[1])
    d_type = sys.argv[2]

# cache dir
cache_dir = 'cache'
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)


class WAVELET(object):
    def wavelet_transform(self, input, level=1, wavelet='db1', normalize=False, resize=True, flatten=False):
        ''' Calculates Slantelet Transform of grayscale image

                arguments
                  input   : input single channel only image or 2D array (preferably square of shape 2**n)
                  level   : level of wavelet transform
                  wavelet : type of wavelet used; pywt.wavelist(family) to view options

                return
                      a numpy 2D array with size equal to iput shape
        '''

        # read image
        if isinstance(input, np.ndarray):  # check if input is path to image or an array
            img = input.copy()
        else:
            img = imageio.imread(input, pilmode='RGB')
        # convert RGB image into grayscale since this is a shape based feature type
        img = skimage.color.rgb2gray(img).astype(np.float32)

        if resize:
            img = skimage.transform.resize(img, (200, 200))

        # calculate wavelet transform
        w = pywt.wavedec2(img, wavelet, mode='periodization', level=level)
        w, _ = pywt.coeffs_to_array(w)  # convert to single array

        if normalize:
            w[0] /= np.abs(w[0]).max()

        if flatten:
            w = w.flatten()

        return w

    def make_samples(self, db, verbose=True):
        sample_cache = "wavelet_cache"
        try:
            samples = cPickle.load(
                open(os.path.join(cache_dir, sample_cache), "rb", True))
            if verbose:
                print("Using cache..., config=%s, distance=%s, depth=%s" %
                      (sample_cache, d_type, depth))
        except:
            if verbose:
                print("Counting wavelet..., config=%s, distance=%s, depth=%s" % (
                    sample_cache, d_type, depth))
            samples = []
            data = db.get_data()
            for d in tqdm(data.itertuples(), total=len(data)):
                d_img, d_cls = getattr(d, "img"), getattr(d, "cls")
                d_wavelet = self.wavelet_transform(
                    d_img, level=level, wavelet=wavelet, normalize=False, resize=True)
                samples.append({
                    'img':  d_img,
                    'cls':  d_cls,
                    'hist': d_wavelet
                })
            cPickle.dump(samples, open(os.path.join(
                cache_dir, sample_cache), "wb", True))
        return samples


if __name__ == "__main__":
    db = Database()
    data = db.get_data()
    wave = WAVELET()

    # test wavelet transform on one instance
    w = wave.wavelet_transform(
        data.iloc[0, 0], level=level, wavelet=wavelet, resize=True, flatten=False)

    APs = evaluate_class(db, f_class=WAVELET, d_type=d_type, depth=depth)
    cls_MAPs = []
    for cls, cls_APs in APs.items():
        MAP = np.mean(cls_APs)
        print("Class {}, MAP {}".format(cls, MAP))
        cls_MAPs.append(MAP)
    print("MMAP", np.mean(cls_MAPs))
