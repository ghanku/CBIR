#!env python
# -*- coding: utf-8 -*-
# LBP(Local Binary Pattern) of 2D image.
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
from skimage.feature import local_binary_pattern
import itertools
from tqdm import tqdm


# configs for LBP
mu = 12        # GVF regularization coefficient

d_type = 'd1'      # distance type (similarity measure)
depth = 3         # retrieved depth, set to None will count the ap for whole database


# cache dir
cache_dir = 'cache'
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)


class LBP(object):
    def Local_Binary_Pattern(self, input, radius=3, points=24, method='default', resize=True,  normalize=True, flatten=False):
        ''' calc Local Binary Pattern of input image

          arguments
            input  : input single channel only image or 2D array
            radius : Radius of circle hyperparameter (spatial resolution of the operator)
            points : Number of neighbor set points
            method : {‘default’, ‘ror’, ‘uniform’, ‘var’}
                     Method to determine the pattern.

          return
            flatten == False
                a tuple of two numpy 2D arrays each with shape equal to input image
            flatten == True
                a numpy array with size equal to 2*height(input)*width(input)
        '''

        # read image
        if isinstance(input, np.ndarray):  # check if input is path to image or an array
            img = input.copy()
        else:
            img = imageio.imread(input, pilmode='RGB')

        height, width, depth = img.shape
        # resize images into 200x200
        if resize:
            img = skimage.transform.resize(img, (200, 200, 3))
        img = img.astype(np.uint8)  # make sure its values are between 0-255

        lbp = img.copy()  # just initialize it with whatever
        # calculate LBP for each color channel
        for i in range(depth):
            lbp[:, :, i] = local_binary_pattern(
                img[:, :, i], P=points, R=radius, method=method)

        if normalize:
            lbp = lbp.astype(float) / 255

        if flatten:
            lbp = lbp.flatten()

        return lbp

    def make_samples(self, db, verbose=True):
        sample_cache = "lbp_cache"
        try:
            samples = cPickle.load(
                open(os.path.join(cache_dir, sample_cache), "rb", True))
            if verbose:
                print("Using cache..., config=%s, distance=%s, depth=%s" %
                      (sample_cache, d_type, depth))
        except:
            if verbose:
                print("Counting lbp..., config=%s, distance=%s, depth=%s" % (
                    sample_cache, d_type, depth))

        data = db.get_data()

        samples = []
        for d in tqdm(data.itertuples(), total=len(data)):
            d_img, d_cls = getattr(d, "img"), getattr(d, "cls")
            d_lbp = self.Local_Binary_Pattern(
                d_img, radius=3, points=24, resize=True, method='default', flatten=True)
            samples.append({
                'img':  d_img,
                'cls':  d_cls,
                'hist': d_lbp
            })

        cPickle.dump(samples, open(os.path.join(
            cache_dir, sample_cache), "wb", True))

        return samples


if __name__ == "__main__":
    db = Database()
    data = db.get_data()
    lbp = LBP()

    # test lbp feature extraction on one instance
    lbp_img = lbp.Local_Binary_Pattern(
        input=data.iloc[0, 0], radius=3, points=24, method='default', resize=True,  normalize=True, flatten=False)

    APs = evaluate_class(db, f_class=LBP, d_type=d_type, depth=depth)
    cls_MAPs = []
    for cls, cls_APs in APs.items():
        MAP = np.mean(cls_APs)
        print("Class {}, MAP {}".format(cls, MAP))
        cls_MAPs.append(MAP)
    print("MMAP", np.mean(cls_MAPs))
