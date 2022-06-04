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
radius = 3
points = 24
method = 'default'

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
                a single channel 2D array with shape equal to input image
            flatten == True
                a numpy array with size equal to height(input)*width(input)
        '''

        # read image
        if isinstance(input, np.ndarray):  # check if input is path to image or an array
            img = input.copy()
        else:
            img = imageio.imread(input, pilmode='RGB')

        # convert to grayscale
        img = skimage.color.rgb2gray(img)
        # resize images into 200x200
        if resize:
            img = skimage.transform.resize(img, (200, 200))

        # img = img.astype(np.uint8)  # make sure its values are between 0-255

        # calculate LBP
        lbp = local_binary_pattern(img, P=points, R=radius, method=method)
        # print(lbp.max())
        # print(lbp.min())
        if normalize:
            lbp = 255 * lbp / np.max(lbp)

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
                    d_img, radius=3, points=24, resize=True, method='default', flatten=False)
                samples.append({
                    'img':  d_img,
                    'cls':  d_cls,
                    'hist': d_lbp.astype(np.uint8)
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
        input=data.iloc[0, 0], radius=3, points=24, method='default', resize=True,  normalize=False, flatten=False)

    APs = evaluate_class(db, f_class=LBP, d_type=d_type, depth=depth)
    cls_MAPs = []
    for cls, cls_APs in sorted(APs.items()):
        MAP = np.mean(cls_APs)
        print("Class {}, MAP {}".format(cls, MAP))
        cls_MAPs.append(MAP)
    print("MMAP", np.mean(cls_MAPs))
