#!env python
# -*- coding: utf-8 -*-
# Slantlet Transform of 2D image.
# file created by A.Chabira
# original matlab implementaion by https://github.com/sanjumaramattam/Image-Transforms
# License: Public Domain
#
# reference:
# Selesnick, I. W. (1999). The Slantlet Transform. In IEEE TRANSACTIONS ON SIGNAL PROCESSING (Vol. 47, Issue 5)


from __future__ import print_function

from evaluate import distance, evaluate_class
from DB import Database

import sys
import os
from six.moves import cPickle
import numpy as np
from numpy import sqrt, zeros, eye
from numpy import concatenate as con
import imageio
import skimage.color
import skimage.data
import skimage.transform
import skimage.filters as skimage_filter
import itertools
from tqdm import tqdm


# configs for slantlet
n = 8        # order of slantlet matrix

d_type = 'd1'      # distance type (similarity measure)
depth = 3         # retrieved depth, set to None will count the ap for whole database

if (len(sys.argv) > 1):
    depth = int(sys.argv[1])
    if depth == "None":
        depth = None
    d_type = sys.argv[2]

# cache dir
cache_dir = 'cache'
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)


class SLANT(object):
    def slant_transform(self, input, order, normalize=False, resize=True, flatten=False):
        ''' Calculates Slantelet Transform of grayscale image

                arguments
                  input   : input single channel only image or 2D array (preferably square of shape 2**n)
                  order : order of slant transform, either 'default' calculated from image shape , or int

                return
                      a numpy 2D array with size equal to size 2**n, where n is the order of slant matrix required
        '''

        # read image
        if isinstance(input, np.ndarray):  # check if input is path to image or an array
            img = input.copy()
        else:
            img = imageio.imread(input, pilmode='RGB')
        # convert RGB image into grayscale since this is a shape based feature type
        img = skimage.color.rgb2gray(img)
        N = img.shape[0]

        if (order == 'default'):
            # calculate order of slant matrix
            n = np.ceil(np.log2(N)).astype(int)

        elif (isinstance(order, int)):
            n = order

        if resize:
            img = skimage.transform.resize(img, (2**n, 2**n))

        S = self.slant_matrix(n)
        S_t = S @ img @ np.transpose(S)  # clalculate slant transform

        if normalize:
            S_t /= np.sum(S_t)

        if flatten:
            S_t = S_t.flatten()

        return S_t

    def slant_matrix(self, n):
        ''' Calculates Slantelet matrices of order n

                arguments
                  n  : order of slant matrix

                return
                      a numpy 2D array with size equal to 2**n
              '''
        # init S1
        S = 1 / sqrt(2) * np.array([[1, 1], [1, -1]])
        a = 1

        for i in range(2, n + 1):

            b = 1 / sqrt(1 + 4 * a**2)
            a = 2 * b * a

            q1 = np.array([[1, 0], [a, b]])
            q2 = np.array([[1, 0], [-a, b]])
            q3 = np.array([[0, 1], [-b, a]])
            q4 = np.array([[0, -1], [b, a]])

            Z = con([con([S, zeros(S.shape)], axis=1),
                     con([zeros(S.shape), S], axis=1)])

            if (i == 2):
                B1 = con([q1, q2], axis=1)  # block 1
                B2 = con([q3, q4], axis=1)  # block 2
                S = (1 / sqrt(2)) * con([B1, B2]) @ Z

            else:
                k = int((2**i - 4) / 2)
                B1 = con([q1, zeros([2, k]), q2, zeros([2, k])],
                         axis=1)  # block 1
                B2 = con([zeros([k, 2]), eye(k), zeros(
                    [k, 2]), eye(k)], axis=1)  # block 2
                B3 = con([q3, zeros([2, k]), q4, zeros([2, k])],
                         axis=1)  # block 3
                B4 = con([zeros([k, 2]), eye(k), zeros(
                    [k, 2]), -eye(k)], axis=1)  # block 4

                S = (1 / sqrt(2)) * con([B1, B2, B3, B4]) @ Z

        return S

    def make_samples(self, db, verbose=True):
        sample_cache = "slant_cache"
        try:
            samples = cPickle.load(
                open(os.path.join(cache_dir, sample_cache), "rb", True))
            if verbose:
                print("Using cache..., config=%s, distance=%s, depth=%s" %
                      (sample_cache, d_type, depth))
        except:
            if verbose:
                print("Counting slant..., config=%s, distance=%s, depth=%s" % (
                    sample_cache, d_type, depth))
            samples = []
            data = db.get_data()
            for d in tqdm(data.itertuples(), total=len(data)):
                d_img, d_cls = getattr(d, "img"), getattr(d, "cls")
                d_slant = self.slant_transform(
                    d_img, order=n, resize=True, flatten=False)
                samples.append({
                    'img':  d_img,
                    'cls':  d_cls,
                    'hist': d_slant
                })
            cPickle.dump(samples, open(os.path.join(
                cache_dir, sample_cache), "wb", True))
        return samples


if __name__ == "__main__":
    db = Database()
    data = db.get_data()
    slant = SLANT()

    # test salntlet transform on one instance
    S_t = slant.slant_transform(
        data.iloc[0, 0], order=n, resize=True, flatten=False)

    APs = evaluate_class(db, f_class=SLANT, d_type=d_type, depth=depth)
    cls_MAPs = []
    for cls, cls_APs in APs.items():
        MAP = np.mean(cls_APs)
        print("Class {}, MAP {}".format(cls, MAP))
        cls_MAPs.append(MAP)
    print("MMAP", np.mean(cls_MAPs))
