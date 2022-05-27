#!env python
# -*- coding: utf-8 -*-
# Elleptic Fourier Descriptor of 2D image.
# file created by A.Chabira
# License: Public Domain
#
# reference:
# Kuhl, F. P., & Giardina, C. R. (1982). Elliptic Fourier features of a closed contour. Computer Graphics and Image Processing, 18(3), 236–258. https://doi.org/10.1016/0146-664X(82)90034-X


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
from pyefd import elliptic_fourier_descriptors
from skimage import measure


# configs for EFD
n = 8        # order of function to approxiamte contour by EFD

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


class EFD(object):
    def elleptic_fourier_descriptor(self, input, order=n, normalize=False, resize=True, flatten=False):
        ''' Calculates the Elleptic Feature Descriptor of an image

            arguments
              input  : either an image or a 2D array
              order : order of function to approxiamte contour by EFD

            return
                  a numpy array with shape equal to (, )
        '''

        # read image
        if isinstance(input, np.ndarray):  # check if input is path to image or an array
            img = input.copy()
        else:
            img = imageio.imread(input, pilmode='RGB')

        # convert RGB image into grayscale since this is a shape based feature type
        img = skimage.color.rgb2gray(img)

        if resize:
            img = skimage.transform.resize(img, (200, 200))

        # threshold
        #img = (img < 127)

        # conpute contours
        contours = measure.find_contours(img)
        if (not contours):
            print('empty contour')
        coeffs = []
        for cnt in contours:
            # Find the coefficients of all contours
            coeff = elliptic_fourier_descriptors(
                cnt, order=order, normalize=True)
            coeffs.append(coeff.flatten()[3:])

        coeffs = np.array(coeffs)

        if normalize:
            coeffs /= np.sum(coeffs)

        if flatten:
            coeffs = coeffs.flatten()

        return coeffs

    def make_samples(self, db, verbose=True):
        sample_cache = "efd_cache"
        try:
            samples = cPickle.load(
                open(os.path.join(cache_dir, sample_cache), "rb", True))
            if verbose:
                print("Using cache..., config=%s, distance=%s, depth=%s" %
                      (sample_cache, d_type, depth))
        except:
            if verbose:
                print("Counting efd..., config=%s, distance=%s, depth=%s" % (
                    sample_cache, d_type, depth))
            samples = []
            data = db.get_data()
            for d in tqdm(data.itertuples(), total=len(data)):
                d_img, d_cls = getattr(d, "img"), getattr(d, "cls")
                d_efd = self.elleptic_fourier_descriptor(
                    d_img, order=n, resize=True, flatten=False)
                samples.append({
                    'img':  d_img,
                    'cls':  d_cls,
                    'hist': d_efd
                })
            cPickle.dump(samples, open(os.path.join(
                cache_dir, sample_cache), "wb", True))
        return samples


if __name__ == '__main__':

    db = Database()
    data = db.get_data()
    efd = EFD()

    # test DCD on one instance
    Desciptor = efd.elleptic_fourier_descriptor(
        data.iloc[0, 0], order=n, resize=True, flatten=False)

    APs = evaluate_class(db, f_class=EFD, d_type=d_type, depth=depth)
    cls_MAPs = []
    for cls, cls_APs in APs.items():
        MAP = np.mean(cls_APs)
        print("Class {}, MAP {}".format(cls, MAP))
        cls_MAPs.append(MAP)
    print("MMAP", np.mean(cls_MAPs))
