#!env python
# -*- coding: utf-8 -*-
# GVF(Gradient Vector Flow) of 2D image.
# file created by A.Chabira
# GVF functions implemented by t-suzuki
# original class structure made by Po-Chih Huang
# License: Public Domain
#
# reference:
# Chenyang Xu, et al. "Snakes, Shpes, and Gradient Vector Flow", IEEE TRANACTIONS ON IMAGE PROCESSING, VOL. 7, NO. 3, MARCH 1998

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


# configs for GVF
mu = 12        # GVF regularization coefficient

d_type = 'd1'      # distance type (similarity measure)
depth = 3         # retrieved depth, set to None will count the ap for whole database


# cache dir
cache_dir = 'cache'
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)


class GVF(object):
    def gradient_vector_flow(self, input, mu, dx=1.0, dy=1.0, normalize=True, resize=True, flatten=False):
        ''' calc gradient vector flow of input gradient field fx, fy

          arguments
            input : input egde map
            dx    :
            dy    :
            mu    : GVF regularization coefficient

          return
            flatten == False
                a tuple of two numpy 2D arrays each with shape equal to input image
            flatten == True
                a numpy array with size equal to 2*length(input)*width(input)
        '''

        # read image
        if isinstance(input, np.ndarray):  # check if input is path to image or an array
            img = input.copy()
        else:
            img = imageio.imread(input, pilmode='RGB')
        # convert RGB image into grayscale since this is a shape based feature type
        img = skimage.color.rgb2gray(img)
        # resize images into 200x200
        if resize:
            img = skimage.transform.resize(img, (200, 200))

        img = img.astype(np.float32) / 255.0  # normalize
        # img = add_border(img, 32)

        # calculate the edge map of the image
        edge = self.edge_map(img, sigma=2)
        # calculate the gradient field of the edge map
        fx, fy = self.gradient_field(edge)
        # calc some coefficients.
        b = fx**2.0 + fy**2.0
        c1, c2 = b * fx, b * fy
        # calc dt from scaling parameter r.
        r = 0.25  # (17) r < 1/4 required for convergence.
        dt = dx * dy / (r * mu)
        # max iteration
        N = int(max(1, np.sqrt(img.shape[0] * img.shape[1])))

        # initialize u(x, y), v(x, y) by the input.
        curr_u = fx
        curr_v = fy

        def laplacian(m):
            return np.hstack([m[:, 0:1], m[:, :-1]]) + np.hstack([m[:, 1:], m[:, -2:-1]]) \
                + np.vstack([m[0:1, :], m[:-1, :]]) + np.vstack([m[1:, :], m[-2:-1, :]]) \
                - 4 * m

        # iteration loop
        for i in range(N):
            next_u = (1.0 - b * dt) * curr_u + r * laplacian(curr_u) + c1 * dt
            next_v = (1.0 - b * dt) * curr_v + r * laplacian(curr_v) + c2 * dt
            curr_u, curr_v = next_u, next_v

        feature_vect = np.zeros([curr_u.shape[0], curr_u.shape[1], 2])
        feature_vect[:, :, 0] = curr_u
        feature_vect[:, :, 1] = curr_v

        if normalize:
            feature_vect /= np.sum(feature_vect)

        if flatten:
            feature_vect = feature_vect.flatten()

        return feature_vect

    def edge_map(self, img, sigma):
        blur = skimage.filters.gaussian(img, sigma)
        return skimage.filters.sobel(blur)

    def gradient_field(self, im):
        im = skimage.filters.gaussian(im, 1.0)
        gradx = np.hstack([im[:, 1:], im[:, -2:-1]]) - \
            np.hstack([im[:, 0:1], im[:, :-1]])
        grady = np.vstack([im[1:, :], im[-2:-1, :]]) - \
            np.vstack([im[0:1, :], im[:-1, :]])
        return gradx, grady

    def add_border(self, img, width):
        h, w = img.shape
        val = img[:, 0].mean() + img[:, -1].mean() + \
            img[0, :].mean() + img[-1, :].mean()
        res = np.zeros((h + width * 2, w + width * 2), dtype=img.dtype) + val
        res[width:h + width, width:w + width] = img
        res[:width, :] = res[width, :][np.newaxis, :]
        res[:, :width] = res[:, width][:, np.newaxis]
        res[h + width:, :] = res[h + width - 1, :][np.newaxis, :]
        res[:, w + width:] = res[:, w + width - 1][:, np.newaxis]
        return res

    def make_samples(self, db, verbose=True):
        sample_cache = "gvf_cache"
        try:
            samples = cPickle.load(
                open(os.path.join(cache_dir, sample_cache), "rb", True))
            if verbose:
                print("Using cache..., config=%s, distance=%s, depth=%s" %
                      (sample_cache, d_type, depth))
        except:
            if verbose:
                print("Counting gvf..., config=%s, distance=%s, depth=%s" % (
                    sample_cache, d_type, depth))
        samples = []
        data = db.get_data()
        for d in tqdm(data.itertuples(), total=len(data)):
            d_img, d_cls = getattr(d, "img"), getattr(d, "cls")
            d_gvf = self.gradient_vector_flow(
                d_img, mu=1.0, resize=True, flatten=True)
            samples.append({
                'img':  d_img,
                'cls':  d_cls,
                'hist': d_gvf
            })
        cPickle.dump(samples, open(os.path.join(
            cache_dir, sample_cache), "wb", True))
        return samples


if __name__ == "__main__":
    db = Database()
    data = db.get_data()
    gvf = GVF()

    # test gvf extraction on one instance
    grad_v_flow = gvf.gradient_vector_flow(data.iloc[0, 0], mu=1.0)
    gx, gy = grad_v_flow[0], grad_v_flow[1]

    APs = evaluate_class(db, f_class=GVF, d_type=d_type, depth=depth)
    cls_MAPs = []
    for cls, cls_APs in APs.items():
        MAP = np.mean(cls_APs)
        print("Class {}, MAP {}".format(cls, MAP))
        cls_MAPs.append(MAP)
    print("MMAP", np.mean(cls_MAPs))
