#!env python
# -*- coding: utf-8 -*-
# Color Coherence Vector of 2D image.
# file created by A.Chabira
# original implementaion by https://github.com/kohjingyu
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


# configs for CCV
n = 2       # indicating n^3 discretized colors, will result in feature size of 2 * n^3
tau = 'default'  # threshold for connectet pixel to be classified as coherent or incoherent

d_type = 'd1'      # distance type (similarity measure)
depth = 3         # retrieved depth, set to None will count the ap for whole database

if (len(sys.argv) > 1):
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
              p_tau : threshold for connectet pixel to be classified as coherent or incoherent

            return
                  a numpy array with len equal n^3 * 2 containing the number of coherent and uncoherent pixels for each color bins, grouped together
        '''

        # read image
        if isinstance(input, np.ndarray):  # check if input is path to image or an array
            img = input.copy()
        else:
            img = imageio.imread(input, pilmode='RGB')

        # resize images into 200x200
        if resize:
            img = skimage.transform.resize(img, (200, 200))

        h, w, d = img.shape

        # Blur pixel slightly using avg pooling with 3x3 kernel and then flatten each channel
        blur_img = skimage.filters.gaussian(img, sigma=1)
        blur_flat = blur_img.reshape(h * w, d)

        # Discretize colors
        hist, edges = np.histogramdd(blur_flat, bins=p_n)
        graph = np.zeros((h, w))
        result = np.zeros(blur_img.shape)

        total = 0
        for i in range(0, p_n):
            for j in range(0, p_n):
                for k in range(0, p_n):
                    rgb_val = [edges[0][i + 1], edges[1]
                               [j + 1], edges[2][k + 1]]
                    previous_edge = [edges[0][i], edges[1][j], edges[2][k]]
                    coords = ((blur_img <= rgb_val) & (
                        blur_img >= previous_edge)).all(axis=2)
                    result[coords] = rgb_val
                    graph[coords] = i + j * p_n + k * p_n**2

        result = result.astype(int)
        max_cliques = self.find_max_cliques(graph, p_n, p_tau)

        if normalize:
            max_cliques /= np.sum(max_cliques)

        if flatten:
            max_cliques = max_cliques.flatten()

        return max_cliques

    def is_adjacent(self, x1, y1, x2, y2):
        ''' Returns true if (x1, y1) is adjacent to (x2, y2), and false otherwise '''
        x_diff = abs(x1 - x2)
        y_diff = abs(y1 - y2)
        return not (x_diff == 1 and y_diff == 1) and (x_diff <= 1 and y_diff <= 1)

    def find_max_cliques(self, arr, p_n, p_tau):
        ''' Returns a 2*n dimensional vector
        v_i, v_{i+1} describes the number of coherent and incoherent pixels respectively a given color
        '''
        if p_tau == 'default':
            # Classify as coherent for area is >= 1% of input image area
            p_tau = int(arr.shape[0] * arr.shape[1] * 0.01)

        ccv = [0 for i in range(p_n**3 * 2)]
        unique = np.unique(arr)
        for u in unique:
            x, y = np.where(arr == u)
            groups = []
            coherent = 0
            incoherent = 0

            for i in tqdm(range(len(x))):
                found_group = False
                for group in groups:
                    if found_group:
                        break

                    for coord in group:
                        xj, yj = coord
                        if self.is_adjacent(x[i], y[i], xj, yj):
                            found_group = True
                            group[(x[i], y[i])] = 1
                            break
                if not found_group:
                    groups.append({(x[i], y[i]): 1})

            for group in groups:
                num_pixels = len(group)
                if num_pixels >= p_tau:
                    coherent += num_pixels
                else:
                    incoherent += num_pixels

            assert(coherent + incoherent == len(x))

            index = int(u)
            ccv[index * 2] = coherent
            ccv[index * 2 + 1] = incoherent

        return ccv

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
    for cls, cls_APs in APs.items():
        MAP = np.mean(cls_APs)
        print("Class {}, MAP {}".format(cls, MAP))
        cls_MAPs.append(MAP)
    print("MMAP", np.mean(cls_MAPs))
