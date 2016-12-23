#!/usr/bin/env python


import os
import sys

import numpy as np

FACTORS = 48


def compute_top_20(matrix, weight):
    result = matrix.dot(weight.T)
    rr = -(np.sort(-result)[:20])
    return np.argsort(-result)[:20]


def compute_ap_20(top_indices, relevance):
    Psum = APsum = 0
    k = 1
    for j in top_indices:
        rel = relevance[j]
        Psum += rel
        Pk = Psum / k
        APsum += Pk * rel
        k += 1
    ap = APsum / len(top_indices)
    return ap


if __name__ == '__main__':
    dirname = sys.argv[1]
    matrix = np.fromfile(os.path.join(dirname, "matrix.bin"), dtype=np.float32)
    s = matrix.shape[0]
    matrix.shape = (s/FACTORS, FACTORS)
    weights = np.fromfile(os.path.join(dirname, "weights.bin"), dtype=np.float32)
    s = weights.shape[0]
    weights.shape = (s/FACTORS, FACTORS)
    relevance = np.fromfile(os.path.join(dirname, "relevance.bin"), dtype=np.float32)
    queries = np.fromfile(os.path.join(dirname, "queries.bin"), dtype=np.uint32)

    w_map = []
    for w in weights:
        idx = 0
        APsum = 0
        for q in queries:
            results = matrix[idx: idx + q, :]
            rel = relevance[idx: idx + q]
            idx += q
            top = compute_top_20(results, w)
            APsum += compute_ap_20(top, rel)
        w_map.append(APsum)

    map = np.array(w_map)
    print(map)

    gpu_map = np.fromfile(sys.argv[2], dtype=np.float32)
    print(gpu_map)


