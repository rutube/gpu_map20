#!/usr/bin/env python

import random

import numpy as np
import os
import sys

FACTORS = 48


def generate_query_results(results=1000):
    return np.random.rand(results, FACTORS).astype(np.float32)


def generate_relevance(results=1000, relevant=100):
    rel = np.zeros(results).astype(np.float32)
    positions = list(range(results))
    for _ in range(relevant):
        pos = random.choice(positions)
        positions.remove(pos)
        rel[pos] = 1.0
    return rel


def generate_weights(count=1):
    return np.random.rand(count, FACTORS).astype(np.float32)


def generate_queries_file(count=3):
    queries = []
    matrix = []
    relevance = []
    for _ in range(count):
        results = random.randint(500, 1000)
        matrix.append(generate_query_results(results=results))
        relevance.append(generate_relevance(results=results, relevant=results // 10))
        queries.append(results)

    matrix = np.vstack(matrix)
    relevance = np.hstack(relevance)
    queries = np.array(queries, dtype=np.uint32)

    return matrix, relevance, queries


if __name__ == '__main__':
    dirname = sys.argv[1]
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    queries = int(sys.argv[2])
    if queries:
        matrix, relevance, queries = generate_queries_file(count=queries)

        matrix.tofile(os.path.join(dirname, 'matrix.bin'))
        relevance.tofile(os.path.join(dirname, 'relevance.bin'))
        queries.tofile(os.path.join(dirname, 'queries.bin'))

    variants = int(sys.argv[3])
    if variants:
        weights = generate_weights(count=variants)
        weights.tofile(os.path.join(dirname, 'weights.bin'))


