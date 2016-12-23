#!/usr/bin/env python

import os
import sys

import numpy as np


FACTORS = 48


if __name__ == '__main__':
    dirname = sys.argv[1]
    block_size = int(sys.argv[2])

    queries = np.fromfile(os.path.join(dirname, 'queries.bin'), dtype=np.uint32)

    prev_matrix_offset = 0
    prev_relevance_offset = 0
    prev_queries_offset = 0
    matrix_offset = 0
    relevance_offset = 0
    queries_offset = 0
    i = 0
    rows_count = 0
    first = True
    for rows in queries:
        rows_count += rows
        queries_offset += np.dtype(np.uint32).itemsize
        matrix_offset += np.dtype(np.float32).itemsize * rows * FACTORS
        relevance_offset += np.dtype(np.float32).itemsize * rows
        i += 1
        if i == block_size:
            if first:
                print("./gpu_map20 --rows {c} --queries "
                      "{dir}/queries.bin {dir}/matrix.bin {dir}/relevance.bin {dir}/weights.bin".format(
                    c=rows_count, dir=dirname
                ))
                first = False
            else:
                print("./gpu_map20 --append --moffset {m} --roffset {r} --qoffset {q} --rows {c} --queries "
                      "{dir}/queries.bin {dir}/matrix.bin {dir}/relevance.bin {dir}/weights.bin".format(
                    m=prev_matrix_offset, r=prev_relevance_offset, q=prev_queries_offset, c=rows_count, dir=dirname
                ))
            i = 0
            rows_count = 0
            prev_matrix_offset = matrix_offset
            prev_queries_offset = queries_offset
            prev_relevance_offset = relevance_offset

    if i != 0:
        print("./gpu_map20 --append --moffset {m} --roffset {r} --qoffset {q} --rows {c} --queries "
              "{dir}/queries.bin {dir}/matrix.bin {dir}/relevance.bin {dir}/weights.bin".format(
            m=prev_matrix_offset, r=prev_relevance_offset, q=prev_queries_offset, c=rows_count, dir=dirname
        ))
