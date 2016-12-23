//
// Created by tumbler on 17.11.16.
//

#ifndef GPU_MAP20_TOP_N_H
#define GPU_MAP20_TOP_N_H

#include <host_defines.h>

__global__ void top_n(float *ranks, float *relevance, int* queries, int num_queries, int total_rows, int variants);

#endif //GPU_MAP20_TOP_N_H
