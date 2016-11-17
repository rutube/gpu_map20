//
// Created by tumbler on 17.11.16.
//

#ifndef GPU_MAP20_TOP_N_H
#define GPU_MAP20_TOP_N_H
__global__ void top_n(float *ranks, float *relevance, float *result, int rows);
#endif //GPU_MAP20_TOP_N_H
