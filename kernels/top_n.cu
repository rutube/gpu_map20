#include <device_launch_parameters.h>
#include <math_functions.h>
#include "constants.h"

/*
Сканирует результаты ранжирования, выбирает из них топ-20 лучших и возвращает
в массив result соответствующие им значения релевантности.
*/
__global__ void top_n(float *ranks, float *relevance, int* queries, int num_queries, int total_rows, int variants){
    // variant - номер обрабатываемого варианта

    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_idx >= variants * num_queries)
        return;
    int variant = global_idx / num_queries;
    int query = global_idx % num_queries;
    int offset = queries[query];
    int next_offset = (query==num_queries - 1)? total_rows: queries[query + 1];
    int rows = next_offset - offset;
    int v_offset = variant * total_rows + offset;
    int len = min(N, rows);
    float value, tmp, max_value;
    int max_idx;

    for (int k = v_offset; k < v_offset + len; k++) {
        max_idx = k;
        max_value = ranks[k];
        for (int j = k + 1; j < v_offset + rows; j++) {
            value = ranks[j];
            if (value > max_value) {
                max_value = value;
                max_idx = j;
            }
        }

        tmp = ranks[max_idx];
        ranks[max_idx] = ranks[k];
        ranks[k] = tmp;

        tmp = relevance[max_idx];
        relevance[max_idx] = relevance[k];
        relevance[k] = tmp;
    }
}
