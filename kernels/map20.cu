#include <device_launch_parameters.h>
#include <math_functions.h>
#include "constants.h"

/// Вычисляет AP@N с учетом указанной релевантности
__global__ void average_precision_n(
        const float *relevance, /// [in] матрица признаков релевантности (0, 1) размера rows * variants
        float *result, /// [out] массив для сохранения AveragePrecision размера variants
        int* queries, /// смещения начала каждого запроса в массиве relevance
        int num_queries, /// общее число запросов
        int total_rows, /// [in] общее число документов для всех запросов
        int variants /// [in] общее число вариантов
){
    // variant - номер обрабатываемого варианта
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_idx >= variants * num_queries)
        return;
    int variant = global_idx % variants;
    int query = global_idx / variants;
    int offset = queries[query];
    int next_offset = (query==num_queries - 1)? total_rows: queries[query + 1];
    int rows = next_offset - offset;
    int v_offset = variant * total_rows + offset;
    int len = min(N, rows);
    float Psum = 0;
    float Pk;
    float rel;
    float APsum = 0;
    int k = 1;

    for(int j = v_offset; j < v_offset + len; j++) {
        rel = relevance[j];
        Psum += rel;
        Pk = Psum / k;
        APsum += Pk * rel;
        k += 1;
    }
    // вычисляем и возвращаем AveragePrecision@N, в знаменателе - реальное
    // число запросов в выдаче, если их меньше N.
    atomicAdd(&result[variant], APsum / len);
//    result[query * variants + variant] = APsum / len;
}
