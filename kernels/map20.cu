#include <device_launch_parameters.h>
#include <math_functions.h>
#include "constants.h"

/// Вычисляет AP@N с учетом указанной релевантности
__global__ void average_precision_n(
        const float *relevance, /// [in] матрица признаков релевантности (0, 1) размера rows * variants
        float *result, /// [out] массив для сохранения AveragePrecision размера variants
        int rows, /// [in] число документов в выдаче
        int variants /// [in] общее число вариантов
){
    // variant - номер обрабатываемого варианта
    int variant = blockIdx.x * blockDim.x + threadIdx.x;
    if (variant >= variants)
        return;
    int v_offset = variant * rows;
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
    result[variant] = APsum / len;
}