//
// Created by tumbler on 17.11.16.
//

#ifndef GPU_MAP20_MAP20_H
#define GPU_MAP20_MAP20_H

#include <host_defines.h>

__global__ void average_precision_n(
        const float *relevance, /// [in] матрица признаков релевантности (0, 1) размера rows * variants
        float *result,          /// [out] массив для сохранения AveragePrecision размера variants
        int* queries,           /// [in] смещения начала каждого запроса в массиве relevance
        int num_queries,        /// [in] общее число запросов
        int total_rows,         /// [in] общее число документов для всех запросов
        int variants            /// [in] число вариантов
);
#endif //GPU_MAP20_MAP20_H
