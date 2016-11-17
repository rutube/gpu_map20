//
// Created by tumbler on 17.11.16.
//

#ifndef GPU_MAP20_MAP20_H
#define GPU_MAP20_MAP20_H
__global__ void average_precision_n(
        const float *relevance, /// [in] матрица признаков релевантности (0, 1) размера rows * variants
        float *result, /// [out] массив для сохранения AveragePrecision размера variants
        int rows /// [in] число документов в выдаче
);
#endif //GPU_MAP20_MAP20_H
