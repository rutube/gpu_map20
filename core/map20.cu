#include <iostream>
#include <cublas_v2.h>
#include "map20.h"
#include "../kernels/top_n.h"
#include "../kernels/map20.h"
#include "../utils/utils.h"

using namespace std;

float *compute_map20(cublasHandle_t cublas_handle, float *gpu_ranked, float* gpu_map20, float* relevance,
                     const int rows, const int variants) {

    float *gpu_relevance;
    cudacall(cudaMalloc((void **) &gpu_relevance,
                        rows * variants * sizeof(gpu_relevance[0])));
    cudacall(cudaMemcpy(gpu_relevance, relevance,
                        (size_t)(rows * sizeof(gpu_relevance[0])),
                        cudaMemcpyHostToDevice));
    // размножаем relevance по числу вариантов
    for (int i = 1; i < variants; i++) {
        cudacall(cudaMemcpy(&gpu_relevance[rows * i], gpu_relevance,
                            (size_t)(rows * sizeof(gpu_relevance[0])),
                            cudaMemcpyDeviceToDevice));
    }

    float *gpu_result;
    cudacall(cudaMalloc((void **) &gpu_result, variants * sizeof(gpu_result[0])));

    // Optimal size for grid:
    // http://stackoverflow.com/a/12921834

    int threads = 256;
    int blocks = max((variants + threads / 2) / threads, 1);

    cout << "Computing Top-20 for " << rows << " rows " << variants << " variants with T=" << threads << "B=" << blocks << endl;
    cudakernelcall(top_n, blocks, threads, gpu_ranked, gpu_relevance, rows, variants);

    cout << "Computing AP@20 for " << rows << " rows" << endl;
    cudakernelcall(average_precision_n, blocks, threads, gpu_relevance, gpu_result, rows, variants);

    const float alpha = 1.0;
    cout << "Accumulate AP@20..." << endl;
    cublascall(cublasSaxpy(cublas_handle, variants, &alpha, gpu_result, 1, gpu_map20, 1));

    cout << "Cleanup GPU..." << endl;
    cleanup_gpu(NULL, 0, &gpu_result, 1, NULL, false);
    cleanup_gpu(NULL, 0, &gpu_relevance, 1, NULL, false);

    return gpu_map20;
}
