#include <iostream>
#include <cublas_v2.h>
#include "map20.h"
#include "../kernels/top_n.h"
#include "../kernels/map20.h"
#include "../utils/utils.h"

using namespace std;

float *compute_map20(cublasHandle_t cublas_handle, float *gpu_ranked, float* gpu_map20, float* relevance,
                     int* queries, const int num_queries, const int total_rows, const int variants) {
    float *gpu_relevance;
    cudacall(cudaMalloc((void **) &gpu_relevance,
                        total_rows * variants * sizeof(gpu_relevance[0])));
    cudacall(cudaMemcpy(gpu_relevance, relevance,
                        (size_t)(total_rows * sizeof(gpu_relevance[0])),
                        cudaMemcpyHostToDevice));

    int *gpu_queries;
    cudacall(cudaMalloc((void **) &gpu_queries, num_queries * sizeof(gpu_queries[0])));
    cudacall(cudaMemcpy(gpu_queries, queries,
                        (size_t)(num_queries * sizeof(gpu_queries[0])),
                        cudaMemcpyHostToDevice));


    // размножаем relevance по числу вариантов
    for (int i = 1; i < variants; i++) {
        cudacall(cudaMemcpy(&gpu_relevance[total_rows * i], gpu_relevance,
                            (size_t)(total_rows * sizeof(gpu_relevance[0])),
                            cudaMemcpyDeviceToDevice));
    }

    float *gpu_result;
    cudacall(cudaMalloc((void **) &gpu_result, num_queries * variants * sizeof(gpu_result[0])));

    // Optimal size for grid:
    // http://stackoverflow.com/a/12921834

    int threads = 256;
    int blocks = max((variants * num_queries + threads / 2) / threads, 1);

    cout << "Computing Top-20 for " << num_queries << " queries " << variants << " variants" << endl;
    cudakernelcall(top_n, blocks, threads, gpu_ranked, gpu_relevance, gpu_queries, num_queries, total_rows, variants);
    float * data;

    cout << "queries[0]=" << queries[0] << endl;
    cout << "total_rows=" << total_rows << endl;
    cout << "num_queries=" << num_queries << endl;

    data = download_from_gpu(gpu_ranked,  total_rows * variants);
    for (int i=0; i< 10; i++) {
        cout << "ranked[" << i << "] = " << data[i] << endl;
    }
    cout << "Computing AP@20 for " << total_rows << " rows in " << num_queries << " queries" << endl;
    cudakernelcall(average_precision_n, blocks, threads, gpu_relevance, gpu_result, gpu_queries, num_queries,
                   total_rows, variants);

    data = download_from_gpu(gpu_result,  num_queries * variants);

    cout << "ap[0] = " << data[0] << endl;
    cout << "ap[1] = " << data[1] << endl;
    cout << "ap[2] = " << data[2] << endl;

    const float alpha = 1.0;
    cout << "Accumulate AP@20..." << endl;
    cublascall(cublasSaxpy(cublas_handle, variants, &alpha, gpu_result, 1, gpu_map20, 1));

    cout << "Cleanup GPU..." << endl;
    cleanup_gpu(NULL, 0, &gpu_result, 1, NULL, false);
    cleanup_gpu(NULL, 0, &gpu_relevance, 1, NULL, false);

    return gpu_map20;
}
