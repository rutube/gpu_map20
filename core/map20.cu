#include <iostream>
#include "map20.h"
#include "../kernels/top_n.h"
#include "../kernels/map20.h"
#include "../utils/utils.h"

using namespace std;

float *compute_map20(float *gpu_ranked, const char *relevance_file,
                     const int relevance_offset, const int rows,
                     const int variants) {
    cout << "Loading relevance file..." << endl;
    float *relevance = load_matrix(relevance_file, relevance_offset, 1, rows);

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

    int threads = 50;
    int blocks = (variants + threads / 2) / threads;

    cout << "Computing Top-20 for " << rows << " rows" << endl;
    cudakernelcall(top_n, blocks, threads, gpu_ranked, gpu_relevance, rows, variants);

    cout << "Computing MAP@20 for " << rows << " rows" << endl;
    cudakernelcall(average_precision_n, blocks, threads, gpu_relevance, gpu_result, rows, variants);

    cout << "Downloading result from GPU..." << endl;
    float *result = download_from_gpu(gpu_result, variants);

#ifdef SAVE_TOP
    float* top_ranks = download_from_gpu(gpu_ranked, variants * rows);
    save_matrix("top_ranks.bin", top_ranks, variants, rows);
    float* top_relevance = download_from_gpu(gpu_relevance, variants * rows);
    save_matrix("top_relevance.bin", top_relevance, variants, rows);
#endif

    cout << "Cleanup GPU..." << endl;
    cleanup_gpu(&gpu_relevance, 1, &gpu_result, 1, NULL, false);
    return result;
}
