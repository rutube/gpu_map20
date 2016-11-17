#include <iostream>
#include <cublas_v2.h>
#include "../utils/utils.h"

using namespace std;


float *prepare_ranks(cublasHandle_t cublas_handle, const char *matrix_file, const int matrix_offset,
                     const char *weights_file, const int rows, const int factors, const int variants) {
    float *host_pointers[2];
    float *gpu_pointers[2];

    cout << "Loading matrix file..." << endl;
    // Загружаем матрицу признаков поисковой выдачи
    // матрица <rows> x <factors> построчно
    float *matrix = host_pointers[0] = load_matrix(
            matrix_file, matrix_offset, factors, rows);
    float *gpu_matrix = gpu_pointers[0] = upload_to_gpu(
            matrix, factors * rows);

    cout << "Loading weights file..." << endl;
    // Загружаем матрицу весов ранкера
    // матрица <variants> x <factors> построчно
    float *weights = host_pointers[1] = load_matrix(
            weights_file, 0, factors, variants);
    float *gpu_weights = gpu_pointers[1] = upload_to_gpu(
            weights, factors * variants);

    float *gpu_ranked;
    cudacall(cudaMalloc((void **) &gpu_ranked,
                        rows * variants * sizeof(gpu_ranked[0])));
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cout << "Perform: matrix * weights.T ..." << endl;
    // ranked = (matrix * weights.T).T;
    // http://docs.nvidia.com/cuda/cublas/#data-layout
    // "cuBLAS library uses column-major storage"
    // т.ч. чтобы из файла, созданного numpy, получить нужную матрицу, а не черти-что, надо брать транспонированный
    // вариант для матрицы и нетранспонированный для весов. Результат тоже получается транспонированный, первая строка
    // содержит <rows> рангов документов для первого варианта
    cublascall(cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                           rows, variants, factors,
                           &alpha,
                           gpu_matrix, factors,
                           gpu_weights, factors,
                           &beta,
                           gpu_ranked, rows));
    cudaDeviceSynchronize();
    cout << "Cleanup GPU..." << endl;
    cleanup_gpu(host_pointers, 2, gpu_pointers, 2, NULL, false);
    return gpu_ranked;
}
