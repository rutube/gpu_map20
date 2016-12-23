//
// Created by tumbler on 17.11.16.
//

#ifndef GPU_MAP20_UTILS_H
#define GPU_MAP20_UTILS_H

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>


/// обертка для вывода ошибок от библиотеки CUDA
#define cudacall(call) \
    do\
    {\
	cudaError_t err = (call);\
	if(cudaSuccess != err)\
	    {\
		fprintf(stderr, "CUDA Error:\nFile = %s\nLine = %d\nReason = %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err));\
		cudaDeviceReset();\
		exit(EXIT_FAILURE);\
	    }\
    }\
    while (0)\

/* обертка для вывода ошибок CuBLAS */
#define cublascall(call) \
do\
{\
	cublasStatus_t status = (call);\
	if(CUBLAS_STATUS_SUCCESS != status)\
	{\
		fprintf(stderr,"CUBLAS Error:\nFile = %s\nLine = %d\nCode = %d\n", __FILE__, __LINE__, status);\
		cudaDeviceReset();\
		exit(EXIT_FAILURE);\
	}\
}\
while(0)\


/* запуск CUDA kernel с синхронизацией и проверкой ошибок */
#define cudakernelcall(kernel, blocks, threads, ...)\
do\
{\
    kernel <<< (blocks), (threads) >>> (__VA_ARGS__);\
    cudaDeviceSynchronize();\
    cudaError_t e=cudaGetLastError();\
    if(e!=cudaSuccess) {\
        printf("CUDA failure %s:%d: '%s'\n",__FILE__,__LINE__, cudaGetErrorString(e));\
        exit(EXIT_FAILURE);\
    }\
}\
while(0)

/// загрузка float-матрицы из файла с учетом offset
/// \param matrix_file путь до файла с данными для матрицы - float32 по строкам
/// \param matrix_offset смещение в matrix_file в байтах
/// \param width число столбцов матрицы
/// \param height число строк матрицы
/// \return указатель на хост-память, аллоцированный через cudaMallocGHost
float* load_matrix(const char *matrix_file, const size_t matrix_offset,
                   const int width, const int height);

/// сохранение float-матрицы в файл
/// \param matrix_file путь до выходного файла
/// \param host_ptr указатель на хост-память матрицы (считается, что матрица расположена по строкам)
/// \param width число столбцов матрицы
/// \param height число строк матрицы
void save_matrix(const char *matrix_file, const float *host_ptr,
                 const int width, const int height);


/// освобождает GPU + зачищает массивы на хосте и на GPU и освобождает хендл для
/// библиотеки CuBLAS.
/// \param host_pointers массив указателей на хост-массивы для освобождения памяти
/// \param host_ptr_count длина массива
/// \param gpu_pointers массив указателей на gpu-массивы для освобождения памяти
/// \param gpu_ptr_count длина массива
/// \param blas_handle дексриптор библиотеки CuBLAS
/// \param reset_device флаг необходимости освобождения видеокарты
void cleanup_gpu(float* host_pointers[], const int host_ptr_count,
                 float* gpu_pointers[], const int gpu_ptr_count,
                 cublasHandle_t blas_handle, const bool reset_device);


/// Инициализирует библиотеки CUDA
bool init_gpu();

/// инициализация библиотеку линейной алгебры
/// \return хендл библиотеки CuBLAS
cublasHandle_t init_cublas();


/* аллоцирует память на GPU и загружает туда данные с хоста */
float * upload_to_gpu(const float *host_pointer, int size);

float * download_from_gpu(const float *gpu_pointer, int size);



#endif //GPU_MAP20_UTILS_H
