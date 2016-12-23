#ifndef MAP20_H_
#define MAP20_H_

float* compute_map20(cublasHandle_t cublas_handle, float *gpu_ranked, float * gpu_map20, float *relevance,
                     const int rows, const int variants);

#endif /* MAP20_H_ */
