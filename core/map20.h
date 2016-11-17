#ifndef MAP20_H_
#define MAP20_H_

float* compute_map20(float *gpu_ranked, const char* relevance_file,
                     const int relevance_offset, const int rows,
                     const int variants);

#endif /* MAP20_H_ */
