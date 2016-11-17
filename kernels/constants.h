//
// Created by tumbler on 17.11.16.
//

#ifndef GPU_MAP20_CONSTANTS_H
#define GPU_MAP20_CONSTANTS_H

const int N = 20; // для вычисления MAP@Т
const int MAX_MATCHES = 1000; // максимальное число документов в выдаче


#ifdef CLION_IDE

#define __global__
#include <math_functions.h>

#endif
#endif //GPU_MAP20_CONSTANTS_H
