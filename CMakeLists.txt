cmake_minimum_required(VERSION 3.6)
project(gpu_map20)

find_package(CUDA QUIET REQUIRED)
set(
        CUDA_NVCC_FLAGS
        ${CUDA_NVCC_FLAGS}
        -Xcompiler -DADD_ -gencode arch=compute_35,code=sm_35 -gencode arch=compute_35,code=compute_35
)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES
        core/ranks.cpp
        core/map20.cu
        kernels/map20.cu
        kernels/top_n.cu
        utils/utils.cpp
        utils/argparse.cpp
        main.cpp
        )
set (CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}
        --Wno-deprecated-gpu-targets)
cuda_add_executable(gpu_map20 ${SOURCE_FILES})
cuda_add_cublas_to_target(gpu_map20)
