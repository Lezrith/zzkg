cmake_minimum_required(VERSION 3.6)
project(zzkg-cuda CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --expt-extended-lambda")

find_package(CUDA QUIET REQUIRED)

cuda_add_executable(zzkg src/main.cpp src/cuda_main.cu)
