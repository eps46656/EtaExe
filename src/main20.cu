#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <cstdlib>
#include <ctime>

#include "random.h"
#include "/media/D/EtaDevelop/Eta/utils.cuh"

struct D {
    int data[32];
};

__global__ void Init(int size, D* data) {
    for (int i{ static_cast<int>(GLOBAL_TID) }; i < size;
         i += gridDim.x * blockDim.x) {
        data[i].data[0] = i;
    }
}

__global__ void Arrange(int* dst_size, D* dst, int data_size, D* data) {
    for (int i{ static_cast<int>(GLOBAL_TID) }; i < data_size;
         i += gridDim.x * blockDim.x) {
        D tmp{ data[i] };

        if (tmp.data[0] % 1 != 0) { continue; }

        int idx{ atomicAdd(dst_size, 1) };
        dst[idx] = tmp;
    }
}

__global__ void Arrange2(int* dst_size_, D* dst, int data_size, D* data) {
    __shared__ int dst_size;

    if (GLOBAL_TID == 0) { dst_size = *dst_size_; }

    for (int i{ static_cast<int>(GLOBAL_TID) }; i < data_size;
         i += gridDim.x * blockDim.x) {
        int idx{ atomicAdd(&dst_size, 1) };
        dst[idx] = data[i];
    }

    __syncthreads();

    if (GLOBAL_TID == 0) { *dst_size_ = dst_size; }
}

int main() {
    int capacity{ 1024 * 1024 };

    int* stack_size;
    cudaMallocHost(&stack_size, sizeof(int));

    D* stack;
    cudaMalloc(&stack, sizeof(D) * capacity);

    int data_size{ 512 * 1024 };

    D* data;
    cudaMalloc(&data, sizeof(D) * data_size);

    Init<<<64, 1024, 0>>>(data_size, data);

    Arrange2<<<64, 1024, 0>>>(stack_size, stack, data_size, data);

    cudaDeviceSynchronize();

    //

    return 0;
}
