#include <iostream>
#include <fstream>
#include <chrono>
#include <string>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <cstdlib>
#include <ctime>

#include "/media/D/EtaDevelop/Eta/WBuffer.cuh"
#include "random.h"

using num_t = eta::num_t;
using eta::cpu_malloc;
using eta::cpu_free;
using eta::gpu_malloc;
using eta::gpu_free;

struct Face {
    num_t vertex_coord[3][3];
    num_t vertex_normal[3][3];
}; //

int main() {
    int num_of_faces{ 156 };
    int height{ 1920 };
    int width{ 1080 };

    int mem_size(num_of_faces * 3 * 3 * sizeof(num_t));

    eta::VectorView vertex_coord_cpu;
    vertex_coord_cpu.base = cpu_malloc<num_t>(mem_size);
    vertex_coord_cpu.coeff[0] = 9 * sizeof(num_t);
    vertex_coord_cpu.coeff[1] = 3 * sizeof(num_t);
    vertex_coord_cpu.coeff[2] = sizeof(num_t);

    eta::VectorView vertex_normal_cpu;
    vertex_normal_cpu.base = cpu_malloc<num_t>(mem_size);
    vertex_normal_cpu.coeff[0] = 9 * sizeof(num_t);
    vertex_normal_cpu.coeff[1] = 3 * sizeof(num_t);
    vertex_normal_cpu.coeff[2] = sizeof(num_t);

    eta::VectorView vertex_coord_gpu;
    vertex_coord_gpu.base = gpu_malloc<num_t>(mem_size);
    vertex_coord_gpu.coeff[0] = 9 * sizeof(num_t);
    vertex_coord_gpu.coeff[1] = 3 * sizeof(num_t);
    vertex_coord_gpu.coeff[2] = sizeof(num_t);

    eta::VectorView vertex_normal_gpu;
    vertex_normal_gpu.base = gpu_malloc<num_t>(mem_size);
    vertex_normal_gpu.coeff[0] = 9 * sizeof(num_t);
    vertex_normal_gpu.coeff[1] = 3 * sizeof(num_t);
    vertex_normal_gpu.coeff[2] = sizeof(num_t);

    /*
        load obj data
    */

    cudaMemcpy(vertex_coord_gpu.base, vertex_coord_cpu.base, mem_size,
               cudaMemcpyHostToDevice);
    cudaMemcpy(vertex_normal_gpu.base, vertex_normal_cpu.base, mem_size,
               cudaMemcpyHostToDevice);

    eta::WBuffer w_buffer_cpu;
    eta::CreateWBuffer(&w_buffer_cpu, height, width, ETA__cpu);

    eta::WBuffer w_buffer_gpu;
    eta::CreateWBuffer(&w_buffer_gpu, height, width, ETA__gpu);

    return 0;
}
