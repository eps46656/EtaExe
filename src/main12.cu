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
// #include "/media/D/EtaDevelop/Eta/Vector.cuh"
// #include "/media/D/EtaDevelop/Eta/Matrix.cuh"
// #include "/media/D/EtaDevelop/Eta/utils.cuh"

#include "/media/D/EtaDevelop/Eta/WBuffer.cuh"
// #include "/media/D/EtaDevelop/Eta/PointLight.cu"

#include "/media/D/EtaDevelop/Eta/utils.cuh"
#include "/media/D/EtaDevelop/Eta/VectorView.cuh"
#include "/media/D/EtaDevelop/Eta/Print.cuh"
#include "/media/D/EtaDevelop/Eta/io.cuh"
#include "/media/D/EtaDevelop/Eta/Image.cu"

using namespace eta;

#define MODEL_DIR "/media/D/teapot"

template<typename Iter, typename T>
void increase(Iter begin, Iter end, T init) {
    for (; begin != end; ++begin, ++init) { *begin = init; }
}

void normalize_vertex_coord(std::vector<num>& vertex_coord) {
    int num_of_vertices{ static_cast<int>(vertex_coord.size()) / 3 };

    num min_x{ ETA_inf };
    num max_x{ -ETA_inf };
    num min_y{ ETA_inf };
    num max_y{ -ETA_inf };
    num min_z{ ETA_inf };
    num max_z{ -ETA_inf };

    for (int i{ 0 }; i < num_of_vertices; ++i) {
        num x{ vertex_coord[3 * i + 0] };
        num y{ vertex_coord[3 * i + 1] };
        num z{ vertex_coord[3 * i + 2] };

        min_x = std::min(min_x, x);
        max_x = std::max(max_x, x);
        min_y = std::min(min_y, y);
        max_y = std::max(max_y, y);
        min_z = std::min(min_z, z);
        max_z = std::max(max_z, z);
    }

    num center_x{ (min_x + max_x) / 2 };
    num center_y{ (min_y + max_y) / 2 };
    num center_z{ (min_z + max_z) / 2 };

    num scale_x{ 2 / (max_x - min_x) };
    num scale_y{ 2 / (max_y - min_y) };
    num scale_z{ 2 / (max_z - min_z) };

    num scale{ std::min(scale_x, std::min(scale_y, scale_z)) };

    for (int i{ 0 }; i < num_of_vertices; ++i) {
        num& x{ vertex_coord[3 * i + 0] };
        num& y{ vertex_coord[3 * i + 1] };
        num& z{ vertex_coord[3 * i + 2] };

        x = (x - center_x) * scale;
        y = (y - center_y) * scale;
        z = (z - center_z) * scale;
    }
}

int main() {
    Matrix<num, 4, 4> transform{ Matrix<num, 4, 4>::eye() };

    std::vector<num> vertex_coord{ read_vector<num>(
        join_path(MODEL_DIR, "vertex_coord.txt")) };
    normalize_vertex_coord(vertex_coord);
    // transform = rotation_mat<num>({ 1, 0, 0 }, -90 * ETA_deg) * transform;
    // transform = rotation_mat<num>({ 0, 1, 0 }, 180 * ETA_deg) * transform;

    std::vector<num> vertex_normal{ read_vector<num>(
        join_path(MODEL_DIR, "vertex_normal.txt")) };

    std::vector<num> vertex_color{ read_vector<num>(
        join_path(MODEL_DIR, "vertex_color.txt")) };
    VectorView vertex_color_view;
    vertex_color_view.base = vertex_color.data();
    vertex_color_view.coeff[0] = sizeof(num) * 3 * 3;
    vertex_color_view.coeff[1] = sizeof(num) * 3;
    vertex_color_view.coeff[2] = sizeof(num);

    std::vector<num> vertex_texture_coord{ read_vector<num>(
        join_path(MODEL_DIR, "vertex_texture_coord.txt")) };

    int num_of_vertices{ static_cast<int>(vertex_coord.size()) / 3 };
    int num_of_faces{ num_of_vertices / 3 };

    int height{ 1024 };
    int width{ 1024 };
    int size{ height * width };

    VectorView face_coord_gpu;
    face_coord_gpu.base = gpu_malloc(sizeof(num) * 3 * num_of_vertices);
    face_coord_gpu.coeff[0] = sizeof(num) * 3 * 3;
    face_coord_gpu.coeff[1] = sizeof(num) * 3;
    face_coord_gpu.coeff[2] = sizeof(num);
    cudaMemcpyAsync(face_coord_gpu.base, vertex_coord.data(),
                    sizeof(num) * 3 * num_of_vertices, cudaMemcpyHostToDevice);

    std::vector<int> face_id;
    face_id.resize(num_of_faces);
    increase(face_id.begin(), face_id.end(), 0);

    ETA_CheckCudaError(cudaGetLastError());

    VectorView face_id_gpu;
    face_id_gpu.base = gpu_malloc(sizeof(int) * num_of_faces);
    face_id_gpu.coeff[0] = sizeof(int);
    cudaMemcpyAsync(face_id_gpu.base, face_id.data(),
                    sizeof(int) * num_of_faces, cudaMemcpyHostToDevice);

    ETA_CheckCudaError(cudaGetLastError());

    VectorView id_buffer_cpu;
    id_buffer_cpu.base = cpu_malloc(sizeof(int) * size);
    id_buffer_cpu.coeff[0] = sizeof(int) * width;
    id_buffer_cpu.coeff[1] = sizeof(int);

    VectorView id_buffer_gpu;
    id_buffer_gpu.base = gpu_malloc(sizeof(int) * size);
    id_buffer_gpu.coeff[0] = sizeof(int) * width;
    id_buffer_gpu.coeff[1] = sizeof(int);
    setvalue<int><<<1, 1024, 0, 0>>>(id_buffer_gpu.base, size, -1);

    VectorView w_buffer_gpu;
    w_buffer_gpu.base = gpu_malloc(sizeof(num) * size);
    w_buffer_gpu.coeff[0] = sizeof(num) * width;
    w_buffer_gpu.coeff[1] = sizeof(num);

    VectorView n_buffer_cpu;
    n_buffer_cpu.base = cpu_malloc(sizeof(num) * 2 * size);
    n_buffer_cpu.coeff[0] = sizeof(num) * 2 * width;
    n_buffer_cpu.coeff[1] = sizeof(num) * 2;
    n_buffer_cpu.coeff[2] = sizeof(num);

    VectorView n_buffer_gpu;
    n_buffer_gpu.base = gpu_malloc(sizeof(num) * 2 * size);
    n_buffer_gpu.coeff[0] = sizeof(num) * 2 * width;
    n_buffer_gpu.coeff[1] = sizeof(num) * 2;
    n_buffer_gpu.coeff[2] = sizeof(num);

    Matrix<num, 3, 4> p_mat{ perspective_mat<num>({ 0, 0, -2 }, // origin
                                                  { -1, 0, 0 }, // right
                                                  { 0, 1, 0 }, // up
                                                  { 0, 0, -1 } // front
                                                  ) };
    /* Matrix<num, 3, 4> p_mat{ perspective_mat<num>({ 0, -2, 0 }, // origin
                                                  { 1, 0, 0 }, // right
                                                  { 0, 0, 1 }, // up
                                                  { 0, 1, 0 } // front
                                                  ) }; */

    ETA_CheckCudaError(cudaGetLastError());

    num theata{ 0 };
    num phi{ 0 };

    ETA_CheckCudaError(cudaGetLastError());

    if (true) {
        WBuffering<<<1, 1024, 0, 0>>>(height, width, // height, width

                                      id_buffer_gpu, // dst_id
                                      w_buffer_gpu, // dst_w

                                      true, // record_n
                                      n_buffer_gpu, // dst_n

                                      num_of_faces, // num_of_faces
                                      face_id_gpu, // face_id
                                      face_coord_gpu, // face_coord
                                      p_mat * transform // transform
        );
    }

    ETA_CheckCudaError(cudaGetLastError());

    cudaMemcpyAsync(id_buffer_cpu.base, // dst
                    id_buffer_gpu.base, // src,
                    sizeof(int) * size, // size
                    cudaMemcpyDeviceToHost, // kind
                    0 // stream
    );

    cudaMemcpyAsync(n_buffer_cpu.base, // dst
                    n_buffer_gpu.base, // src,
                    sizeof(num) * 2 * size, // size
                    cudaMemcpyDeviceToHost, // kind
                    0 // stream
    );

    ETA_CheckCudaError(cudaGetLastError());

    cudaDeviceSynchronize();

    ETA_CheckCudaError(cudaGetLastError());

    Print("outputing\n");

    Image o_img;
    o_img.create(height, width, 4);

    for (int i{ 0 }; i < height; ++i) {
        for (int j{ 0 }; j < width; ++j) {
            int id{ id_buffer_cpu.get_ref<int>(i, j) };

            if (id == -1) {
                o_img.get(i, j, 0) = 0;
                o_img.get(i, j, 1) = 0;
                o_img.get(i, j, 2) = 0;
            } else {
                Vector<num, 3> a{
                    vertex_color_view.get_ref<num>(id, 0, 0),
                    vertex_color_view.get_ref<num>(id, 0, 1),
                    vertex_color_view.get_ref<num>(id, 0, 2),
                };

                Vector<num, 3> b{
                    vertex_color_view.get_ref<num>(id, 1, 0),
                    vertex_color_view.get_ref<num>(id, 1, 1),
                    vertex_color_view.get_ref<num>(id, 1, 2),
                };

                Vector<num, 3> c{
                    vertex_color_view.get_ref<num>(id, 2, 0),
                    vertex_color_view.get_ref<num>(id, 2, 1),
                    vertex_color_view.get_ref<num>(id, 2, 2),
                };

                num n1{ n_buffer_cpu.get_ref<num>(i, j, 0) };
                num n2{ n_buffer_cpu.get_ref<num>(i, j, 1) };

                Vector<num, 3> r{ a * (1 - n1 - n2) + b * n1 + c * n2 };

                o_img.get(i, j, 0) = r.data[0] * 255;
                o_img.get(i, j, 1) = r.data[1] * 255;
                o_img.get(i, j, 2) = r.data[2] * 255;
            }

            o_img.get(i, j, 3) = 255;
        }
    }

    o_img.save("/media/D/o_test.png");

    return 0;
}
