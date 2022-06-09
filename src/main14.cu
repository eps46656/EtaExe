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
#include "/media/D/EtaDevelop/Eta/Texture2D.cuh"
#include "/media/D/EtaDevelop/Eta/SimpleModel.cu"

using namespace eta;

#include "/media/D/EtaDevelop/EtaExe/src/MyModel.cuh"

int main() {
    int height{ 1024 };
    int width{ 1024 };
    int size{ height * width };

    MyModel teapot;
    teapot.LoadModel("/media/D/teapot", true);
    increase(teapot.face_id.begin(), teapot.face_id.end(), 0);
    teapot.LoadToGpu();

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

    ////////////////////////////////////////////////////////////////////////////

    Matrix<num, 3, 4> p_mat{ perspective_mat<num>({ 0, 0, -2 }, // origin
                                                  { -1, 0, 0 }, // right
                                                  { 0, 1, 0 }, // up
                                                  { 0, 0, -1 } // front
                                                  ) };

    WBuffering<<<1, 1024, 0, 0>>>(height, width, // height, width

                                  id_buffer_gpu, // dst_id
                                  w_buffer_gpu, // dst_w

                                  true, // record_n
                                  n_buffer_gpu, // dst_n

                                  teapot.num_of_faces, // num_of_faces
                                  teapot.face_id_gpu, // face_id
                                  teapot.face_coord_gpu, // face_coord
                                  p_mat * teapot.transform // transform
    );

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

    ////////////////////////////////////////////////////////////////////////////

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
                num n1{ n_buffer_cpu.get_ref<num>(i, j, 0) };
                num n2{ n_buffer_cpu.get_ref<num>(i, j, 1) };

                Vector<num, 2> tex_coord_0{
                    teapot.vertex_texture_coord_view.get_ref<num>(id, 0, 0),
                    teapot.vertex_texture_coord_view.get_ref<num>(id, 0, 1),
                };
                Vector<num, 2> tex_coord_1{
                    teapot.vertex_texture_coord_view.get_ref<num>(id, 1, 0),
                    teapot.vertex_texture_coord_view.get_ref<num>(id, 1, 1),
                };
                Vector<num, 2> tex_coord_2{
                    teapot.vertex_texture_coord_view.get_ref<num>(id, 2, 0),
                    teapot.vertex_texture_coord_view.get_ref<num>(id, 2, 1),
                };

                Vector<num, 2> tex_coord{ tex_coord_0 * (1 - n1 - n2) +
                                          tex_coord_1 * n1 + tex_coord_2 * n2 };

                Vector<num, 3> r{ teapot.texture.get(tex_coord.data[0],
                                                     tex_coord.data[1]) };

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
