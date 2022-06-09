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
#include "/media/D/EtaDevelop/Eta/Vector.cuh"
#include "/media/D/EtaDevelop/Eta/Matrix.cuh"
#include "/media/D/EtaDevelop/Eta/utils.cuh"
#include "/media/D/EtaDevelop/Eta/SimpleModel.cu"
#include "/media/D/EtaDevelop/Eta/SkyBox.cu"
#include "/media/D/EtaDevelop/Eta/PointLight.cu"

using namespace eta;

int main() {
    int height{ 1080 };
    int width{ 1920 };
    int size{ height * width };

    num fov{ 120 * ETA_deg };
    num aspect{ (width + 0.0f) / height };

    num radius{ 3 };
    num phi{ 120 * ETA_deg };
    num theta{ 90 * ETA_deg };

    Vector<num, 3> view_point{
        radius * cos(phi) * sin(theta), //
        radius * sin(phi) * sin(theta), //
        radius * cos(theta), //
    };
    Vector<num, 3> r;
    Vector<num, 3> u;
    Vector<num, 3> f;

    look_at_origin(r, u, f, view_point, fov, aspect);

    Matrix<num, 3, 4> camera_mat{ perspective_mat(view_point, // view_point
                                                  r, // right
                                                  u, // up
                                                  f // front
                                                  ) };

    cudaStream_t stream{ 0 };
    cudaDeviceSetLimit(cudaLimitStackSize, 128 * 1024);

    ETA_CheckLastCudaError;

    ////////////////////////////////////////////////////////////////////////////

    FaceIdManager face_id_manager;

    ////////////////////////////////////////////////////////////////////////////

    std::vector<Model*> models;

    ////////////////////////////////////////////////////////////////////////////

    SimpleModel teapot;
    models.push_back(&teapot);

    teapot.LoadFromDir("/media/D/teapot", // dir
                       true, // normalize face coord
                       AttrViewMode::VERTEX, // face_normal mode
                       AttrViewMode::TEXTURE // face_color mode
    );

    teapot.transform() =
        rotation_mat({ 1, 0, 0 }, 90 * ETA_deg) * teapot.transform();

    teapot.set_face_id(face_id_manager);

    teapot.face_color_cpu().tex().s_wrapping_mode =
        Texture2DWrappingMode::REPEAT;
    teapot.face_color_cpu().tex().s_filtering_mode =
        Texture2DFilteringMode::LINEAR;

    teapot.face_color_cpu().tex().t_wrapping_mode =
        Texture2DWrappingMode::REPEAT;
    teapot.face_color_cpu().tex().t_filtering_mode =
        Texture2DFilteringMode::LINEAR;

    teapot.PassToGPU(stream);

    ////////////////////////////////////////////////////////////////////////////

    SimpleModel kan;
    models.push_back(&kan);

    kan.LoadFromDir("/media/D/kan", // dir
                    true, // normalize face coord
                    AttrViewMode::VERTEX, // face_normal mode
                    AttrViewMode::VERTEX // face_color mode
    );

    kan.transform() = translation_mat({ 0, 2, 0 }) *
                      rotation_mat({ 0, 0, 1 }, 90 * ETA_deg) * kan.transform();

    kan.set_face_id(face_id_manager);

    kan.face_color_cpu().tex().s_wrapping_mode = Texture2DWrappingMode::REPEAT;
    kan.face_color_cpu().tex().s_filtering_mode =
        Texture2DFilteringMode::LINEAR;

    kan.face_color_cpu().tex().t_wrapping_mode = Texture2DWrappingMode::REPEAT;
    kan.face_color_cpu().tex().t_filtering_mode =
        Texture2DFilteringMode::LINEAR;

    kan.PassToGPU(stream);

    ////////////////////////////////////////////////////////////////////////////

    SkyBox sky_box;

    Cubemap cubemap;
    cubemap.load("/media/D/render/skybox.png");

    sky_box.Load(cubemap);

    sky_box.PassToGPU(stream);

    ////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////

    PointLight point_light;
    point_light.origin = { 0, 3, 3 };
    point_light.intensity = { 10, 10, 10 };

    point_light.InitCage(1000, stream);

    ////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////

    Rasterization rasterization{ CreateRasterization(height, width) };
    InitializeRasterization(rasterization, stream);

    TracingBatch tracing_batch_1{ CreateTracingBatch(height * width) };
    InitializeTracingBatch(tracing_batch_1, stream);

    TracingBatch tracing_batch_2{ CreateTracingBatch(height * width) };
    InitializeTracingBatch(tracing_batch_2, stream);

    ////////////////////////////////////////////////////////////////////////////

    ETA_CheckLastCudaError;

    View result_buffer_cpu;
    result_buffer_cpu.base = Malloc<CPU>(sizeof(Vector<int, 3>) * size);
    result_buffer_cpu.coeff[0] = sizeof(Vector<int, 3>) * width;
    result_buffer_cpu.coeff[1] = sizeof(Vector<int, 3>);

    ETA_CheckLastCudaError;

    View result_buffer_gpu;
    result_buffer_gpu.base = Malloc<GPU>(sizeof(Vector<int, 3>) * size);
    result_buffer_gpu.coeff[0] = sizeof(Vector<int, 3>);
    setvalue<num><<<1, 1024, 0, stream>>>(result_buffer_gpu.base, 3 * size, 0);

    ETA_CheckLastCudaError;

    ////////////////////////////////////////////////////////////////////////////

    cudaDeviceSynchronize();

    std::chrono::high_resolution_clock::time_point begin_time(
        std::chrono::high_resolution_clock::now());

    ETA_CheckLastCudaError;

    ////////////////////////////////////////////////////////////////////////////

    for (size_t i{ 0 }; i < models.size(); ++i) {
        point_light.GenerateCage(models[i], stream);
    }

    ETA_CheckLastCudaError;

    ////////////////////////////////////////////////////////////////////////////

    // rasterize

    for (size_t i{ 0 }; i < models.size(); ++i) {
        models[i]->Rasterize(rasterization, // dst

                             camera_mat, // camera_mat

                             stream // stream
        );
    }

    ETA_CheckLastCudaError;

    ////////////////////////////////////////////////////////////////////////////

    Buffer<CPU> face_id;
    face_id.Resize(sizeof(int) * size);

    Memcpy<CPU, GPU>(face_id.data(), rasterization.face_id, sizeof(int) * size);

    ////////////////////////////////////////////////////////////////////////////

    ETA_CheckLastCudaError;

    ////////////////////////////////////////////////////////////////////////////

    cudaDeviceSynchronize();

    ETA_CheckLastCudaError;

    std::chrono::high_resolution_clock::time_point end_time(
        std::chrono::high_resolution_clock::now());

    size_t duration(
        size_t(std::chrono::duration_cast<std::chrono::milliseconds>(end_time -
                                                                     begin_time)
                   .count()));

    Print("duration: ", duration, "\n");

    Image o_img;
    o_img.create(height, width);

    int white_count{ 0 };

    for (int i{ 0 }; i < height; ++i) {
        for (int j{ 0 }; j < width; ++j) {
            if (0 < static_cast<int*>(face_id.data())[width * i + j]) {
                o_img.get(i, j, 0) = 255;
                o_img.get(i, j, 1) = 255;
                o_img.get(i, j, 2) = 255;
                ++white_count;
            } else {
                o_img.get(i, j, 0) = 0;
                o_img.get(i, j, 1) = 0;
                o_img.get(i, j, 2) = 0;
            }

            o_img.get(i, j, 3) = 255;
        }
    }

    printf("white_count: %d\n", white_count);

    o_img.save("/media/D/o_test.png");

    ////////////////////////////////////////////////////////////////////////////

    return 0;
}
