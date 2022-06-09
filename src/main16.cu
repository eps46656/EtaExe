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

void Render(int height, int width, //

            Vector<num, 3> view_point, // view_point
            Vector<num, 3> r, // right
            Vector<num, 3> u, // up
            Vector<num, 3> f, // front

            const std::vector<Model*> models, //
            const std::vector<Light*> lights, //

            cudaStream_t stream) {
    cudaDeviceSetLimit(cudaLimitStackSize, 128 * 1024);

    Matrix<num, 3, 4> camera_mat{ perspective_mat(view_point, // view_point
                                                  r, // right
                                                  u, // up
                                                  f // front
                                                  ) };
}

int main() {
    int height{ 1080 };
    int width{ 1920 };
    int size{ height * width };

    num fov{ 120 * ETA_deg };
    num aspect{ (width + 0.0f) / height };

    num radius{ 3 };
    num phi{ 90 * ETA_deg };
    num theta{ 70 * ETA_deg };

    Vector<num, 3> view_point{
        radius * cos(phi) * sin(theta), //
        radius * sin(phi) * sin(theta), //
        radius * cos(theta), //
    };
    Vector<num, 3> r;
    Vector<num, 3> u;
    Vector<num, 3> f;

    look_at_origin(r, u, f, view_point, fov, aspect);

    cudaDeviceSetLimit(cudaLimitStackSize, 128 * 1024);

    Matrix<num, 3, 4> camera_mat{ perspective_mat(view_point, // view_point
                                                  r, // right
                                                  u, // up
                                                  f // front
                                                  ) };

    cudaStream_t stream{ 0 };

    int max_depth{ 1 };

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
                       AttrViewMode::VERTEX // face_color mode
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
    ////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////

    std::vector<Light*> lights;

    ////////////////////////////////////////////////////////////////////////////

    SkyBox sky_box;
    lights.push_back(&sky_box);

    Cubemap cubemap;
    cubemap.load("/media/D/render/skybox.png");

    sky_box.Load(cubemap);

    sky_box.PassToGPU(stream);

    ////////////////////////////////////////////////////////////////////////////

    PointLight point_light;
    lights.push_back(&point_light);

    point_light.origin = { 0, 3, 3 };
    point_light.intensity = { 10.0f, 10.0f, 10.0f };

    point_light.InitCage(1000, stream);

    ////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////

    std::vector<TracingBatch> tracing_batches;
    tracing_batches.resize((1 << (max_depth + 1)) - 1);

    ::printf("tracing_batches.size(): %d\n",
             static_cast<int>(tracing_batches.size()));

    for (size_t i{ 0 }; i < tracing_batches.size(); ++i) {
        tracing_batches[i] = CreateTracingBatch(height * width);
        InitializeTracingBatch(height * width, tracing_batches[i], stream);
    }

    ////////////////////////////////////////////////////////////////////////////

    Vector<int, 3>* result_cpu{ Malloc<CPU, Vector<int, 3>>(
        sizeof(Vector<int, 3>) * size) };
    Vector<int, 3>* result_gpu{ Malloc<GPU, Vector<int, 3>>(
        sizeof(Vector<int, 3>) * size) };
    setvalue<Vector<int, 3>>
        <<<1, 1024, 0, stream>>>(result_gpu, size, { 0, 0, 0 });

    ////////////////////////////////////////////////////////////////////////////

    cudaDeviceSynchronize();

    std::chrono::high_resolution_clock::time_point begin_time(
        std::chrono::high_resolution_clock::now());

    ////////////////////////////////////////////////////////////////////////////

    /* for (size_t i{ 0 }; i < models.size(); ++i) {
        point_light.GenerateCage(models[i], stream);
    } */

    ////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////

    for (size_t model_i{ 0 }; model_i < models.size(); ++model_i) {
        models[model_i]->Rasterize(
            height, width, //

            tracing_batches.front().face_id, // dst_face_id
            tracing_batches.front().w, // dst_w
            true, // record_n
            tracing_batches.front().n, // dst_n

            camera_mat, // camera_mat

            stream // stream
        );
    }

    RasterizationToTracingBatch(height, width, //

                                tracing_batches.front(), // dst

                                view_point, // view_point
                                r, // right
                                u, // up
                                f, // front

                                stream);

    for (size_t model_i{ 0 }; model_i < models.size(); ++model_i) {
        models[model_i]->Shade(size, tracing_batches.front(), stream);
    }

    for (size_t i{ 0 }; i < tracing_batches.size() / 2; ++i) {
        TracingBatchBraching<<<1, 1024, 0, stream>>>(
            size, tracing_batches[i * 2 + 1], tracing_batches[i * 2 + 2],
            tracing_batches[i]);

        for (size_t model_i{ 0 }; model_i < models.size(); ++model_i) {
            models[model_i]->RayCast(
                size, //

                tracing_batches[i * 2 + 1].face_id, //
                tracing_batches[i * 2 + 1].w, //
                true, //
                tracing_batches[i * 2 + 1].n, //

                tracing_batches[i * 2 + 1].previous_acc_path_length,
                tracing_batches[i * 2 + 1].ray_origin,
                tracing_batches[i * 2 + 1].ray_direct,

                stream);
        }

        for (size_t model_i{ 0 }; model_i < models.size(); ++model_i) {
            models[model_i]->Shade(size, tracing_batches[i * 2 + 1], stream);
        }

        for (size_t model_i{ 0 }; model_i < models.size(); ++model_i) {
            models[model_i]->RayCast(
                size, //

                tracing_batches[i * 2 + 2].face_id,
                tracing_batches[i * 2 + 2].w,
                true, //
                tracing_batches[i * 2 + 2].n,

                tracing_batches[i * 2 + 2].previous_acc_path_length,
                tracing_batches[i * 2 + 2].ray_origin,
                tracing_batches[i * 2 + 2].ray_direct,

                stream);
        }

        for (size_t model_i{ 0 }; model_i < models.size(); ++model_i) {
            models[model_i]->Shade(size, tracing_batches[i * 2 + 2], stream);
        }
    }

    for (size_t i{ 0 }; i < tracing_batches.size(); ++i) {
        for (size_t light_i{ 0 }; light_i < lights.size(); ++light_i) {
            lights[light_i]->Shade(size, result_gpu, tracing_batches[i],
                                   stream);
        }
    }

    ////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////

    // copy result from gpu

    MemcpyAsync<CPU, GPU>(result_cpu, // dst
                          result_gpu, // src
                          sizeof(Vector<int, 3>) * size, // size
                          stream // stream
    );

    ////////////////////////////////////////////////////////////////////////////

    cudaDeviceSynchronize();

    std::chrono::high_resolution_clock::time_point end_time(
        std::chrono::high_resolution_clock::now());

    size_t duration(
        size_t(std::chrono::duration_cast<std::chrono::milliseconds>(end_time -
                                                                     begin_time)
                   .count()));

    Print("duration: ", duration, "\n");

    Image o_img;
    o_img.create(height, width);

    for (int i{ 0 }; i < height; ++i) {
        for (int j{ 0 }; j < width; ++j) {
            for (int c{ 0 }; c < 3; ++c) {
                int value{ result_cpu[width * i + j].data[c] };

                o_img.get(i, j, c) = static_cast<unsigned char>(clamp<int>(
                    static_cast<int>(::round(255 * value / ETA_MUL)), 0, 255));
            }

            o_img.get(i, j, 3) = 255;
        }
    }

    o_img.save("/media/D/o_test.png");

    ////////////////////////////////////////////////////////////////////////////

    return 0;
}
