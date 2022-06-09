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
#include "/media/D/EtaDevelop/Eta/Renderer.cu"
#include "/media/D/EtaDevelop/Eta/Shader.cu"

using namespace eta;

int main() {
    int height{ 1080 };
    int width{ 1920 };

    float fov{ 90 * ETA_deg };
    float aspect{ (width + 0.0f) / height };

    float radius{ 5 };
    float phi{ 135 * ETA_deg };
    float theta{ 80 * ETA_deg };

    Vector<float, 3> view_point{
        radius * cos(phi) * sin(theta), //
        radius * sin(phi) * sin(theta), //
        radius * cos(theta), //
    };
    Vector<float, 3> r;
    Vector<float, 3> u;
    Vector<float, 3> f;

    look_at_origin(r, u, f, view_point, fov, aspect);

    cudaDeviceSetLimit(cudaLimitMallocHeapSize, 2ll * 1024 * 1024 * 1024);

    ETA_CheckLastCudaError;

    cudaStream_t stream{ 0 };

    ////////////////////////////////////////////////////////////////////////////

    Renderer renderer;

    renderer.height = height;
    renderer.width = width;

    renderer.view_point = view_point;
    renderer.r = r;
    renderer.u = u;
    renderer.f = f;

    renderer.max_depth = 4;

    renderer.Reserve();

    ////////////////////////////////////////////////////////////////////////////

    SimpleModel teapot;
    renderer.models.push_back(&teapot);

    teapot.LoadFromDir("/media/D/teapot", // dir
                       true, // normalize face coord
                       AttrViewMode::VERTEX, // face_normal mode
                       AttrViewMode::VERTEX // face_color mode
    );

    teapot.transform() = translation_mat({ 0, 0, 0 }) *
                         rotation_mat({ 0, 0, 1 }, -90 * ETA_deg) *
                         rotation_mat({ 1, 0, 0 }, 90 * ETA_deg) *
                         teapot.transform();

    teapot.face_color_cpu().tex().s_wrapping_mode =
        Texture2DWrappingMode::REPEAT;
    teapot.face_color_cpu().tex().s_filtering_mode =
        Texture2DFilteringMode::LINEAR;

    teapot.face_color_cpu().tex().t_wrapping_mode =
        Texture2DWrappingMode::REPEAT;
    teapot.face_color_cpu().tex().t_filtering_mode =
        Texture2DFilteringMode::LINEAR;

    teapot.LoadOn(stream);

    ////////////////////////////////////////////////////////////////////////////

    SimpleModel kan;
    renderer.models.push_back(&kan);

    kan.LoadFromDir("/media/D/kan", // dir
                    true, // normalize face coord
                    AttrViewMode::VERTEX, // face_normal mode
                    AttrViewMode::VERTEX // face_color mode
    );

    kan.transform() = translation_mat({ 0, 2, 0 }) *
                      rotation_mat({ 0, 0, 1 }, 90 * ETA_deg) * kan.transform();

    kan.face_color_cpu().tex().s_wrapping_mode = Texture2DWrappingMode::REPEAT;
    kan.face_color_cpu().tex().s_filtering_mode =
        Texture2DFilteringMode::LINEAR;

    kan.face_color_cpu().tex().t_wrapping_mode = Texture2DWrappingMode::REPEAT;
    kan.face_color_cpu().tex().t_filtering_mode =
        Texture2DFilteringMode::LINEAR;

    kan.LoadOn(stream);

    ////////////////////////////////////////////////////////////////////////////

    SimpleModel floor;
    renderer.models.push_back(&floor);

    floor.LoadFromDir("/media/D/floor", // dir
                      false, // normalize face coord
                      AttrViewMode::VERTEX, // face_normal mode
                      AttrViewMode::VERTEX // face_color mode
    );

    floor.transform() = translation_mat({ 0, 0, -1 }) *
                        scale_mat({ 200, 200, 200 }) * floor.transform();

    floor.face_color_cpu().tex().s_wrapping_mode =
        Texture2DWrappingMode::REPEAT;
    floor.face_color_cpu().tex().s_filtering_mode =
        Texture2DFilteringMode::LINEAR;

    floor.face_color_cpu().tex().t_wrapping_mode =
        Texture2DWrappingMode::REPEAT;
    floor.face_color_cpu().tex().t_filtering_mode =
        Texture2DFilteringMode::LINEAR;

    floor.LoadOn(stream);

    ////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////

    std::vector<Light*> lights;

    ////////////////////////////////////////////////////////////////////////////

    SkyBox sky_box;
    renderer.lights.push_back(&sky_box);

    Cubemap cubemap;
    cubemap.load("/media/D/render/skybox.png");

    sky_box.Load(cubemap);

    sky_box.LoadOn(stream);

    ////////////////////////////////////////////////////////////////////////////

    PointLight point_light_a;
    renderer.lights.push_back(&point_light_a);

    point_light_a.origin = { 3, 0, 0 };
    point_light_a.intensity = { 5.0f, 5.0f, 5.0f };

    point_light_a.InitCage(500, stream);

    for (size_t model_i{ 0 }; model_i < renderer.models.size(); ++model_i) {
        point_light_a.GenerateCage(renderer.models[model_i], stream);
    }

    point_light_a.LoadOn(stream);

    ////////////////////////////////////////////////////////////////////////////

    PointLight point_light_b;
    renderer.lights.push_back(&point_light_b);

    point_light_b.origin = { 3, 0, 0 };
    point_light_b.intensity = { 5.0f, 5.0f, 5.0f };

    point_light_b.InitCage(500, stream);

    for (size_t model_i{ 0 }; model_i < renderer.models.size(); ++model_i) {
        point_light_b.GenerateCage(renderer.models[model_i], stream);
    }

    point_light_b.LoadOn(stream);

    ////////////////////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////

    std::vector<Vector<int, 3>> result;

    ////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////

#define TEST_NUM (1)

    {
        cudaDeviceSynchronize();

        cudaEvent_t begin_event;
        cudaEventCreate(&begin_event);
        cudaEventRecord(begin_event, stream);

        ////////////////////////////////////////////////////////////////////////

        for (int i{ 0 }; i < TEST_NUM; ++i) {
            result = renderer.Render(stream);
        }

        ////////////////////////////////////////////////////////////////////////

        cudaEvent_t end_event;
        cudaEventCreate(&end_event);
        cudaEventRecord(end_event, stream);

        cudaDeviceSynchronize();

        float duration;

        cudaEventElapsedTime(&duration, begin_event, end_event);

        Print("duration: ", duration, "\n");
        Print("average duration: ", duration / TEST_NUM, "\n");
    }

    ////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////

    Image o_img;
    o_img.create(height, width);

    for (int i{ 0 }; i < height; ++i) {
        for (int j{ 0 }; j < width; ++j) {
            for (int c{ 0 }; c < 3; ++c) {
                int value{ result[width * i + j].data[c] };

                o_img.get(i, j, c) = static_cast<unsigned char>(clamp<int>(
                    static_cast<int>(::round(255 * value / ETA_MUL)), 0, 255));
            }

            o_img.get(i, j, 3) = 255;
        }
    }

    o_img.save("/media/D/gallery_3.png");

    ////////////////////////////////////////////////////////////////////////////

    return 0;
}
