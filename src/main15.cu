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

#include "/media/D/EtaDevelop/Eta/WBuffer.cuh"
// #include "/media/D/EtaDevelop/Eta/PointLight.cu"

#include "/media/D/EtaDevelop/Eta/utils.cuh"
#include "/media/D/EtaDevelop/Eta/View.cuh"
#include "/media/D/EtaDevelop/Eta/Print.cuh"
#include "/media/D/EtaDevelop/Eta/io.cuh"
#include "/media/D/EtaDevelop/Eta/Image.cu"
#include "/media/D/EtaDevelop/Eta/Texture2D.cuh"
#include "/media/D/EtaDevelop/Eta/SimpleModel.cu"
#include "/media/D/EtaDevelop/Eta/Cubemap.cu"
#include "/media/D/EtaDevelop/Eta/SkyBox.cu"
#include "/media/D/EtaDevelop/Eta/PointLight.cu"

#include <chrono>

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

    Buffer<GPU> normal_buffer;
    View normal_buffer_gpu;
    normal_buffer_gpu.base =
        normal_buffer.Resize(sizeof(Vector<num, 3>) * size);
    normal_buffer_gpu.coeff[0] = sizeof(Vector<num, 3>);

    Buffer<GPU> diffuse_buffer;
    View diffuse_buffer_gpu;
    diffuse_buffer_gpu.base =
        diffuse_buffer.Resize(sizeof(Vector<num, 3>) * size);
    diffuse_buffer_gpu.coeff[0] = sizeof(Vector<num, 3>);

    Buffer<GPU> specular_buffer;
    View specular_buffer_gpu;
    specular_buffer_gpu.base =
        specular_buffer.Resize(sizeof(Vector<num, 3>) * size);
    specular_buffer_gpu.coeff[0] = sizeof(Vector<num, 3>);

    Buffer<GPU> shininess_buffer;
    View shininess_buffer_gpu;
    shininess_buffer_gpu.base =
        shininess_buffer.Resize(sizeof(Vector<num, 3>) * size);
    shininess_buffer_gpu.coeff[0] = sizeof(Vector<num, 3>);

    Buffer<GPU> transparent_buffer;
    View transparent_buffer_gpu;
    transparent_buffer_gpu.base =
        transparent_buffer.Resize(sizeof(Vector<num, 3>) * size);
    transparent_buffer_gpu.coeff[0] = sizeof(Vector<num, 3>);

    View face_id_buffer_gpu;
    face_id_buffer_gpu.base = Malloc<GPU>(sizeof(int) * size);
    face_id_buffer_gpu.coeff[0] = sizeof(int) * width;
    face_id_buffer_gpu.coeff[1] = sizeof(int);
    setvalue<int><<<1, 1024, 0, stream>>>(face_id_buffer_gpu.base, size, -1);

    View w_buffer_gpu;
    w_buffer_gpu.base = Malloc<GPU>(sizeof(int) * size);
    w_buffer_gpu.coeff[0] = sizeof(int) * width;
    w_buffer_gpu.coeff[1] = sizeof(int);
    setvalue<int><<<1, 1024, 0, stream>>>(w_buffer_gpu.base, size, 0);

    View n_buffer_gpu;
    n_buffer_gpu.base = Malloc<GPU>(sizeof(Vector<num, 2>) * size);
    n_buffer_gpu.coeff[0] = sizeof(Vector<num, 2>) * width;
    n_buffer_gpu.coeff[1] = sizeof(Vector<num, 2>);

    ////////////////////////////////////////////////////////////////////////////

    View result_buffer_cpu;
    result_buffer_cpu.base = Malloc<CPU>(sizeof(Vector<num, 3>) * size);
    result_buffer_cpu.coeff[0] = sizeof(Vector<num, 3>) * width;
    result_buffer_cpu.coeff[1] = sizeof(Vector<num, 3>);

    View result_buffer_gpu;
    result_buffer_gpu.base = Malloc<GPU>(sizeof(Vector<num, 3>) * size);
    result_buffer_gpu.coeff[0] = sizeof(Vector<num, 3>);
    setvalue<num><<<1, 1024, 0, stream>>>(result_buffer_gpu.base, 3 * size, 0);

    ////////////////////////////////////////////////////////////////////////////

    cudaDeviceSynchronize();

    std::chrono::high_resolution_clock::time_point begin_time(
        std::chrono::high_resolution_clock::now());

    ////////////////////////////////////////////////////////////////////////////

    for (size_t i{ 0 }; i < models.size(); ++i) {
        point_light.GenerateCage(models[i], stream);
    }

    ////////////////////////////////////////////////////////////////////////////

    // rasterize

    for (size_t i{ 0 }; i < models.size(); ++i) {
        models[i]->Rasterize(height, width, // height, width

                             face_id_buffer_gpu, // dst_id
                             w_buffer_gpu, // dst_w

                             true, // record_n
                             n_buffer_gpu, // dst_n

                             camera_mat, // camera_mat

                             stream // stream
        );
    }

    ////////////////////////////////////////////////////////////////////////////

    // flatten buffers

    face_id_buffer_gpu.coeff[0] = sizeof(int);
    w_buffer_gpu.coeff[0] = sizeof(num);
    n_buffer_gpu.coeff[0] = sizeof(Vector<num, 2>);

    ////////////////////////////////////////////////////////////////////////////

    // shade

    for (size_t i{ 0 }; i < models.size(); ++i) {
        models[i]->Shade(
            height, width, //

            normal_buffer_gpu, // Vector<num, 3> [height*width] gpu
            diffuse_buffer_gpu, // Vector<num, 3> [height*width] gpu
            specular_buffer_gpu, // Vector<num, 3> [height*width] gpu
            shininess_buffer_gpu, // Vector<num, 3> [height*width] gpu
            transparent_buffer_gpu, // Vector<num, 3> [height*width] gpu

            view_point, // view_point
            r, // right
            u, // up
            f, // front

            face_id_buffer_gpu, // face_id
            w_buffer_gpu, // w
            n_buffer_gpu, // n

            stream // stream
        );
    }

    ////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////

    point_light.Shade(
        height, width,

        result_buffer_gpu, // dst

        view_point, // view point
        r, // right
        u, // up
        f, // front

        normal_buffer_gpu, // Vector<num, 3> [height*width] gpu
        diffuse_buffer_gpu, // Vector<num, 3> [height*width] gpu
        specular_buffer_gpu, // Vector<num, 3> [height*width] gpu
        shininess_buffer_gpu, // Vector<num, 3> [height*width] gpu
        transparent_buffer_gpu, // Vector<num, 3> [height*width] gpu

        face_id_buffer_gpu, // face_id
        w_buffer_gpu, // w
        n_buffer_gpu, // n

        stream // stream
    );

    ////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////

    sky_box.Shade(height, width, //

                  result_buffer_gpu, // dst

                  view_point, // view_point
                  r, // right
                  u, // up
                  f, // front

                  face_id_buffer_gpu, // face_id
                  w_buffer_gpu, // w
                  n_buffer_gpu, // n

                  stream // stream
    );

    ////////////////////////////////////////////////////////////////////////////

    // copy result from gpu

    MemcpyAsync<CPU, GPU>(result_buffer_cpu.base, // dst
                          result_buffer_gpu.base, // src
                          sizeof(Vector<num, 3>) * size, // size
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
                o_img.get(i, j, c) = static_cast<unsigned char>(clamp<int>(
                    static_cast<int>(::round(
                        255 * result_buffer_cpu.get_ref<Vector<num, 3>>(i, j)
                                  .data[c])),
                    0, 255));
            }

            o_img.get(i, j, 3) = 255;
        }
    }

    o_img.save("/media/D/o_test.png");

    ////////////////////////////////////////////////////////////////////////////

    return 0;
}
