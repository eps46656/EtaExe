#include <iostream>
#include <fstream>
#include <chrono>
#include <string>
#include <vector>

// #include <cuda.h>
// #include <cuda_runtime.h>
// #include <cuda_runtime_api.h>
// #include <math_constants.h>

#include <algorithm>

#include "render/Renderer.cu"
#include "render/Triangle.cu"
#include "utility/RGBAImage.cu"
#include "utility/STLFile.cpp"
#include "render/PointLight.cu"
#include "math/Interp.cu"
#include "render/RenderingTriangle.cu"
#include "render/Model.cu"
#include "render/ModelManager.cu"
// #include "render/QuestBManager.cu"
// #include "render/QuestAManager.cu"
// #include "render/QuestBManager.cu"
// #include "render/TriangleSet.cu"
// #include "render/RayCasting.cu"
// #include "render/SolidTriangleTexture.cu"

// #include "TriangleBatch.cu"

// #include "Camera.cu"

int System(const char* command) { return system(command); }
int System(const std::string& command) { return system(command.c_str()); }

unsigned int SetRandom(unsigned int seed) {
    ::srand(seed);
    return seed;
}

unsigned int SetRandom() {
    time_t t;
    unsigned int seed((long int)time(&t));
    ::srand(seed);
    return seed;
}

int Random() { return ::rand(); }

int Random(int lower, int upper) { return Random() % (upper - lower) + lower; }

float RandomNum(float lower, float upper) {
    float b(4096);
    return Random(lower * b, upper * b) / b;
}

float RandomNormalNum() { return RandomNum(-256, 256); }

float sq(float x) { return x * x; }

std::string get_date() {
    char r[64] = { '\0' };
    std::time_t t = std::time(nullptr);
    std::strftime(r, sizeof(r), "%Y-%m-%d-%H%M", std::localtime(&t));
    return r;
}

void make_intensity_to_rgb(
    std::vector<eta::utility::RGBA<unsigned char>>& dst_rgba_vec,
    const std::vector<eta::utility::RGB<float>>& src_rgb_vec) {
    std::size_t pixel_size(src_rgb_vec.size());

    std::vector<float> intensity;

    for (std::size_t i(0); i < pixel_size; ++i) {
        float in((src_rgb_vec[i].r + src_rgb_vec[i].g + src_rgb_vec[i].b) / 3);

        if (in != 0) { intensity.push_back(in); }
    }

    if (intensity.empty()) {
        dst_rgba_vec.resize(src_rgb_vec.size());

        for (auto rgb_iter(dst_rgba_vec.begin()), rgb_end(dst_rgba_vec.end());
             rgb_iter != rgb_end; ++rgb_iter) {
            *rgb_iter = { 0, 0, 0, 255 };
        }

        return;
    }

    std::sort(intensity.begin(), intensity.end(), std::greater<float>());

    std::vector<float> acc_intensity(intensity.size());
    acc_intensity.back() = intensity.back();

    for (std::size_t i(1); i < intensity.size(); ++i) {
        acc_intensity[acc_intensity.size() - 1 - i] =
            intensity[intensity.size() - 1 - i] +
            acc_intensity[acc_intensity.size() - i];
    }

    float target_avg_intensity(196);
    float target_sum_intensity(target_avg_intensity * pixel_size);

    float mul(-1);

    for (std::size_t i(1); i < intensity.size(); ++i) {
        float min_mul(255 / intensity[i - 1]);
        float max_mul(255 / intensity[i]);

        float mul_((target_sum_intensity - 255 * i) / acc_intensity[i]);

        if (min_mul <= mul_ && mul_ <= max_mul) {
            mul = mul_;
            break;
        }
    }

    if (mul == -1) {
        dst_rgba_vec.resize(src_rgb_vec.size());

        for (auto rgb_iter(dst_rgba_vec.begin()), rgb_end(dst_rgba_vec.end());
             rgb_iter != rgb_end; ++rgb_iter) {
            *rgb_iter = { 255, 255, 255, 255 };
        }

        return;
    }

    dst_rgba_vec.resize(src_rgb_vec.size());

    for (std::size_t i(0); i < src_rgb_vec.size(); ++i) {
        const eta::utility::RGB<float>& rgb(src_rgb_vec[i]);

        dst_rgba_vec[i] = {
            static_cast<unsigned char>(std::min(255.0f, rgb.r * mul)),
            static_cast<unsigned char>(std::min(255.0f, rgb.g * mul)),
            static_cast<unsigned char>(std::min(255.0f, rgb.b * mul)), 255
        };
    }
}

void make_rgb_avg(std::vector<eta::utility::RGBA<unsigned char>>& dst_rgba_vec,
                  const std::vector<eta::utility::RGB<float>>& src_rgb_vec) {
    if (true) {
        float min_intensity(ETA__inf);
        float max_intensity(-ETA__inf);

        for (auto rgb_iter(src_rgb_vec.begin()), rgb_end(src_rgb_vec.end());
             rgb_iter != rgb_end; ++rgb_iter) {
            float intensity((rgb_iter->r + rgb_iter->g + rgb_iter->b) / 3);

            min_intensity = std::min(min_intensity, intensity);
            max_intensity = std::max(max_intensity, intensity);
        }

        float diff_intenisty(max_intensity - min_intensity);

        dst_rgba_vec.resize(src_rgb_vec.size());

        for (std::size_t i(0); i < src_rgb_vec.size(); ++i) {
            const eta::utility::RGB<float>& rgb(src_rgb_vec[i]);

            dst_rgba_vec[i] = {
                static_cast<unsigned char>((rgb.r - min_intensity) /
                                           diff_intenisty * 255),
                static_cast<unsigned char>((rgb.g - min_intensity) /
                                           diff_intenisty * 255),
                static_cast<unsigned char>((rgb.b - min_intensity) /
                                           diff_intenisty * 255),
                255
            };
        }

        eta::PrintFormat("max_intensity: ", max_intensity, "\n");
        eta::PrintFormat("min_intensity: ", min_intensity, "\n");
        eta::PrintFormat("255 / diff_intenisty: ", diff_intenisty * 255, "\n");

    } else {
#define ETA__MUL (100000)

        dst_rgba_vec.resize(src_rgb_vec.size());

        for (std::size_t i(0); i < src_rgb_vec.size(); ++i) {
            const eta::utility::RGB<float>& rgb(src_rgb_vec[i]);

            dst_rgba_vec[i] = { static_cast<unsigned char>(
                                    std::min(float(255), rgb.r * 1000000)),
                                static_cast<unsigned char>(
                                    std::min(float(255), rgb.g * 1000000)),
                                static_cast<unsigned char>(
                                    std::min(float(255), rgb.b * 1000000)),
                                255 };
        }
    }
}

/* eta::render::TriangleBatch* CreateTriangleBatch(int tri_num) {
    eta::render::TriangleBatch* tri_batch(new eta::render::TriangleBatch());
    eta::render::TriangleBatch::Create(tri_batch, tri_num);

    eta::render::SolidTexture* texture(
        new eta::render::SolidTexture({ 255, 255, 255, 127 }));

    tri_batch->size = tri_num;

    for (size_t i(0); i < tri_num; ++i) {
        tri_batch->tris[i].origin.set(RandomNormalNum(), RandomNormalNum(),
                                      RandomNormalNum());
        tri_batch->tris[i].axis[0].set(RandomNormalNum(), RandomNormalNum(),
                                       RandomNormalNum());
        tri_batch->tris[i].axis[1].set(RandomNormalNum(), RandomNormalNum(),
                                       RandomNormalNum());
        tri_batch->tris[i].rela_refractive_index.set(1, 1, 1);
        tri_batch->tris[i].decay_index_in.set(1, 1, 1);
        tri_batch->tris[i].decay_index_ex.set(1, 1, 1);
        tri_batch->tris[i].texture = texture;
    }

    return tri_batch;
} */

eta::render::Model*
CreateModelFromSTLFile(const eta::utility::STLFile& stl_file) {
    eta::render::Model* model{ new eta::render::Model{} };

    model->transmit_decay_index_in = { 0.1, 0.1, 0.1 };
    model->transmit_decay_index_ex = { 0.1, 0.1, 0.1 };

    model->triangle_vec.resize(stl_file.tris.size());

    for (int i{ 0 }; i < stl_file.tris.size(); ++i) {
        const eta::utility::STLFile::Triangle& stl_tri(stl_file.tris[i]);

        eta::render::Triangle* triangle{ new eta::render::Triangle{} };

        triangle->origin = { stl_tri.vertex1x, stl_tri.vertex1y,
                             stl_tri.vertex1z };

        triangle->axis[0] = eta::math::Vector<3>{
            stl_tri.vertex2x, stl_tri.vertex2y, stl_tri.vertex2z
        } - triangle->origin;

        triangle->axis[1] = eta::math::Vector<3>{
            stl_tri.vertex3x, stl_tri.vertex3y, stl_tri.vertex3z
        } - triangle->origin;

        triangle->Setup();

        model->triangle_vec[i] = triangle;
    }

    return model;
}

void test_render() {
    eta::render::Renderer renderer;

    unsigned int width(1920);
    unsigned int height(1080);

    std::vector<eta::utility::Ray> pixel_rays(
        eta::render::Renderer::GetPixelRays(
            width, // width
            height, // height
            { -15, -15, 15 }, // origin
            { 15, 15, -15 }, // axis_f
            ETA__deg_to_rad(140), // angle_of_view
            0 // angle_of_rotate
            ));
    renderer.set_pixel_rays(&pixel_rays);

    /*for (int i(0); i < 64; ++i) {
        printf(ETA__format_num ", " ETA__format_num ", " ETA__format_num "\n",
               pixel_rays[i].direct.x, pixel_rays[i].direct.y,
               pixel_rays[i].direct.z);
    }

    return;*/

    // eta::render::TriangleBatch* tri_batch(CreateTetrahedron());
    // eta::render::TriangleBatch* tri_batch(CreateCube());
    // eta::render::TriangleBatch* tri_batch(CreateTriangleBatch(128));

    eta::render::ModelManager* model_manager{ new eta::render::ModelManager{} };
    renderer.set_model_manager(model_manager);

    eta::utility::STLFile stl_file("/home/eps/desktop/Jump_Pack.stl");
    model_manager->model_vec.push_back(CreateModelFromSTLFile(stl_file));

    model_manager->Synchronize();

    cudaDeviceSynchronize();

    eta::render::PointLight* point_light_0((new eta::render::PointLight())
                                               ->set_origin({ -15, -15, 15 })
                                               ->set_intensity({ 1, 1, 1 })
                                               ->set_range(1024));

    /*eta::render::PointLight* point_light_1(
        (new eta::render::PointLight())
            ->set_origin({ 0, -15, 15 })
            ->set_intensity({ 10000, 10000, 10000 })
            ->set_range(1024)); */

    eta::render::PointLight* point_light_2((new eta::render::PointLight())
                                               ->set_origin({ 0, 0, 30 })
                                               ->set_intensity({ 1, 1, 1 })
                                               ->set_range(1024));

    std::vector<const eta::render::DivergentLight*> divergent_lights;
    divergent_lights.push_back(point_light_0);
    // divergent_lights.push_back(point_light_1);
    divergent_lights.push_back(point_light_2);
    renderer.set_divergent_lights(&divergent_lights);

    eta::Print("renderg begin\n");

    std::chrono::high_resolution_clock::time_point begin_time(
        std::chrono::high_resolution_clock::now());

    renderer.Render();

    std::chrono::high_resolution_clock::time_point end_time(
        std::chrono::high_resolution_clock::now());

    eta::Print("renderg end\n");

    eta::Print("duration : ",
               std::chrono::duration_cast<std::chrono::milliseconds>(end_time -
                                                                     begin_time)
                       .count() /
                   1000.0,
               " s\n");

    std::vector<eta::utility::RGB<float>> result(renderer.result());

    std::vector<eta::utility::RGBA<unsigned char>> color;
    make_rgb_avg(color, result);

    eta::utility::RGBAImage image;

    std::string render_dir("/home/eps/desktop/render/m");
    std::string date(get_date());
    std::string rgba_file(render_dir + "/" + "m-" + date + ".rgba");
    std::string img_file(render_dir + "/" + "m-" + date + ".png");

    std::string rgba_to_img_cmd(
        "python3 '/home/eps/desktop/test/python/rgba.py' --rgba-to-img " +
        rgba_file + " " + img_file);

    image.set(width, height, color);
    image.write(rgba_file);

    System(rgba_to_img_cmd);

    /* for (int i(0); i < result.size(); ++i) {
        printf("(%f,%f,%f)", result[i].r, result[i].g, result[i].b);
    } */

    // -8, 8, -8,
    // -8, 8, 16,
    // 25, 25, 0

    // 8, 8, -8,
    // 25, -25, 0
    // 8, 8, 16,
}

void test_cubic_interp() {
    unsigned int test_num(1024);

    for (unsigned int test_i(0); test_i < test_num; ++test_i) {
        float x0(RandomNormalNum());
        float span(RandomNum(1, 256));
        float a3(RandomNormalNum());
        float a2(RandomNormalNum());
        float a1(RandomNormalNum());
        float a0(RandomNormalNum());

        float y[4];

        for (std::size_t x_i(0); x_i < 4; ++x_i) {
            float x(x0 + x_i * span);
            y[x_i] = ((a3 * x + a2) * x + a1) * x + a0;
        }

        float x_target(RandomNum(x0, x0 + span * 4));
        float y_target_interp(
            eta::math::CubicInterp(x0, span, y[0], y[1], y[2], y[3], x_target));

        float y_target_real(((a3 * x_target + a2) * x_target + a1) * x_target +
                            a0);

        eta::PrintFormat(test_i, ": ", abs(y_target_interp - y_target_real),
                         "\n");
    }
}

void test_stl() {
    eta::utility::STLFile stl("/home/eps/desktop/Jump_Pack.stl");

    // size = 170610

    float min_x(ETA__inf);
    float max_x(-ETA__inf);
    float min_y(ETA__inf);
    float max_y(-ETA__inf);
    float min_z(ETA__inf);
    float max_z(-ETA__inf);

    for (std::size_t i(0); i < stl.tris.size(); ++i) {
        eta::utility::STLFile::Triangle& tri(stl.tris[i]);

        min_x = std::min(min_x, static_cast<float>(tri.vertex1x));
        min_x = std::min(min_x, static_cast<float>(tri.vertex2x));
        min_x = std::min(min_x, static_cast<float>(tri.vertex3x));

        max_x = std::max(max_x, static_cast<float>(tri.vertex1x));
        max_x = std::max(max_x, static_cast<float>(tri.vertex2x));
        max_x = std::max(max_x, static_cast<float>(tri.vertex3x));

        min_y = std::min(min_y, static_cast<float>(tri.vertex1y));
        min_y = std::min(min_y, static_cast<float>(tri.vertex2y));
        min_y = std::min(min_y, static_cast<float>(tri.vertex3y));

        max_y = std::max(max_y, static_cast<float>(tri.vertex1y));
        max_y = std::max(max_y, static_cast<float>(tri.vertex2y));
        max_y = std::max(max_y, static_cast<float>(tri.vertex3y));

        min_z = std::min(min_z, static_cast<float>(tri.vertex1z));
        min_z = std::min(min_z, static_cast<float>(tri.vertex2z));
        min_z = std::min(min_z, static_cast<float>(tri.vertex3z));

        max_z = std::max(max_z, static_cast<float>(tri.vertex1z));
        max_z = std::max(max_z, static_cast<float>(tri.vertex2z));
        max_z = std::max(max_z, static_cast<float>(tri.vertex3z));
    }

    eta::PrintFormat("x: (", min_x, ", ", max_x, ")\n");
    eta::PrintFormat("y: (", min_y, ", ", max_y, ")\n");
    eta::PrintFormat("z: (", min_z, ", ", max_z, ")\n");
}

void test_date() {
    std::string ss("abc");

    // eta::Print(ss + "def\n");
}

int main() {
    eta::PrintFormat("seed: ", SetRandom(), '\n');
    // test_quest_a_manager();
    // test_channel();

    // test_quest_a_manager();

    test_render();

    // test_cubic_interp();

    // test_stl();

    cudaDeviceSynchronize();

    return 0;
}