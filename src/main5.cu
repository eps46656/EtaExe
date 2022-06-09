#include <iostream>
#include <fstream>
#include <chrono>
#include <string>
#include <vector>

// #include <cuda.h>
// #include <cuda_runtime.h>
// #include <cuda_runtime_api.h>
// #include <math_constants.h>

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

double RandomNum(double lower, double upper) {
    double b(4096);
    return Random(lower * b, upper * b) / b;
}

double RandomNormalNum() { return RandomNum(-256, 256); }

float sq(float x) { return x * x; }

std::string get_date() {
    char r[64] = { '\0' };
    std::time_t t = std::time(nullptr);
    std::strftime(r, sizeof(r), "%Y-%m-%d-%H%M", std::localtime(&t));
    return r;
}

#include "render/Renderer.cu"
#include "render/Rasterizer.cu"
#include "render/Triangle.cu"

#include "utility/RGBAImage.cu"
#include "utility/STLFile.cpp"

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

void make_color(int width, int height,
                std::vector<eta::utility::RGBA<unsigned char>>& dst_rgba_vec,
                const eta::render::Rasterizer::Pixel* map) {
    int size{ width * height };

    dst_rgba_vec.resize(size);

    for (int i{ 0 }; i < size; ++i) {
        if (map[i].ray_s < ETA__inf) {
            dst_rgba_vec[i] = { 255, 255, 255, 255 };
        } else {
            dst_rgba_vec[i] = { 0, 0, 0, 255 };
        }
    }
}

int main() {
    eta::PrintFormat("seed: ", SetRandom(), '\n');

    eta::render::Rasterizer rasterizer;

    unsigned int width(1920);
    unsigned int height(1080);

    eta::math::Vector<3> origin{ -15, -15, 15 };
    eta::math::Vector<3> axis_f{ 15, 15, -15 };
    eta::math::Vector<3> axis_w;
    eta::math::Vector<3> axis_h;

    eta::render::Renderer::GetAxisWH(axis_w, // dst_axis_w
                                     axis_h, //dst_axis_h
                                     width, // width
                                     height, // height
                                     axis_f, // axis_f
                                     ETA__deg_to_rad(140), // angle_of_view
                                     0 // angle_of_rotate
    );

    rasterizer.set_width(width);
    rasterizer.set_height(height);
    rasterizer.set_origin(origin);
    rasterizer.set_axis_f(axis_f);
    rasterizer.set_axis_w(axis_w);
    rasterizer.set_axis_h(axis_h);

    eta::render::ModelManager* model_manager{ new eta::render::ModelManager{} };
    rasterizer.set_model_manager(model_manager);

    eta::utility::STLFile stl_file("/home/eps/desktop/Jump_Pack.stl");
    model_manager->model_vec.push_back(CreateModelFromSTLFile(stl_file));

    model_manager->Synchronize();

    rasterizer.Rasterize();

    cudaDeviceSynchronize();

    const eta::render::Rasterizer::Pixel* map{ rasterizer.map() };

    std::vector<eta::utility::RGBA<unsigned char>> color;
    make_color(width, height, color, map);

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

    return 0;
}