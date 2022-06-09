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

#include "render/Camera.cu"
#include "render/Triangle.cu"
#include "utility/RGBAImage.cu"
#include "utility/STLFile.cpp"
#include "render/PointLight.cu"
#include "math/Interp.cu"
#include "render/Model.cu"
#include "render/ModelManager.cu"
#include "math/Quaternion.cu"
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
    std::strftime(r, sizeof(r), "%Y-%m%d-%H%M%S", std::localtime(&t));
    return r;
}

void make_rgb_avg(std::vector<eta::utility::RGBA<unsigned char>>& dst_rgba_vec,
                  const std::vector<eta::utility::RGB<float>>& src_rgb_vec) {
    if (false) {
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
        eta::PrintFormat("255 / diff_intenisty: ", 255 / diff_intenisty, "\n");

    } else {
#define ETA__MUL (2.25e7f)

        dst_rgba_vec.resize(src_rgb_vec.size());

        for (std::size_t i(0); i < src_rgb_vec.size(); ++i) {
            const eta::utility::RGB<float>& rgb(src_rgb_vec[i]);

            if (rgb.r < 0 || rgb.g < 0 || rgb.b < 0) { ETA__print_pos; }

            dst_rgba_vec[i] = { static_cast<unsigned char>(
                                    std::min(float(255), rgb.r* ETA__MUL)),
                                static_cast<unsigned char>(
                                    std::min(float(255), rgb.g* ETA__MUL)),
                                static_cast<unsigned char>(
                                    std::min(float(255), rgb.b* ETA__MUL)),
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

#define SCALE (1)
#define CENTER (eta::math::Vector<3>{ 0, 0, 0 })

eta::render::Model*
CreateModelFromSTLFile(const eta::utility::STLFile& stl_file) {
    int size{ static_cast<int>(stl_file.tris.size()) };

    eta::render::Model* model{ new eta::render::Model{} };

    model->transmit_decay_index_in = { 0.1, 0.1, 0.1 };
    model->transmit_decay_index_ex = { 0.1, 0.1, 0.1 };

    model->triangle_vec.resize(size);

    std::vector<int> indexes;
    indexes.resize(size);

    for (int i{ 0 }; i < size; ++i) { indexes[i] = i; }

    std::random_shuffle(indexes.begin(), indexes.end());

    eta::math::Quaternion q{ eta::math::Quaternion::Make(ETA__deg_to_rad(0),
                                                         { 0, 0, 1 }) };

    for (int i{ 0 }; i < size; ++i) {
        const eta::utility::STLFile::Triangle& stl_tri(stl_file.tris[i]);

        eta::render::Triangle* triangle{ new eta::render::Triangle{} };

        eta::math::Vector<3> vertex[3]{
            { stl_tri.vertex1x, stl_tri.vertex1y, stl_tri.vertex1z },
            { stl_tri.vertex2x, stl_tri.vertex2y, stl_tri.vertex2z },
            { stl_tri.vertex3x, stl_tri.vertex3y, stl_tri.vertex3z }
        };

        triangle->vertex[0] = q.rotate(vertex[0] * SCALE);
        triangle->vertex[1] = q.rotate(vertex[1] * SCALE);
        triangle->vertex[2] = q.rotate(vertex[2] * SCALE);

        model->triangle_vec[indexes[i]] = triangle;
    }

    return model;
}

void NormalizeModel(eta::render::Model* model) {
    eta::math::Vector<3> bbox_min{ ETA__inf, ETA__inf, ETA__inf };
    eta::math::Vector<3> bbox_max{ -ETA__inf, -ETA__inf, -ETA__inf };

    int num_of_triangles{ static_cast<int>(model->triangle_vec.size()) };

    for (int triangle_i{ 0 }; triangle_i < num_of_triangles; ++triangle_i) {
        bbox_min = eta::math::Vector<3>::min(
            eta::math::Vector<3>::min(
                eta::math::Vector<3>::min(
                    bbox_min, model->triangle_vec[triangle_i]->vertex[0]),
                model->triangle_vec[triangle_i]->vertex[1]),
            model->triangle_vec[triangle_i]->vertex[2]);

        bbox_max = eta::math::Vector<3>::max(
            eta::math::Vector<3>::max(
                eta::math::Vector<3>::max(
                    bbox_max, model->triangle_vec[triangle_i]->vertex[0]),
                model->triangle_vec[triangle_i]->vertex[1]),
            model->triangle_vec[triangle_i]->vertex[2]);
    }

    eta::math::Vector<3> center{ (bbox_min + bbox_max) / 2 };

    eta::num_t scale{ 30 / std::max(std::max(bbox_max.x - bbox_min.x,
                                             bbox_max.y - bbox_min.y),
                                    bbox_max.z - bbox_min.z) };

    for (int triangle_i{ 0 }; triangle_i < num_of_triangles; ++triangle_i) {
        ((model->triangle_vec[triangle_i]->vertex[0] -= center) *= scale) +=
            CENTER;
        ((model->triangle_vec[triangle_i]->vertex[1] -= center) *= scale) +=
            CENTER;
        ((model->triangle_vec[triangle_i]->vertex[2] -= center) *= scale) +=
            CENTER;
    }
}

void test_render() {
    eta::render::Camera camera;

    unsigned int width(1920);
    unsigned int height(1080);

    eta::math::Vector<3> origin{ -15, -15, 15 };
    eta::math::Vector<3> axis_f{ 15, 15, -15 };
    eta::math::Vector<3> axis_w;
    eta::math::Vector<3> axis_h;

    origin *= SCALE;
    axis_f *= SCALE;

    origin += CENTER;

    eta::render::Camera::GetAxisWH(axis_w, // dst_axis_w
                                   axis_h, //dst_axis_h
                                   width, // width
                                   height, // height
                                   axis_f, // axis_f
                                   ETA__deg_to_rad(140), // angle_of_view
                                   0 // angle_of_rotate
    );

    camera.set_width(width);
    camera.set_height(height);
    camera.set_origin(origin);
    camera.set_axis_f(axis_f);
    camera.set_axis_w(axis_w);
    camera.set_axis_h(axis_h);

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
    camera.set_model_manager(model_manager);

    eta::utility::STLFile stl_file("/home/eps/desktop/Jump_Pack.stl");
    // eta::utility::STLFile stl_file("/home/eps/desktop/Aquila_Shield.stl");

    eta::render::Model* model{ CreateModelFromSTLFile(stl_file) };
    NormalizeModel(model);
    model->SetupAllTriangles();

    model_manager->model_vec.push_back(model);

    model_manager->Synchronize();

    cudaDeviceSynchronize();

    eta::render::PointLight* point_light_0(
        (new eta::render::PointLight())
            ->set_origin(eta::math::Vector<3>{ 0, -30, 0 } + CENTER)
            // ->set_origin({ -15, -15, 15 })
            ->set_intensity({ SCALE * SCALE, SCALE * SCALE, SCALE * SCALE })
            ->set_range(1024));

    /*eta::render::PointLight* point_light_1(
        (new eta::render::PointLight())
            ->set_origin({ 0, -15, 15 })
            ->set_intensity({ 10000, 10000, 10000 })
            ->set_range(1024)); */

    eta::render::PointLight* point_light_2(
        (new eta::render::PointLight())
            ->set_origin({ 0, 0, 30 })
            ->set_intensity({ SCALE * SCALE, SCALE * SCALE, SCALE * SCALE })
            ->set_range(1024));

    std::vector<const eta::render::DivergentLight*> divergent_lights;
    divergent_lights.push_back(point_light_0);
    // divergent_lights.push_back(point_light_1);
    // divergent_lights.push_back(point_light_2);
    camera.set_divergent_lights(&divergent_lights);

    eta::Print("renderg begin\n");

    std::chrono::high_resolution_clock::time_point begin_time(
        std::chrono::high_resolution_clock::now());

    camera.Render();

    std::chrono::high_resolution_clock::time_point end_time(
        std::chrono::high_resolution_clock::now());

    eta::Print("renderg end\n");

    eta::Print("duration : ",
               std::chrono::duration_cast<std::chrono::milliseconds>(end_time -
                                                                     begin_time)
                       .count() /
                   1000.0,
               " s\n");

    camera.GenerateColorMap();

    // std::vector<eta::utility::RGB<float>> result(camera.result());

    /* std::vector<eta::utility::RGBA<unsigned char>> color;
    make_rgb_avg(color, result); */

    eta::utility::RGBAImage image;

    std::string render_dir("/home/eps/desktop/render/m");
    std::string date(get_date());
    std::string rgba_file(render_dir + "/" + "m-" + date + ".rgba");
    std::string img_file(render_dir + "/" + "m-" + date + ".png");

    std::string rgba_to_img_cmd(
        "python3 '/home/eps/desktop/Eta/utility/rgba.py' --rgba-to-img " +
        rgba_file + " " + img_file);

    image.set(width, height, camera.color());
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

int main() {
    eta::PrintFormat("seed: ", SetRandom(), '\n');
    test_render();
    cudaDeviceSynchronize();

    return 0;
}