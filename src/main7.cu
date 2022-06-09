#include <iostream>
// #include <fstream>
// #include <chrono>
// #include <string>
// #include <vector>

// #include <cuda.h>
// #include <cuda_runtime.h>
// #include <cuda_runtime_api.h>
// #include <math_constants.h>

// #include "utility/Array.cuh"
// #include "render/QuestBManager.cu"
// #include "render/QuestAManager.cu"
// #include "render/QuestBManager.cu"
// #include "render/TriangleSet.cu"
// #include "render/RayCasting.cu"
// #include "render/SolidTexture.cpp"
// #include "render/Renderer.cu"
// #include "render/RenderingTriangle.cu"

#include "define.cuh"
#include "utility/Matrix.cu"
#include "utility/RGBA.cuh"

// #include "utility/RGBAImage.cu"

// #include "TriangleBatch.cu"

// #include "Camera.cu"

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

eta::num_t RandomNum(eta::num_t lower, eta::num_t upper) {
    eta::num_t b(4096);
    return Random(lower * b, upper * b) / b;
}

eta::num_t RandomNormalNum() { return RandomNum(-256, 256); }

eta::num_t sq(eta::num_t x) { return x * x; }

void test_matrix() {
    eta::utility::Matrix<4, 4> m;

    int test_num(128);

    for (int test_i(0); test_i < test_num; ++test_i) {
        m.value[0][0] = RandomNormalNum();
        m.value[0][1] = RandomNormalNum();
        m.value[0][2] = RandomNormalNum();
        m.value[0][3] = RandomNormalNum();

        m.value[1][0] = RandomNormalNum();
        m.value[1][1] = RandomNormalNum();
        m.value[1][2] = RandomNormalNum();
        m.value[1][3] = RandomNormalNum();

        m.value[2][0] = RandomNormalNum();
        m.value[2][1] = RandomNormalNum();
        m.value[2][2] = RandomNormalNum();
        m.value[2][3] = RandomNormalNum();

        m.value[3][0] = RandomNormalNum();
        m.value[3][1] = RandomNormalNum();
        m.value[3][2] = RandomNormalNum();
        m.value[3][3] = RandomNormalNum();

        eta::utility::Matrix<4, 4> m_inv(m.inv());

        eta::utility::Matrix<4, 4> k(m * m_inv);

        bool b(k == eta::utility::Matrix<4, 4>::identity());

        if (!b) { eta::FormatPrint(b, "\n", k, "\n"); }
    }
}

int main() {
    eta::FormatPrint("seed: ", SetRandom(), '\n');

    test_matrix();

    // testing_func();

    // eta::utility::Vector3 vec;

    // eta::render::QuestAManager quest_a_manager;

    return 0;
}