#include <iostream>
#include <fstream>
#include <chrono>
#include <string>
#include <vector>

// #include <cuda.h>
// #include <cuda_runtime.h>
// #include <cuda_runtime_api.h>
// #include <math_constants.h>

// #include "render/QuestAManager.cu"
// #include "render/QuestBManager.cu"
// #include "render/TriangleSet.cu"
// #include "render/RayCasting.cu"

// #include "TriangleBatch.cu"

// #include "Camera.cu"

#include "Printer.cuh"

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

int main() {
    printf("seed: %d\n", SetRandom());

    // test_quat();

    // eta::Print(eta::NameGetter<eta::math::Matrix<2, 3>>::name(), "\n");

    std::string str("abc");

    // eta::Print(str, "\n");

    return 0;
}